import io
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dev_worker import cli, run_journal
from dev_worker.contract import MandateInput, MandateOutput
from dev_worker.github import GitHubRepository
from dev_worker.header import HeaderError
from dev_worker.instance import Instance, Repository, Tracker
from dev_worker.jira import JiraTracker
from dev_worker.repositories import RepositoryError
from dev_worker.tickets import Blocker, Ticket, TrackerError

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "examples" / "instance.yaml"
TORN_FILE = "journal.torn"
MAKE_TRACKER = cli.make_tracker
MAKE_REPOSITORY = cli.make_repository


def valid_input(**overrides: object) -> dict[str, object]:
    """Return a valid `deliver_ticket` input, with `overrides` applied."""
    data: dict[str, object] = {
        "contract_version": "0.1",
        "task_id": "task-1",
        "mandate": "deliver_ticket",
        "caller": "tests",
        "params": {"ticket_key": "DEMO-1"},
    }
    data.update(overrides)
    return data


@pytest.fixture
def instance(tmp_path: Path) -> Path:
    """Return a copy of the example instance file in a temporary directory."""
    return Path(shutil.copy(INSTANCE, tmp_path / "instance.yaml"))


@pytest.fixture(autouse=True)
def examples_stay_clean() -> Iterator[None]:
    """Fail a test that leaves a file in `examples/` it did not find there."""
    before = sorted(INSTANCE.parent.rglob("*"))
    yield
    assert sorted(INSTANCE.parent.rglob("*")) == before


class ReadyTracker:
    """A tracker whose every ticket is in the example `ready` status, with no blocker."""

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`, ready and unblocked."""
        return Ticket(key=key, status="To Do", labels=(), blockers=())

    def set_status(self, key: str, status: str) -> None:
        """Change nothing."""

    def check_access(self) -> None:
        """Grant access."""


class ReachableRepository:
    """A repository whose access check passes."""

    def check_access(self) -> None:
        """Grant access."""


@pytest.fixture(autouse=True)
def ready_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `run` read every ticket from a `ReadyTracker` instead of the real tracker."""
    monkeypatch.setattr(cli, "make_tracker", lambda tracker: ReadyTracker())


@pytest.fixture(autouse=True)
def reachable_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `health` check a `ReachableRepository` instead of the real repository."""
    monkeypatch.setattr(
        cli, "make_repository", lambda repository: ReachableRepository()
    )


def journal_lines(instance: Path) -> list[dict[str, object]]:
    """Return the parsed lines of the journal of `instance`, none when it is absent."""
    path = instance.with_suffix("") / run_journal.JOURNAL_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    argv: list[str],
    stdin: bytes = b"",
) -> dict[str, object]:
    """Call `main(argv)` with `stdin` as raw input; check exit 0 and return the parsed answer."""
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin)))

    code = cli.main(argv)

    assert code == 0
    out = capsysbinary.readouterr().out
    assert out.endswith(b"\n")
    return json.loads(out.decode("utf-8"))


def run_mandate(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    stdin: bytes,
    mandate: str = "deliver_ticket",
    instance: Path = INSTANCE,
) -> dict[str, object]:
    """Run `run <mandate> --instance <instance>` with `stdin` and return the parsed answer."""
    return run(
        monkeypatch,
        capsysbinary,
        ["run", mandate, "--instance", str(instance)],
        stdin,
    )


def encode(data: object) -> bytes:
    """Encode `data` as UTF-8 JSON."""
    return json.dumps(data).encode("utf-8")


def assert_failed(
    answer: dict[str, object], task_id: str | None, fragment: str
) -> None:
    """Check that `answer` is a failed output for `task_id` whose errors mention `fragment`."""
    assert answer["status"] == "failed"
    assert answer["task_id"] == task_id
    assert answer["summary"] == "The mandate could not be run."
    errors = answer["errors"]
    assert isinstance(errors, list)
    assert any(fragment in str(error) for error in errors), errors


def test_valid_ticket_key_completes_and_journals_the_run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["task_id"] == "task-1"
    assert answer["errors"] == []
    assert "Nothing was done" in str(answer["summary"])
    running, line = journal_lines(instance)
    assert running == {
        "task_id": "task-1",
        "ticket_key": "DEMO-1",
        "status": "running",
        "start": line["start"],
        "duration": None,
        "tokens": 0,
    }
    assert line.keys() == {
        "task_id",
        "ticket_key",
        "status",
        "start",
        "duration",
        "tokens",
    }
    assert line["task_id"] == "task-1"
    assert line["ticket_key"] == "DEMO-1"
    assert line["status"] == "completed"
    assert datetime.fromisoformat(str(line["start"])).utcoffset() == timedelta(0)
    duration = line["duration"]
    assert isinstance(duration, float) and duration >= 0
    assert line["tokens"] == 0
    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ({}, "params.ticket_key is missing."),
        ({"ticket_key": 12}, "params.ticket_key must be a string."),
        ({"ticket_key": "DEMO-01"}, "is not of the form DEMO-<number>."),
        ({"ticket_key": "DEMO-0"}, "is not of the form DEMO-<number>."),
        ({"ticket_key": "DEMO"}, "is not of the form DEMO-<number>."),
        ({"ticket_key": "DEMO-1\n"}, "is not of the form DEMO-<number>."),
        ({"ticket_key": "DEMO-1٣"}, "is not of the form DEMO-<number>."),
        ({"ticket_key": "OTHER-1"}, "is not in the tracker project 'DEMO'."),
        ({"ticket_key": "demo-1"}, "is not in the tracker project 'DEMO'."),
        ({"ticket_key": "DEMO-1-2"}, "is not in the tracker project 'DEMO'."),
        ({"ticket_key": "XDEMO-1"}, "is not in the tracker project 'DEMO'."),
    ],
)
def test_refused_ticket_key_fails_without_journaling(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    params: dict[str, object],
    fragment: str,
) -> None:
    stdin = encode(valid_input(params=params))

    answer = run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    assert_failed(answer, "task-1", fragment)
    assert not instance.with_suffix("").exists()


class StubTracker:
    """A tracker that answers every read with `answer`, raising it when it is an error.

    `reads` records each key read and `trackers` each instance tracker it was made for.
    """

    def __init__(self, answer: Ticket | TrackerError) -> None:
        """Keep `answer` for every later read."""
        self.answer = answer
        self.reads: list[str] = []
        self.trackers: list[Tracker] = []

    def make(self, tracker: Tracker) -> "StubTracker":
        """Record `tracker` and return this stub, in place of `cli.make_tracker`."""
        self.trackers.append(tracker)
        return self

    def read(self, key: str) -> Ticket:
        """Record `key`, then return or raise the answer."""
        self.reads.append(key)
        if isinstance(self.answer, TrackerError):
            raise self.answer
        return self.answer

    def set_status(self, key: str, status: str) -> None:
        """Change nothing."""

    def check_access(self) -> None:
        """Grant access."""


def serve(
    monkeypatch: pytest.MonkeyPatch, answer: Ticket | TrackerError
) -> StubTracker:
    """Make `run` read every ticket from a `StubTracker` answering `answer`."""
    stub = StubTracker(answer)
    monkeypatch.setattr(cli, "make_tracker", stub.make)
    return stub


def demo_ticket(status: str = "To Do", *blockers: Blocker) -> Ticket:
    """Return the ticket DEMO-1 in `status`, blocked by `blockers`."""
    return Ticket(key="DEMO-1", status=status, labels=(), blockers=blockers)


def test_run_reads_the_mandate_ticket_from_the_tracker_of_the_instance(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert stub.reads == ["DEMO-1"]
    assert [tracker.project for tracker in stub.trackers] == ["DEMO"]
    assert stub.trackers[0].token_env == "JIRA_TOKEN"


def test_refused_ticket_key_is_not_read_from_the_tracker(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    stdin = encode(valid_input(params={"ticket_key": "OTHER-1"}))

    answer = run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    assert_failed(answer, "task-1", "is not in the tracker project 'DEMO'.")
    assert stub.reads == []


@pytest.mark.parametrize("status", ["In Progress", "Done", "to do", "To Do "])
def test_ticket_not_in_the_ready_status_fails_naming_both_statuses(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    status: str,
) -> None:
    serve(monkeypatch, demo_ticket(status))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer, "task-1", f"Ticket 'DEMO-1' is in status {status!r}, not 'To Do'."
    )
    assert not instance.with_suffix("").exists()


def test_ticket_whose_blockers_are_all_done_completes(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    serve(
        monkeypatch,
        demo_ticket("To Do", Blocker("DEMO-2", "Done"), Blocker("DEMO-3", "Done")),
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "completed",
    ]


def test_ticket_with_an_open_blocker_fails_naming_only_the_open_one(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    serve(
        monkeypatch,
        demo_ticket(
            "To Do", Blocker("DEMO-2", "Done"), Blocker("DEMO-3", "In Progress")
        ),
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "DEMO-3")
    assert answer["errors"] == [
        "Ticket 'DEMO-1' is blocked by DEMO-3 ('In Progress'), not in status 'Done'."
    ]
    assert not instance.with_suffix("").exists()


def test_ticket_with_several_open_blockers_fails_naming_each_in_order(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    serve(
        monkeypatch,
        demo_ticket(
            "To Do",
            Blocker("DEMO-4", "Blocked"),
            Blocker("DEMO-2", "Done"),
            Blocker("DEMO-3", "done"),
        ),
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' is blocked by DEMO-4 ('Blocked'), DEMO-3 ('done'), "
        "not in status 'Done'.",
    )


def test_ticket_the_tracker_cannot_read_fails_naming_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    serve(monkeypatch, TrackerError("twg failed while reading DEMO-1: timeout"))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' cannot be read from the tracker: "
        "twg failed while reading DEMO-1: timeout",
    )
    assert not instance.with_suffix("").exists()


def test_ticket_not_ready_is_refused_before_the_lock_and_leaves_the_journal(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)
    directory = instance.with_suffix("")
    journal = (directory / run_journal.JOURNAL_FILE).read_bytes()
    lock = directory / run_journal.LOCK_FILE
    lock.write_bytes(b"held by another run")
    files = sorted(directory.iterdir())
    serve(monkeypatch, demo_ticket("In Progress"))

    answer = run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input(task_id="task-2")),
        instance=instance,
    )

    assert_failed(answer, "task-2", "is in status 'In Progress', not 'To Do'.")
    assert lock.read_bytes() == b"held by another run"
    assert (directory / run_journal.JOURNAL_FILE).read_bytes() == journal
    assert sorted(directory.iterdir()) == files


def test_make_tracker_returns_a_jira_tracker() -> None:
    tracker = Tracker(
        type="jira",
        site="example.atlassian.net",
        user="worker@example.com",
        project="DEMO",
        token_env="JIRA_TOKEN",
    )

    assert isinstance(MAKE_TRACKER(tracker), JiraTracker)


def test_run_reads_through_jira_with_the_token_variable_of_the_instance(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    monkeypatch.setattr(cli, "make_tracker", MAKE_TRACKER)
    monkeypatch.delenv("JIRA_TOKEN", raising=False)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' cannot be read from the tracker: the variable JIRA_TOKEN, "
        "which should hold the Jira token, is not set",
    )
    assert not instance.with_suffix("").exists()


def test_repeated_task_id_fails_and_is_journaled_once(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stdin = encode(valid_input())
    run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    answer = run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "completed",
    ]


def test_other_task_id_after_a_run_is_journaled(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input(task_id="task-2")),
        instance=instance,
    )

    assert answer["status"] == "completed"
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "task-1",
        "task-1",
        "task-2",
        "task-2",
    ]


def test_journal_records_the_status_the_mandate_returns(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def refuse(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        return cli.failed(mandate.task_id, "The ticket cannot be delivered.")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", refuse)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "The ticket cannot be delivered.")
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "failed",
    ]


def test_journal_records_the_duration_of_the_mandate_in_seconds(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    clock = [100.0, 102.5]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock.pop(0))

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    running, final = journal_lines(instance)
    assert running["duration"] is None
    assert final["duration"] == 2.5


def test_held_lock_fails_leaving_it_untouched(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    lock = directory / run_journal.LOCK_FILE
    lock.write_bytes(b"held by another run")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert lock.read_bytes() == b"held by another run"
    assert journal_lines(instance) == []


def test_lock_holds_the_run_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def peek(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        lock = instance.with_suffix("") / run_journal.LOCK_FILE
        seen.append(json.loads(lock.read_text(encoding="utf-8")))
        return cli.deliver_ticket(mandate, resolved)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    (record,) = seen
    assert record.keys() == {"task_id", "ticket_key", "start", "pid"}
    assert record["pid"] == os.getpid()
    assert record["task_id"] == "task-1"
    assert record["ticket_key"] == "DEMO-1"
    assert record["start"] == journal_lines(instance)[0]["start"]


def test_lock_is_released_when_the_mandate_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def explode(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)

    with pytest.raises(RuntimeError, match="boom"):
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()
    assert [line["status"] for line in journal_lines(instance)] == ["running"]


def test_instance_directory_that_cannot_be_created_fails_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.write_bytes(b"a file where the directory should be")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer, "task-1", f"Instance directory '{directory}' cannot be created: "
    )


def test_instance_file_without_a_suffix_fails_creating_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = Path(shutil.copy(INSTANCE, tmp_path / "acme"))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "has no suffix")
    assert list(tmp_path.iterdir()) == [instance]


def test_lock_that_cannot_be_written_fails_naming_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def denied(directory: Path, record: object) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "take_lock", denied)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "cannot be written: Permission denied")


def test_journal_that_cannot_be_read_fails_and_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    (directory / run_journal.JOURNAL_FILE).mkdir(parents=True)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer, "task-1", f"Journal in '{directory}' cannot be read or repaired: "
    )
    assert not (directory / run_journal.LOCK_FILE).exists()


def test_journal_with_a_line_that_is_not_an_object_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.JOURNAL_FILE).write_bytes(b"[1]\n")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "line 1 is not a JSON object.")


def test_journal_with_a_terminated_line_that_is_not_json_fails_wherever_it_is(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    journal = directory / run_journal.JOURNAL_FILE
    journal.write_bytes(b'{"task_id": "a"}\n{"task_id": "b", "sta\n{"task_id": "c"}\n')
    before = journal.read_bytes()

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "line 2 is not valid JSON")
    assert journal.read_bytes() == before
    assert not (directory / TORN_FILE).exists()


def test_journal_with_a_line_nested_too_deeply_to_parse_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    journal = directory / run_journal.JOURNAL_FILE
    journal.write_bytes(b'{"task_id": "a"}\n' + b"[" * 100_000 + b"]" * 100_000 + b"\n")
    before = journal.read_bytes()

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "line 2 is not valid JSON")
    assert journal.read_bytes() == before
    assert not (directory / run_journal.LOCK_FILE).exists()


TORN_TAILS = [
    pytest.param(b'{"task_id": "t1", "sta', id="cut-json"),
    pytest.param(b'{"task_id": "t\xc3', id="cut-utf8"),
    pytest.param(b"\xff\xfe", id="not-utf8"),
    pytest.param(b"[1]", id="json-not-an-object"),
]


def write_torn_journal(instance: Path, tail: bytes, *entries: object) -> Path:
    """Write `entries` as complete journal lines followed by `tail` unterminated."""
    directory = write_journal(instance, *entries)
    with (directory / run_journal.JOURNAL_FILE).open("ab") as journal:
        journal.write(tail)
    return directory


@pytest.mark.parametrize("tail", TORN_TAILS)
def test_run_after_a_torn_journal_line_completes_and_journals_after_the_whole_lines(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    tail: bytes,
) -> None:
    directory = write_torn_journal(instance, tail, {"task_id": "a"}, {"task_id": "b"})

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    content = (directory / run_journal.JOURNAL_FILE).read_bytes()
    assert content.startswith(b'{"task_id": "a"}\n{"task_id": "b"}\n')
    assert tail not in content
    assert all(line.endswith(b"\n") for line in content.splitlines(keepends=True))
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "a",
        "b",
        "task-1",
        "task-1",
    ]


def test_run_after_a_journal_holding_only_a_torn_line_journals_the_run_alone(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_torn_journal(instance, b'{"task_id": "t1", "sta')

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "task-1",
        "task-1",
    ]


@pytest.mark.parametrize("tail", TORN_TAILS)
def test_run_sets_the_torn_journal_line_aside_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    tail: bytes,
) -> None:
    directory = write_torn_journal(instance, tail, {"task_id": "a"})
    torn = directory / TORN_FILE

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert torn.is_file()
    assert torn.read_bytes() == tail + b"\n"


def test_two_torn_journal_lines_are_set_aside_one_after_the_other(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "x", "st', {"task_id": "a"})
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)
    with (directory / run_journal.JOURNAL_FILE).open("ab") as journal:
        journal.write(b'{"task_id": "y"')

    answer = run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input(task_id="task-2")),
        instance=instance,
    )

    assert answer["status"] == "completed"
    assert (directory / TORN_FILE).read_bytes() == (
        b'{"task_id": "x", "st\n{"task_id": "y"\n'
    )
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "a",
        "task-1",
        "task-1",
        "task-2",
        "task-2",
    ]


def test_run_after_a_whole_unterminated_journal_line_keeps_it_and_terminates_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "b"}', {"task_id": "a"})

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    content = (directory / run_journal.JOURNAL_FILE).read_bytes()
    assert content.startswith(b'{"task_id": "a"}\n{"task_id": "b"}\n')
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "a",
        "b",
        "task-1",
        "task-1",
    ]
    assert not (directory / TORN_FILE).exists()


def test_task_of_a_whole_unterminated_journal_line_is_not_run_again(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "task-1"}')

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert (directory / run_journal.JOURNAL_FILE).read_bytes() == (
        b'{"task_id": "task-1"}\n'
    )
    assert not (directory / TORN_FILE).exists()


def test_run_after_a_whole_journal_sets_nothing_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_journal(instance, {"task_id": "a"})

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert not (directory / TORN_FILE).exists()


def test_journal_that_cannot_be_written_fails_naming_the_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def full(directory: Path, entry: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "append_entry", full)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "cannot be written: No space left on device")
    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()


def test_mandate_is_not_run_when_its_running_line_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    called: list[str] = []

    def record(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        called.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved)

    def full(directory: Path, entry: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", record)
    monkeypatch.setattr(cli, "append_entry", full)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    directory = instance.with_suffix("")
    assert_failed(
        answer,
        "task-1",
        f"Journal in '{directory}' cannot be written: No space left on device",
    )
    assert called == []
    assert journal_lines(instance) == []


def fail_on_final_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `append_entry` write the `running` line, then fail on every other line."""
    real_append_entry = cli.append_entry

    def full_after_running(directory: Path, entry: dict[str, object]) -> None:
        if entry["status"] != "running":
            raise OSError(28, "No space left on device")
        real_append_entry(directory, entry)

    monkeypatch.setattr(cli, "append_entry", full_after_running)


def test_final_line_that_cannot_be_written_keeps_the_output_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    fail_on_final_line(monkeypatch)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    directory = instance.with_suffix("")
    assert answer["status"] == "completed"
    assert "Nothing was done" in str(answer["summary"])
    assert answer["errors"] == [
        f"Journal in '{directory}' cannot be written: No space left on device"
    ]
    assert [line["status"] for line in journal_lines(instance)] == ["running"]
    assert not (directory / run_journal.LOCK_FILE).exists()


def test_final_line_that_cannot_be_written_follows_the_errors_of_the_mandate(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def refuse(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        return cli.failed(mandate.task_id, "The ticket cannot be delivered.")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", refuse)
    fail_on_final_line(monkeypatch)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    directory = instance.with_suffix("")
    assert answer["status"] == "failed"
    assert answer["errors"] == [
        "The ticket cannot be delivered.",
        f"Journal in '{directory}' cannot be written: No space left on device",
    ]


def test_task_whose_final_line_could_not_be_written_is_not_run_again(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    with monkeypatch.context() as failing:
        fail_on_final_line(failing)
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert [line["status"] for line in journal_lines(instance)] == ["running"]


def test_task_whose_mandate_raised_is_not_run_again(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def explode(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        raise RuntimeError("boom")

    with monkeypatch.context() as raising:
        raising.setitem(cli.MANDATES, "deliver_ticket", explode)
        with pytest.raises(RuntimeError, match="boom"):
            run_mandate(
                monkeypatch, capsysbinary, encode(valid_input()), instance=instance
            )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert [line["status"] for line in journal_lines(instance)] == ["running"]


def test_lock_that_cannot_be_removed_is_reported_with_the_output(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def stuck(path: Path, *record: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "release_lock", stuck)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    lock = instance.with_suffix("") / run_journal.LOCK_FILE
    assert answer["status"] == "completed"
    assert answer["errors"] == [
        f"Lock file '{lock}' cannot be removed: Permission denied"
    ]


def test_run_leaves_the_lock_of_the_run_that_took_it_over_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    successor = encode({"task_id": "successor", "ticket_key": "DEMO-2"})
    lock = instance.with_suffix("") / run_journal.LOCK_FILE

    def taken_over_while_running(
        mandate: MandateInput, resolved: Instance
    ) -> MandateOutput:
        lock.write_bytes(successor)
        return cli.deliver_ticket(mandate, resolved)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", taken_over_while_running)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert run_journal.read_lock(lock.parent) == {
        "task_id": "successor",
        "ticket_key": "DEMO-2",
    }


def test_input_that_is_not_utf8_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run_mandate(monkeypatch, capsysbinary, b"\xff\xfe")

    assert_failed(answer, None, "Input is not valid UTF-8")


def test_malformed_json_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run_mandate(monkeypatch, capsysbinary, b'{"task_id": ')

    assert_failed(answer, None, "Input is not valid JSON")


def test_json_nested_too_deeply_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run_mandate(monkeypatch, capsysbinary, b"[" * 100_000)

    assert_failed(answer, None, "Input JSON is nested too deeply.")


def test_json_that_is_not_an_object_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run_mandate(monkeypatch, capsysbinary, b"[1, 2]")

    assert_failed(answer, None, "Input must be a JSON object.")


def test_lone_surrogate_elsewhere_fails_keeping_the_task_id(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(caller="\ud800"))

    answer = run_mandate(monkeypatch, capsysbinary, stdin)

    assert_failed(answer, "task-1", "cannot be encoded as UTF-8")


def test_lone_surrogate_in_task_id_fails_without_a_task_id(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(task_id="\ud800"))

    answer = run_mandate(monkeypatch, capsysbinary, stdin)

    assert_failed(answer, None, "cannot be encoded as UTF-8")


def test_unsupported_contract_version_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(contract_version="9.9"))

    answer = run_mandate(monkeypatch, capsysbinary, stdin)

    assert_failed(answer, "task-1", "Unsupported contract_version '9.9'")


def test_input_breaking_the_contract_fails_with_a_non_string_task_id(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(task_id=7, params=[]))

    answer = run_mandate(monkeypatch, capsysbinary, stdin)

    assert_failed(answer, None, "task_id: Input should be a valid string.")
    assert_failed(answer, None, "params: Input should be a valid dictionary.")


def test_mandate_differing_from_the_command_line_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(mandate="other"))

    answer = run_mandate(monkeypatch, capsysbinary, stdin)

    assert_failed(answer, "task-1", "differs from 'other' in the input")


def test_unknown_mandate_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    stdin = encode(valid_input(mandate="frame_ticket"))

    answer = run_mandate(monkeypatch, capsysbinary, stdin, mandate="frame_ticket")

    assert_failed(answer, "task-1", "Unknown mandate 'frame_ticket'.")


def test_worker_file_with_an_invalid_header_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    def broken(path: Path) -> object:
        raise HeaderError("front matter is not valid YAML")

    monkeypatch.setattr(cli, "load_worker_file", broken)

    answer = run_mandate(monkeypatch, capsysbinary, encode(valid_input()))

    assert_failed(
        answer, "task-1", "WORKER.md does not load: front matter is not valid YAML"
    )


def test_unreadable_worker_file_fails(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    def missing(path: Path) -> object:
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(cli, "load_worker_file", missing)

    answer = run_mandate(monkeypatch, capsysbinary, encode(valid_input()))

    assert_failed(answer, "task-1", "WORKER.md cannot be read")


def test_invalid_instance_fails_naming_the_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance.yaml"
    instance.write_text("tracker: [\n", encoding="utf-8")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", f"Instance file '{instance}': ")


def test_missing_instance_fails_naming_the_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "missing.yaml"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", f"Instance file '{instance}' cannot be read")


def test_instance_not_in_utf8_fails_naming_the_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance.yaml"
    instance.write_bytes("tracker:\n  project: Café\n".encode("latin-1"))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", f"Instance file '{instance}': ")


def test_instance_nested_too_deeply_fails_naming_the_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "instance.yaml"
    instance.write_text("tracker: " + "[" * 5000 + "\n", encoding="utf-8")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", f"Instance file '{instance}': ")


def test_instance_path_with_a_lone_surrogate_fails_naming_it_escaped(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "caf\udce9.yaml"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "caf\\udce9.yaml")


NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
EMPTY_TODAY = {
    "common": {
        "current": [],
        "recent": [],
        "pending": [],
        "cost": {"day": "2026-11-05", "runs": 0, "tokens": 0},
    },
    "domain": {},
}


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Freeze the clock the command line reads at `NOW`."""
    clock = SimpleNamespace(
        now=lambda tz=None: NOW.astimezone(tz), fromisoformat=datetime.fromisoformat
    )
    monkeypatch.setattr(cli, "datetime", clock)
    return NOW


def state(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> dict[str, object]:
    """Run `state --instance <instance>` and return the parsed answer."""
    return run(monkeypatch, capsysbinary, ["state", "--instance", str(instance)])


def common(answer: dict[str, object]) -> dict[str, object]:
    """Return the `common` part of a state answer."""
    part = answer["common"]
    assert isinstance(part, dict)
    return part


def write_journal(instance: Path, *entries: object) -> Path:
    """Write `entries` as the JSON lines of the journal of `instance`; return its directory."""
    directory = instance.with_suffix("")
    directory.mkdir(exist_ok=True)
    (directory / run_journal.JOURNAL_FILE).write_bytes(
        b"".join(json.dumps(entry).encode("utf-8") + b"\n" for entry in entries)
    )
    return directory


def snapshot(root: Path) -> dict[Path, bytes | str | None]:
    """Return every path under `root` with its content.

    A symbolic link maps to its target as text, a directory to `None`.
    """
    return {
        path: str(path.readlink())
        if path.is_symlink()
        else None
        if path.is_dir()
        else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.usefixtures("frozen_now")
def test_state_of_an_instance_never_run_is_empty_with_today_at_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    assert state(monkeypatch, capsysbinary, instance) == EMPTY_TODAY


@pytest.mark.usefixtures("frozen_now")
def test_state_of_an_instance_path_without_a_suffix_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / run_journal.LOCK_FILE).write_bytes(b"{}")

    assert state(monkeypatch, capsysbinary, tmp_path / "acme") == EMPTY_TODAY


def test_state_day_is_the_current_utc_day(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    before = datetime.now(UTC).date().isoformat()
    cost = common(state(monkeypatch, capsysbinary, instance))["cost"]
    after = datetime.now(UTC).date().isoformat()

    assert isinstance(cost, dict)
    assert cost["day"] in {before, after}


@pytest.mark.usefixtures("frozen_now")
def test_state_reports_the_run_holding_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    record = {"task_id": "t", "ticket_key": "DEMO-1", "start": NOW.isoformat()}
    directory.mkdir()
    run_journal.take_lock(directory, record)

    assert common(state(monkeypatch, capsysbinary, instance))["current"] == [record]


@pytest.mark.parametrize("content", [b"", b"held by another run", b"[1]"])
def test_state_reports_a_run_of_unknown_details_for_a_lock_that_is_not_an_object(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    content: bytes,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.LOCK_FILE).write_bytes(content)

    assert common(state(monkeypatch, capsysbinary, instance))["current"] == [{}]


def test_state_lists_the_last_ten_runs_most_recent_first(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    entries = [{"task_id": f"task-{n}", "status": "completed"} for n in range(12)]
    write_journal(instance, *entries)

    recent = common(state(monkeypatch, capsysbinary, instance))["recent"]

    assert recent == entries[:1:-1]


def test_state_skips_journal_lines_that_are_not_objects(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_journal(instance, {"task_id": "t"})
    with (directory / run_journal.JOURNAL_FILE).open("ab") as journal:
        journal.write(b"{\n\xff\n[1]\n")

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"task_id": "t"}
    ]


@pytest.mark.usefixtures("frozen_now")
def test_state_cost_counts_the_runs_started_on_the_current_utc_day(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    todays = [
        {"start": "2026-11-05T00:00:00+00:00", "tokens": 5},
        {"start": "2026-11-04T23:30:00-02:00", "tokens": 3},
        {"start": "2026-11-05T23:59:59+00:00", "tokens": True},
        {"start": "2026-11-05T09:00:00+00:00", "tokens": "7"},
        {"start": "2026-11-05T10:00:00+00:00", "tokens": 1.5},
        {"start": "2026-11-05T11:00:00+00:00"},
    ]
    others = [
        {"start": "2026-11-05T01:00:00+02:00", "tokens": 100},
        {"start": "2026-11-06T00:00:00+00:00", "tokens": 100},
        {"start": "2026-11-04T23:59:59+00:00", "tokens": 100},
        {"start": "2026-11-05T10:00:00", "tokens": 100},
        {"start": "today", "tokens": 100},
        {"start": 20261105, "tokens": 100},
        {"tokens": 100},
    ]
    write_journal(instance, *todays, *others)

    cost = common(state(monkeypatch, capsysbinary, instance))["cost"]

    assert cost == {"day": "2026-11-05", "runs": 6, "tokens": 8}


@pytest.mark.usefixtures("frozen_now")
def test_state_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_journal(instance, {"task_id": "t", "start": NOW.isoformat()})
    (directory / run_journal.LOCK_FILE).write_bytes(b"")
    before = snapshot(instance.parent)

    state(monkeypatch, capsysbinary, instance)

    assert snapshot(instance.parent) == before


def test_state_of_an_instance_never_run_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    before = snapshot(instance.parent)

    state(monkeypatch, capsysbinary, instance)

    assert snapshot(instance.parent) == before


@pytest.mark.usefixtures("frozen_now")
def test_state_after_a_run_lists_it_and_counts_it_today(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = common(state(monkeypatch, capsysbinary, instance))

    _, entry = journal_lines(instance)
    assert entry["status"] == "completed"
    assert answer["current"] == []
    assert answer["recent"] == [entry]
    assert answer["cost"] == {"day": "2026-11-05", "runs": 1, "tokens": 0}


@pytest.mark.usefixtures("frozen_now")
def test_state_answers_while_a_run_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    seen: list[dict[str, object]] = []

    def peek(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        seen.append(cli.read_state(instance).model_dump(mode="json"))
        return cli.deliver_ticket(mandate, resolved)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    (during,) = seen
    running, _ = journal_lines(instance)
    assert running["status"] == "running"
    assert common(during)["current"] == [
        {
            "task_id": "task-1",
            "ticket_key": "DEMO-1",
            "start": NOW.isoformat(),
            "pid": os.getpid(),
        }
    ]
    assert common(during)["recent"] == [running]


INTERRUPTED = "The run stopped before its final journal line was written."


def test_state_lists_each_task_once_by_its_last_line_in_the_order_of_its_first(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(
        instance,
        {"task_id": "a", "status": "running"},
        {"task_id": "b", "status": "running"},
        {"task_id": "b", "status": "failed"},
        {"task_id": "a", "status": "completed"},
    )

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"task_id": "b", "status": "failed"},
        {"task_id": "a", "status": "completed"},
    ]


def test_state_shows_a_task_left_running_without_the_lock_as_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(
        instance, {"task_id": "a", "ticket_key": "DEMO-1", "status": "running"}
    )

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {
            "task_id": "a",
            "ticket_key": "DEMO-1",
            "status": "failed",
            "errors": [INTERRUPTED],
        }
    ]


def test_state_shows_the_line_a_run_left_before_its_body_returned_as_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    left = {
        "task_id": "a",
        "ticket_key": "DEMO-1",
        "status": "running",
        "start": NOW.isoformat(),
        "duration": None,
        "tokens": 0,
    }
    write_journal(instance, left)

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        left | {"status": "failed", "errors": [INTERRUPTED]}
    ]


@pytest.mark.parametrize(
    "lock",
    [
        pytest.param(b"held by another run", id="unknown-details"),
        pytest.param(b'{"pid": 1}', id="no-task-id"),
        pytest.param(b'{"task_id": "a"}', id="same-task"),
    ],
)
def test_state_keeps_a_task_running_while_the_lock_may_be_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    lock: bytes,
) -> None:
    directory = write_journal(
        instance,
        {"task_id": "a", "status": "running"},
        {"task_id": "b", "status": "running"},
        {"task_id": "b", "status": "completed"},
    )
    (directory / run_journal.LOCK_FILE).write_bytes(lock)

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"task_id": "b", "status": "completed"},
        {"task_id": "a", "status": "running"},
    ]


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("status", ["running", "waiting"])
def test_state_keeps_the_status_a_finished_run_returned_while_the_lock_is_free(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    status: str,
) -> None:
    def still_going(mandate: MandateInput, resolved: Instance) -> MandateOutput:
        output = cli.deliver_ticket(mandate, resolved)
        return output.model_copy(update={"status": status})

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", still_going)
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = common(state(monkeypatch, capsysbinary, instance))

    _, entry = journal_lines(instance)
    assert entry["status"] == status
    assert answer["recent"] == [entry]


def test_state_shows_a_task_left_running_as_interrupted_while_another_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_journal(
        instance,
        {"task_id": "a", "status": "running"},
        {"task_id": "b", "status": "running"},
    )
    (directory / run_journal.LOCK_FILE).write_bytes(b'{"task_id": "b"}')

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"task_id": "b", "status": "running"},
        {"task_id": "a", "status": "failed", "errors": [INTERRUPTED]},
    ]


def test_state_keeps_running_a_task_that_takes_the_lock_while_state_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    running = {"task_id": "a", "status": "running"}
    real_read_entries = cli.read_entries

    def run_starts_first(read_from: Path) -> list[dict[str, object]]:
        if not (read_from / run_journal.LOCK_FILE).exists():
            run_journal.take_lock(read_from, {"task_id": "a"})
            run_journal.append_entry(read_from, running)
        return real_read_entries(read_from)

    monkeypatch.setattr(cli, "read_entries", run_starts_first)

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [running]


def test_state_takes_each_line_without_a_string_task_id_as_a_task_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(
        instance,
        {"task_id": None, "status": "completed"},
        {"task_id": None, "status": "failed"},
        {"task_id": 1, "status": "completed"},
        {"task_id": 1, "status": "failed"},
        {"status": "running"},
    )

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"status": "failed", "errors": [INTERRUPTED]},
        {"task_id": 1, "status": "failed"},
        {"task_id": 1, "status": "completed"},
        {"task_id": None, "status": "failed"},
        {"task_id": None, "status": "completed"},
    ]


@pytest.mark.usefixtures("frozen_now")
def test_state_cost_counts_tasks_with_the_tokens_of_their_last_line(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    start = "2026-11-05T09:00:00+00:00"
    write_journal(
        instance,
        {"task_id": "a", "status": "running", "start": start, "tokens": 0},
        {"task_id": "b", "status": "running", "start": start, "tokens": 0},
        {"task_id": "a", "status": "completed", "start": start, "tokens": 4},
        {"task_id": "b", "status": "failed", "start": start, "tokens": 3},
    )

    cost = common(state(monkeypatch, capsysbinary, instance))["cost"]

    assert cost == {"day": "2026-11-05", "runs": 2, "tokens": 7}


def test_health_is_ok(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run(monkeypatch, capsysbinary, ["health", "--instance", str(INSTANCE)])

    assert answer == {"status": "ok", "reason": None, "active": True}


def health(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> dict[str, object]:
    """Run `health --instance <instance>` and return the parsed answer."""
    return run(monkeypatch, capsysbinary, ["health", "--instance", str(instance)])


def assert_down(answer: dict[str, object], reason_start: str) -> None:
    """Check that `answer` is down, active, with a reason starting with `reason_start`."""
    assert answer["status"] == "down"
    assert answer["active"] is True
    reason = answer["reason"]
    assert isinstance(reason, str)
    assert reason.startswith(reason_start)


def hold_lock(instance: Path, content: bytes) -> Path:
    """Write `content` as the lock of the directory of `instance`; return the lock path."""
    directory = instance.with_suffix("")
    directory.mkdir(exist_ok=True)
    lock = directory / run_journal.LOCK_FILE
    lock.write_bytes(content)
    return lock


def lock_started(seconds_ago: float) -> bytes:
    """Return a lock record whose run started `seconds_ago` seconds before `NOW`."""
    start = (NOW - timedelta(seconds=seconds_ago)).isoformat()
    return json.dumps({"task_id": "t", "ticket_key": "DEMO-1", "start": start}).encode()


def test_health_of_an_instance_run_before_is_ok(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(instance, {"task_id": "t"})

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {"status": "ok", "reason": None, "active": True}


def test_health_of_a_relative_instance_path_is_ok(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    monkeypatch.chdir(instance.parent)

    answer = health(monkeypatch, capsysbinary, Path(instance.name))

    assert answer["status"] == "ok"


def test_health_is_down_when_the_worker_file_does_not_load(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def broken(path: Path) -> object:
        raise HeaderError("front matter is not valid YAML")

    monkeypatch.setattr(cli, "load_worker_file", broken)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "down",
        "reason": "WORKER.md does not load: front matter is not valid YAML",
        "active": True,
    }


def test_health_is_down_when_the_worker_file_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def missing(path: Path) -> object:
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(cli, "load_worker_file", missing)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "WORKER.md cannot be read: No such file or directory")


def test_health_is_down_when_the_instance_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    instance.write_text("tracker: [\n", encoding="utf-8")

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, f"Instance file '{instance}': ")


def test_health_is_down_when_the_instance_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    instance = tmp_path / "missing.yaml"

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, f"Instance file '{instance}' cannot be read")


def test_health_names_an_instance_path_with_a_lone_surrogate_escaped(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    answer = health(monkeypatch, capsysbinary, tmp_path / "caf\udce9.yaml")

    assert_down(answer, "Instance file '")
    assert "caf\\udce9.yaml" in str(answer["reason"])


def test_health_is_down_when_the_instance_path_has_no_suffix(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    bare = instance.rename(instance.with_suffix(""))

    answer = health(monkeypatch, capsysbinary, bare)

    assert answer["reason"] == f"Instance file '{bare}' has no suffix."
    assert_down(answer, "Instance file")


@pytest.mark.parametrize("exists", [True, False])
def test_health_is_down_when_the_instance_directory_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    exists: bool,
) -> None:
    directory = instance.with_suffix("")
    if exists:
        directory.mkdir()
    checked: list[tuple[Path, int]] = []

    def refused(path: Path, mode: int) -> bool:
        checked.append((path, mode))
        return False

    monkeypatch.setattr(cli.os, "access", refused)

    answer = health(monkeypatch, capsysbinary, instance)

    nearest = directory if exists else instance.parent
    assert checked == [(nearest, os.W_OK | os.X_OK)]
    assert answer["reason"] == (
        f"Instance directory '{directory}' cannot be written: "
        f"'{nearest}' is not a writable directory."
    )
    assert_down(answer, "Instance directory")


def test_health_is_down_when_the_instance_directory_is_a_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    instance.with_suffix("").write_bytes(b"")

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, f"Instance directory '{instance.with_suffix('')}'")


def refuse_journal(monkeypatch: pytest.MonkeyPatch, instance: Path) -> Path:
    """Make `os.access` refuse the journal of `instance` only; return the journal path."""
    journal = instance.with_suffix("") / run_journal.JOURNAL_FILE
    allowed = os.access

    def access(path: Path, mode: int) -> bool:
        return Path(path) != journal and allowed(path, mode)

    monkeypatch.setattr(cli.os, "access", access)
    return journal


def test_health_is_down_when_the_journal_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(instance, {"task_id": "t"})
    journal = refuse_journal(monkeypatch, instance)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


def test_health_is_down_when_the_guard_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(instance, {"task_id": "t"})
    guard = instance.with_suffix("") / run_journal.GUARD_FILE
    guard.touch()
    allowed = os.access

    def access(path: Path, mode: int) -> bool:
        return Path(path) != guard and allowed(path, mode)

    monkeypatch.setattr(cli.os, "access", access)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(guard) in str(answer["reason"])


def test_health_is_ok_when_the_journal_does_not_exist_yet(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    instance.with_suffix("").mkdir()
    refuse_journal(monkeypatch, instance)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {"status": "ok", "reason": None, "active": True}


def assert_degraded_naming(answer: dict[str, object], path: Path) -> None:
    """Check that `answer` is degraded, active, with a reason naming `path`."""
    assert answer["status"] == "degraded"
    assert answer["active"] is True
    assert str(path) in str(answer["reason"])


def test_health_is_degraded_while_torn_journal_lines_are_set_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_journal(instance, {"task_id": "t"})
    torn = directory / TORN_FILE
    torn.write_bytes(b'{"task_id": "x", "st\n')

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, torn)


def test_health_is_degraded_when_the_journal_ends_with_a_torn_line(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "x", "st', {"task_id": "t"})

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, directory / run_journal.JOURNAL_FILE)


def test_health_sets_no_torn_journal_line_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_torn_journal(instance, b'{"task_id": "x", "st', {"task_id": "t"})
    before = snapshot(instance.parent)

    health(monkeypatch, capsysbinary, instance)

    assert snapshot(instance.parent) == before


def test_health_is_degraded_when_the_journal_ends_with_a_torn_line_beside_a_torn_file(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    torn.write_bytes(b'{"task_id": "x", "st\n')

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, torn)


def make_torn_directory(torn: Path) -> None:
    """Create `torn` as a directory instead of a regular file."""
    torn.mkdir()


def make_torn_file(torn: Path) -> None:
    """Create `torn` as a regular file holding one set-aside line."""
    torn.write_bytes(b'{"task_id": "x", "st\n')


def refuse_nothing(monkeypatch: pytest.MonkeyPatch, torn: Path) -> None:
    """Leave `torn` reachable as it is."""


def refuse_reading(monkeypatch: pytest.MonkeyPatch, torn: Path) -> None:
    """Make every read or open of `torn` raise `PermissionError`."""
    opened = Path.open
    read = Path.read_bytes

    def open_refusing_torn(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == torn:
            raise PermissionError(13, "Permission denied", str(path))
        return opened(path, *args, **kwargs)

    def read_refusing_torn(path: Path) -> bytes:
        if path == torn:
            raise PermissionError(13, "Permission denied", str(path))
        return read(path)

    monkeypatch.setattr(Path, "open", open_refusing_torn)
    monkeypatch.setattr(Path, "read_bytes", read_refusing_torn)


def refuse_writing(monkeypatch: pytest.MonkeyPatch, torn: Path) -> None:
    """Make `cli.os.access` refuse `torn` only."""
    allowed = os.access

    def access(path: Path, mode: int) -> bool:
        return Path(path) != torn and allowed(path, mode)

    monkeypatch.setattr(cli.os, "access", access)


UNSETTABLE_TORN_FILES = pytest.mark.parametrize(
    ("make", "refuse"),
    [
        (make_torn_directory, refuse_nothing),
        (make_torn_file, refuse_reading),
        (make_torn_file, refuse_writing),
    ],
    ids=["directory", "unreadable", "unwritable"],
)


@UNSETTABLE_TORN_FILES
def test_health_is_down_when_a_torn_journal_line_cannot_be_set_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    make: Any,
    refuse: Any,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    make(torn)
    refuse(monkeypatch, torn)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(torn) in str(answer["reason"])


@UNSETTABLE_TORN_FILES
def test_health_writes_nothing_when_a_torn_journal_line_cannot_be_set_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    make: Any,
    refuse: Any,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    make(torn)
    before = snapshot(instance.parent)
    refuse(monkeypatch, torn)

    health(monkeypatch, capsysbinary, instance)

    monkeypatch.undo()
    assert snapshot(instance.parent) == before


@UNSETTABLE_TORN_FILES
def test_health_is_degraded_when_an_unsettable_torn_file_sits_beside_a_whole_journal(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    make: Any,
    refuse: Any,
) -> None:
    directory = write_journal(instance, {"task_id": "t"})
    torn = directory / TORN_FILE
    make(torn)
    refuse(monkeypatch, torn)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, torn)


def link_torn_file_into_a_missing_directory(
    instance: Path, _: pytest.MonkeyPatch
) -> Path:
    """Write a torn journal whose torn file links into an absent directory.

    Return the torn file path.
    """
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    torn.symlink_to(instance.parent / "missing-dir" / "target")
    return torn


def link_torn_file_into_an_unwritable_directory(
    instance: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Write a torn journal whose torn file links into a directory `cli.os.access` refuses.

    Return the torn file path.
    """
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    locked = instance.parent / "locked"
    locked.mkdir()
    torn.symlink_to(locked / "target")
    refuse_writing(monkeypatch, Path(os.path.realpath(locked)))
    return torn


def link_torn_file_to_itself(instance: Path, _: pytest.MonkeyPatch) -> Path:
    """Write a torn journal whose torn file is a link to itself.

    Return the torn file path.
    """
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    torn.symlink_to(torn)
    return torn


UNREACHABLE_TORN_LINKS = pytest.mark.parametrize(
    "link",
    [
        link_torn_file_into_a_missing_directory,
        link_torn_file_into_an_unwritable_directory,
        link_torn_file_to_itself,
    ],
    ids=["missing-directory", "unwritable-directory", "loop"],
)


@UNREACHABLE_TORN_LINKS
def test_health_is_down_when_the_torn_file_links_where_a_run_cannot_append(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    link: Any,
) -> None:
    torn = link(instance, monkeypatch)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(torn) in str(answer["reason"])


@UNREACHABLE_TORN_LINKS
def test_health_writes_nothing_when_the_torn_file_links_where_a_run_cannot_append(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    link: Any,
) -> None:
    link(instance, monkeypatch)
    before = snapshot(instance.parent)

    health(monkeypatch, capsysbinary, instance)

    monkeypatch.undo()
    assert snapshot(instance.parent) == before


def test_health_is_degraded_when_the_torn_file_links_to_an_absent_file_in_a_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "y", "st', {"task_id": "t"})
    torn = directory / TORN_FILE
    elsewhere = instance.parent / "elsewhere"
    elsewhere.mkdir()
    torn.symlink_to(elsewhere / "target")

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, directory / run_journal.JOURNAL_FILE)


def test_health_is_degraded_when_an_unwritable_torn_file_already_ends_with_the_torn_line(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tail = b'{"task_id": "y", "st'
    directory = write_torn_journal(instance, tail, {"task_id": "t"})
    torn = directory / TORN_FILE
    torn.write_bytes(b'{"task_id": "x", "st\n' + tail + b"\n")
    refuse_writing(monkeypatch, torn)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, torn)


def test_health_is_degraded_after_a_run_set_a_torn_journal_line_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "x", "st')
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, directory / TORN_FILE)


def test_health_is_ok_when_the_last_journal_line_lacks_only_its_newline(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_torn_journal(instance, b'{"task_id": "b"}', {"task_id": "a"})

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {"status": "ok", "reason": None, "active": True}


def test_health_is_ok_when_the_journal_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(instance)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {"status": "ok", "reason": None, "active": True}


def test_health_is_down_rather_than_degraded_when_the_journal_cannot_be_written(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b'{"task_id": "x", "st', {"task_id": "t"})
    (directory / TORN_FILE).write_bytes(b'{"task_id": "y"\n')
    journal = refuse_journal(monkeypatch, instance)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


INVALID_JOURNAL_LINES = [
    pytest.param(b'{"task_id": "b", "sta\n', id="not-json"),
    pytest.param(b"\xff\xfe\n", id="not-utf8"),
    pytest.param(b"[" * 100_000 + b"]" * 100_000 + b"\n", id="too-deeply-nested"),
    pytest.param(b"[1]\n", id="json-not-an-object"),
]


def write_invalid_journal(instance: Path, line: bytes, tail: bytes = b"") -> Path:
    """Write a journal with `line` between two valid lines, then `tail`; return its path."""
    directory = write_journal(instance, {"task_id": "a"})
    journal = directory / run_journal.JOURNAL_FILE
    with journal.open("ab") as file:
        file.write(line + b'{"task_id": "c"}\n' + tail)
    return journal


@pytest.mark.parametrize("line", INVALID_JOURNAL_LINES)
def test_health_is_down_when_a_complete_journal_line_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    line: bytes,
) -> None:
    journal = write_invalid_journal(instance, line)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


@pytest.mark.parametrize("line", INVALID_JOURNAL_LINES)
def test_health_is_down_rather_than_degraded_when_an_invalid_line_precedes_a_torn_one(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    line: bytes,
) -> None:
    journal = write_invalid_journal(instance, line, tail=b'{"task_id": "x", "st')

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


@pytest.mark.parametrize("line", INVALID_JOURNAL_LINES)
def test_health_writes_nothing_when_a_journal_line_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    line: bytes,
) -> None:
    write_invalid_journal(instance, line, tail=b'{"task_id": "x", "st')
    before = snapshot(instance.parent)

    health(monkeypatch, capsysbinary, instance)

    assert snapshot(instance.parent) == before


def test_health_is_down_when_the_journal_is_a_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    journal = instance.with_suffix("") / run_journal.JOURNAL_FILE
    journal.mkdir(parents=True)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


def test_health_is_down_when_the_journal_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    journal = write_journal(instance, {"task_id": "t"}) / run_journal.JOURNAL_FILE
    opened = Path.open
    read = Path.read_bytes

    def open_refusing_journal(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == journal:
            raise PermissionError(13, "Permission denied", str(path))
        return opened(path, *args, **kwargs)

    def read_refusing_journal(path: Path) -> bytes:
        if path == journal:
            raise PermissionError(13, "Permission denied", str(path))
        return read(path)

    monkeypatch.setattr(Path, "open", open_refusing_journal)
    monkeypatch.setattr(Path, "read_bytes", read_refusing_journal)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        pytest.param(
            b'{"task_id": "a"}\n{"task_id": "b", "sta\n{"task_id": "c"}\n',
            "line 2 is not valid JSON",
            id="not-json",
        ),
        pytest.param(
            b'{"task_id": "a"}\n\n  \n[1]\n',
            "line 4 is not a JSON object.",
            id="after-blank-lines",
        ),
        pytest.param(
            b'{"task_id": "a"}\n[1]\n{"task_id": "x", "st',
            "line 2 is not a JSON object.",
            id="before-a-torn-line",
        ),
    ],
)
def test_health_names_the_invalid_journal_line_a_run_fails_on(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    content: bytes,
    fragment: str,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    journal = directory / run_journal.JOURNAL_FILE
    journal.write_bytes(content)

    answer = health(monkeypatch, capsysbinary, instance)
    failed = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_down(answer, f"Journal '{journal}': {fragment}")
    assert_failed(failed, "task-1", fragment)
    health_detail = str(answer["reason"]).removeprefix(f"Journal '{journal}': ")
    run_prefix = f"Journal in '{directory}': "
    errors = failed["errors"]
    assert isinstance(errors, list)
    run_details = [
        str(error).split(run_prefix, 1)[1]
        for error in errors
        if run_prefix in str(error)
    ]
    assert run_details == [health_detail]


def test_health_is_ok_when_the_journal_holds_blank_lines(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.JOURNAL_FILE).write_bytes(
        b'\n{"task_id": "a"}\n\n \t\n \r\n{"task_id": "b"}\n'
    )

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {"status": "ok", "reason": None, "active": True}


def test_health_is_degraded_rather_than_down_when_the_unterminated_last_line_is_not_an_object(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = write_torn_journal(instance, b"[1]", {"task_id": "a"})

    answer = health(monkeypatch, capsysbinary, instance)

    assert_degraded_naming(answer, directory / run_journal.JOURNAL_FILE)


@pytest.mark.usefixtures("frozen_now")
def test_health_is_degraded_when_the_run_holding_the_lock_outlives_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_started(1801))

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "degraded",
        "reason": (
            f"The run holding the lock of instance directory "
            f"'{instance.with_suffix('')}' started 1801 seconds ago, beyond the "
            f"run timeout of 1800 seconds."
        ),
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("seconds_ago", [1800, 0])
def test_health_is_ok_while_the_run_holding_the_lock_is_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    seconds_ago: int,
) -> None:
    hold_lock(instance, lock_started(seconds_ago))

    assert health(monkeypatch, capsysbinary, instance)["status"] == "ok"


@pytest.mark.usefixtures("frozen_now")
def test_health_uses_the_run_timeout_of_the_instance(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    with instance.open("a", encoding="utf-8") as file:
        file.write("  run_timeout_seconds: 60\n")
    hold_lock(instance, lock_started(61))

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer["status"] == "degraded"
    assert "run timeout of 60 seconds" in str(answer["reason"])


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"held by another run",
        b'{"task_id": "t"}',
        b'{"start": 20261105}',
        b'{"start": "yesterday"}',
        b'{"start": "2026-11-05T10:00:00"}',
    ],
)
@pytest.mark.parametrize(("seconds_ago", "status"), [(1801, "degraded"), (60, "ok")])
def test_health_ages_a_lock_without_a_usable_start_by_its_modification_time(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    content: bytes,
    seconds_ago: int,
    status: str,
) -> None:
    lock = hold_lock(instance, content)
    modified = NOW.timestamp() - seconds_ago
    os.utime(lock, (modified, modified))

    assert health(monkeypatch, capsysbinary, instance)["status"] == status


def test_health_is_ok_when_the_lock_vanishes_before_its_age_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    instance.with_suffix("").mkdir()
    monkeypatch.setattr(cli, "read_lock", lambda directory: {})

    assert health(monkeypatch, capsysbinary, instance)["status"] == "ok"


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("seconds_ago", [1801, 0])
def test_health_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    seconds_ago: int,
) -> None:
    write_journal(instance, {"task_id": "t"})
    hold_lock(instance, lock_started(seconds_ago))
    before = snapshot(instance.parent)

    health(monkeypatch, capsysbinary, instance)

    assert snapshot(instance.parent) == before


@pytest.mark.usefixtures("frozen_now")
def test_health_touches_no_modification_time(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_journal(instance, {"task_id": "t"})
    hold_lock(instance, lock_started(1801))
    directory = instance.with_suffix("")
    paths = [instance, directory, *sorted(directory.rglob("*"))]
    past = NOW.timestamp() - 3600
    for path in paths:
        os.utime(path, (past, past))
    before = {path: path.stat().st_mtime_ns for path in paths}

    health(monkeypatch, capsysbinary, instance)

    assert {path: path.stat().st_mtime_ns for path in paths} == before


def test_health_of_an_instance_never_run_creates_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    before = snapshot(instance.parent)

    assert health(monkeypatch, capsysbinary, instance)["status"] == "ok"
    assert snapshot(instance.parent) == before


@pytest.mark.parametrize("command", ["state", "health"])
def test_instance_is_required(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([command])

    assert raised.value.code == 2


def test_run_requires_the_instance() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["run", "deliver_ticket"])

    assert raised.value.code == 2


def test_dev_worker_script_points_at_main() -> None:
    (script,) = entry_points(group="console_scripts", name="dev-worker")

    assert script.load() is cli.main


def test_python_m_runs_the_command_line(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["dev-worker", "health", "--instance", str(INSTANCE)]
    )

    with pytest.raises(SystemExit) as raised:
        runpy.run_module("dev_worker", run_name="__main__")

    assert raised.value.code == 0
    assert json.loads(capsysbinary.readouterr().out)["status"] == "ok"


def test_run_reads_and_writes_utf8_whatever_the_system_encoding(
    instance: Path,
) -> None:
    task_id = "tâche ✓"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}

    completed = subprocess.run(
        [sys.executable, "-m", "dev_worker", "run", "deliver_ticket"]
        + ["--instance", str(instance)],
        input=json.dumps(valid_input(task_id=task_id), ensure_ascii=False).encode(
            "utf-8"
        ),
        capture_output=True,
        env=env,
        check=True,
    )

    assert task_id.encode("utf-8") in completed.stdout
    assert json.loads(completed.stdout.decode("utf-8"))["task_id"] == task_id


def nested_object(depth: int) -> bytes:
    """Return a JSON object holding `depth` nested arrays."""
    return b'{"task_id": "deep", "x": ' + b"[" * depth + b"]" * depth + b"}"


UNREADABLE_RECORDS = [
    pytest.param(b"[" * 100_000 + b"]" * 100_000, id="too-deep-for-json"),
    pytest.param(nested_object(300), id="too-deep-to-validate"),
    pytest.param(b'{"task_id": "\\ud800", "status": "completed"}', id="lone-surrogate"),
]


@pytest.mark.parametrize("line", UNREADABLE_RECORDS)
def test_state_skips_a_journal_line_it_cannot_answer_in_json(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    line: bytes,
) -> None:
    directory = write_journal(instance, {"task_id": "t"})
    with (directory / run_journal.JOURNAL_FILE).open("ab") as journal:
        journal.write(line + b"\n")

    assert common(state(monkeypatch, capsysbinary, instance))["recent"] == [
        {"task_id": "t"}
    ]


@pytest.mark.parametrize("content", UNREADABLE_RECORDS)
def test_state_reports_a_run_of_unknown_details_for_a_lock_it_cannot_answer_in_json(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    content: bytes,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.LOCK_FILE).write_bytes(content)

    assert common(state(monkeypatch, capsysbinary, instance))["current"] == [{}]


def dead_pid() -> int:
    """Return the PID of a process that has run and exited."""
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    return finished.pid


def lock_of(seconds_ago: float, **fields: object) -> bytes:
    """Return a lock record of task `old` started `seconds_ago` seconds before `NOW`."""
    record: dict[str, object] = {
        "task_id": "old",
        "ticket_key": "DEMO-7",
        "start": (NOW - timedelta(seconds=seconds_ago)).isoformat(),
    }
    record.update(fields)
    return json.dumps(record).encode()


def gone_line(seconds_ago: float, reason: str) -> dict[str, object]:
    """Return the journal line of the run of `lock_of(seconds_ago)`, gone for `reason`."""
    return {
        "task_id": "old",
        "ticket_key": "DEMO-7",
        "status": "failed",
        "start": (NOW - timedelta(seconds=seconds_ago)).isoformat(),
        "duration": None,
        "tokens": 0,
        "errors": [reason],
    }


def timed_out(instance: Path, seconds: int) -> str:
    """Return the reason given for a run of `instance` gone past the timeout."""
    return (
        f"The run holding the lock of instance directory '{instance.with_suffix('')}' "
        f"started {seconds} seconds ago, beyond the run timeout of 1800 seconds."
    )


def process_gone(instance: Path, pid: int) -> str:
    """Return the reason given for a run of `instance` whose process `pid` is gone."""
    return (
        f"The run holding the lock of instance directory '{instance.with_suffix('')}' "
        f"is gone: its process {pid} no longer runs."
    )


def assert_taken_over(
    answer: dict[str, object], instance: Path, gone: dict[str, object]
) -> None:
    """Check the run completed after journaling `gone`, and left only its journal."""
    assert answer["status"] == "completed"
    old, running, own = journal_lines(instance)
    assert old == gone
    assert running["status"] == "running"
    assert own["task_id"] == "task-1"
    assert own["status"] == "completed"
    assert own["start"] == NOW.isoformat()
    directory = instance.with_suffix("")
    assert sorted(path.name for path in directory.iterdir()) == [
        run_journal.JOURNAL_FILE,
        run_journal.GUARD_FILE,
    ]


@pytest.mark.usefixtures("frozen_now")
def test_run_takes_over_a_lock_held_beyond_the_timeout_and_journals_the_gone_run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801, pid=os.getpid()))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_taken_over(answer, instance, gone_line(1801, timed_out(instance, 1801)))


@pytest.mark.usefixtures("frozen_now")
def test_run_takes_over_a_lock_whose_process_is_gone_and_journals_the_gone_run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    pid = dead_pid()
    hold_lock(instance, lock_of(60, pid=pid))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_taken_over(answer, instance, gone_line(60, process_gone(instance, pid)))


@pytest.mark.usefixtures("frozen_now")
def test_run_takes_over_a_lock_of_unknown_details_held_beyond_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, b"held by another run")
    modified = NOW.timestamp() - 1801
    os.utime(lock, (modified, modified))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    old, _, own = journal_lines(instance)
    assert old["task_id"] is None
    assert old["ticket_key"] is None
    assert old["start"] is None
    assert old["errors"] == [timed_out(instance, 1801)]
    assert own["task_id"] == "task-1"


def test_run_refuses_a_lock_whose_process_still_runs_leaving_it_untouched(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    content = json.dumps(
        {
            "task_id": "old",
            "ticket_key": "DEMO-7",
            "start": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
        }
    ).encode()
    lock = hold_lock(instance, content)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert lock.read_bytes() == content
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_run_journals_the_gone_run_after_setting_a_torn_journal_line_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    write_torn_journal(instance, b'{"task_id": "to', {"task_id": "first"})
    hold_lock(instance, lock_of(1801))

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert [line["task_id"] for line in journal_lines(instance)] == [
        "first",
        "old",
        "task-1",
        "task-1",
    ]


def test_run_fails_as_already_in_progress_when_the_lock_is_released_before_it_is_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, b"held by another run")
    real_take_lock = cli.take_lock

    def take_lock_then_see_it_released(
        directory: Path, record: dict[str, object]
    ) -> Path:
        try:
            return real_take_lock(directory, record)
        except FileExistsError:
            lock.unlink()
            raise

    monkeypatch.setattr(cli, "take_lock", take_lock_then_see_it_released)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_run_keeps_the_gone_run_journaled_when_another_run_takes_the_lock_first(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801))
    real_take_lock = cli.take_lock
    calls: list[Path] = []

    def take_lock_after_another_run(directory: Path, record: dict[str, object]) -> Path:
        calls.append(directory)
        if len(calls) == 2:
            real_take_lock(directory, {"task_id": "other"})
        return real_take_lock(directory, record)

    monkeypatch.setattr(cli, "take_lock", take_lock_after_another_run)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert run_journal.read_lock(instance.with_suffix("")) == {"task_id": "other"}
    assert journal_lines(instance) == [gone_line(1801, timed_out(instance, 1801))]


@pytest.mark.usefixtures("frozen_now")
def test_run_keeps_the_gone_run_journaled_when_the_lock_cannot_be_taken_after_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801))
    real_take_lock = cli.take_lock
    calls: list[Path] = []

    def take_lock_then_run_out_of_space(
        directory: Path, record: dict[str, object]
    ) -> Path:
        calls.append(directory)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real_take_lock(directory, record)

    monkeypatch.setattr(cli, "take_lock", take_lock_then_run_out_of_space)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"Lock file in '{instance.with_suffix('')}' cannot be written: "
        "No space left on device",
    )
    assert journal_lines(instance) == [gone_line(1801, timed_out(instance, 1801))]
    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()


@pytest.mark.usefixtures("frozen_now")
def test_run_fails_naming_the_journal_when_the_gone_run_cannot_be_journaled(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801))

    def full(directory: Path, entry: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "append_entry", full)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"Journal in '{instance.with_suffix('')}' cannot be written: "
        "No space left on device",
    )
    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()


def test_run_fails_naming_the_lock_when_the_held_lock_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = instance.with_suffix("") / run_journal.LOCK_FILE
    lock.mkdir(parents=True)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"Lock file in '{instance.with_suffix('')}' cannot be written: ",
    )
    assert lock.is_dir()
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_run_fails_naming_the_lock_when_its_guard_cannot_be_opened(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, lock_of(1801))
    real_open = os.open

    def refuse_the_guard(path: Path, flags: int, mode: int = 0o777) -> int:
        if Path(path).name == run_journal.GUARD_FILE:
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, mode)

    monkeypatch.setattr(run_journal.os, "open", refuse_the_guard)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"Lock file in '{instance.with_suffix('')}' cannot be written: "
        "Permission denied",
    )
    assert lock.exists()
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_run_fails_as_already_in_progress_when_the_lock_changes_before_it_is_set_aside(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, lock_of(1801))
    real_gone_reason = cli.gone_reason

    def gone_then_taken_by_another_run(
        directory: Path, held: dict[str, object], timeout: int
    ) -> str | None:
        reason = real_gone_reason(directory, held, timeout)
        lock.write_bytes(b'{"task_id": "other"}')
        return reason

    monkeypatch.setattr(cli, "gone_reason", gone_then_taken_by_another_run)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert lock.read_bytes() == b'{"task_id": "other"}'
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_health_is_degraded_when_the_process_of_the_run_holding_the_lock_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    pid = dead_pid()
    hold_lock(instance, lock_of(60, pid=pid))

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "degraded",
        "reason": process_gone(instance, pid),
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("pid", [True, 0, -1, "1", 1.0, None])
def test_health_ignores_a_pid_that_is_not_a_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    pid: object,
) -> None:
    hold_lock(instance, lock_of(60, pid=pid))

    assert health(monkeypatch, capsysbinary, instance)["status"] == "ok"


def test_health_ignores_a_gone_process_when_the_lock_has_no_usable_start(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, json.dumps({"pid": dead_pid()}).encode())

    assert health(monkeypatch, capsysbinary, instance)["status"] == "ok"


class Unreachable:
    """A tracker or repository whose access check raises `error`; `checks` counts the calls."""

    def __init__(self, error: Exception) -> None:
        """Keep `error` for every later check."""
        self.error = error
        self.checks = 0

    def check_access(self) -> None:
        """Count the call, then raise the error."""
        self.checks += 1
        raise self.error


def test_health_is_down_when_the_tracker_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = Unreachable(TrackerError("twg could not be started: not found"))
    repository = Unreachable(RepositoryError("never asked"))
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    monkeypatch.setattr(cli, "make_repository", lambda made: repository)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "down",
        "reason": "The tracker cannot be reached: twg could not be started: not found",
        "active": True,
    }
    assert (tracker.checks, repository.checks) == (1, 0)


def test_health_is_down_when_the_repository_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    repository = Unreachable(
        RepositoryError("gh did not answer within 60 seconds while checking access")
    )
    monkeypatch.setattr(cli, "make_repository", lambda made: repository)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "down",
        "reason": "The repository cannot be reached: "
        "gh did not answer within 60 seconds while checking access",
        "active": True,
    }


def test_health_checks_the_connectors_of_the_instance_with_its_token_variables(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    monkeypatch.setattr(cli, "make_tracker", MAKE_TRACKER)
    monkeypatch.setattr(cli, "make_repository", MAKE_REPOSITORY)
    monkeypatch.setenv("JIRA_TOKEN", "")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    tracker_down = health(monkeypatch, capsysbinary, instance)
    monkeypatch.setattr(cli, "make_tracker", lambda made: ReadyTracker())
    repository_down = health(monkeypatch, capsysbinary, instance)

    assert tracker_down["reason"] == (
        "The tracker cannot be reached: the variable JIRA_TOKEN, "
        "which should hold the Jira token, is not set"
    )
    assert repository_down["reason"] == (
        "The repository cannot be reached: the variable GITHUB_TOKEN, "
        "which should hold the GitHub token, is not set"
    )
    assert not instance.with_suffix("").exists()


def test_health_is_down_rather_than_degraded_when_a_connector_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.JOURNAL_FILE).write_bytes(b'{"torn')
    monkeypatch.setattr(
        cli, "make_tracker", lambda made: Unreachable(TrackerError("refused"))
    )

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer["status"] == "down"
    assert answer["reason"] == "The tracker cannot be reached: refused"


def test_make_repository_returns_a_github_repository() -> None:
    repository = Repository(
        type="github", name="example/project", token_env="GITHUB_TOKEN"
    )

    assert isinstance(MAKE_REPOSITORY(repository), GitHubRepository)


def test_health_does_not_check_the_connectors_when_a_local_check_is_down(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    journal = write_invalid_journal(instance, b"[1]\n")
    tracker = Unreachable(TrackerError("refused"))
    repository = Unreachable(RepositoryError("refused"))
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    monkeypatch.setattr(cli, "make_repository", lambda made: repository)

    answer = health(monkeypatch, capsysbinary, instance)

    assert_down(answer, "")
    assert str(journal) in str(answer["reason"])
    assert (tracker.checks, repository.checks) == (0, 0)


def test_health_is_degraded_when_both_connectors_can_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    directory = instance.with_suffix("")
    directory.mkdir()
    (directory / run_journal.JOURNAL_FILE).write_bytes(b'{"torn')
    tracker = ReadyTracker()
    repository = ReachableRepository()
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    monkeypatch.setattr(cli, "make_repository", lambda made: repository)

    assert health(monkeypatch, capsysbinary, instance)["status"] == "degraded"
