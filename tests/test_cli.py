import io
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from dev_worker import cli, run_journal
from dev_worker.container import (
    Container,
    ContainerError,
    ContainerExit,
    agent_command,
    review_command,
)
from dev_worker.contract import MandateInput, MandateOutput
from dev_worker.github import GitHubRepository
from dev_worker.header import HeaderError, Plugin
from dev_worker.instance import GitIdentity, Instance, Repository, Tracker
from dev_worker.jira import TWG_TIMEOUT_SECONDS, JiraTracker
from dev_worker.outcome import Outcome
from dev_worker.plugin import PluginCheckout, PluginError
from dev_worker.repositories import (
    BaseCiFailure,
    ChangedFile,
    CiState,
    HeadMovedError,
    PullRequest,
    RepositoryError,
)
from dev_worker.text import writable_text
from dev_worker.tickets import (
    REQUEUE_MARKER,
    RETURN_MARKER,
    Blocker,
    Claim,
    Ticket,
    TrackerError,
)
from dev_worker.workspace import Workspace, WorkspaceError

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "examples" / "instance.yaml"
TORN_FILE = "journal.torn"
MAKE_TRACKER = cli.make_tracker
MAKE_REPOSITORY = cli.make_repository
MAKE_WORKSPACE = cli.make_workspace
MAKE_CONTAINER = cli.make_container
MAKE_PLUGIN = cli.make_plugin
CHECKOUT = Path("plugin")
IDENTITY = "12345678+example-worker@users.noreply.github.com"
RESCUED = ("rescue", "agent/DEMO-1", "Example Worker", IDENTITY)
REVIEWED = [
    ("refresh",),
    ("open_review", "main", "0123456789abcdef0123456789abcdef01234567"),
    ("close_review",),
]
SUMMARY = "The login page asks for an email."
DONE = json.dumps(
    {"type": "result", "structured_output": {"outcome": "done", "message": SUMMARY}}
)
NO_DEFECT = json.dumps(
    {
        "type": "result",
        "structured_output": {"reviewed": True, "reason": "", "defects": []},
    }
)
BASE_COMMIT = "89abcdef0123456789abcdef0123456789abcdef"


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
        return Ticket(
            key=key,
            status="To Do",
            labels=(),
            blockers=(),
            title="Add a login page",
            description="The page asks for an email.",
        )

    def claim(self, key: str, task_id: str) -> int:
        """Return the claim id 1."""
        return 1

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Return the claim 1 alone."""
        return (Claim(1, "task-1", CLAIM_EPOCH),)

    def requeues(self, key: str, statuses: frozenset[str]) -> int:
        """Return 0: the ticket was never requeued."""
        return 0

    def last_return(self, key: str, statuses: frozenset[str]) -> str | None:
        """Return None: the ticket never came back."""
        return None

    def comment(self, key: str, body: str) -> None:
        """Change nothing."""

    def set_status(self, key: str, status: str) -> None:
        """Change nothing."""

    def check_access(self) -> None:
        """Grant access."""


PULL = PullRequest(
    number=7,
    url="https://github.com/example-org/example-app/pull/7",
    head="0123456789abcdef0123456789abcdef01234567",
    title="feat: a login page",
    base="main",
)


class ReachableRepository:
    """A repository whose access check passes and whose branches each have `pull`, its CI `ci`.

    `pull` None gives a branch no pull request. A call raises the error `errors` maps its
    name to, when there is one. `calls` records each call, by name and argument.
    """

    def __init__(self) -> None:
        self.pull: PullRequest | None = PULL
        self.ci: CiState = "passed"
        self.base_failure: BaseCiFailure | None = None
        self.files: tuple[ChangedFile, ...] = ()
        self.errors: dict[str, RepositoryError] = {}
        self.calls: list[tuple[str, object]] = []

    def _step(self, name: str, argument: object) -> None:
        self.calls.append((name, argument))
        if name in self.errors:
            raise self.errors[name]

    def check_access(self) -> None:
        """Grant access."""

    def find_pull_request(self, branch: str) -> PullRequest | None:
        """Record the call; return `pull`."""
        self._step("find_pull_request", branch)
        return self.pull

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Record the call; return `ci`."""
        self._step("read_ci", pull_request)
        return self.ci

    def base_ci_failure(self) -> BaseCiFailure | None:
        """Record the call; return `base_failure`."""
        self._step("base_ci_failure", None)
        return self.base_failure

    def changed_files(self, pull_request: PullRequest) -> tuple[ChangedFile, ...]:
        """Record the call; return `files`."""
        self._step("changed_files", pull_request)
        return self.files

    def merge(self, pull_request: PullRequest) -> None:
        """Record the call."""
        self._step("merge", pull_request)


@pytest.fixture(autouse=True)
def ready_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `run` read every ticket from a `ReadyTracker` instead of the real tracker."""
    monkeypatch.setattr(cli, "make_tracker", lambda tracker: ReadyTracker())


@pytest.fixture(autouse=True)
def reachable_repository(monkeypatch: pytest.MonkeyPatch) -> ReachableRepository:
    """Make `run` and `health` use a `ReachableRepository` instead of the real repository; return it."""
    fake = ReachableRepository()
    monkeypatch.setattr(cli, "make_repository", lambda *made: fake)
    return fake


class FakeWorkspace:
    """A workspace that records its calls and raises `WorkspaceError` at the step named in `fail`."""

    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.left: str | None = None
        self.calls: list[tuple[str, ...]] = []
        self.worktree = Path("worktree")
        self.review = Path("review")

    def _step(self, *call: str) -> None:
        self.calls.append(call)
        if call[0] == self.fail:
            raise WorkspaceError(f"{call[0]} broke")

    def left_branch(self) -> str | None:
        """Record the look for a left worktree; return the branch in `left`."""
        self._step("left_branch")
        return self.left

    def drop_left_branches(self) -> None:
        """Record the deletion of the local branches left by earlier runs."""
        self._step("drop_left_branches")

    def rescue(self, branch: str, identity: GitIdentity) -> None:
        """Record the rescue of `branch` as `identity`."""
        self._step("rescue", branch, identity.name, identity.email)

    def refresh(self) -> None:
        """Record the refresh."""
        self._step("refresh")

    def open(self, branch: str) -> Path:
        """Record the opening on `branch`."""
        self._step("open", branch)
        return Path("worktree")

    def close(self, branch: str) -> None:
        """Record the closing of `branch`."""
        self._step("close", branch)

    def open_review(self, base: str, head: str) -> str:
        """Record the review checkout of `head` against `base`; return `BASE_COMMIT`."""
        self._step("open_review", base, head)
        return BASE_COMMIT

    def close_review(self) -> None:
        """Record the deletion of the review checkout."""
        self._step("close_review")


@pytest.fixture(autouse=True)
def workspace(monkeypatch: pytest.MonkeyPatch) -> FakeWorkspace:
    """Make `run` use a `FakeWorkspace` instead of a git mirror; return it."""
    fake = FakeWorkspace()
    monkeypatch.setattr(cli, "make_workspace", lambda repository, directory: fake)
    return fake


class FakePlugin:
    """A plugin checkout that records what it prepares and raises `error`, when set."""

    def __init__(self) -> None:
        self.error: PluginError | None = None
        self.prepared: list[tuple[Path, Plugin]] = []
        self.directory = Path()

    def make(self, directory: Path) -> "FakePlugin":
        """Keep `directory`; return this fake, in place of `cli.make_plugin`."""
        self.directory = directory
        return self

    def prepare(self, plugin: Plugin) -> Path:
        """Record `plugin` with the directory; return `CHECKOUT`."""
        self.prepared.append((self.directory, plugin))
        if self.error is not None:
            raise self.error
        return CHECKOUT


@pytest.fixture(autouse=True)
def plugin(monkeypatch: pytest.MonkeyPatch) -> FakePlugin:
    """Make `run` use a `FakePlugin` instead of a git checkout; return it."""
    fake = FakePlugin()
    monkeypatch.setattr(cli, "make_plugin", fake.make)
    return fake


class FakeContainer:
    """A container that records its calls and raises `error`, when set, from those named in `failing`.

    `failing` None names every call.
    """

    def __init__(self, error: ContainerError | None = None) -> None:
        self.error = error
        self.exited = ContainerExit(0, DONE, "")
        self.verdict = ContainerExit(0, NO_DEFECT, "")
        self.readonly: list[bool] = []
        self.failing: tuple[str, ...] | None = None
        self.images: list[str] = []
        self.directories: list[Path] = []
        self.calls: list[tuple[object, ...]] = []
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []

    def make(self, image: str, directory: Path) -> "FakeContainer":
        """Record `image` and `directory`; return this fake, in place of `cli.make_container`."""
        self.images.append(image)
        self.directories.append(directory)
        return self

    def _step(self, *call: object) -> None:
        self.calls.append(call)
        if self.error is not None and (self.failing is None or call[0] in self.failing):
            raise self.error

    def check_access(self) -> None:
        """Record the check."""
        self._step("check_access")

    def run(
        self,
        worktree: Path,
        time_left: float,
        plugin: Path,
        command: Sequence[str],
        environment: Mapping[str, str],
        readonly: bool = False,
    ) -> ContainerExit:
        """Record the run on `worktree` with `time_left` seconds left and `plugin` mounted, then its command, environment and whether `worktree` is read only; return `verdict` when it is, `exited` otherwise."""
        self.commands.append(tuple(command))
        self.environments.append(dict(environment))
        self.readonly.append(readonly)
        self._step("run", worktree, time_left, plugin)
        return self.verdict if readonly else self.exited

    def check_none_running(self) -> None:
        """Record the check that no container of the instance is running."""
        self._step("check_none_running")

    def kill_running(self) -> None:
        """Record the killing of the containers of the instance still running."""
        self._step("kill_running")


@pytest.fixture(autouse=True)
def tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the token variables the example instance names."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "agent-token")
    monkeypatch.setenv("GITHUB_AGENT_TOKEN", "github-token")
    monkeypatch.setenv("GITHUB_WORKER_TOKEN", "worker-github-token")
    monkeypatch.setenv("JIRA_TOKEN", "jira-token")


@pytest.fixture(autouse=True)
def container(monkeypatch: pytest.MonkeyPatch) -> FakeContainer:
    """Make `run` and `health` use a `FakeContainer` instead of Docker; return it."""
    fake = FakeContainer()
    monkeypatch.setattr(cli, "make_container", fake.make)
    return fake


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
    text = instance.read_text(encoding="utf-8")
    instance.write_text(
        text.replace("  review_model: opus\n", "  review_model: sonnet\n"),
        encoding="utf-8",
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["task_id"] == "task-1"
    assert answer["errors"] == []
    assert answer["summary"] == SUMMARY
    running, line = journal_lines(instance)
    assert running == {
        "task_id": "task-1",
        "ticket_key": "DEMO-1",
        "model": "opus",
        "review_model": "sonnet",
        "status": "running",
        "start": line["start"],
        "duration": None,
        "tokens": 0,
    }
    assert line.keys() == {
        "task_id",
        "ticket_key",
        "model",
        "review_model",
        "status",
        "start",
        "duration",
        "tokens",
    }
    assert line["task_id"] == "task-1"
    assert line["ticket_key"] == "DEMO-1"
    assert line["model"] == "opus"
    assert line["review_model"] == "sonnet"
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

    A status change raises `move_error` when one is given; otherwise it becomes the
    status of the ticket later reads answer. `reads` records each key read, `moves` each
    status change asked, `posted` the task_id of each claim, `comments` each comment
    posted, as (key, body), and `trackers` each instance tracker it was made for. A
    comment raises `comment_error` when one is set.
    """

    def __init__(
        self, answer: Ticket | TrackerError, move_error: TrackerError | None = None
    ) -> None:
        """Keep `answer` for every later read and `move_error` for every status change."""
        self.answer = answer
        self.move_error = move_error
        self.reads: list[str] = []
        self.moves: list[tuple[str, str]] = []
        self.trackers: list[Tracker] = []
        self.posted: list[str] = []
        self.comments: list[tuple[str, str]] = []
        self.comment_error: TrackerError | None = None

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

    def claim(self, key: str, task_id: str) -> int:
        """Record the claim of `task_id`; return its id, its rank from 1."""
        self.posted.append(task_id)
        return len(self.posted)

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Return every claim recorded, oldest first."""
        return tuple(
            Claim(rank, task, CLAIM_EPOCH) for rank, task in enumerate(self.posted, 1)
        )

    def requeues(self, key: str, statuses: frozenset[str]) -> int:
        """Return 0: the ticket was never requeued."""
        return 0

    def last_return(self, key: str, statuses: frozenset[str]) -> str | None:
        """Return None: the ticket never came back."""
        return None

    def comment(self, key: str, body: str) -> None:
        """Record the comment, then raise `comment_error` when set."""
        self.comments.append((key, body))
        if self.comment_error is not None:
            raise self.comment_error

    def set_status(self, key: str, status: str) -> None:
        """Record the change, then raise `move_error` or give the answer `status`."""
        self.moves.append((key, status))
        if self.move_error is not None:
            raise self.move_error
        if isinstance(self.answer, Ticket):
            self.answer = replace(self.answer, status=status)

    def check_access(self) -> None:
        """Grant access."""


def serve(
    monkeypatch: pytest.MonkeyPatch,
    answer: Ticket | TrackerError,
    move_error: TrackerError | None = None,
) -> StubTracker:
    """Make `run` use a `StubTracker` answering `answer` and raising `move_error`."""
    stub = StubTracker(answer, move_error)
    monkeypatch.setattr(cli, "make_tracker", stub.make)
    return stub


def demo_ticket(status: str = "To Do", *blockers: Blocker) -> Ticket:
    """Return the ticket DEMO-1 in `status`, blocked by `blockers`."""
    return Ticket(
        key="DEMO-1",
        status=status,
        labels=(),
        blockers=blockers,
        title="",
        description="",
    )


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
    assert stub.reads == ["DEMO-1", "DEMO-1", "DEMO-1", "DEMO-1"]
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


def test_ready_ticket_is_moved_to_in_progress_once_under_the_lock_before_the_journal(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    directory = instance.with_suffix("")
    seen: list[tuple[bool, list[dict[str, object]]]] = []
    move = stub.set_status

    def spy(key: str, status: str) -> None:
        seen.append(
            ((directory / run_journal.LOCK_FILE).exists(), journal_lines(instance))
        )
        move(key, status)

    monkeypatch.setattr(stub, "set_status", spy)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert stub.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]
    assert seen[0] == (True, [])


def test_ticket_is_claimed_under_the_lock_before_it_is_moved_and_the_run_journaled(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    directory = instance.with_suffix("")
    seen: list[tuple[bool, list[tuple[str, str]], list[dict[str, object]]]] = []
    post = stub.claim

    def spy(key: str, task_id: str) -> int:
        seen.append(
            (
                (directory / run_journal.LOCK_FILE).exists(),
                list(stub.moves),
                journal_lines(instance),
            )
        )
        return post(key, task_id)

    monkeypatch.setattr(stub, "claim", spy)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert stub.posted == ["task-1"]
    assert seen == [(True, [], [])]


def test_ticket_not_ready_is_not_moved(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket("Done"))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "is in status 'Done', not 'To Do'.")
    assert stub.moves == []
    assert stub.posted == []


def test_ticket_the_tracker_cannot_read_is_not_moved(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, TrackerError("twg failed while reading DEMO-1: timeout"))

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert stub.moves == []


def test_ticket_taken_by_a_run_is_refused_to_a_second_run_under_another_task_id(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input(task_id="task-2")),
        instance=instance,
    )

    assert_failed(answer, "task-2", "is in status 'Done', not 'To Do'.")
    assert stub.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "task-1",
        "task-1",
    ]


def test_ticket_the_tracker_cannot_move_fails_as_of_unknown_status_journaling_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    serve(
        monkeypatch,
        demo_ticket(),
        TrackerError("twg failed while moving DEMO-1: timeout"),
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' cannot be moved to 'In Progress': "
        "twg failed while moving DEMO-1: timeout "
        "Its status is unknown and is to be read again in the tracker.",
    )
    assert journal_lines(instance) == []
    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()


class InterleavedTracker(ReadyTracker):
    """A tracker whose ticket stays ready to every read, as read by runs before any move.

    Claims are kept in the order posted, with ids from 1; `asked` records the status
    each claims read is asked about, and `moves` each status change.
    """

    def __init__(self) -> None:
        """Start with no claim, no claims read and no move."""
        self.posted: list[str] = []
        self.asked: list[str] = []
        self.moves: list[tuple[str, str]] = []

    def claim(self, key: str, task_id: str) -> int:
        """Keep the claim of `task_id`; return its id."""
        self.posted.append(task_id)
        return len(self.posted)

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Record `statuses`, sorted; return every claim kept, oldest first."""
        self.asked.extend(sorted(statuses))
        return tuple(
            Claim(rank, task, CLAIM_EPOCH) for rank, task in enumerate(self.posted, 1)
        )

    def set_status(self, key: str, status: str) -> None:
        """Record the change."""
        self.moves.append((key, status))


def test_two_runs_that_both_read_the_ticket_ready_move_it_once(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = InterleavedTracker()
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    answer = run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input(task_id="task-2")),
        instance=instance,
    )

    assert_failed(
        answer,
        "task-2",
        "Ticket 'DEMO-1' is claimed by task 'task-1'. "
        "The claim of task 'task-2' was posted.",
    )
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]
    assert tracker.asked == ["To Do", "To Do"]
    assert [line["task_id"] for line in journal_lines(instance)] == [
        "task-1",
        "task-1",
    ]


class UnclaimableTracker(ReadyTracker):
    """A tracker that fails at the claim step `step` with `TrackerError`, or loses the claim."""

    def __init__(self, step: str) -> None:
        """Keep the step that fails: `claim`, `claims`, or `lost` for a claim not read back."""
        self.step = step
        self.moves: list[tuple[str, str]] = []

    def claim(self, key: str, task_id: str) -> int:
        """Raise when the claim step fails; otherwise return the id 7."""
        if self.step == "claim":
            raise TrackerError("twg failed while claiming DEMO-1: timeout")
        return 7

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Raise when the read step fails; otherwise return a claim other than 7."""
        if self.step == "claims":
            raise TrackerError(
                "twg failed while reading the comments of DEMO-1: timeout"
            )
        return (Claim(3, "task-0", CLAIM_EPOCH),)

    def set_status(self, key: str, status: str) -> None:
        """Record the change."""
        self.moves.append((key, status))


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        (
            "claim",
            (
                "Ticket 'DEMO-1' cannot be claimed for task 'task-1': "
                "twg failed while claiming DEMO-1: timeout "
                "The claim may have been posted."
            ),
        ),
        (
            "claims",
            (
                "The claims on ticket 'DEMO-1' cannot be read: "
                "twg failed while reading the comments of DEMO-1: timeout "
                "The claim of task 'task-1' was posted."
            ),
        ),
        (
            "lost",
            (
                "The claim of task 'task-1' is not found among the claims on "
                "ticket 'DEMO-1' since it last entered 'To Do'. "
                "The claim of task 'task-1' was posted."
            ),
        ),
    ],
)
def test_ticket_that_cannot_be_claimed_is_not_moved_nor_journaled(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    step: str,
    reason: str,
) -> None:
    tracker = UnclaimableTracker(step)
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert tracker.moves == []
    assert journal_lines(instance) == []


CLAIM_EPOCH = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)


def timed_claim(claim_id: int, task_id: str, created: datetime) -> Claim:
    """Return the claim `claim_id` of `task_id`, created on the tracker at `created`."""
    return Claim(claim_id, task_id, created)


class TimedTracker(ReadyTracker):
    """A tracker whose claims carry the server time they were created at.

    `earlier` holds the claims already on the ticket, as (task_id, created) pairs, oldest
    first; the run's own claim is created at `own_created`, and the claims of `later`
    are posted right after it, with higher ids. Reading the claims advances the
    monotonic `clock` and the `wall` clock by `read_seconds`, and the `wall` clock alone
    by `suspended_seconds`. Once a claim is posted, reading the ticket advances both
    clocks by `reread_seconds` and the `wall` clock alone by `reread_suspended_seconds`,
    and answers `status_after_claim`. `moves` records each status change.
    """

    def __init__(
        self,
        earlier: list[tuple[str, datetime]],
        own_created: datetime,
        read_seconds: float = 0.0,
        status_after_claim: str = "To Do",
        later: tuple[tuple[str, datetime], ...] = (),
        suspended_seconds: float = 0.0,
        reread_seconds: float = 0.0,
        reread_suspended_seconds: float = 0.0,
    ) -> None:
        """Keep the claims already posted, the own claim's time, the delays and the later claims."""
        self.posted = list(earlier)
        self.own_created = own_created
        self.later = later
        self.read_seconds = read_seconds
        self.status_after_claim = status_after_claim
        self.suspended_seconds = suspended_seconds
        self.reread_seconds = reread_seconds
        self.reread_suspended_seconds = reread_suspended_seconds
        self.clock = [1000.0]
        self.wall = [CLAIM_EPOCH.timestamp()]
        self.claimed = False
        self.moves: list[tuple[str, str]] = []

    def monotonic(self) -> float:
        """Return the fake monotonic clock, in seconds."""
        return self.clock[0]

    def time(self) -> float:
        """Return the fake wall clock, in seconds since the epoch."""
        return self.wall[0]

    def read(self, key: str) -> Ticket:
        """Return the ticket ready, or in `status_after_claim` once a claim is posted."""
        if self.claimed:
            self.clock[0] += self.reread_seconds
            self.wall[0] += self.reread_seconds + self.reread_suspended_seconds
        status = self.status_after_claim if self.claimed else "To Do"
        return Ticket(
            key=key,
            status=status,
            labels=(),
            blockers=(),
            title="",
            description="",
        )

    def claim(self, key: str, task_id: str) -> int:
        """Keep the claim of `task_id`, created at `own_created`, then `later`; return its id."""
        self.posted.append((task_id, self.own_created))
        own = len(self.posted)
        self.posted.extend(self.later)
        self.claimed = True
        return own

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Advance the clocks by the read delays; return every claim kept, oldest first."""
        self.clock[0] += self.read_seconds
        self.wall[0] += self.read_seconds + self.suspended_seconds
        return tuple(
            timed_claim(rank, task, created)
            for rank, (task, created) in enumerate(self.posted, 1)
        )

    def set_status(self, key: str, status: str) -> None:
        """Record the change."""
        self.moves.append((key, status))


def serve_timed(monkeypatch: pytest.MonkeyPatch, tracker: TimedTracker) -> None:
    """Make `run` use `tracker` and read time from its monotonic and wall clocks."""
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    monkeypatch.setattr(cli.time, "monotonic", tracker.monotonic)
    monkeypatch.setattr(cli.time, "time", tracker.time)


@pytest.mark.parametrize("orphan_task", ["task-0", "task-1"])
def test_claim_created_more_than_the_expiry_before_its_own_does_not_block_the_run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    orphan_task: str,
) -> None:
    tracker = TimedTracker(
        [(orphan_task, CLAIM_EPOCH)],
        CLAIM_EPOCH + timedelta(seconds=601),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed", answer["errors"]
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]


@pytest.mark.parametrize("age", [1, 600])
def test_rival_claim_created_within_the_expiry_still_wins_over_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    age: int,
) -> None:
    tracker = TimedTracker(
        [("task-0", CLAIM_EPOCH)],
        CLAIM_EPOCH + timedelta(seconds=age),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' is claimed by task 'task-0'. "
        "The claim of task 'task-1' was posted.",
    )
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_losing_claim_does_not_block_the_run_once_the_claim_it_lost_to_expired(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker(
        [("task-0", CLAIM_EPOCH), ("task-2", CLAIM_EPOCH + timedelta(seconds=300))],
        CLAIM_EPOCH + timedelta(seconds=700),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed", answer["errors"]
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]


def test_winning_claim_created_the_expiry_before_its_own_wins_over_a_losing_one(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker(
        [("task-0", CLAIM_EPOCH), ("task-2", CLAIM_EPOCH + timedelta(seconds=300))],
        CLAIM_EPOCH + timedelta(seconds=600),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' is claimed by task 'task-0'. "
        "The claim of task 'task-1' was posted.",
    )
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_claim_that_won_after_an_expired_one_wins_over_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker(
        [
            ("task-0", CLAIM_EPOCH),
            ("task-2", CLAIM_EPOCH + timedelta(seconds=300)),
            ("task-3", CLAIM_EPOCH + timedelta(seconds=700)),
        ],
        CLAIM_EPOCH + timedelta(seconds=1000),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' is claimed by task 'task-3'. "
        "The claim of task 'task-1' was posted.",
    )
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_losing_claim_names_the_claim_it_lost_to_not_a_later_winning_one(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker(
        [("task-0", CLAIM_EPOCH)],
        CLAIM_EPOCH + timedelta(seconds=599),
        later=(("task-2", CLAIM_EPOCH + timedelta(seconds=601)),),
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "Ticket 'DEMO-1' is claimed by task 'task-0'. "
        "The claim of task 'task-1' was posted.",
    )
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_run_whose_claims_are_read_back_too_late_gives_up_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker([], CLAIM_EPOCH, read_seconds=301.0)
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer, "task-1", "were read back more than 300 seconds after the claim"
    )
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_run_whose_claims_are_read_back_within_the_delay_takes_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker([], CLAIM_EPOCH, read_seconds=300.0)
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed", answer["errors"]
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]


TOO_LATE_TO_MOVE = (
    "Ticket 'DEMO-1' was about to be moved more than 300 seconds after the claim of "
    "task 'task-1' began to be posted. The claim was posted; the ticket was not moved."
)
TOO_LATE_TO_READ = "were read back more than 300 seconds after the claim"


@pytest.mark.parametrize(
    ("read", "suspended", "reread", "reread_suspended", "reason"),
    [
        (0.0, 0.0, 301.0, 0.0, TOO_LATE_TO_MOVE),
        (1.0, 0.0, 300.0, 0.0, TOO_LATE_TO_MOVE),
        (0.0, 301.0, 0.0, 0.0, TOO_LATE_TO_READ),
        (0.0, 0.0, 0.0, 301.0, TOO_LATE_TO_MOVE),
        (0.0, 1.0, 0.0, 300.0, TOO_LATE_TO_MOVE),
        (301.0, -301.0, 0.0, 0.0, TOO_LATE_TO_READ),
        (0.0, 0.0, 301.0, -301.0, TOO_LATE_TO_MOVE),
    ],
)
def test_run_that_reaches_the_move_too_late_after_its_claim_gives_up_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    read: float,
    suspended: float,
    reread: float,
    reread_suspended: float,
    reason: str,
) -> None:
    tracker = TimedTracker(
        [],
        CLAIM_EPOCH,
        read_seconds=read,
        suspended_seconds=suspended,
        reread_seconds=reread,
        reread_suspended_seconds=reread_suspended,
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_claim_delay_starts_after_the_worktree_left_by_an_earlier_run_is_rescued(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    tracker = TimedTracker([], CLAIM_EPOCH)
    serve_timed(monkeypatch, tracker)
    rescue = workspace.rescue

    def slow_rescue(branch: str, identity: GitIdentity) -> None:
        rescue(branch, identity)
        if branch == "agent/OLD-7":
            tracker.clock[0] += 1000.0
            tracker.wall[0] += 1000.0

    monkeypatch.setattr(workspace, "rescue", slow_rescue)
    workspace.left = "agent/OLD-7"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed", answer["errors"]
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]


@pytest.mark.parametrize(
    ("read", "suspended", "reread", "reread_suspended"),
    [
        (0.0, 0.0, 300.0, 0.0),
        (150.0, 0.0, 150.0, 0.0),
        (0.0, 300.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 300.0),
        (0.0, 150.0, 0.0, 150.0),
    ],
)
def test_run_that_reaches_the_move_within_the_delay_after_its_claim_takes_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    read: float,
    suspended: float,
    reread: float,
    reread_suspended: float,
) -> None:
    tracker = TimedTracker(
        [],
        CLAIM_EPOCH,
        read_seconds=read,
        suspended_seconds=suspended,
        reread_seconds=reread,
        reread_suspended_seconds=reread_suspended,
    )
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed", answer["errors"]
    assert tracker.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]


def test_winning_run_gives_up_a_ticket_read_again_out_of_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = TimedTracker([], CLAIM_EPOCH, status_after_claim="In Progress")
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "no longer in 'To Do'")
    assert tracker.moves == []
    assert journal_lines(instance) == []


class RereadFailingTracker(TimedTracker):
    """A timed tracker that cannot read the ticket again once a claim is posted."""

    def read(self, key: str) -> Ticket:
        """Return the ticket ready, or raise once a claim is posted."""
        if self.claimed:
            raise TrackerError("twg failed while reading DEMO-1: timeout")
        return super().read(key)


def test_winning_run_gives_up_a_ticket_that_cannot_be_read_again(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    tracker = RereadFailingTracker([], CLAIM_EPOCH)
    serve_timed(monkeypatch, tracker)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "cannot be read again")
    assert tracker.moves == []
    assert journal_lines(instance) == []


def test_ticket_is_not_moved_when_a_live_run_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    live_run = {
        "task_id": "old",
        "ticket_key": "DEMO-7",
        "start": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    hold_lock(instance, json.dumps(live_run).encode())

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert stub.moves == []
    assert stub.posted == []


def test_ticket_is_not_moved_again_for_a_task_id_already_run(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    stdin = encode(valid_input())
    run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)
    stub.answer = demo_ticket()

    answer = run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert stub.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "Done")]
    assert stub.posted == ["task-1"]


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
    def refuse(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
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
    clock = [90.0, 90.0, 90.0, 90.0, 100.0, 101.0, 101.5, 102.5]
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

    def peek(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        lock = instance.with_suffix("") / run_journal.LOCK_FILE
        seen.append(json.loads(lock.read_text(encoding="utf-8")))
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)
    text = instance.read_text(encoding="utf-8")
    instance.write_text(
        text.replace("  review_model: opus\n", "  review_model: sonnet\n"),
        encoding="utf-8",
    )

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    (record,) = seen
    assert record.keys() == {
        "task_id",
        "ticket_key",
        "model",
        "review_model",
        "start",
        "pid",
    }
    assert record["pid"] == os.getpid()
    assert record["task_id"] == "task-1"
    assert record["ticket_key"] == "DEMO-1"
    assert record["model"] == "opus"
    assert record["review_model"] == "sonnet"
    assert record["start"] == journal_lines(instance)[0]["start"]


def test_lock_is_released_when_the_mandate_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)

    with pytest.raises(RuntimeError, match="boom"):
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert not (instance.with_suffix("") / run_journal.LOCK_FILE).exists()
    assert [line["status"] for line in journal_lines(instance)] == ["running"]


def test_mandate_runs_in_a_worktree_on_the_ticket_branch_closed_afterwards(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    seen: list[list[tuple[str, ...]]] = []

    def peek(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        seen.append(list(workspace.calls))
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert seen == [
        [
            ("left_branch",),
            ("drop_left_branches",),
            ("refresh",),
            ("open", "agent/DEMO-1"),
        ]
    ]
    assert workspace.calls == [
        ("left_branch",),
        ("drop_left_branches",),
        ("refresh",),
        ("open", "agent/DEMO-1"),
        RESCUED,
        ("close", "agent/DEMO-1"),
        *REVIEWED,
    ]


def test_workspace_is_made_for_the_instance_repository_in_its_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    made: list[tuple[Repository, Path]] = []

    def make(repository: Repository, directory: Path) -> FakeWorkspace:
        made.append((repository, directory))
        return workspace

    monkeypatch.setattr(cli, "make_workspace", make)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert [(repository.name, directory) for repository, directory in made] == [
        ("example-org/example-app", instance.with_suffix(""))
    ] * 3


def test_worktree_branch_follows_the_instance_branch_template(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    text = instance.read_text(encoding="utf-8")
    assert "branch: agent/{key}\n" in text
    instance.write_text(
        text.replace("branch: agent/{key}\n", "branch: feature/{key}-auto\n"),
        encoding="utf-8",
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert workspace.calls == [
        ("left_branch",),
        ("drop_left_branches",),
        ("refresh",),
        ("open", "feature/DEMO-1-auto"),
        ("rescue", "feature/DEMO-1-auto", "Example Worker", IDENTITY),
        ("close", "feature/DEMO-1-auto"),
        *REVIEWED,
    ]


def test_make_workspace_keeps_the_mirror_and_worktree_in_the_instance_directory(
    tmp_path: Path,
) -> None:
    repository = Repository(
        type="github",
        name="octo/widgets",
        agent_token_env="A",
        worker_token_env="W",
    )

    made = MAKE_WORKSPACE(repository, tmp_path)

    assert isinstance(made, Workspace)
    assert made.mirror == tmp_path / "mirror.git"
    assert made.worktree == tmp_path / "worktree"


@pytest.mark.parametrize("step", ["refresh", "open"])
def test_workspace_that_cannot_be_prepared_fails_naming_why_and_journals_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    step: str,
) -> None:
    ran: list[str] = []

    def record(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        ran.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", record)
    workspace.fail = step

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", f"The worktree cannot be prepared: {step} broke")
    assert ran == []
    assert ("close", "agent/DEMO-1") not in workspace.calls
    assert RESCUED not in workspace.calls
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "failed",
    ]


def test_container_runs_from_the_instance_image_on_the_worktree_before_the_mandate(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    seen: list[tuple[list[tuple[str, ...]], list[tuple[object, ...]]]] = []

    def peek(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        seen.append((list(workspace.calls), list(container.calls)))
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)
    slow_refresh(monkeypatch, workspace, 0)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert set(container.images) == {"ghcr.io/example-org/example-app-agent:1.0.0"}
    assert set(container.directories) == {instance.with_suffix("")}
    assert seen == [
        (
            [
                ("left_branch",),
                ("drop_left_branches",),
                ("refresh",),
                ("open", "agent/DEMO-1"),
            ],
            [("run", workspace.worktree, 1200, CHECKOUT)],
        )
    ]


def test_container_runs_claude_on_the_ticket_with_the_worker_rules_and_only_the_run_variables(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    frozen_now: datetime,
) -> None:
    slow_refresh(monkeypatch, workspace, 0)
    text = instance.read_text(encoding="utf-8")
    instance.write_text(
        text.replace("  review_model: opus\n", "  review_model: sonnet\n"),
        encoding="utf-8",
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    worker = cli.load_worker()
    ticket = ReadyTracker().read("DEMO-1")
    assert container.commands == [
        agent_command(
            ticket,
            worker.body,
            40,
            ["done", "blocked", "needs_human", "abandoned"],
            frozen_now + timedelta(seconds=1020),
            "opus",
        ),
        review_command(BASE_COMMIT, PULL.head, 40, "sonnet"),
    ]
    assert container.environments == [
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "agent-token",
            "GH_TOKEN": "github-token",
            "TWG_TOKEN": "jira-token",
            "TWG_SITE": "example.atlassian.net",
            "TWG_USER": "worker@example.com",
            "IS_SANDBOX": "1",
        },
        {
            "CLAUDE_CODE_OAUTH_TOKEN": "agent-token",
            "IS_SANDBOX": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/workspace",
        },
    ]
    assert container.readonly == [False, True]
    assert "worker-github-token" not in container.environments[0].values()


@pytest.mark.parametrize(
    ("variable", "service"),
    [
        ("CLAUDE_CODE_OAUTH_TOKEN", "Claude Code subscription token"),
        ("GITHUB_AGENT_TOKEN", "agent's GitHub token"),
        ("JIRA_TOKEN", "Jira token"),
    ],
)
@pytest.mark.parametrize("unset", [True, False], ids=["unset", "empty"])
def test_a_token_variable_unset_fails_the_run_naming_it_without_starting_the_container(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    variable: str,
    service: str,
    unset: bool,
) -> None:
    if unset:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, "")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"The container was not started: the variable {variable}, which should "
        f"hold the {service}, is not set.",
    )
    assert container.commands == []
    errors = json.dumps(answer["errors"])
    assert not any(
        value in errors for value in ("agent-token", "github-token", "jira-token")
    )


def use_api_key(instance: Path) -> None:
    """Make the agent of the instance file at `instance` sign in with the API key in `ANTHROPIC_KEY`."""
    text = instance.read_text(encoding="utf-8")
    instance.write_text(
        text.replace(
            "  provider: subscription\n  token_env: CLAUDE_CODE_OAUTH_TOKEN\n",
            "  provider: api_key\n  token_env: ANTHROPIC_KEY\n",
        ),
        encoding="utf-8",
    )


def test_container_of_an_api_key_agent_gets_the_key_as_anthropic_api_key_and_no_oauth_token(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    use_api_key(instance)
    monkeypatch.setenv("ANTHROPIC_KEY", "agent-key")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert container.environments == [
        {
            "ANTHROPIC_API_KEY": "agent-key",
            "GH_TOKEN": "github-token",
            "TWG_TOKEN": "jira-token",
            "TWG_SITE": "example.atlassian.net",
            "TWG_USER": "worker@example.com",
            "IS_SANDBOX": "1",
        },
        {
            "ANTHROPIC_API_KEY": "agent-key",
            "IS_SANDBOX": "1",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "/workspace",
        },
    ]


@pytest.mark.parametrize("unset", [True, False], ids=["unset", "empty"])
def test_an_api_key_variable_unset_fails_the_run_naming_it_without_starting_the_container(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    unset: bool,
) -> None:
    use_api_key(instance)
    if unset:
        monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_KEY", "")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The container was not started: the variable ANTHROPIC_KEY, which should "
        "hold the Anthropic API key, is not set.",
    )
    assert container.commands == []
    errors = json.dumps(answer["errors"])
    assert not any(
        value in errors for value in ("agent-token", "github-token", "jira-token")
    )


def test_run_prepares_the_plugin_the_header_names_in_the_instance_directory(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    plugin: FakePlugin,
) -> None:
    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert (
        plugin.prepared
        == [
            (
                instance.with_suffix(""),
                Plugin(
                    repository="https://github.com/AxiaCoder/claude-config.git",
                    tag="atelier--v1.0.0",
                ),
            )
        ]
        * 2
    )


def test_plugin_that_cannot_be_fetched_fails_the_run_naming_why_before_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    plugin: FakePlugin,
) -> None:
    ran: list[str] = []

    def record(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        ran.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", record)
    plugin.error = PluginError("git failed while cloning: no such tag")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The plugin cannot be fetched: git failed while cloning: no such tag",
    )
    assert ran == []
    assert ("refresh",) not in workspace.calls
    assert container.calls == []
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "failed",
    ]


def test_make_plugin_keeps_the_checkout_in_the_instance_directory(
    tmp_path: Path,
) -> None:
    made = MAKE_PLUGIN(tmp_path)

    assert isinstance(made, PluginCheckout)
    assert made.checkout == tmp_path / "plugin"


def test_container_runs_under_the_run_timeout_of_the_instance_less_its_review_reserve(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    with instance.open("a", encoding="utf-8") as file:
        file.write("  run_timeout_seconds: 200\n  review_timeout_seconds: 1\n")

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    [(step, worktree, time_left, checkout), checked, reviewed] = container.calls
    assert checked == ("check_none_running",)
    assert reviewed[0] == "run"
    assert (step, worktree, checkout) == ("run", Path("worktree"), CHECKOUT)
    assert isinstance(time_left, float)
    assert 198 < time_left <= 199


def test_prompt_carries_the_ci_deadline_of_the_container_time_less_the_margin(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    frozen_now: datetime,
) -> None:
    slow_refresh(monkeypatch, workspace, 600)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    prompt = container.commands[0][2]
    assert "\nCI deadline: 2026-11-05T12:07:00Z\n" in prompt


def slow_refresh(
    monkeypatch: pytest.MonkeyPatch, workspace: FakeWorkspace, seconds: float
) -> None:
    """Make the clock of `cli` stand still, except during the refresh, which takes `seconds`."""
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    refresh = workspace.refresh

    def slow() -> None:
        refresh()
        clock[0] += seconds

    monkeypatch.setattr(workspace, "refresh", slow)


def test_container_is_given_the_time_left_once_the_worktree_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    slow_refresh(monkeypatch, workspace, 600)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.calls == [
        ("run", workspace.worktree, 600, CHECKOUT),
        ("check_none_running",),
        ("run", workspace.review, 600, CHECKOUT),
    ]


def test_container_time_left_counts_the_time_spent_taking_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    class SlowTransitionTracker(ReadyTracker):
        def set_status(self, key: str, status: str) -> None:
            clock[0] += 300

    monkeypatch.setattr(cli, "make_tracker", lambda made: SlowTransitionTracker())

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.calls == [
        ("run", workspace.worktree, 900, CHECKOUT),
        ("check_none_running",),
        ("run", workspace.review, 1500, CHECKOUT),
    ]


@pytest.mark.parametrize("seconds", [1200, 2400])
def test_run_that_reached_its_ceiling_before_the_container_fails_without_starting_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    seconds: float,
) -> None:
    ran: list[str] = []

    def record(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        ran.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", record)
    slow_refresh(monkeypatch, workspace, seconds)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The container was not started: the run ceiling of 1800 seconds, less the "
        "600 seconds kept for the guard review, was reached before it.",
    )
    assert container.calls == [("check_none_running",)]
    assert ran == []
    assert workspace.calls[-2:] == [RESCUED, ("close", "agent/DEMO-1")]


def test_run_left_with_only_the_ci_margin_fails_without_starting_the_container(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    slow_refresh(monkeypatch, workspace, 1200 - 180)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "The container was not started")
    assert_failed(answer, "task-1", "180 seconds")
    assert_failed(answer, "task-1", "CI")
    assert container.calls == [("check_none_running",)]
    assert workspace.calls[-2:] == [RESCUED, ("close", "agent/DEMO-1")]


def test_run_left_one_second_past_the_ci_margin_starts_the_container(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    slow_refresh(monkeypatch, workspace, 1200 - 181)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.calls[0] == ("run", workspace.worktree, 181, CHECKOUT)


def test_container_that_fails_fails_the_run_naming_why_without_the_mandate(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    ran: list[str] = []

    def record(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        ran.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", record)
    container.error = ContainerError("docker failed while running the container: no")
    container.failing = ("run",)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The container failed: docker failed while running the container: no",
    )
    assert ran == []
    assert workspace.calls[-2:] == [RESCUED, ("close", "agent/DEMO-1")]
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "failed",
    ]


def test_make_container_returns_a_container() -> None:
    assert isinstance(MAKE_CONTAINER("example/image:1.0", Path("instance")), Container)


def test_worktree_is_closed_when_the_mandate_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def refuse(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        return cli.failed(mandate.task_id, "refused")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", refuse)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == ["refused"]
    assert workspace.calls[-5:-3] == [RESCUED, ("close", "agent/DEMO-1")]


def test_worktree_is_closed_when_the_mandate_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)

    with pytest.raises(RuntimeError, match="boom"):
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert workspace.calls[-2:] == [RESCUED, ("close", "agent/DEMO-1")]


def test_worktree_that_cannot_be_removed_keeps_the_output_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def refuse(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        return cli.failed(mandate.task_id, "refused")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", refuse)
    workspace.fail = "close"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "failed"
    assert answer["errors"] == [
        "refused",
        "The worktree cannot be removed: close broke",
    ]


def test_worktree_that_cannot_be_removed_after_the_mandate_raised_notes_why_on_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)
    workspace.fail = "close"

    with pytest.raises(RuntimeError, match="boom") as caught:
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert workspace.calls[-2:] == [RESCUED, ("close", "agent/DEMO-1")]
    assert caught.value.__notes__ == ["The worktree cannot be removed: close broke"]


def test_rescue_that_fails_keeps_the_reason_the_run_stopped_and_the_worktree(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def refuse(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        return cli.failed(mandate.task_id, "refused")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", refuse)
    workspace.fail = "rescue"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "failed"
    assert answer["errors"] == [
        "refused",
        "The work of the run cannot be rescued, its worktree is kept: rescue broke",
    ]
    assert workspace.calls[-1] == RESCUED


def test_rescue_that_fails_after_the_mandate_raised_notes_why_on_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        raise RuntimeError("boom")

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)
    workspace.fail = "rescue"

    with pytest.raises(RuntimeError, match="boom") as caught:
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert workspace.calls[-1] == RESCUED
    assert caught.value.__notes__ == [
        "The work of the run cannot be rescued, its worktree is kept: rescue broke"
    ]


def test_worktree_left_by_an_earlier_run_is_rescued_and_removed_before_the_ticket_is_taken(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    moved: list[list[tuple[str, ...]]] = []
    move = stub.set_status

    def spy(key: str, status: str) -> None:
        moved.append(list(workspace.calls))
        move(key, status)

    monkeypatch.setattr(stub, "set_status", spy)
    workspace.left = "agent/OLD-7"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert moved[0] == [
        ("left_branch",),
        ("refresh",),
        ("rescue", "agent/OLD-7", "Example Worker", IDENTITY),
        ("close", "agent/OLD-7"),
    ]
    assert workspace.calls[4:6] == [("refresh",), ("open", "agent/DEMO-1")]


@pytest.mark.parametrize(
    ("step", "reason"),
    [
        (
            "left_branch",
            "The worktree left by an earlier run cannot be rescued: left_branch broke",
        ),
        (
            "refresh",
            "The worktree left by an earlier run cannot be rescued: refresh broke",
        ),
        (
            "rescue",
            (
                "The worktree left by an earlier run on the branch agent/OLD-7: The work"
                " of the run cannot be rescued, its worktree is kept: rescue broke"
            ),
        ),
        (
            "close",
            (
                "The worktree left by an earlier run on the branch agent/OLD-7: The"
                " worktree cannot be removed: close broke"
            ),
        ),
    ],
)
def test_worktree_left_that_cannot_be_rescued_fails_the_run_without_taking_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    step: str,
    reason: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.left = "agent/OLD-7"
    workspace.fail = step

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.moves == []
    assert stub.posted == []
    assert ("open", "agent/DEMO-1") not in workspace.calls
    assert journal_lines(instance) == []


FAKE_DOCKER = Path(__file__).with_name("fake_docker.py")


def fake_docker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **answers: str
) -> Path:
    """Make `run` drive `tests/fake_docker.py` in place of Docker; return its record.

    Each keyword sets the environment variable `FAKE_DOCKER_<KEYWORD>`, such as
    `ps_stdout_1` for `FAKE_DOCKER_PS_STDOUT_1`. The container writes the `done`
    outcome `DONE` unless `start_stdout` says otherwise, and the guard review container
    after it the verdict `NO_DEFECT` unless `start_stdout_1` does.
    """
    record = tmp_path / "docker.jsonl"
    monkeypatch.setenv("FAKE_DOCKER_RECORD", str(record))
    monkeypatch.setenv("FAKE_DOCKER_START_STDOUT", DONE)
    monkeypatch.setenv("FAKE_DOCKER_START_STDOUT_1", NO_DEFECT)
    for name, value in answers.items():
        monkeypatch.setenv(f"FAKE_DOCKER_{name.upper()}", value)
    monkeypatch.setattr(cli, "make_container", MAKE_CONTAINER)
    monkeypatch.setattr(
        cli, "Container", partial(Container, command=[sys.executable, str(FAKE_DOCKER)])
    )
    return record


def docker_calls(record: Path) -> list[str]:
    """Return the subcommand of each call `record` holds, with the name a `kill` was given."""
    calls = [json.loads(line) for line in record.read_text("utf-8").splitlines()]
    return [" ".join(call[:2]) if call[0] == "kill" else call[0] for call in calls]


def test_rescue_does_not_run_while_a_container_that_could_not_be_killed_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    tmp_path: Path,
) -> None:
    record = fake_docker(
        monkeypatch,
        tmp_path,
        start_sleep="5",
        rm_exit="1",
        rm_stderr="permission denied\n",
        ps_stdout="dev-worker-stuck\n",
    )
    slow_refresh(monkeypatch, workspace, 1799.5)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The work of the run cannot be rescued, its worktree is kept: the container "
        "dev-worker-stuck of the instance is still running",
    )
    assert docker_calls(record)[-1] == "ps"
    assert RESCUED not in workspace.calls
    assert ("close", "agent/DEMO-1") not in workspace.calls


def test_container_left_by_a_killed_worker_is_killed_before_its_worktree_is_rescued(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    tmp_path: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    record = fake_docker(monkeypatch, tmp_path, ps_stdout_0="dev-worker-left\n")
    workspace.left = "agent/OLD-7"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert docker_calls(record) == [
        "ps",
        "kill dev-worker-left",
        "ps",
        "image",
        "create",
        "start",
        "ps",
        "image",
        "create",
        "start",
    ]
    assert ("rescue", "agent/OLD-7", "Example Worker", IDENTITY) in workspace.calls
    assert stub.moves != []


@pytest.mark.parametrize(
    ("answers", "reason"),
    [
        (
            {
                "ps_stdout": "dev-worker-left\n",
                "kill_exit": "1",
                "kill_stderr": "permission denied\n",
            },
            (
                "The worktree left by an earlier run cannot be rescued: docker failed "
                "while killing the container dev-worker-left: permission denied"
            ),
        ),
        (
            {"ps_stdout": "dev-worker-left\n"},
            (
                "The worktree left by an earlier run on the branch agent/OLD-7: The work "
                "of the run cannot be rescued, its worktree is kept: the container "
                "dev-worker-left of the instance is still running"
            ),
        ),
    ],
)
def test_container_left_that_cannot_be_stopped_fails_the_run_without_taking_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    tmp_path: Path,
    answers: dict[str, str],
    reason: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    record = fake_docker(monkeypatch, tmp_path, **answers)
    workspace.left = "agent/OLD-7"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert "create" not in docker_calls(record)
    assert ("rescue", "agent/OLD-7", "Example Worker", IDENTITY) not in workspace.calls
    assert stub.moves == []
    assert journal_lines(instance) == []


def spell_instance_path(
    instance: Path, spelling: str, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Return another path to the instance file, written the way `spelling` names."""
    if spelling == "relative":
        monkeypatch.chdir(instance.parent)
        return Path(instance.name)
    if spelling == "parent-step":
        (instance.parent / "elsewhere").mkdir()
        return instance.parent / "elsewhere" / ".." / instance.name
    if spelling == "symlink":
        link = instance.parent / "link"
        try:
            link.symlink_to(instance.parent, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symbolic links cannot be created here: {error}")
        return link / instance.name
    return instance


@pytest.mark.parametrize("spelling", ["absolute", "relative", "parent-step", "symlink"])
def test_containers_of_an_instance_carry_one_label_whatever_the_path_spelling(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    tmp_path: Path,
    spelling: str,
) -> None:
    record = fake_docker(monkeypatch, tmp_path)

    run_mandate(
        monkeypatch,
        capsysbinary,
        encode(valid_input()),
        instance=spell_instance_path(instance, spelling, monkeypatch),
    )

    owner = f"dev-worker.instance={instance.with_suffix('').resolve()}"
    calls = [json.loads(line) for line in record.read_text("utf-8").splitlines()]
    create_calls = [call for call in calls if call[0] == "create"]
    assert len(create_calls) == 2
    assert all(owner in create_call for create_call in create_calls)
    assert all(f"label={owner}" in call for call in calls if call[0] == "ps")


def test_local_branches_that_cannot_be_dropped_fail_the_run_without_taking_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "drop_left_branches"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        "The worktree left by an earlier run cannot be rescued: "
        "drop_left_branches broke",
    )
    assert stub.moves == []
    assert ("refresh",) not in workspace.calls
    assert journal_lines(instance) == []


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

    def record(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        called.append(mandate.task_id)
        return cli.deliver_ticket(mandate, resolved, outcome)

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
    assert answer["summary"] == SUMMARY
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
    def refuse(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
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
    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
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
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        lock.write_bytes(successor)
        return cli.deliver_ticket(mandate, resolved, outcome)

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

    def peek(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        seen.append(cli.read_state(instance).model_dump(mode="json"))
        return cli.deliver_ticket(mandate, resolved, outcome)

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", peek)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    (during,) = seen
    running, _ = journal_lines(instance)
    assert running["status"] == "running"
    assert common(during)["current"] == [
        {
            "task_id": "task-1",
            "ticket_key": "DEMO-1",
            "model": "opus",
            "review_model": "opus",
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
    def still_going(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        output = cli.deliver_ticket(mandate, resolved, outcome)
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
def test_health_is_degraded_naming_the_live_process_of_a_lock_beyond_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    pid = os.getpid()
    hold_lock(instance, lock_of(1801, pid=pid))
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "degraded",
        "reason": (
            f"The run holding the lock of instance directory "
            f"'{instance.with_suffix('')}' started 1801 seconds ago, beyond the "
            f"run timeout of 1800 seconds; its process {pid} still runs, so the "
            f"lock is not taken over."
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
        file.write("  run_timeout_seconds: 900\n")
    hold_lock(instance, lock_started(901))

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer["status"] == "degraded"
    assert "run timeout of 900 seconds" in str(answer["reason"])


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
        "model": "claude-opus-4-1",
        "review_model": "sonnet",
        "start": (NOW - timedelta(seconds=seconds_ago)).isoformat(),
    }
    record.update(fields)
    return json.dumps(record).encode()


def gone_line(seconds_ago: float, reason: str) -> dict[str, object]:
    """Return the journal line of the run of `lock_of(seconds_ago)`, gone for `reason`."""
    return {
        "task_id": "old",
        "ticket_key": "DEMO-7",
        "model": "claude-opus-4-1",
        "review_model": "sonnet",
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
def test_run_takes_over_a_lock_without_a_process_held_beyond_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_taken_over(answer, instance, gone_line(1801, timed_out(instance, 1801)))


@pytest.mark.usefixtures("frozen_now")
def test_run_refuses_a_lock_held_beyond_the_timeout_while_its_process_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    content = lock_of(1801, pid=os.getpid())
    lock = hold_lock(instance, content)
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert lock.read_bytes() == content
    assert journal_lines(instance) == []


@pytest.mark.usefixtures("frozen_now")
def test_run_takes_over_a_lock_held_beyond_the_timeout_whose_process_is_gone(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801, pid=dead_pid()))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    old, running, own = journal_lines(instance)
    assert {key: old[key] for key in ("task_id", "ticket_key", "status")} == {
        "task_id": "old",
        "ticket_key": "DEMO-7",
        "status": "failed",
    }
    assert running["status"] == "running"
    assert own["status"] == "completed"


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


def deny_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every process refuse to be inspected, as one of another user does."""

    def refuse(pid: int) -> psutil.Process:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", refuse)


@pytest.mark.usefixtures("frozen_now")
def test_run_takes_over_a_lock_beyond_the_timeout_whose_process_cannot_be_inspected(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801, pid=os.getpid()))
    deny_inspection(monkeypatch)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_taken_over(answer, instance, gone_line(1801, timed_out(instance, 1801)))


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("seconds_ago", [1800, 60])
def test_run_refuses_a_lock_within_the_timeout_whose_process_cannot_be_inspected(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    seconds_ago: int,
) -> None:
    content = lock_of(seconds_ago, pid=os.getpid())
    lock = hold_lock(instance, content)
    deny_inspection(monkeypatch)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", "A run is already in progress")
    assert lock.read_bytes() == content
    assert journal_lines(instance) == []


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
    assert old["model"] is None
    assert old["review_model"] is None
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


@pytest.mark.usefixtures("frozen_now")
def test_health_names_the_gone_process_rather_than_the_timeout_of_a_lock_beyond_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    pid = dead_pid()
    hold_lock(instance, lock_of(1801, pid=pid))

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "degraded",
        "reason": process_gone(instance, pid),
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("seconds_ago", [1800, 0])
def test_health_is_ok_while_the_live_process_of_the_lock_is_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    seconds_ago: int,
) -> None:
    hold_lock(instance, lock_of(seconds_ago, pid=os.getpid()))
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)

    assert health(monkeypatch, capsysbinary, instance) == {
        "status": "ok",
        "reason": None,
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
def test_health_ages_a_lock_whose_process_cannot_be_inspected_by_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801, pid=os.getpid()))
    deny_inspection(monkeypatch)

    assert health(monkeypatch, capsysbinary, instance) == {
        "status": "degraded",
        "reason": timed_out(instance, 1801),
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("seconds_ago", [1800, 0])
def test_health_is_ok_while_a_lock_whose_process_cannot_be_inspected_is_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    seconds_ago: int,
) -> None:
    hold_lock(instance, lock_of(seconds_ago, pid=os.getpid()))
    deny_inspection(monkeypatch)

    assert health(monkeypatch, capsysbinary, instance) == {
        "status": "ok",
        "reason": None,
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("pid", [True, 0, -1, "1", 1.0, None])
def test_health_ages_a_lock_whose_pid_is_not_a_positive_integer_by_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    pid: object,
) -> None:
    hold_lock(instance, lock_of(1801, pid=pid))
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)

    assert health(monkeypatch, capsysbinary, instance) == {
        "status": "degraded",
        "reason": timed_out(instance, 1801),
        "active": True,
    }


@pytest.mark.usefixtures("frozen_now")
@pytest.mark.parametrize("start", [None, "yesterday", "2026-11-05T10:00:00"])
def test_health_ages_a_lock_with_a_live_pid_but_no_usable_start_by_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    start: object,
) -> None:
    record: dict[str, object] = {"pid": os.getpid()}
    if start is not None:
        record["start"] = start
    lock = hold_lock(instance, json.dumps(record).encode())
    modified = NOW.timestamp() - 1801
    os.utime(lock, (modified, modified))
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer["status"] == "degraded"
    assert str(answer["reason"]).endswith("beyond the run timeout of 1800 seconds.")


@pytest.mark.usefixtures("frozen_now")
def test_run_journals_a_run_gone_beyond_the_timeout_by_its_gone_process(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    pid = dead_pid()
    hold_lock(instance, lock_of(1801, pid=pid))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert journal_lines(instance)[0]["errors"] == [process_gone(instance, pid)]


@pytest.mark.usefixtures("frozen_now")
def test_wake_does_not_take_over_a_lock_within_the_timeout_while_its_process_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    content = lock_of(60, pid=os.getpid(), ticket_key="DEMO-1")
    lock = hold_lock(instance, content)
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, _ = wake(capsysbinary, instance)

    assert code == 0
    assert stub.comments == []
    assert stub.moves == []
    assert lock.read_bytes() == content


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
    monkeypatch.setattr(cli, "make_repository", lambda *made: repository)

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
    monkeypatch.setattr(cli, "make_repository", lambda *made: repository)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "down",
        "reason": "The repository cannot be reached with the worker's token: "
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
    monkeypatch.delenv("GITHUB_WORKER_TOKEN", raising=False)

    tracker_down = health(monkeypatch, capsysbinary, instance)
    monkeypatch.setattr(cli, "make_tracker", lambda made: ReadyTracker())
    repository_down = health(monkeypatch, capsysbinary, instance)

    assert tracker_down["reason"] == (
        "The tracker cannot be reached: the variable JIRA_TOKEN, "
        "which should hold the Jira token, is not set"
    )
    assert repository_down["reason"] == (
        "The repository cannot be reached with the worker's token: the variable "
        "GITHUB_WORKER_TOKEN, which should hold the GitHub token, is not set"
    )
    assert not instance.with_suffix("").exists()


def test_health_is_down_naming_the_agent_token_when_only_it_cannot_reach_the_repository(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    checked: list[str | None] = []

    def make(made: Repository, token_env: str | None = None) -> object:
        checked.append(token_env)
        if token_env == "GITHUB_AGENT_TOKEN":
            return Unreachable(RepositoryError("refused"))
        return ReachableRepository()

    monkeypatch.setattr(cli, "make_repository", make)

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer["reason"] == (
        "The repository cannot be reached with the agent's token: refused"
    )
    assert checked == ["GITHUB_WORKER_TOKEN", "GITHUB_AGENT_TOKEN"]


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
        type="github",
        name="example/project",
        agent_token_env="GITHUB_AGENT_TOKEN",
        worker_token_env="GITHUB_WORKER_TOKEN",
    )

    assert isinstance(MAKE_REPOSITORY(repository), GitHubRepository)


@pytest.mark.parametrize(
    ("token_env", "read"),
    [
        (None, "GITHUB_WORKER_TOKEN"),
        ("GITHUB_WORKER_TOKEN", "GITHUB_WORKER_TOKEN"),
        ("GITHUB_AGENT_TOKEN", "GITHUB_AGENT_TOKEN"),
    ],
)
def test_make_repository_reads_the_token_variable_it_is_given_or_the_worker_one(
    monkeypatch: pytest.MonkeyPatch, token_env: str | None, read: str
) -> None:
    repository = Repository(
        type="github",
        name="example/project",
        agent_token_env="GITHUB_AGENT_TOKEN",
        worker_token_env="GITHUB_WORKER_TOKEN",
    )
    monkeypatch.delenv("GITHUB_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_WORKER_TOKEN", raising=False)

    with pytest.raises(RepositoryError, match=f"the variable {read},"):
        MAKE_REPOSITORY(repository, token_env).check_access()


def test_health_does_not_check_the_connectors_when_a_local_check_is_down(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    journal = write_invalid_journal(instance, b"[1]\n")
    tracker = Unreachable(TrackerError("refused"))
    repository = Unreachable(RepositoryError("refused"))
    monkeypatch.setattr(cli, "make_tracker", lambda made: tracker)
    monkeypatch.setattr(cli, "make_repository", lambda *made: repository)

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
    monkeypatch.setattr(cli, "make_repository", lambda *made: repository)

    assert health(monkeypatch, capsysbinary, instance)["status"] == "degraded"


def test_health_is_down_when_docker_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    container.error = ContainerError("docker could not be started: not found")

    answer = health(monkeypatch, capsysbinary, instance)

    assert answer == {
        "status": "down",
        "reason": "Docker cannot be reached: docker could not be started: not found",
        "active": True,
    }
    assert container.images == ["ghcr.io/example-org/example-app-agent:1.0.0"]
    assert container.calls == [("check_access",)]


def test_health_does_not_check_docker_when_a_connector_cannot_be_reached(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    monkeypatch.setattr(
        cli, "make_tracker", lambda made: Unreachable(TrackerError("refused"))
    )

    health(monkeypatch, capsysbinary, instance)

    assert container.calls == []


IN_PROGRESS = ("DEMO-1", "In Progress")
IN_REVIEW = ("DEMO-1", "In Review")
BLOCKED = ("DEMO-1", "Blocked")
MERGED = ("DEMO-1", "Done")
MERGED_BODY = f"{SUMMARY}\n\nMerged: {PULL.url}"


def ending_on(outcome: str, message: str) -> ContainerExit:
    """Return how a container ends when Claude Code ends on `outcome` with `message`."""
    structured = {"outcome": outcome, "message": message}
    return ContainerExit(
        0,
        json.dumps(
            {"type": "result", "subtype": "success", "structured_output": structured}
        ),
        "",
    )


def assert_blocked_run_fails(
    answer: dict[str, object], instance: Path, reason: str
) -> None:
    """Check that a run left blocked fails naming `reason` and is journaled failed."""
    assert_failed(answer, "task-1", reason)
    assert [line["status"] for line in journal_lines(instance)] == [
        "running",
        "failed",
    ]


def test_done_merges_the_pull_request_once_the_container_is_gone_then_comments_and_moves_the_ticket_to_done(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    seen: list[list[tuple[str, ...]]] = []
    find = reachable_repository.find_pull_request

    def spy(branch: str) -> PullRequest | None:
        seen.append(list(workspace.calls))
        return find(branch)

    monkeypatch.setattr(reachable_repository, "find_pull_request", spy)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["summary"] == SUMMARY
    assert answer["errors"] == []
    assert reachable_repository.calls == [
        ("find_pull_request", "agent/DEMO-1"),
        ("read_ci", PULL),
        ("merge", PULL),
    ]
    assert stub.comments == [("DEMO-1", MERGED_BODY)]
    assert stub.moves == [IN_PROGRESS, MERGED]
    assert seen[0][-2:] == [RESCUED, ("close", "agent/DEMO-1")]


def test_done_on_a_ticket_carrying_the_hold_label_leaves_the_pull_request_open_and_moves_it_to_review(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    find = reachable_repository.find_pull_request

    def label_then_find(branch: str) -> PullRequest | None:
        stub.answer = replace(demo_ticket("In Progress"), labels=("hold-merge",))
        return find(branch)

    monkeypatch.setattr(reachable_repository, "find_pull_request", label_then_find)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe pull request {PULL.url} is left open: the ticket "
                "carries the label 'hold-merge', which holds its merge back."
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, IN_REVIEW]


def test_done_without_a_pull_request_comments_why_moves_the_ticket_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.pull = None

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        "no open pull request has the branch agent/DEMO-1 as its head.",
    )
    assert reachable_repository.calls == [("find_pull_request", "agent/DEMO-1")]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe ticket is blocked: no open pull request has the "
                "branch agent/DEMO-1 as its head."
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_ci_has_not_finished_is_not_merged_moves_the_ticket_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.ci = "running"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer, instance, f"the CI of the pull request {PULL.url} has not finished"
    )
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe ticket is blocked: the CI of the pull request "
                f"{PULL.url} has not finished; it was not merged."
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_ci_failed_is_not_merged_and_goes_back_to_the_agent(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.ci = "failed"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    reason = f"the CI of the pull request {PULL.url} failed; it was not merged."
    assert_blocked_run_fails(answer, instance, f"went back to the agent: {reason}")
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert stub.comments == [("DEMO-1", f"{RETURN_MARKER}{reason} (attempt 1 of 3).")]
    assert stub.moves == [IN_PROGRESS, ("DEMO-1", "To Do")]


def test_done_whose_ci_has_reported_no_check_yet_is_not_merged_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.ci = "unreported"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    reason = (
        f"no CI check is reported yet on the head commit {PULL.head} of the pull "
        f"request {PULL.url}; it was not merged."
    )
    assert_blocked_run_fails(answer, instance, reason)
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert stub.comments == [
        ("DEMO-1", f"{SUMMARY}\n\nThe ticket is blocked: {reason}")
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


@pytest.mark.parametrize(
    ("step", "error", "calls"),
    [
        (
            "find_pull_request",
            RepositoryError("gh failed while finding the pull request: HTTP 502"),
            ["find_pull_request"],
        ),
        (
            "read_ci",
            HeadMovedError("the head of pull request #7 has moved"),
            ["find_pull_request", "read_ci"],
        ),
        (
            "read_ci",
            RepositoryError(
                "the repository octo/widgets has no active GitHub Actions workflow file "
                "and reports no CI check on pull request #7"
            ),
            ["find_pull_request", "read_ci"],
        ),
        (
            "merge",
            HeadMovedError("the head of pull request #7 has moved; it was not merged"),
            ["find_pull_request", "read_ci", "merge"],
        ),
    ],
)
def test_done_whose_repository_step_fails_comments_the_reason_moves_the_ticket_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
    step: str,
    error: RepositoryError,
    calls: list[str],
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.errors[step] = error

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(answer, instance, str(error))
    assert [name for name, _ in reachable_repository.calls] == calls
    assert stub.comments == [("DEMO-1", f"{SUMMARY}\n\nThe ticket is blocked: {error}")]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_merge_fails_says_it_may_have_been_merged_moves_the_ticket_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.errors["merge"] = RepositoryError(
        "gh did not answer within 60 seconds while merging pull request #7"
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        "gh did not answer within 60 seconds while merging pull request #7",
    )
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
        "merge",
    ]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe ticket is blocked: the pull request {PULL.url} "
                "cannot be merged: gh did not answer within 60 seconds while merging "
                "pull request #7 It may have been merged and is to be read again."
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_ticket_cannot_be_read_again_is_not_merged_moves_it_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    find = reachable_repository.find_pull_request

    def unreadable_then_find(branch: str) -> PullRequest | None:
        stub.answer = TrackerError("twg failed while reading DEMO-1: timeout")
        return find(branch)

    monkeypatch.setattr(reachable_repository, "find_pull_request", unreadable_then_find)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer, instance, "twg failed while reading DEMO-1: timeout"
    )
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe ticket is blocked: twg failed while reading "
                "DEMO-1: timeout"
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


@pytest.mark.parametrize("outcome", ["blocked", "needs_human", "abandoned"])
def test_outcome_other_than_done_comments_the_reason_moves_the_ticket_to_blocked_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
    outcome: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.exited = ending_on(outcome, "The ticket has no acceptance criteria.")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(
        answer,
        "task-1",
        f"The run ended on the outcome {outcome!r}: "
        "The ticket has no acceptance criteria.",
    )
    assert stub.comments == [("DEMO-1", "The ticket has no acceptance criteria.")]
    assert stub.moves == [IN_PROGRESS, BLOCKED]
    assert reachable_repository.calls == []


@pytest.mark.parametrize(
    ("exited", "reason"),
    [
        pytest.param(
            ContainerExit(0, "Error: boom", ""),
            "The run ended without a usable outcome: "
            "the output of Claude Code does not end on a result.",
            id="not-json",
        ),
        pytest.param(
            ContainerExit(
                1,
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "error_max_turns",
                        "is_error": True,
                        "result": "Max.",
                    }
                ),
                "",
            ),
            "The run ended without a usable outcome: Claude Code exited with code 1 "
            "(subtype error_max_turns, result: Max.).",
            id="claude-code-fails",
        ),
        pytest.param(
            ending_on("merged", "Done."),
            "The run ended without a usable outcome: the structured output of "
            "Claude Code does not match the outcome schema: its outcome 'merged' is "
            "not one of done, blocked, needs_human, abandoned.",
            id="off-schema",
        ),
    ],
)
def test_run_without_a_usable_outcome_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
    exited: ContainerExit,
    reason: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.exited = exited

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]
    assert reachable_repository.calls == []


def test_container_reaching_the_ceiling_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.error = ContainerError(
        "the container dev-worker-1 was still running when the run reached its "
        "ceiling and was removed"
    )
    container.failing = ("run",)
    reason = f"The container failed: {container.error}"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


@pytest.mark.parametrize("message", ["x\x00y", "x\ud800y"], ids=["nul", "surrogate"])
def test_container_failing_with_a_nul_or_a_lone_surrogate_comments_a_writable_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    message: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.error = ContainerError(message)
    container.failing = ("run",)
    reason = f"The container failed: {container.error}"

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert stub.comments == [("DEMO-1", writable_text(reason))]
    body = stub.comments[0][1]
    assert "\x00" not in body
    body.encode("utf-8")
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_plugin_that_cannot_be_fetched_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    plugin: FakePlugin,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    plugin.error = PluginError("git failed while cloning: no such tag")
    reason = "The plugin cannot be fetched: git failed while cloning: no such tag"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_worktree_that_cannot_be_opened_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "open"
    reason = "The worktree cannot be prepared: open broke"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_missing_token_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
    reason = (
        "The container was not started: the variable CLAUDE_CODE_OAUTH_TOKEN, which "
        "should hold the Claude Code subscription token, is not set."
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_ceiling_reached_before_the_container_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    slow_refresh(monkeypatch, workspace, 1200)
    reason = (
        "The container was not started: the run ceiling of 1800 seconds, less the "
        "600 seconds kept for the guard review, was reached before it."
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_running_line_that_cannot_be_written_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())

    def full(directory: Path, entry: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cli, "append_entry", full)
    reason = (
        f"Journal in '{instance.with_suffix('')}' cannot be written: "
        "No space left on device"
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_failed(answer, "task-1", reason)
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_work_cannot_be_rescued_comments_why_and_moves_the_ticket_to_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "rescue"
    reason = "The work of the run cannot be rescued, its worktree is kept: rescue broke"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "failed"
    assert answer["errors"] == [reason]
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_run_without_outcome_whose_work_cannot_be_rescued_comments_both_reasons(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "rescue"
    container.exited = ContainerExit(0, "Error: boom", "")

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    reason = (
        "The run ended without a usable outcome: the output of Claude Code does "
        "not end on a result. The work of the run cannot be rescued, its worktree "
        "is kept: rescue broke"
    )
    assert stub.comments == [("DEMO-1", reason)]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_whose_worktree_cannot_be_removed_still_merges_and_moves_the_ticket_to_done(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "close"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == ["The worktree cannot be removed: close broke"]
    assert stub.comments == [("DEMO-1", MERGED_BODY)]
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_done_whose_worktree_cannot_be_removed_and_ticket_ends_blocked_reports_both_reasons(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    workspace.fail = "close"
    reachable_repository.ci = "running"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    errors: Any = answer["errors"]
    assert answer["status"] == "failed"
    assert errors == [
        (
            "The run ended on 'done' but its ticket is blocked: the CI of the pull "
            f"request {PULL.url} has not finished; it was not merged."
        ),
        "The worktree cannot be removed: close broke",
    ]


def test_comment_that_fails_is_reported_and_the_ticket_is_still_moved(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    stub.comment_error = TrackerError("twg failed while commenting on DEMO-1: timeout")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["errors"] == [
        (
            "Ticket 'DEMO-1' cannot be commented on: twg failed while commenting on "
            "DEMO-1: timeout The comment may have been posted."
        )
    ]
    assert stub.comments == [("DEMO-1", MERGED_BODY)]
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_ticket_that_cannot_be_moved_at_the_end_is_reported_with_its_status_unknown(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    move = stub.set_status

    def refuse_done(key: str, status: str) -> None:
        move(key, status)
        if status == "Done":
            raise TrackerError("twg failed while moving DEMO-1: timeout")

    monkeypatch.setattr(stub, "set_status", refuse_done)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["errors"] == [
        (
            "Ticket 'DEMO-1' cannot be moved to 'Done': twg failed while moving "
            "DEMO-1: timeout Its status is unknown and is to be read again in the "
            "tracker."
        )
    ]
    assert stub.comments == [("DEMO-1", MERGED_BODY)]
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_ticket_is_commented_on_before_it_is_moved_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    calls: list[str] = []
    post, move = stub.comment, stub.set_status

    def spy_comment(key: str, body: str) -> None:
        calls.append("comment")
        post(key, body)

    def spy_move(key: str, status: str) -> None:
        calls.append(status)
        move(key, status)

    monkeypatch.setattr(stub, "comment", spy_comment)
    monkeypatch.setattr(stub, "set_status", spy_move)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert calls == ["In Progress", "comment", "Done"]


def test_comment_and_move_that_both_fail_at_the_end_are_both_reported_in_order(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.exited = ending_on("blocked", "No spec.")
    stub.comment_error = TrackerError("comment broke.")
    move = stub.set_status

    def refuse_blocked(key: str, status: str) -> None:
        move(key, status)
        if status == "Blocked":
            raise TrackerError("move broke.")

    monkeypatch.setattr(stub, "set_status", refuse_blocked)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "failed"
    assert answer["errors"] == [
        "The run ended on the outcome 'blocked': No spec.",
        (
            "Ticket 'DEMO-1' cannot be commented on: comment broke. "
            "The comment may have been posted."
        ),
        (
            "Ticket 'DEMO-1' cannot be moved to 'Blocked': move broke. "
            "Its status is unknown and is to be read again in the tracker."
        ),
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_ticket_is_never_moved_back_to_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, replace(demo_ticket(), labels=("hold-merge",)))
    worker = cli.load_worker()
    outcomes = worker.header.outcomes.model_copy(update={"done": "ready"})
    header = worker.header.model_copy(update={"outcomes": outcomes})
    monkeypatch.setattr(cli, "load_worker", lambda: replace(worker, header=header))

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == [
        (
            "Ticket 'DEMO-1' was not moved: the run ended tied to 'To Do', and a "
            "ticket is never moved back to the ready status."
        )
    ]
    assert [key for key, _ in stub.comments] == ["DEMO-1"]
    assert stub.moves == [IN_PROGRESS]


def test_ticket_is_left_as_it_is_when_the_mandate_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = serve(monkeypatch, demo_ticket())

    def explode(
        mandate: MandateInput, resolved: Instance, outcome: Outcome
    ) -> MandateOutput:
        raise KeyboardInterrupt

    monkeypatch.setitem(cli.MANDATES, "deliver_ticket", explode)

    with pytest.raises(KeyboardInterrupt):
        run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert stub.comments == []
    assert stub.moves == [IN_PROGRESS]


@pytest.mark.parametrize("message", ["x\x00y", "x\ud800y"], ids=["nul", "surrogate"])
def test_done_whose_summary_holds_a_nul_or_a_lone_surrogate_still_comments_moves_and_answers(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    message: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.exited = ending_on("done", message)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    summary = answer["summary"]
    assert isinstance(summary, str)
    assert "\x00" not in summary
    assert [key for key, _ in stub.comments] == ["DEMO-1"]
    body = stub.comments[0][1]
    assert "\x00" not in body
    body.encode("utf-8")
    assert stub.moves == [IN_PROGRESS, MERGED]


def review_verdict(*defects: dict[str, object]) -> ContainerExit:
    """Return a guard review container that ended on a verdict listing `defects`."""
    return ContainerExit(
        0,
        json.dumps(
            {
                "type": "result",
                "structured_output": {
                    "reviewed": True,
                    "reason": "",
                    "defects": list(defects),
                },
            }
        ),
        "",
    )


def test_done_is_reviewed_read_only_without_repository_or_tracker_tokens_then_merged_at_the_reviewed_head(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert container.readonly == [False, True]
    step, mounted, time_left, plugin_checkout = container.calls[-1]
    assert (step, mounted, plugin_checkout) == ("run", workspace.review, CHECKOUT)
    assert isinstance(time_left, float)
    assert 0 < time_left <= 1800
    assert container.commands[-1] == review_command(BASE_COMMIT, PULL.head, 40, "opus")
    review_environment = container.environments[-1]
    assert "GH_TOKEN" not in review_environment
    assert "TWG_TOKEN" not in review_environment
    assert review_environment["GIT_CONFIG_COUNT"] == "1"
    assert review_environment["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert review_environment["GIT_CONFIG_VALUE_0"] == "/workspace"
    assert "worker-github-token" not in review_environment.values()
    assert workspace.calls[-3:] == REVIEWED
    assert reachable_repository.calls[-1] == ("merge", PULL)
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_review_max_turns_of_the_instance_caps_the_review(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    with instance.open("a", encoding="utf-8") as file:
        file.write("  review_max_turns: 12\n")

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.commands[-1] == review_command(BASE_COMMIT, PULL.head, 12, "opus")


def test_done_whose_review_finds_defects_is_not_merged_and_goes_back_to_the_agent_listing_them(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.verdict = review_verdict(
        {"file": "src/app.py", "line": 12, "summary": "The token is logged."},
        {"file": "src/db.py", "line": 3, "summary": "The query is concatenated."},
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    reason = (
        f"the guard review of the pull request {PULL.url} found reachable defects; "
        "it was not merged.\n"
        "- src/app.py:12 The token is logged.\n"
        "- src/db.py:3 The query is concatenated."
    )
    assert_blocked_run_fails(answer, instance, f"went back to the agent: {reason}")
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.comments == [("DEMO-1", f"{RETURN_MARKER}{reason} (attempt 1 of 3).")]
    assert stub.moves == [IN_PROGRESS, ("DEMO-1", "To Do")]
    assert workspace.calls[-1] == ("close_review",)


@pytest.mark.parametrize(
    ("verdict", "reason"),
    [
        pytest.param(
            ContainerExit(
                0,
                json.dumps(
                    {
                        "structured_output": {
                            "reviewed": True,
                            "reason": "",
                            "defects": 3,
                        }
                    }
                ),
                "",
            ),
            "gave no verdict: the structured output of Claude Code does not match "
            "the verdict schema: its defects are missing or not a list.",
            id="off-schema",
        ),
        pytest.param(
            ContainerExit(0, json.dumps({"structured_output": {"defects": []}}), ""),
            "gave no verdict: the structured output of Claude Code does not match "
            "the verdict schema: its reviewed is missing or not a boolean.",
            id="reviewed-missing",
        ),
        pytest.param(
            ContainerExit(
                0,
                json.dumps(
                    {
                        "structured_output": {
                            "reviewed": False,
                            "reason": "gh pr checks needs a token.",
                            "defects": [],
                        }
                    }
                ),
                "",
            ),
            "gave no verdict: the review did not take place: gh pr checks needs a "
            "token..",
            id="not-reviewed",
        ),
        pytest.param(
            ContainerExit(
                1,
                json.dumps({"subtype": "error_max_turns", "is_error": True}),
                "",
            ),
            "gave no verdict: Claude Code exited with code 1 "
            "(subtype error_max_turns).",
            id="turn-ceiling",
        ),
    ],
)
def test_done_whose_review_gives_no_verdict_is_not_merged_naming_why(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
    verdict: ContainerExit,
    reason: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    container.verdict = verdict

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    full = f"the guard review of the pull request {PULL.url} {reason}"
    assert_blocked_run_fails(answer, instance, full)
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.moves == [IN_PROGRESS, BLOCKED]
    assert workspace.calls[-1] == ("close_review",)


def test_done_whose_review_container_fails_is_not_merged_naming_why(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    run = container.run

    def fail_review(*arguments: Any, readonly: bool = False) -> ContainerExit:
        if readonly:
            raise ContainerError("the image is gone")
        return run(*arguments)

    monkeypatch.setattr(container, "run", fail_review)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        f"the guard review of the pull request {PULL.url} failed: the image is gone.",
    )
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.moves == [IN_PROGRESS, BLOCKED]
    assert workspace.calls[-1] == ("close_review",)


@pytest.mark.parametrize("step", ["open_review", "refresh"])
def test_done_whose_review_checkout_cannot_be_made_is_not_reviewed_nor_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
    step: str,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    open_worktree = workspace.open

    def open_then_fail(branch: str) -> Path:
        opened = open_worktree(branch)
        workspace.fail = step
        return opened

    monkeypatch.setattr(workspace, "open", open_then_fail)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        f"the guard review of the pull request {PULL.url} did not run, its "
        f"checkout cannot be made: {step} broke",
    )
    assert container.readonly == [False]
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.moves == [IN_PROGRESS, BLOCKED]
    assert workspace.calls[-1] == ("close_review",)


def test_done_whose_review_plugin_cannot_be_fetched_is_not_reviewed_nor_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    plugin: FakePlugin,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    find = reachable_repository.find_pull_request

    def find_then_break_plugin(branch: str) -> PullRequest | None:
        plugin.error = PluginError("the tag is gone")
        return find(branch)

    monkeypatch.setattr(
        reachable_repository, "find_pull_request", find_then_break_plugin
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        f"the guard review of the pull request {PULL.url} did not run, its plugin "
        "cannot be fetched: the tag is gone",
    )
    assert container.readonly == [False]
    assert "merge" not in [name for name, _ in reachable_repository.calls]


def test_done_whose_review_credential_is_gone_is_not_reviewed_nor_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    find = reachable_repository.find_pull_request

    def find_then_drop_credential(branch: str) -> PullRequest | None:
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN")
        return find(branch)

    monkeypatch.setattr(
        reachable_repository, "find_pull_request", find_then_drop_credential
    )

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        f"the guard review of the pull request {PULL.url} did not run. The "
        "container was not started: the variable CLAUDE_CODE_OAUTH_TOKEN, which "
        "should hold the Claude Code subscription token, is not set.",
    )
    assert container.readonly == [False]
    assert "merge" not in [name for name, _ in reachable_repository.calls]


def test_done_with_no_time_left_for_the_review_is_not_reviewed_nor_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    clock = [1000.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    refresh = workspace.refresh

    def refresh_late() -> None:
        refresh()
        if ("close", "agent/DEMO-1") in workspace.calls:
            clock[0] += 1800

    monkeypatch.setattr(workspace, "refresh", refresh_late)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(
        answer,
        instance,
        f"the guard review of the pull request {PULL.url} did not run: the run "
        "ceiling of 1800 seconds was reached before it.",
    )
    assert container.readonly == [False]
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert workspace.calls[-1] == ("close_review",)


def test_done_whose_ci_has_not_passed_runs_no_review(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    reachable_repository.ci = "failed"

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.readonly == [False]
    assert ("close_review",) not in workspace.calls


def test_done_on_a_ticket_carrying_the_hold_label_runs_no_review(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    find = reachable_repository.find_pull_request

    def label_then_find(branch: str) -> PullRequest | None:
        stub.answer = replace(demo_ticket("In Progress"), labels=("hold-merge",))
        return find(branch)

    monkeypatch.setattr(reachable_repository, "find_pull_request", label_then_find)

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert container.readonly == [False]
    assert ("close_review",) not in workspace.calls


def test_done_whose_ticket_gets_the_hold_label_during_the_review_is_left_open(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    run = container.run

    def label_during_review(*arguments: Any, readonly: bool = False) -> ContainerExit:
        if readonly:
            stub.answer = replace(demo_ticket("In Progress"), labels=("hold-merge",))
        return run(*arguments, readonly=readonly)

    monkeypatch.setattr(container, "run", label_during_review)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert container.readonly == [False, True]
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{SUMMARY}\n\nThe pull request {PULL.url} is left open: the ticket "
                "carries the label 'hold-merge', which holds its merge back."
            ),
        )
    ]
    assert stub.moves == [IN_PROGRESS, IN_REVIEW]


def test_done_whose_ticket_cannot_be_read_after_the_review_is_blocked_and_not_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    run = container.run

    def tracker_down_after_review(
        *arguments: Any, readonly: bool = False
    ) -> ContainerExit:
        if readonly:
            stub.answer = TrackerError("the tracker is down.")
        return run(*arguments, readonly=readonly)

    monkeypatch.setattr(container, "run", tracker_down_after_review)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(answer, instance, "the tracker is down.")
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


UNDELETED = "The review checkout cannot be deleted: close_review broke"


def test_done_whose_clean_review_checkout_cannot_be_deleted_is_merged_reporting_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    workspace.fail = "close_review"

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["errors"] == [UNDELETED]
    assert reachable_repository.calls[-1] == ("merge", PULL)
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_done_whose_review_gives_no_verdict_and_whose_checkout_cannot_be_deleted_is_blocked_reporting_both(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    serve(monkeypatch, demo_ticket())
    workspace.fail = "close_review"
    container.verdict = ContainerExit(0, json.dumps({"structured_output": {}}), "")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    reason = (
        f"the guard review of the pull request {PULL.url} gave no verdict: the "
        "structured output of Claude Code does not match the verdict schema: its "
        "reviewed is missing or not a boolean."
    )
    assert answer["status"] == "failed"
    assert answer["errors"] == [
        f"The run ended on 'done' but its ticket is blocked: {reason}",
        UNDELETED,
    ]
    assert "merge" not in [name for name, _ in reachable_repository.calls]


@pytest.mark.parametrize("removable", [True, False], ids=["removed", "kept"])
def test_review_interrupted_deletes_its_checkout_and_raises_again(
    monkeypatch: pytest.MonkeyPatch,
    workspace: FakeWorkspace,
    container: FakeContainer,
    removable: bool,
) -> None:
    instance = cli.load_instance(INSTANCE, cli.load_worker().header)

    def interrupt(*arguments: Any, readonly: bool = False) -> ContainerExit:
        raise KeyboardInterrupt

    monkeypatch.setattr(container, "run", interrupt)
    if not removable:
        workspace.fail = "close_review"

    with pytest.raises(KeyboardInterrupt) as raised:
        cli.review_pull_request(
            PULL, instance, cli.load_worker(), Path("instance"), cli.time.monotonic()
        )

    assert workspace.calls[-1] == ("close_review",)
    notes = getattr(raised.value, "__notes__", [])
    assert notes == (
        []
        if removable
        else ["The review checkout cannot be deleted: close_review broke"]
    )


class ListingTracker(StubTracker):
    """A `StubTracker` that lists `listed` as the tickets in the example `ready` status, raising it when it is an error, and no ticket in any other status.

    `listings` records each status listed.
    """

    def __init__(
        self, listed: tuple[Ticket, ...] | TrackerError, answer: Ticket | TrackerError
    ) -> None:
        super().__init__(answer)
        self.listed = listed
        self.listings: list[str] = []

    def tickets_in(self, status: str) -> tuple[Ticket, ...]:
        """Record `status`, then return or raise `listed` for `To Do`, nothing for another status."""
        self.listings.append(status)
        if status != "To Do":
            return ()
        if isinstance(self.listed, TrackerError):
            raise self.listed
        return self.listed


def list_tickets(
    monkeypatch: pytest.MonkeyPatch,
    listed: tuple[Ticket, ...] | TrackerError | None = None,
    answer: Ticket | TrackerError | None = None,
) -> ListingTracker:
    """Make `wake` list `listed` and read `answer` from a `ListingTracker`; return it.

    Both default to the ready ticket DEMO-1 alone.
    """
    stub = ListingTracker(
        (demo_ticket(),) if listed is None else listed,
        demo_ticket() if answer is None else answer,
    )
    monkeypatch.setattr(cli, "make_tracker", stub.make)
    return stub


def wake(
    capsysbinary: pytest.CaptureFixture[bytes], instance: Path
) -> tuple[int, dict[str, Any] | None]:
    """Call `main` with `wake`; return its exit code and parsed answer, None when it wrote nothing."""
    code = cli.main(["wake", "--instance", str(instance)])
    out = capsysbinary.readouterr().out
    return code, json.loads(out.decode("utf-8")) if out else None


def ticket_on(key: str, *blockers: Blocker) -> Ticket:
    """Return the ready ticket `key`, blocked by `blockers`."""
    return replace(demo_ticket("To Do", *blockers), key=key)


def test_wake_without_a_ready_ticket_does_nothing_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = list_tickets(monkeypatch, listed=())

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.listings == ["In Progress", "To Do"]
    assert stub.reads == []
    assert not (instance.with_suffix("") / cli.LOCK_FILE).exists()


def test_wake_whose_ready_tickets_are_all_blocked_does_nothing_and_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = list_tickets(
        monkeypatch,
        listed=(
            ticket_on("DEMO-1", Blocker("DEMO-9", "In Progress")),
            ticket_on("DEMO-2", Blocker("DEMO-8", "Done"), Blocker("DEMO-7", "To Do")),
        ),
    )

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.reads == []
    assert stub.posted == []


def test_wake_runs_the_first_ticket_by_rank_whose_blockers_are_all_done(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    picked = ticket_on("DEMO-2", Blocker("DEMO-8", "Done"))
    stub = list_tickets(
        monkeypatch,
        listed=(
            ticket_on("DEMO-3", Blocker("DEMO-9", "In Progress")),
            picked,
            ticket_on("DEMO-1"),
        ),
        answer=picked,
    )

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert answer["contract_version"] == "0.1"
    task_id = answer["task_id"]
    assert isinstance(task_id, str) and task_id.startswith("wake-")
    assert set(stub.reads) == {"DEMO-2"}
    assert stub.posted == [task_id]
    assert ("DEMO-2", "In Progress") in stub.moves
    assert [
        (line["task_id"], line["ticket_key"]) for line in journal_lines(instance)
    ] == [
        (task_id, "DEMO-2"),
        (task_id, "DEMO-2"),
    ]


def test_each_wake_runs_under_a_task_id_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    list_tickets(monkeypatch)
    _, first = wake(capsysbinary, instance)
    list_tickets(monkeypatch)
    _, second = wake(capsysbinary, instance)

    assert first is not None and second is not None
    assert first["task_id"] != second["task_id"]


def test_wake_on_an_instance_that_cannot_be_loaded_fails_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    tmp_path: Path,
) -> None:
    stub = list_tickets(monkeypatch)

    code, answer = wake(capsysbinary, tmp_path / "missing.yaml")

    assert code == 1
    assert answer is not None
    assert_failed(answer, None, "cannot be read")
    assert stub.listings == []


def test_wake_whose_tickets_cannot_be_listed_fails_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    list_tickets(monkeypatch, listed=TrackerError("twg failed: down"))

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        None,
        "The tickets in 'To Do' cannot be listed from the tracker: twg failed: down",
    )
    assert not (instance.with_suffix("") / cli.LOCK_FILE).exists()


def blocked_outcome(container: FakeContainer) -> None:
    """Make the run container end on the `blocked` outcome."""
    container.exited = ContainerExit(
        0,
        json.dumps(
            {
                "type": "result",
                "structured_output": {"outcome": "blocked", "message": "Unclear."},
            }
        ),
        "",
    )


@dataclass
class Wake:
    """What a wake test may arrange before the wake."""

    monkeypatch: pytest.MonkeyPatch
    instance: Path
    stub: ListingTracker
    workspace: FakeWorkspace
    container: FakeContainer
    plugin: FakePlugin
    repository: ReachableRepository


def hold_the_lock(setup: Wake) -> None:
    hold_lock(setup.instance, b"held by another run")


def lose_claim(setup: Wake) -> None:
    setup.stub.posted.append("rival")


def leave_ready_on_claim(setup: Wake) -> None:
    claim = setup.stub.claim

    def claim_then_move(key: str, task_id: str) -> int:
        setup.stub.answer = demo_ticket("In Progress")
        return claim(key, task_id)

    setup.monkeypatch.setattr(setup.stub, "claim", claim_then_move)


def taken_before_read(setup: Wake) -> None:
    setup.stub.answer = demo_ticket("In Progress")


def blocked_before_read(setup: Wake) -> None:
    setup.stub.answer = demo_ticket("To Do", Blocker("DEMO-9", "In Progress"))


def end_blocked(setup: Wake) -> None:
    blocked_outcome(setup.container)


def merge(setup: Wake) -> None:
    pass


def hold_merge(setup: Wake) -> None:
    setup.stub.answer = replace(demo_ticket(), labels=("hold-merge",))


def fail_ci(setup: Wake) -> None:
    setup.repository.ci = "failed"


def unfinished_ci(setup: Wake) -> None:
    setup.repository.ci = "running"


def unreported_ci(setup: Wake) -> None:
    setup.repository.ci = "unreported"


def no_pull_request(setup: Wake) -> None:
    setup.repository.pull = None


def find_defects(setup: Wake) -> None:
    setup.container.verdict = review_verdict(
        {"file": "src/app.py", "line": 12, "summary": "The token is logged."}
    )


def move_the_head(setup: Wake) -> None:
    setup.repository.errors["merge"] = HeadMovedError(
        "the head of pull request #7 moved since its review"
    )


def after_the_run(setup: Wake, act: Callable[[], None]) -> None:
    """Make the run container call `act` once it has run, before the ticket is settled."""
    run = setup.container.run

    def run_then_act(*arguments: Any, readonly: bool = False) -> ContainerExit:
        exited = run(*arguments, readonly=readonly)
        if not readonly:
            act()
        return exited

    setup.monkeypatch.setattr(setup.container, "run", run_then_act)


def during_the_review(setup: Wake, act: Callable[[], None]) -> None:
    """Make the guard review container call `act` once it has run."""
    run = setup.container.run

    def review_then_act(*arguments: Any, readonly: bool = False) -> ContainerExit:
        exited = run(*arguments, readonly=readonly)
        if readonly:
            act()
        return exited

    setup.monkeypatch.setattr(setup.container, "run", review_then_act)


def hold_merge_during_the_review(setup: Wake) -> None:
    def label() -> None:
        setup.stub.answer = replace(demo_ticket("In Progress"), labels=("hold-merge",))

    during_the_review(setup, label)


@pytest.mark.parametrize(
    ("arrange", "fragment"),
    [
        (unfinished_ci, "The run ended on 'done' but its ticket is blocked"),
        (unreported_ci, "no CI check is reported yet on the head commit"),
        (no_pull_request, "no open pull request has the branch"),
        (find_defects, "found reachable defects"),
        (move_the_head, "the head of pull request #7 moved since its review"),
        (hold_merge_during_the_review, None),
        (hold_the_lock, "A run is already in progress"),
        (lose_claim, "is claimed by task 'rival'"),
        (leave_ready_on_claim, "is no longer in 'To Do'"),
        (taken_before_read, "is in status 'In Progress', not 'To Do'"),
        (blocked_before_read, "is blocked by DEMO-9 ('In Progress')"),
        (end_blocked, "The run ended on the outcome 'blocked': Unclear."),
        (merge, None),
        (hold_merge, None),
        (fail_ci, "The run ended on 'done' but its ticket went back to the agent"),
    ],
)
def test_wake_that_yields_or_settles_its_ticket_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    plugin: FakePlugin,
    reachable_repository: ReachableRepository,
    arrange: Callable[[Wake], None],
    fragment: str | None,
) -> None:
    stub = list_tickets(monkeypatch)
    arrange(
        Wake(
            monkeypatch,
            instance,
            stub,
            workspace,
            container,
            plugin,
            reachable_repository,
        )
    )

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    if fragment is None:
        assert answer["status"] == "completed"
        assert answer["errors"] == []
    else:
        assert_failed(answer, answer["task_id"], fragment)
        assert len(answer["errors"]) == 1


def unreadable_ticket(setup: Wake) -> None:
    setup.stub.answer = TrackerError("twg failed: down")


def unwritable_lock(setup: Wake) -> None:
    def refuse(directory: Path, record: object) -> Path:
        raise PermissionError(13, "Permission denied")

    setup.monkeypatch.setattr(cli, "take_lock", refuse)


def unreadable_journal(setup: Wake) -> None:
    def refuse(directory: Path, task_id: str) -> bool:
        raise PermissionError(13, "Permission denied")

    setup.monkeypatch.setattr(cli, "journal_holds", refuse)


def unrescuable_worktree(setup: Wake) -> None:
    setup.workspace.left = "agent/DEMO-7"
    setup.workspace.fail = "rescue"


def unpostable_claim(setup: Wake) -> None:
    def refuse(key: str, task_id: str) -> int:
        raise TrackerError("twg failed: down")

    setup.monkeypatch.setattr(setup.stub, "claim", refuse)


def unreadable_claims(setup: Wake) -> None:
    def refuse(key: str, status: str) -> tuple[Claim, ...]:
        raise TrackerError("twg failed: down")

    setup.monkeypatch.setattr(setup.stub, "claims", refuse)


def unmovable_ticket(setup: Wake) -> None:
    setup.stub.move_error = TrackerError("twg failed: down")


def unmovable_ticket_at_the_end(setup: Wake) -> None:
    set_status = setup.stub.set_status

    def refuse_done(key: str, status: str) -> None:
        if status == "Done":
            raise TrackerError("twg failed: down")
        set_status(key, status)

    setup.monkeypatch.setattr(setup.stub, "set_status", refuse_done)


def uncommentable_ticket(setup: Wake) -> None:
    blocked_outcome(setup.container)
    setup.stub.comment_error = TrackerError("twg failed: down")


def unwritable_running_line(setup: Wake) -> None:
    def full(directory: Path, entry: object) -> None:
        raise OSError(28, "No space left on device")

    setup.monkeypatch.setattr(cli, "append_entry", full)


def unwritable_final_line(setup: Wake) -> None:
    fail_on_final_line(setup.monkeypatch)


def unremovable_lock(setup: Wake) -> None:
    def stuck(path: Path, *record: object) -> None:
        raise PermissionError(13, "Permission denied")

    setup.monkeypatch.setattr(cli, "release_lock", stuck)


def unfetchable_plugin(setup: Wake) -> None:
    setup.plugin.error = PluginError("git failed while cloning: no such tag")


def unremovable_worktree(setup: Wake) -> None:
    setup.workspace.fail = "close"


def unfound_claim(setup: Wake) -> None:
    def none_found(key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        return ()

    setup.monkeypatch.setattr(setup.stub, "claims", none_found)


def overdue_claims(setup: Wake) -> None:
    wall = [CLAIM_EPOCH.timestamp()]
    setup.monkeypatch.setattr(cli.time, "time", lambda: wall[0])
    claims = setup.stub.claims

    def late(key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        wall[0] += cli.CLAIM_TAKE_LIMIT_SECONDS + 1
        return claims(key, statuses)

    setup.monkeypatch.setattr(setup.stub, "claims", late)


def unpreparable_worktree(setup: Wake) -> None:
    setup.workspace.fail = "refresh"


def failing_container(setup: Wake) -> None:
    setup.container.error = ContainerError("docker failed while running: no")
    setup.container.failing = ("run",)


def ceiling_reached(setup: Wake) -> None:
    slow_refresh(setup.monkeypatch, setup.workspace, 1801.0)


def unreadable_outcome(setup: Wake) -> None:
    setup.container.exited = ContainerExit(0, "not an outcome", "")


def uncreatable_directory(setup: Wake) -> None:
    setup.instance.with_suffix("").write_bytes(b"a file, not a directory")


@pytest.mark.parametrize(
    ("arrange", "fragment"),
    [
        (unreadable_ticket, "cannot be read from the tracker: twg failed: down"),
        (unwritable_lock, "cannot be written: Permission denied"),
        (unreadable_journal, "cannot be read or repaired: Permission denied"),
        (unrescuable_worktree, "The worktree left by an earlier run"),
        (unpostable_claim, "cannot be claimed for task"),
        (unreadable_claims, "cannot be read: twg failed: down"),
        (unmovable_ticket, "cannot be moved to 'In Progress'"),
        (unmovable_ticket_at_the_end, "cannot be moved to 'Done'"),
        (uncommentable_ticket, "cannot be commented on"),
        (unwritable_running_line, "cannot be written: No space left on device"),
        (unwritable_final_line, "cannot be written: No space left on device"),
        (unremovable_lock, "cannot be removed: Permission denied"),
        (unfetchable_plugin, "The plugin cannot be fetched"),
        (unremovable_worktree, "The worktree cannot be removed"),
        (unfound_claim, "is not found among the claims"),
        (overdue_claims, "were read back more than 300 seconds"),
        (unpreparable_worktree, "The worktree cannot be prepared: refresh broke"),
        (failing_container, "The container failed: docker failed while running"),
        (ceiling_reached, "less the 600 seconds kept for the guard review"),
        (unreadable_outcome, "The run ended without a usable outcome"),
        (uncreatable_directory, "cannot be created"),
    ],
)
def test_wake_whose_worker_fails_exits_with_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    plugin: FakePlugin,
    reachable_repository: ReachableRepository,
    arrange: Callable[[Wake], None],
    fragment: str,
) -> None:
    stub = list_tickets(monkeypatch)
    arrange(
        Wake(
            monkeypatch,
            instance,
            stub,
            workspace,
            container,
            plugin,
            reachable_repository,
        )
    )

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    errors = answer["errors"]
    assert any(fragment in error for error in errors), errors


def unfindable_pull_request(setup: Wake) -> None:
    setup.repository.errors["find_pull_request"] = RepositoryError(
        "gh failed while listing pull requests: down"
    )


def unreadable_ci(setup: Wake) -> None:
    setup.repository.errors["read_ci"] = RepositoryError(
        "gh failed while reading checks: down"
    )


def ci_without_workflow(setup: Wake) -> None:
    setup.repository.errors["read_ci"] = RepositoryError(
        "the repository octo/widgets has no active GitHub Actions workflow file and "
        "reports no CI check on pull request #7"
    )


def unreadable_ticket_before_the_review(setup: Wake) -> None:
    def tracker_down() -> None:
        setup.stub.answer = TrackerError("twg failed while reading DEMO-1: down")

    after_the_run(setup, tracker_down)


def unreadable_ticket_after_the_review(setup: Wake) -> None:
    def tracker_down() -> None:
        setup.stub.answer = TrackerError("twg failed while reading DEMO-1: down")

    during_the_review(setup, tracker_down)


def unmergeable_pull_request(setup: Wake) -> None:
    setup.repository.errors["merge"] = RepositoryError(
        "gh did not answer within 60 seconds while merging pull request #7"
    )


def unfetchable_review_plugin(setup: Wake) -> None:
    def break_plugin() -> None:
        setup.plugin.error = PluginError("the tag is gone")

    after_the_run(setup, break_plugin)


def unmakeable_review_checkout(setup: Wake) -> None:
    def break_checkout() -> None:
        setup.workspace.fail = "open_review"

    after_the_run(setup, break_checkout)


def missing_review_credential(setup: Wake) -> None:
    after_the_run(setup, lambda: setup.monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN"))


def failing_review_container(setup: Wake) -> None:
    run = setup.container.run

    def fail_review(*arguments: Any, readonly: bool = False) -> ContainerExit:
        if readonly:
            raise ContainerError("the image is gone")
        return run(*arguments, readonly=readonly)

    setup.monkeypatch.setattr(setup.container, "run", fail_review)


def give_no_verdict(setup: Wake) -> None:
    setup.container.verdict = ContainerExit(
        0, json.dumps({"structured_output": {"defects": []}}), ""
    )


def review_not_taken_place(setup: Wake) -> None:
    setup.container.verdict = ContainerExit(
        0,
        json.dumps(
            {
                "structured_output": {
                    "reviewed": False,
                    "reason": "gh pr checks needs a token.",
                    "defects": [],
                }
            }
        ),
        "",
    )


def review_at_its_turn_ceiling(setup: Wake) -> None:
    setup.container.verdict = ContainerExit(
        1, json.dumps({"subtype": "error_max_turns", "is_error": True}), ""
    )


def no_time_left_for_the_review(setup: Wake) -> None:
    clock = [1000.0]
    setup.monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])

    def spend_the_run_time() -> None:
        clock[0] += 1800

    after_the_run(setup, spend_the_run_time)


@pytest.mark.parametrize(
    ("arrange", "fragment"),
    [
        (give_no_verdict, "gave no verdict: the structured output"),
        (review_not_taken_place, "the review did not take place"),
        (review_at_its_turn_ceiling, "(subtype error_max_turns)"),
        (no_time_left_for_the_review, "ceiling of 1800 seconds was reached before it"),
        (unfindable_pull_request, "gh failed while listing pull requests: down"),
        (unreadable_ci, "gh failed while reading checks: down"),
        (ci_without_workflow, "has no active GitHub Actions workflow"),
        (unreadable_ticket_before_the_review, "twg failed while reading DEMO-1"),
        (unreadable_ticket_after_the_review, "twg failed while reading DEMO-1"),
        (unmergeable_pull_request, "It may have been merged"),
        (unfetchable_review_plugin, "its plugin cannot be fetched: the tag is gone"),
        (unmakeable_review_checkout, "its checkout cannot be made: open_review"),
        (missing_review_credential, "the variable CLAUDE_CODE_OAUTH_TOKEN"),
        (failing_review_container, "failed: the image is gone."),
    ],
)
def test_wake_whose_done_run_the_worker_cannot_land_exits_with_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    plugin: FakePlugin,
    reachable_repository: ReachableRepository,
    arrange: Callable[[Wake], None],
    fragment: str,
) -> None:
    stub = list_tickets(monkeypatch)
    arrange(
        Wake(
            monkeypatch,
            instance,
            stub,
            workspace,
            container,
            plugin,
            reachable_repository,
        )
    )

    code, answer = wake(capsysbinary, instance)

    assert answer is not None
    errors = answer["errors"]
    assert any(fragment in error for error in errors), errors
    assert code == 1


def test_wake_whose_listed_key_is_refused_exits_with_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    list_tickets(monkeypatch, listed=(ticket_on("OTHER-1"),))

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(answer, answer["task_id"], "is not in the tracker project 'DEMO'.")


def test_wake_whose_task_id_is_already_journaled_exits_with_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    fixed = cli.uuid.UUID(int=7)
    monkeypatch.setattr(cli.uuid, "uuid4", lambda: fixed)
    stub = list_tickets(monkeypatch)
    first_code, _ = wake(capsysbinary, instance)
    stub.answer = demo_ticket()

    code, answer = wake(capsysbinary, instance)

    assert first_code == 0
    assert code == 1
    assert answer is not None
    assert_failed(answer, f"wake-{fixed}", f"Task 'wake-{fixed}' has already been run.")


REQUEUE = "dev-worker returns this ticket to the queue: "
ENTERS_IN_PROGRESS = ("status", "In Progress")
ENTERS_READY = ("status", "To Do")
ENTERS_BLOCKED = ("status", "Blocked")


class TimelineTracker(StubTracker):
    """A `StubTracker` for one ticket that keeps its history as events, oldest first.

    An event is a status entered `("status", name)`, a claim `("claim", task_id)`, a
    comment `("comment", body)` or a pause `("wait", seconds)`; a claim's id is its rank
    among the events, from 1. Every event happens at `CLAIM_EPOCH` plus the seconds of
    the pauses before it. Claims and requeues count from the last entry in any of the
    statuses asked, every one when none was ever entered. `listings` records each status
    listed; `list_error` and `claims_error` are raised by those reads when set.
    """

    def __init__(self, ticket: Ticket, *events: tuple[str, str]) -> None:
        super().__init__(ticket)
        self.events = list(events)
        self.listings: list[str] = []
        self.list_error: TrackerError | None = None
        self.claims_error: TrackerError | None = None

    def since(self, statuses: frozenset[str]) -> list[tuple[int, tuple[str, str]]]:
        """Return the events since the ticket last entered any of `statuses`, with their rank from 1."""
        ranked = list(enumerate(self.events, 1))
        entries = [
            rank
            for rank, (kind, name) in ranked
            if kind == "status" and name in statuses
        ]
        start = max(entries, default=0)
        return [(rank, event) for rank, event in ranked if rank > start]

    def moment(self, rank: int) -> datetime:
        """Return when the event of rank `rank`, from 1, happened."""
        pauses = sum(
            int(seconds) for kind, seconds in self.events[:rank] if kind == "wait"
        )
        return CLAIM_EPOCH + timedelta(seconds=pauses)

    def entered(self, key: str, statuses: frozenset[str]) -> datetime | None:
        """Return when the ticket last entered any of `statuses`, None when it never did."""
        entries = [
            rank
            for rank, (kind, name) in enumerate(self.events, 1)
            if kind == "status" and name in statuses
        ]
        return self.moment(max(entries)) if entries else None

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Return the claims since the ticket last entered any of `statuses`; raise `claims_error` when set."""
        if self.claims_error is not None:
            raise self.claims_error
        return tuple(
            Claim(rank, task, self.moment(rank))
            for rank, (kind, task) in self.since(statuses)
            if kind == "claim"
        )

    def requeues(self, key: str, statuses: frozenset[str]) -> int:
        """Return how many requeue comments were posted since the ticket last entered any of `statuses`."""
        return sum(
            kind == "comment" and body.startswith(REQUEUE)
            for _, (kind, body) in self.since(statuses)
        )

    def claim(self, key: str, task_id: str) -> int:
        """Record the claim of `task_id`; return its rank among the events."""
        super().claim(key, task_id)
        self.events.append(("claim", task_id))
        return len(self.events)

    def comment(self, key: str, body: str) -> None:
        """Record the comment as `StubTracker` does, then as an event."""
        super().comment(key, body)
        self.events.append(("comment", body))

    def set_status(self, key: str, status: str) -> None:
        """Record the change as `StubTracker` does, then as an event."""
        super().set_status(key, status)
        self.events.append(("status", status))

    def tickets_in(self, status: str) -> tuple[Ticket, ...]:
        """Record `status`; return the ticket when it is in `status`, raising `list_error` when set."""
        self.listings.append(status)
        if self.list_error is not None:
            raise self.list_error
        assert isinstance(self.answer, Ticket)
        return (self.answer,) if self.answer.status == status else ()


def timeline(
    monkeypatch: pytest.MonkeyPatch, *events: tuple[str, str], blocked: bool = True
) -> TimelineTracker:
    """Make `wake` use a `TimelineTracker` on DEMO-1, in progress after `events`; return it.

    A `blocked` ticket waits on DEMO-9, so that the wake does not run it once ready.
    """
    blockers = (Blocker("DEMO-9", "To Do"),) if blocked else ()
    stub = TimelineTracker(demo_ticket("In Progress", *blockers), *events)
    monkeypatch.setattr(cli, "make_tracker", stub.make)
    return stub


DEAD_RUN = (ENTERS_READY, ("claim", "t1"), ENTERS_IN_PROGRESS)


def test_wake_hands_back_a_ticket_its_dead_run_left_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{REQUEUE}the run that took it is gone without settling it "
                "(attempt 1 of 3)."
            ),
        )
    ]
    assert stub.moves == [("DEMO-1", "To Do")]
    assert stub.listings == ["In Progress", "To Do"]
    assert not (instance.with_suffix("") / cli.LOCK_FILE).exists()


def test_wake_takes_the_lock_over_from_the_dead_run_before_handing_its_ticket_back(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, lock_started(1801))
    write_journal(
        instance,
        {"task_id": "t", "ticket_key": "DEMO-1", "status": "running", "start": "x"},
    )
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.moves == [("DEMO-1", "To Do")]
    assert not lock.exists()
    assert journal_lines(instance)[-1]["status"] == "failed"


def test_wake_hands_back_a_ticket_whose_runs_died_before_as_the_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(
        monkeypatch,
        *DEAD_RUN,
        ("comment", f"{REQUEUE}first"),
        ENTERS_READY,
        ("claim", "t2"),
        ENTERS_IN_PROGRESS,
    )

    wake(capsysbinary, instance)

    assert stub.comments[0][1].endswith("(attempt 2 of 3).")
    assert stub.moves == [("DEMO-1", "To Do")]


def test_wake_blocks_a_ticket_whose_runs_died_as_many_times_as_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    died = (
        ("comment", f"{REQUEUE}earlier"),
        ENTERS_READY,
        ("claim", "t"),
        ENTERS_IN_PROGRESS,
    )
    stub = timeline(monkeypatch, *DEAD_RUN, *died, *died)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == [
        (
            "DEMO-1",
            (
                "The runs that took this ticket died 3 times without settling it. "
                "A person needs to look at it, then move it back to 'To Do'."
            ),
        )
    ]
    assert stub.moves == [("DEMO-1", "Blocked")]


def test_wake_blocks_at_the_attempt_limit_of_the_instance(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    with instance.open("a", encoding="utf-8") as file:
        file.write("  max_attempts: 1\n")
    stub = timeline(monkeypatch, *DEAD_RUN)

    wake(capsysbinary, instance)

    assert stub.comments[0][1].startswith("The runs that took this ticket died 1 ")
    assert stub.moves == [("DEMO-1", "Blocked")]


def test_attempts_count_again_once_a_blocked_ticket_is_handed_back(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(
        monkeypatch,
        *DEAD_RUN,
        ("comment", f"{REQUEUE}first"),
        ("comment", f"{REQUEUE}second"),
        ENTERS_BLOCKED,
        *DEAD_RUN,
    )

    wake(capsysbinary, instance)

    assert stub.comments[0][1].endswith("(attempt 1 of 3).")
    assert stub.moves == [("DEMO-1", "To Do")]


@pytest.mark.parametrize(
    "events",
    [
        (ENTERS_READY, ENTERS_IN_PROGRESS),
        (ENTERS_READY, ("claim", "t1"), ENTERS_BLOCKED, ENTERS_IN_PROGRESS),
        (("claim", "t1"), ENTERS_READY, ENTERS_IN_PROGRESS),
    ],
    ids=["never-claimed", "claimed-before-blocked", "claimed-before-ready"],
)
def test_wake_leaves_a_ticket_a_person_moved_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    events: tuple[tuple[str, str], ...],
) -> None:
    stub = timeline(monkeypatch, *events)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == []
    assert stub.moves == []


def test_wake_leaves_a_ticket_a_person_moved_in_progress_hours_after_an_orphan_claim(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(
        monkeypatch,
        ENTERS_READY,
        ("claim", "t1"),
        ("wait", str(3 * 3600)),
        ENTERS_IN_PROGRESS,
    )

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == []
    assert stub.moves == []


def test_wake_leaves_a_ticket_moved_in_progress_before_its_latest_claim(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(
        monkeypatch,
        ENTERS_READY,
        ENTERS_IN_PROGRESS,
        ("wait", "60"),
        ("claim", "t1"),
    )

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == []
    assert stub.moves == []


@pytest.mark.parametrize(
    ("delay", "moves"),
    [
        (
            cli.CLAIM_TAKE_LIMIT_SECONDS + int(TWG_TIMEOUT_SECONDS),
            [("DEMO-1", "To Do")],
        ),
        (cli.CLAIM_TAKE_LIMIT_SECONDS + int(TWG_TIMEOUT_SECONDS) + 1, []),
    ],
    ids=[
        "at-the-take-limit-plus-the-move-timeout",
        "past-the-take-limit-plus-the-move-timeout",
    ],
)
def test_wake_hands_back_a_ticket_only_when_it_entered_in_progress_within_the_take_limit_plus_the_move_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    delay: int,
    moves: list[tuple[str, str]],
) -> None:
    stub = timeline(
        monkeypatch,
        ENTERS_READY,
        ("claim", "t1"),
        ("wait", str(delay)),
        ENTERS_IN_PROGRESS,
    )

    wake(capsysbinary, instance)

    assert stub.moves == moves


def test_wake_hands_back_a_ticket_whose_latest_claim_follows_its_last_block(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(
        monkeypatch,
        ENTERS_READY,
        ("claim", "t1"),
        ENTERS_BLOCKED,
        ("claim", "t2"),
        ENTERS_IN_PROGRESS,
    )

    wake(capsysbinary, instance)

    assert stub.moves == [("DEMO-1", "To Do")]


def test_wake_does_not_recover_while_a_live_run_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    live = {"task_id": "t", "start": datetime.now(UTC).isoformat(), "pid": os.getpid()}
    lock = hold_lock(instance, json.dumps(live).encode())
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.listings == ["To Do"]
    assert stub.comments == []
    assert lock.exists()


def test_wake_runs_the_ticket_it_handed_back_in_the_same_wake(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN, blocked=False)

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert stub.moves[:2] == [("DEMO-1", "To Do"), ("DEMO-1", "In Progress")]


def test_wake_whose_in_progress_tickets_cannot_be_listed_fails_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN)
    stub.list_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        None,
        "The tickets in 'In Progress' cannot be listed from the tracker: "
        "twg failed: down",
    )
    assert stub.listings == ["In Progress"]
    assert not (instance.with_suffix("") / cli.LOCK_FILE).exists()


def test_wake_whose_claims_cannot_be_read_fails_without_moving_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN)
    stub.claims_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        None,
        "The comments of ticket 'DEMO-1' cannot be read from the tracker: "
        "twg failed: down",
    )
    assert stub.moves == []


def test_wake_whose_requeue_comment_fails_does_not_move_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN)
    stub.comment_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        None,
        "Ticket 'DEMO-1' cannot be commented on: twg failed: down The comment may "
        "have been posted; the ticket was not moved.",
    )
    assert stub.moves == []


def test_wake_whose_requeued_ticket_cannot_be_moved_fails_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN)
    stub.move_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        None,
        "Ticket 'DEMO-1' cannot be moved to 'To Do': twg failed: down Its status is "
        "unknown and is to be read again in the tracker.",
    )


def test_wake_whose_instance_directory_cannot_be_made_fails_before_recovering(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    instance.with_suffix("").write_text("", encoding="utf-8")
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(answer, None, "cannot be created")
    assert stub.listings == []


def test_wake_whose_lock_cannot_be_written_fails_before_recovering(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def refuse(directory: Path, record: Mapping[str, object]) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "take_lock", refuse)
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(answer, None, "cannot be written: Permission denied")
    assert stub.listings == []


def test_wake_whose_recovery_lock_cannot_be_released_fails_with_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def refuse(path: Path, record: Mapping[str, object]) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "release_lock", refuse)
    stub = timeline(monkeypatch, *DEAD_RUN)
    stub.list_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    errors = answer["errors"]
    assert errors[0].startswith("The tickets in 'In Progress' cannot be listed")
    assert errors[1].startswith("Lock file '")
    assert errors[1].endswith("cannot be removed: Permission denied")


ENTERS_IN_REVIEW = ("status", "In Review")
ENTERS_DONE = ("status", "Done")


@pytest.mark.parametrize(
    "settled", [ENTERS_IN_REVIEW, ENTERS_DONE], ids=["in-review", "done"]
)
def test_wake_leaves_a_settled_ticket_a_person_moved_back_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    settled: tuple[str, str],
) -> None:
    stub = timeline(monkeypatch, *DEAD_RUN, settled, ENTERS_IN_PROGRESS)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == []
    assert stub.moves == []


@pytest.mark.parametrize(
    "settled", [ENTERS_IN_REVIEW, ENTERS_DONE], ids=["in-review", "done"]
)
def test_attempts_count_again_once_a_settled_ticket_is_handed_back(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    settled: tuple[str, str],
) -> None:
    died = (
        ("comment", f"{REQUEUE}earlier"),
        ENTERS_READY,
        ("claim", "t"),
        ENTERS_IN_PROGRESS,
    )
    stub = timeline(monkeypatch, *DEAD_RUN, *died, *died, settled, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert [body for _, body in stub.comments] == [
        f"{REQUEUE}the run that took it is gone without settling it (attempt 1 of 3)."
    ]
    assert stub.moves == [("DEMO-1", "To Do")]


@pytest.mark.usefixtures("frozen_now")
def test_wake_recovers_from_a_lock_beyond_the_timeout_whose_process_cannot_be_inspected(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    lock = hold_lock(instance, lock_of(1801, pid=os.getpid(), ticket_key="DEMO-1"))
    deny_inspection(monkeypatch)
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.moves == [("DEMO-1", "To Do")]
    assert not lock.exists()


@pytest.mark.usefixtures("frozen_now")
def test_wake_does_not_recover_while_the_process_of_the_lock_it_took_over_still_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    hold_lock(instance, lock_of(1801, pid=os.getpid(), ticket_key="DEMO-1"))
    monkeypatch.setattr(cli, "process_runs_since", lambda pid, moment: True)
    stub = timeline(monkeypatch, *DEAD_RUN)

    code, _ = wake(capsysbinary, instance)

    assert code == 0
    assert stub.comments == []
    assert stub.moves == []


USAGE_RESET = datetime.fromtimestamp(1792854600, UTC)
LIMIT_RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": True,
    "result": "You've hit your limit",
}


def usage_limit_event(status: str) -> dict[str, object]:
    """Return a `rate_limit_event` at `status`, resetting at `USAGE_RESET`."""
    return {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": status,
            "resetsAt": 1792854600,
            "rateLimitType": "five_hour",
        },
    }


def ending_at_the_usage_limit(*events: object) -> ContainerExit:
    """Return how a run container ends when Claude Code writes `events` then the result of a run stopped by the usage limit."""
    lines = [*events, LIMIT_RESULT]
    return ContainerExit(0, "".join(json.dumps(line) + "\n" for line in lines), "")


def usage_limit_file(instance: Path) -> Path:
    """Return the usage limit file of the instance directory of `instance`."""
    return instance.with_suffix("") / cli.USAGE_LIMIT_FILE


def recorded_reset(instance: Path) -> datetime:
    """Return the reset moment written in the usage limit file of `instance`."""
    recorded = json.loads(usage_limit_file(instance).read_text(encoding="utf-8"))
    return datetime.fromisoformat(recorded["resets_at"])


def write_reset(instance: Path, content: str) -> None:
    """Write `content` as the usage limit file of `instance`, its directory created."""
    path = usage_limit_file(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.usefixtures("workspace", "plugin", "reachable_repository")
def test_wake_whose_run_is_rejected_by_the_usage_limit_puts_its_ticket_back_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = list_tickets(monkeypatch)
    container.exited = ending_at_the_usage_limit(
        usage_limit_event("allowed"), usage_limit_event("rejected")
    )

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert_failed(answer, answer["task_id"], "Claude Code gave no structured output")
    assert stub.moves == [("DEMO-1", "In Progress"), ("DEMO-1", "To Do")]
    assert recorded_reset(instance) == USAGE_RESET
    ((_, body),) = stub.comments
    assert not body.startswith(REQUEUE_MARKER)
    assert "usage limit" in body
    assert "2026-10-24 15:10:00 UTC" in body
    assert len(container.readonly) == 1


@pytest.mark.usefixtures("workspace", "plugin", "reachable_repository")
def test_wake_whose_run_fails_with_a_429_waits_the_instance_delay(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = list_tickets(monkeypatch)
    container.exited = ContainerExit(
        1, json.dumps(LIMIT_RESULT | {"api_error_status": 429}), ""
    )
    before = datetime.now(UTC)

    code, _ = wake(capsysbinary, instance)

    after = datetime.now(UTC)
    assert code == 0
    assert stub.moves[-1] == ("DEMO-1", "To Do")
    wait = timedelta(seconds=3600)
    assert before + wait <= recorded_reset(instance) <= after + wait


def test_wake_before_the_usage_limit_resets_takes_no_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stub = list_tickets(monkeypatch)
    resets_at = datetime(2999, 1, 2, 3, 4, 5, tzinfo=UTC)
    write_reset(instance, json.dumps({"resets_at": resets_at.isoformat()}))

    code = cli.main(["wake", "--instance", str(instance)])

    captured = capsysbinary.readouterr()
    assert code == 0
    assert captured.out == b""
    assert captured.err.decode("utf-8") == (
        "dev-worker takes no ticket: the agent's account reached its usage limit, "
        "which resets at 2999-01-02 03:04:05 UTC.\n"
    )
    assert stub.listings == ["In Progress"]
    assert stub.reads == []
    assert stub.posted == []


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(json.dumps({"resets_at": "2000-01-01T00:00:00+00:00"}), id="past"),
        pytest.param(json.dumps({"resets_at": "2999-01-01T00:00:00"}), id="naive"),
        pytest.param(json.dumps(["2999-01-01T00:00:00+00:00"]), id="not-object"),
        pytest.param("{", id="not-json"),
    ],
)
@pytest.mark.usefixtures("workspace", "container", "plugin", "reachable_repository")
def test_wake_with_a_past_or_unreadable_usage_limit_takes_its_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    content: str,
) -> None:
    stub = list_tickets(monkeypatch)
    write_reset(instance, content)

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert stub.listings == ["In Progress", "To Do"]


@pytest.mark.usefixtures("workspace", "plugin", "reachable_repository")
def test_usage_limit_that_cannot_be_recorded_still_puts_the_ticket_back_ready(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = list_tickets(monkeypatch)
    container.exited = ending_at_the_usage_limit(usage_limit_event("rejected"))
    usage_limit_file(instance).mkdir(parents=True)

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(answer, answer["task_id"], "cannot be written")
    assert stub.moves[-1] == ("DEMO-1", "To Do")
    assert len(stub.comments) == 1


@pytest.mark.usefixtures("workspace", "plugin", "reachable_repository")
def test_usage_limit_whose_ticket_cannot_be_commented_on_nor_moved_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    stub = list_tickets(monkeypatch)
    container.exited = ending_at_the_usage_limit(usage_limit_event("rejected"))
    stub.comment_error = TrackerError("twg failed: down")
    set_status = stub.set_status

    def refuse_ready(key: str, status: str) -> None:
        set_status(key, status)
        if status == "To Do":
            raise TrackerError("twg failed: down")

    monkeypatch.setattr(stub, "set_status", refuse_ready)

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(answer, answer["task_id"], "cannot be commented on")
    assert_failed(answer, answer["task_id"], "cannot be moved to 'To Do'")
    assert recorded_reset(instance) == USAGE_RESET


@pytest.mark.usefixtures("plugin", "reachable_repository")
def test_usage_limit_whose_work_cannot_be_rescued_blocks_the_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
) -> None:
    stub = list_tickets(monkeypatch)
    container.exited = ending_at_the_usage_limit(usage_limit_event("rejected"))
    workspace.fail = "rescue"

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert stub.moves[-1] == ("DEMO-1", "Blocked")
    ((_, body),) = stub.comments
    assert body.startswith("The run ended without a usable outcome: ")
    assert not usage_limit_file(instance).exists()


@pytest.mark.usefixtures("workspace", "plugin", "reachable_repository")
def test_a_run_dead_after_its_ticket_came_back_from_the_usage_limit_is_its_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    limited = list_tickets(monkeypatch)
    container.exited = ending_at_the_usage_limit(usage_limit_event("rejected"))
    wake(capsysbinary, instance)
    assert limited.moves[-1] == ("DEMO-1", "To Do")
    ((_, limit_comment),) = limited.comments
    stub = timeline(
        monkeypatch,
        *DEAD_RUN,
        ("comment", limit_comment),
        ENTERS_READY,
        ("claim", "t2"),
        ENTERS_IN_PROGRESS,
    )

    code, answer = wake(capsysbinary, instance)

    assert (code, answer) == (0, None)
    assert stub.comments == [
        (
            "DEMO-1",
            (
                f"{REQUEUE}the run that took it is gone without settling it "
                "(attempt 1 of 3)."
            ),
        )
    ]
    assert stub.moves == [("DEMO-1", "To Do")]


def test_wake_on_a_red_default_branch_takes_no_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = list_tickets(monkeypatch)
    reachable_repository.base_failure = BaseCiFailure(
        branch="main",
        commit="abc1234def5678abc1234def5678abc1234def56",
        checks=("lint", "tests"),
    )

    code = cli.main(["wake", "--instance", str(instance)])

    captured = capsysbinary.readouterr()
    assert code == 0
    assert captured.out == b""
    assert captured.err.decode("utf-8") == (
        "dev-worker takes no ticket: the CI of main is red at abc1234 (lint, tests).\n"
    )
    assert stub.reads == []
    assert stub.posted == []
    assert reachable_repository.calls == [("base_ci_failure", None)]


@pytest.mark.usefixtures("workspace", "container", "plugin")
def test_wake_whose_default_branch_ci_cannot_be_read_takes_its_ticket(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    list_tickets(monkeypatch)
    reachable_repository.errors["base_ci_failure"] = RepositoryError("gh broke")

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert reachable_repository.calls[0] == ("base_ci_failure", None)


def test_wake_without_a_ready_ticket_does_not_read_the_default_branch_ci(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    list_tickets(monkeypatch, listed=())

    assert wake(capsysbinary, instance) == (0, None)
    assert reachable_repository.calls == []


def unfilled_ticket() -> Ticket:
    """Return the ready ticket DEMO-1 with its acceptance criteria blank and its environment filled."""
    return replace(
        demo_ticket(),
        fields=(("Acceptance criteria", " \n"), ("Environment", "Staging")),
    )


def test_wake_blocks_a_ticket_with_an_empty_required_field_without_running_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    ticket = unfilled_ticket()
    stub = list_tickets(monkeypatch, listed=(ticket,), answer=ticket)

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert_failed(
        answer,
        answer["task_id"],
        "Ticket 'DEMO-1' has empty required fields: Acceptance criteria.",
    )
    assert stub.posted == [answer["task_id"]]
    assert stub.comments == [
        (
            "DEMO-1",
            (
                "This ticket cannot be taken, these required fields are empty: "
                "Acceptance criteria. Fill them, then move it back to 'To Do'."
            ),
        )
    ]
    assert stub.moves == [("DEMO-1", "Blocked")]
    assert journal_lines(instance) == []


def test_wake_whose_blocking_comment_fails_leaves_the_ticket_and_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    ticket = unfilled_ticket()
    stub = list_tickets(monkeypatch, listed=(ticket,), answer=ticket)
    stub.comment_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        answer["task_id"],
        "Ticket 'DEMO-1' cannot be commented on: twg failed: down The comment may "
        "have been posted; the ticket was not moved.",
    )
    assert stub.moves == []
    assert journal_lines(instance) == []


def test_wake_whose_blocking_move_fails_fails_with_the_status_unknown(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    ticket = unfilled_ticket()
    stub = list_tickets(monkeypatch, listed=(ticket,), answer=ticket)
    stub.move_error = TrackerError("twg failed: down")

    code, answer = wake(capsysbinary, instance)

    assert code == 1
    assert answer is not None
    assert_failed(
        answer,
        answer["task_id"],
        "Ticket 'DEMO-1' cannot be moved to 'Blocked': twg failed: down Its status "
        "is unknown and is to be read again in the tracker.",
    )
    assert stub.moves == [("DEMO-1", "Blocked")]
    assert journal_lines(instance) == []


def test_wake_takes_a_ticket_whose_required_fields_are_all_filled(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    ticket = replace(
        demo_ticket(),
        fields=(("Acceptance criteria", "It logs in."), ("Environment", "Staging")),
    )
    stub = list_tickets(monkeypatch, listed=(ticket,), answer=ticket)

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert stub.moves[0] == ("DEMO-1", "In Progress")
    assert "Blocked" not in [status for _, status in stub.moves]


def test_wake_that_loses_the_claim_leaves_a_ticket_with_an_empty_required_field_unblocked(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    ticket = unfilled_ticket()
    stub = list_tickets(monkeypatch, listed=(ticket,), answer=ticket)
    stub.posted.append("rival")

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert_failed(answer, answer["task_id"], "is claimed by task 'rival'")
    assert stub.comments == []
    assert stub.moves == []
    assert journal_lines(instance) == []


class ChangedOnClaimTracker(ListingTracker):
    """A `ListingTracker` whose reads answer `before` until a claim is posted, then `after`."""

    def __init__(self, before: Ticket, after: Ticket) -> None:
        super().__init__((before,), before)
        self.after = after

    def claim(self, key: str, task_id: str) -> int:
        """Record the claim, then answer `after` to every later read."""
        self.answer = self.after
        return super().claim(key, task_id)


def changed_on_claim(
    monkeypatch: pytest.MonkeyPatch, before: Ticket, after: Ticket
) -> ChangedOnClaimTracker:
    """Make `wake` list and read `before`, then read `after` once its claim is posted; return the tracker."""
    stub = ChangedOnClaimTracker(before, after)
    monkeypatch.setattr(cli, "make_tracker", stub.make)
    return stub


def test_wake_blocks_a_ticket_whose_required_field_is_emptied_by_the_time_it_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    listed = replace(
        demo_ticket(),
        fields=(("Acceptance criteria", "It logs in."), ("Environment", "Staging")),
    )
    stub = changed_on_claim(monkeypatch, listed, unfilled_ticket())

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert_failed(
        answer,
        answer["task_id"],
        "Ticket 'DEMO-1' has empty required fields: Acceptance criteria.",
    )
    assert stub.moves == [("DEMO-1", "Blocked")]
    assert container.commands == []
    assert journal_lines(instance) == []


def test_wake_runs_a_ticket_whose_required_field_is_filled_by_the_time_it_is_claimed_with_the_value_read_again(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
) -> None:
    claimed = replace(
        demo_ticket(),
        fields=(("Acceptance criteria", "It logs in."), ("Environment", "Staging")),
    )
    stub = changed_on_claim(monkeypatch, unfilled_ticket(), claimed)

    code, answer = wake(capsysbinary, instance)

    assert code == 0
    assert answer is not None
    assert answer["status"] == "completed"
    assert stub.moves[0] == ("DEMO-1", "In Progress")
    assert "Blocked" not in [status for _, status in stub.moves]
    assert "It logs in." in container.commands[0][2]


HANDOVER_BLOCK = """
handover:
  sensitive_paths:
    - .github/workflows/**
  max_changed_lines: 10
  test_paths:
    - tests/**
"""
LEFT_FOR_HUMAN = f"{SUMMARY}\n\nThe pull request {PULL.url} is left open for a human: "


def declare_handover(instance: Path) -> None:
    """Append a `handover` block to the instance file at `instance`."""
    with instance.open("a", encoding="utf-8") as file:
        file.write(HANDOVER_BLOCK)


def test_done_touching_a_sensitive_path_stays_open_unreviewed_and_moves_to_review_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    workspace: FakeWorkspace,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    declare_handover(instance)
    reachable_repository.files = (ChangedFile(".github/workflows/ci.yml", None, 1, 0),)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["status"] == "completed"
    assert answer["errors"] == []
    assert reachable_repository.calls == [
        ("find_pull_request", "agent/DEMO-1"),
        ("read_ci", PULL),
        ("changed_files", PULL),
    ]
    assert container.readonly == [False]
    assert ("close_review",) not in workspace.calls
    assert stub.comments == [
        (
            "DEMO-1",
            f"{LEFT_FOR_HUMAN}it touches sensitive paths: .github/workflows/ci.yml.",
        )
    ]
    assert stub.moves == [IN_PROGRESS, IN_REVIEW]


def test_done_above_the_size_bound_stays_open_and_moves_to_review_naming_the_count(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    declare_handover(instance)
    reachable_repository.files = (ChangedFile("src/app.py", None, 6, 5),)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert container.readonly == [False]
    assert stub.comments == [
        (
            "DEMO-1",
            f"{LEFT_FOR_HUMAN}it changes 11 lines outside tests, above the bound of 10.",
        )
    ]
    assert stub.moves == [IN_PROGRESS, IN_REVIEW]


def test_done_with_a_handover_and_clean_files_is_reviewed_then_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    declare_handover(instance)
    reachable_repository.files = (ChangedFile("src/app.py", None, 6, 4),)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert reachable_repository.calls == [
        ("find_pull_request", "agent/DEMO-1"),
        ("read_ci", PULL),
        ("changed_files", PULL),
        ("merge", PULL),
    ]
    assert container.readonly == [False, True]
    assert stub.comments == [("DEMO-1", MERGED_BODY)]
    assert stub.moves == [IN_PROGRESS, MERGED]


def test_done_whose_files_cannot_be_read_is_blocked_unreviewed_and_not_merged(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    container: FakeContainer,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    declare_handover(instance)
    reachable_repository.errors["changed_files"] = RepositoryError("files unreadable")

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert_blocked_run_fails(answer, instance, "files unreadable")
    assert "merge" not in [name for name, _ in reachable_repository.calls]
    assert container.readonly == [False]
    assert stub.comments == [
        ("DEMO-1", f"{SUMMARY}\n\nThe ticket is blocked: files unreadable")
    ]
    assert stub.moves == [IN_PROGRESS, BLOCKED]


def test_done_on_a_held_ticket_does_not_read_the_files_of_its_pull_request(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    declare_handover(instance)
    reachable_repository.errors["changed_files"] = RepositoryError("files unreadable")
    find = reachable_repository.find_pull_request

    def label_then_find(branch: str) -> PullRequest | None:
        stub.answer = replace(demo_ticket("In Progress"), labels=("hold-merge",))
        return find(branch)

    monkeypatch.setattr(reachable_repository, "find_pull_request", label_then_find)

    answer = run_mandate(
        monkeypatch, capsysbinary, encode(valid_input()), instance=instance
    )

    assert answer["errors"] == []
    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
    ]
    assert "carries the label 'hold-merge'" in stub.comments[0][1]
    assert stub.moves == [IN_PROGRESS, IN_REVIEW]


def test_done_without_a_handover_never_reads_the_files_even_when_they_would_match(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
    reachable_repository: ReachableRepository,
) -> None:
    stub = serve(monkeypatch, demo_ticket())
    reachable_repository.files = (
        ChangedFile(".github/workflows/ci.yml", None, 1000, 0),
    )

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    assert [name for name, _ in reachable_repository.calls] == [
        "find_pull_request",
        "read_ci",
        "merge",
    ]
    assert stub.moves == [IN_PROGRESS, MERGED]
