import io
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from dev_worker import cli, run_journal
from dev_worker.contract import MandateInput, MandateOutput
from dev_worker.header import HeaderError
from dev_worker.instance import Instance

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "examples" / "instance.yaml"


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
    (line,) = journal_lines(instance)
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


def test_repeated_task_id_fails_and_is_journaled_once(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    stdin = encode(valid_input())
    run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    answer = run_mandate(monkeypatch, capsysbinary, stdin, instance=instance)

    assert_failed(answer, "task-1", "Task 'task-1' has already been run.")
    assert len(journal_lines(instance)) == 1


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
    (line,) = journal_lines(instance)
    assert line["status"] == "failed"


def test_journal_records_the_duration_of_the_mandate_in_seconds(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    clock = [100.0, 102.5]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock.pop(0))

    run_mandate(monkeypatch, capsysbinary, encode(valid_input()), instance=instance)

    (line,) = journal_lines(instance)
    assert line["duration"] == 2.5


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
    assert record.keys() == {"task_id", "ticket_key", "start"}
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

    assert_failed(answer, "task-1", f"Journal in '{directory}' cannot be read: ")
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


def test_lock_that_cannot_be_removed_is_reported_with_the_output(
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
    instance: Path,
) -> None:
    def stuck(path: Path) -> None:
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


def test_state_is_empty(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run(monkeypatch, capsysbinary, ["state", "--instance", str(INSTANCE)])

    assert answer == {
        "common": {"current": [], "recent": [], "pending": [], "cost": {}},
        "domain": {},
    }


def test_health_is_ok(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run(monkeypatch, capsysbinary, ["health", "--instance", str(INSTANCE)])

    assert answer == {"status": "ok", "reason": None, "active": True}


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
    monkeypatch.setattr(sys, "argv", ["dev-worker", "health", "--instance", "x"])

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
