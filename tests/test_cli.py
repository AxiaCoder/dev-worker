import io
import json
import os
import runpy
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from dev_worker import cli
from dev_worker.header import HeaderError

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "examples" / "instance.yaml"


def valid_input(**overrides: object) -> dict[str, object]:
    """Return a valid `deliver_ticket` input, with `overrides` applied."""
    data: dict[str, object] = {
        "contract_version": "0.1",
        "task_id": "task-1",
        "mandate": "deliver_ticket",
        "caller": "tests",
        "params": {},
    }
    data.update(overrides)
    return data


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


def test_valid_deliver_ticket_input_fails_as_not_built_yet(
    monkeypatch: pytest.MonkeyPatch, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    answer = run_mandate(monkeypatch, capsysbinary, encode(valid_input()))

    assert answer == {
        "contract_version": "0.1",
        "task_id": "task-1",
        "status": "failed",
        "summary": "The mandate could not be run.",
        "deliverables": [],
        "approval_request": None,
        "errors": ["The deliver_ticket mandate is not built yet."],
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


def test_run_reads_and_writes_utf8_whatever_the_system_encoding() -> None:
    task_id = "tâche ✓"
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}

    completed = subprocess.run(
        [sys.executable, "-m", "dev_worker", "run", "deliver_ticket"]
        + ["--instance", str(INSTANCE)],
        input=json.dumps(valid_input(task_id=task_id), ensure_ascii=False).encode(
            "utf-8"
        ),
        capture_output=True,
        env=env,
        check=True,
    )

    assert task_id.encode("utf-8") in completed.stdout
    assert json.loads(completed.stdout.decode("utf-8"))["task_id"] == task_id
