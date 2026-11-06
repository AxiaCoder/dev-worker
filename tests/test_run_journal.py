import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from dev_worker import run_journal
from dev_worker.run_journal import (
    JOURNAL_FILE,
    LOCK_FILE,
    JournalError,
    append_entry,
    instance_directory,
    journal_entry,
    journal_holds,
    process_runs_since,
    read_entries,
    read_lock,
    release_lock,
    set_aside_lock,
    take_lock,
)


def test_instance_directory_drops_the_suffix(tmp_path: Path) -> None:
    assert instance_directory(tmp_path / "acme.yaml") == tmp_path / "acme"


def test_instance_file_without_a_suffix_has_no_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has no suffix"):
        instance_directory(tmp_path / "acme")


def test_take_lock_writes_the_record(tmp_path: Path) -> None:
    lock = take_lock(tmp_path, {"task_id": "t"})

    assert lock == tmp_path / LOCK_FILE
    assert json.loads(lock.read_text(encoding="utf-8")) == {"task_id": "t"}


def test_take_lock_refuses_a_held_lock(tmp_path: Path) -> None:
    take_lock(tmp_path, {"task_id": "t"})

    with pytest.raises(FileExistsError):
        take_lock(tmp_path, {"task_id": "u"})


def test_lock_that_cannot_be_written_is_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def full(descriptor: int, data: bytes) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", full)

    with pytest.raises(OSError, match="No space left"):
        take_lock(tmp_path, {"task_id": "t"})

    monkeypatch.undo()
    assert not (tmp_path / LOCK_FILE).exists()


def test_release_lock_removes_it(tmp_path: Path) -> None:
    lock = take_lock(tmp_path, {"task_id": "t"})

    release_lock(lock, {"task_id": "t"})

    assert not lock.exists()


def test_release_lock_leaves_a_lock_taken_over_by_another_run(tmp_path: Path) -> None:
    lock = take_lock(tmp_path, {"task_id": "t"})
    lock.write_bytes(json.dumps({"task_id": "successor"}).encode("utf-8"))

    release_lock(lock, {"task_id": "t"})

    assert read_lock(tmp_path) == {"task_id": "successor"}


def test_release_lock_of_a_lock_already_gone_does_nothing(tmp_path: Path) -> None:
    release_lock(tmp_path / LOCK_FILE, {"task_id": "t"})

    assert list(tmp_path.iterdir()) == []


def test_release_lock_raises_when_its_own_lock_cannot_be_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lock = take_lock(tmp_path, {"task_id": "t"})

    def refuse(path: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", refuse)

    with pytest.raises(PermissionError):
        release_lock(lock, {"task_id": "t"})

    monkeypatch.undo()
    assert read_lock(tmp_path) == {"task_id": "t"}


def test_journal_holds_nothing_when_absent(tmp_path: Path) -> None:
    assert journal_holds(tmp_path, "t") is False


def test_journal_holds_an_appended_task_id_only(tmp_path: Path) -> None:
    append_entry(tmp_path, {"task_id": "t"})
    append_entry(tmp_path, {"task_id": "u"})

    assert journal_holds(tmp_path, "u") is True
    assert journal_holds(tmp_path, "v") is False
    assert (tmp_path / JOURNAL_FILE).read_bytes().count(b"\n") == 2


def test_journal_skips_blank_lines(tmp_path: Path) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(b'\n{"task_id": "t"}\n')

    assert journal_holds(tmp_path, "t") is True


@pytest.mark.parametrize(
    "line",
    [
        pytest.param(b"{", id="cut-json"),
        pytest.param(b"\xff", id="not-utf8"),
        pytest.param(b"[" * 100_000 + b"]" * 100_000, id="too-deeply-nested"),
    ],
)
def test_journal_line_that_is_not_json_raises(tmp_path: Path, line: bytes) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(line + b"\n")

    with pytest.raises(JournalError, match="line 1 is not valid JSON"):
        journal_holds(tmp_path, "t")


def test_journal_entry_returns_the_object_of_its_line() -> None:
    assert journal_entry(7, b'{"task_id": "t", "n": 1}\n') == {"task_id": "t", "n": 1}


@pytest.mark.parametrize(
    ("line", "message"),
    [
        pytest.param(b"{", "line 7 is not valid JSON", id="cut-json"),
        pytest.param(b"\xff", "line 7 is not valid JSON", id="not-utf8"),
        pytest.param(
            b"[" * 100_000 + b"]" * 100_000,
            "line 7 is not valid JSON",
            id="too-deeply-nested",
        ),
        pytest.param(b'"t"', "line 7 is not a JSON object.", id="not-an-object"),
    ],
)
def test_journal_entry_raises_naming_its_line_number(line: bytes, message: str) -> None:
    with pytest.raises(JournalError, match=f"^{message}"):
        journal_entry(7, line)


def test_journal_line_that_is_not_an_object_raises(tmp_path: Path) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(b'{"task_id": "a"}\n\n[1]\n')

    with pytest.raises(JournalError, match="line 3 is not a JSON object"):
        journal_holds(tmp_path, "t")


@pytest.mark.parametrize(
    "line",
    [
        {"task_id": "task-10"},
        {"task_id": "pre-task-1"},
        {"task_id": "other", "ticket_key": "task-1"},
        {"run": {"task_id": "task-1"}},
    ],
)
def test_journal_holds_only_an_exact_task_id(
    tmp_path: Path, line: dict[str, object]
) -> None:
    append_entry(tmp_path, line)

    assert journal_holds(tmp_path, "task-1") is False


def test_concurrent_take_lock_lets_exactly_one_run_hold_it(tmp_path: Path) -> None:
    contenders = 8
    barrier = threading.Barrier(contenders)
    outcomes: list[str] = []

    def contend(number: int) -> None:
        barrier.wait()
        try:
            take_lock(tmp_path, {"task_id": f"t{number}"})
        except FileExistsError:
            outcomes.append("refused")
        else:
            outcomes.append("held")

    threads = [
        threading.Thread(target=contend, args=(number,)) for number in range(contenders)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["held"] + ["refused"] * (contenders - 1)


def test_read_lock_gives_none_when_no_run_holds_it(tmp_path: Path) -> None:
    assert read_lock(tmp_path) is None


def test_read_lock_gives_the_record_of_the_run_in_progress(tmp_path: Path) -> None:
    take_lock(tmp_path, {"task_id": "t", "ticket_key": "DEMO-1"})

    assert read_lock(tmp_path) == {"task_id": "t", "ticket_key": "DEMO-1"}


@pytest.mark.parametrize("content", [b"", b"{", b"\xff", b"[1]", b'"held"', b"null"])
def test_read_lock_gives_unknown_details_for_a_lock_that_is_not_an_object(
    tmp_path: Path, content: bytes
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(content)

    assert read_lock(tmp_path) == {}


def test_read_lock_gives_unknown_details_for_a_lock_that_cannot_be_read(
    tmp_path: Path,
) -> None:
    (tmp_path / LOCK_FILE).mkdir()

    assert read_lock(tmp_path) == {}


def test_read_lock_leaves_the_lock_in_place(tmp_path: Path) -> None:
    lock = take_lock(tmp_path, {"task_id": "t"})
    before = lock.read_bytes()

    read_lock(tmp_path)

    assert lock.read_bytes() == before


def test_read_entries_gives_nothing_when_the_journal_is_absent(tmp_path: Path) -> None:
    assert read_entries(tmp_path) == []


def test_read_entries_gives_nothing_when_the_journal_cannot_be_read(
    tmp_path: Path,
) -> None:
    (tmp_path / JOURNAL_FILE).mkdir()

    assert read_entries(tmp_path) == []


def test_read_entries_gives_the_appended_entries_oldest_first(tmp_path: Path) -> None:
    append_entry(tmp_path, {"task_id": "t"})
    append_entry(tmp_path, {"task_id": "u"})

    assert read_entries(tmp_path) == [{"task_id": "t"}, {"task_id": "u"}]


def test_read_entries_skips_lines_that_are_not_utf8_json_objects(
    tmp_path: Path,
) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(
        b'{"task_id": "t"}\n\n{\n\xff\n[1]\n"u"\n{"task_id": "v"}'
    )

    assert read_entries(tmp_path) == [{"task_id": "t"}, {"task_id": "v"}]


TOO_DEEP_FOR_JSON = b"[" * 100_000 + b"]" * 100_000


def test_read_lock_gives_unknown_details_for_a_lock_nested_too_deeply(
    tmp_path: Path,
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(TOO_DEEP_FOR_JSON)

    assert read_lock(tmp_path) == {}


def test_read_entries_skips_a_line_nested_too_deeply(tmp_path: Path) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(
        b'{"task_id": "t"}\n' + TOO_DEEP_FOR_JSON + b'\n{"task_id": "v"}'
    )

    assert read_entries(tmp_path) == [{"task_id": "t"}, {"task_id": "v"}]


def dead_pid() -> int:
    """Return the PID of a process that has run and exited."""
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    return finished.pid


def test_process_runs_since_is_false_for_a_process_that_exited() -> None:
    assert process_runs_since(dead_pid(), time.time()) is False


def test_process_runs_since_is_true_for_a_process_started_before_the_moment() -> None:
    assert process_runs_since(os.getpid(), time.time()) is True


def test_process_runs_since_allows_a_second_of_imprecision() -> None:
    created = psutil.Process().create_time()

    assert process_runs_since(os.getpid(), created - 0.9) is True


def test_process_runs_since_is_false_for_a_pid_reused_after_the_moment() -> None:
    created = psutil.Process().create_time()

    assert process_runs_since(os.getpid(), created - 1.1) is False


@pytest.mark.parametrize(
    ("error", "runs"),
    [
        (psutil.AccessDenied(1), True),
        (OSError("refused"), True),
        (psutil.ZombieProcess(1), False),
    ],
)
def test_process_runs_since_on_a_process_it_cannot_inspect(
    monkeypatch: pytest.MonkeyPatch, error: Exception, runs: bool
) -> None:
    def refuse(pid: int) -> psutil.Process:
        raise error

    monkeypatch.setattr(psutil, "Process", refuse)

    assert process_runs_since(1, time.time()) is runs


def leftovers(directory: Path) -> list[str]:
    """Return the names of the set-aside lock files left in `directory`."""
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.name.startswith(f"{LOCK_FILE}.")
    )


def test_set_aside_lock_removes_a_lock_still_holding_what_was_seen(
    tmp_path: Path,
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"seen")

    assert set_aside_lock(tmp_path, b"seen") is True

    assert not (tmp_path / LOCK_FILE).exists()
    assert leftovers(tmp_path) == []


def test_set_aside_lock_puts_back_a_lock_changed_since_it_was_seen(
    tmp_path: Path,
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"newer run")

    assert set_aside_lock(tmp_path, b"seen") is False

    assert (tmp_path / LOCK_FILE).read_bytes() == b"newer run"
    assert leftovers(tmp_path) == []


def test_set_aside_lock_gives_false_when_the_lock_is_gone(tmp_path: Path) -> None:
    assert set_aside_lock(tmp_path, b"seen") is False

    assert list(tmp_path.iterdir()) == []


def test_set_aside_lock_keeps_a_lock_taken_while_a_changed_one_was_aside(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"newer run")
    real_link = os.link

    def link_after_a_new_lock(source: Path, target: Path) -> None:
        take_lock(tmp_path, {"task_id": "newest"})
        real_link(source, target)

    monkeypatch.setattr(run_journal.os, "link", link_after_a_new_lock)

    assert set_aside_lock(tmp_path, b"seen") is False

    assert read_lock(tmp_path) == {"task_id": "newest"}
    assert leftovers(tmp_path) == []


def contend_for_one_lock(directory: Path, contenders: int) -> list[str]:
    """Have `contenders` threads set aside the same lock of `directory` at once.

    Returns one outcome per thread, sorted: `removed`, `lost` (returned `False`), or
    the representation of the error raised.
    """
    (directory / LOCK_FILE).write_bytes(b"seen")
    barrier = threading.Barrier(contenders)
    outcomes: list[str] = []

    def contend() -> None:
        barrier.wait()
        try:
            removed = set_aside_lock(directory, b"seen")
        except OSError as error:
            outcomes.append(repr(error))
        else:
            outcomes.append("removed" if removed else "lost")

    threads = [threading.Thread(target=contend) for _ in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sorted(outcomes)


@pytest.mark.skipif(sys.platform == "win32", reason="a rename is atomic off Windows")
def test_concurrent_set_aside_lock_lets_exactly_one_run_remove_it_off_windows(
    tmp_path: Path,
) -> None:
    contenders = 8
    for _ in range(50):
        outcomes = contend_for_one_lock(tmp_path, contenders)

        assert outcomes == ["lost"] * (contenders - 1) + ["removed"]
        assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="a rename is not atomic on Windows")
def test_concurrent_set_aside_lock_lets_at_most_one_run_remove_it_on_windows(
    tmp_path: Path,
) -> None:
    contenders = 8
    for _ in range(50):
        for leftover in tmp_path.iterdir():
            leftover.unlink()

        outcomes = contend_for_one_lock(tmp_path, contenders)

        assert set(outcomes) <= {"lost", "removed"}, outcomes
        assert outcomes.count("removed") <= 1


def test_set_aside_lock_counts_a_rename_refused_for_lack_of_rights_as_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"seen")

    def refuse(source: Path, target: Path) -> Path:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "rename", refuse)

    assert set_aside_lock(tmp_path, b"seen") is False

    assert (tmp_path / LOCK_FILE).read_bytes() == b"seen"


def test_set_aside_lock_counts_a_lock_moved_again_before_it_is_read_as_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"seen")
    real_rename = Path.rename

    def rename_then_moved_by_another_run(source: Path, target: Path) -> Path:
        moved = real_rename(source, target)
        real_rename(moved, tmp_path / "elsewhere")
        return moved

    monkeypatch.setattr(Path, "rename", rename_then_moved_by_another_run)

    assert set_aside_lock(tmp_path, b"seen") is False

    assert [path.name for path in tmp_path.iterdir()] == ["elsewhere"]


def test_set_aside_lock_leaves_a_changed_lock_aside_when_it_cannot_be_put_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / LOCK_FILE).write_bytes(b"newer run")

    def unsupported(source: Path, target: Path) -> None:
        raise OSError(1, "Operation not permitted")

    monkeypatch.setattr(run_journal.os, "link", unsupported)

    with pytest.raises(OSError, match="Operation not permitted"):
        set_aside_lock(tmp_path, b"seen")

    (aside,) = leftovers(tmp_path)
    assert (tmp_path / aside).read_bytes() == b"newer run"
    assert not (tmp_path / LOCK_FILE).exists()
