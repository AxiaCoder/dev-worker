import json
import os
import threading
from pathlib import Path

import pytest

from dev_worker.run_journal import (
    JOURNAL_FILE,
    LOCK_FILE,
    JournalError,
    append_entry,
    instance_directory,
    journal_holds,
    release_lock,
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

    release_lock(lock)

    assert not lock.exists()


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


@pytest.mark.parametrize("line", [b"{", b"\xff"])
def test_journal_line_that_is_not_json_raises(tmp_path: Path, line: bytes) -> None:
    (tmp_path / JOURNAL_FILE).write_bytes(line + b"\n")

    with pytest.raises(JournalError, match="line 1 is not valid JSON"):
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
