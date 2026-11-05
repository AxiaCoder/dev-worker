import json
import os
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
