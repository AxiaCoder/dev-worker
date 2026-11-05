from pathlib import Path

import pytest

from dev_worker.run_journal import JOURNAL_FILE, set_aside_torn_tail, torn_tail

TORN_FILE = "journal.torn"


def test_set_aside_creates_nothing_when_the_journal_is_absent(tmp_path: Path) -> None:
    set_aside_torn_tail(tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [b"", b'{"task_id": "a"}\n'])
def test_set_aside_leaves_a_whole_journal_untouched(
    tmp_path: Path, content: bytes
) -> None:
    journal = tmp_path / JOURNAL_FILE
    journal.write_bytes(content)

    set_aside_torn_tail(tmp_path)

    assert journal.read_bytes() == content
    assert not (tmp_path / TORN_FILE).exists()


def test_set_aside_moves_the_torn_tail_to_the_torn_file(tmp_path: Path) -> None:
    journal = tmp_path / JOURNAL_FILE
    journal.write_bytes(b'{"task_id": "a"}\n{"task_id": "b", "sta')

    set_aside_torn_tail(tmp_path)

    assert journal.read_bytes() == b'{"task_id": "a"}\n'
    assert (tmp_path / TORN_FILE).read_bytes() == b'{"task_id": "b", "sta\n'


def test_set_aside_appends_to_an_existing_torn_file(tmp_path: Path) -> None:
    (tmp_path / TORN_FILE).write_bytes(b"\xff\n")
    (tmp_path / JOURNAL_FILE).write_bytes(b'{"task_id": "b"')

    set_aside_torn_tail(tmp_path)

    assert (tmp_path / JOURNAL_FILE).read_bytes() == b""
    assert (tmp_path / TORN_FILE).read_bytes() == b'\xff\n{"task_id": "b"\n'


def test_set_aside_raises_when_the_torn_file_cannot_be_written(
    tmp_path: Path,
) -> None:
    (tmp_path / TORN_FILE).mkdir()
    (tmp_path / JOURNAL_FILE).write_bytes(b'{"task_id": "b"')

    with pytest.raises(OSError):
        set_aside_torn_tail(tmp_path)

    assert (tmp_path / JOURNAL_FILE).read_bytes() == b'{"task_id": "b"'


@pytest.mark.parametrize(
    ("content", "tail"),
    [
        (b'{"task_id": "a"}\n[1]', b"[1]"),
        (b'"x"', b'"x"'),
        (b"42", b"42"),
        (b'{"task_id": "a"}\n{"task_id": "b', b'{"task_id": "b'),
    ],
)
def test_torn_tail_is_an_unterminated_last_line_that_is_not_a_json_object(
    content: bytes, tail: bytes
) -> None:
    assert torn_tail(content) == tail


@pytest.mark.parametrize(
    "content", [b"", b'{"task_id": "a"}\n', b'{"task_id": "a"}\n{"task_id": "b"}']
)
def test_torn_tail_is_none_for_a_whole_journal(content: bytes) -> None:
    assert torn_tail(content) is None


def test_set_aside_terminates_a_last_line_that_lacks_only_its_newline(
    tmp_path: Path,
) -> None:
    journal = tmp_path / JOURNAL_FILE
    journal.write_bytes(b'{"task_id": "a"}\n{"task_id": "b"}')

    set_aside_torn_tail(tmp_path)

    assert journal.read_bytes() == b'{"task_id": "a"}\n{"task_id": "b"}\n'
    assert not (tmp_path / TORN_FILE).exists()


def test_set_aside_moves_a_tail_that_is_json_but_not_an_object(
    tmp_path: Path,
) -> None:
    journal = tmp_path / JOURNAL_FILE
    journal.write_bytes(b'{"task_id": "a"}\n[1]')

    set_aside_torn_tail(tmp_path)

    assert journal.read_bytes() == b'{"task_id": "a"}\n'
    assert (tmp_path / TORN_FILE).read_bytes() == b"[1]\n"
