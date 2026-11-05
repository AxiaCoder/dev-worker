"""The run journal and the run lock of an instance, kept in the instance directory.

The instance directory is the instance file path without its suffix
(`instances/acme.yaml` gives `instances/acme/`). It holds `journal.jsonl`, one JSON
line per run, `run.lock`, present while a run is in progress, and `journal.torn`, the
journal lines a run cut off while writing them, once a later run has set them aside.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

JOURNAL_FILE = "journal.jsonl"
LOCK_FILE = "run.lock"
TORN_FILE = "journal.torn"


class JournalError(ValueError):
    """Raised when the journal holds a line that is not a JSON object."""


def instance_directory(instance_path: Path) -> Path:
    """Return the directory of the instance read from `instance_path`.

    Raises `ValueError` when `instance_path` has no suffix, since the directory would
    then be the instance file itself.
    """
    if not instance_path.suffix:
        raise ValueError(f"Instance file '{instance_path}' has no suffix.")
    return instance_path.with_suffix("")


def take_lock(directory: Path, record: Mapping[str, object]) -> Path:
    """Create the lock file of `directory` atomically, holding `record` as JSON.

    Returns the lock file path. Raises `FileExistsError` when the lock is already
    held, another `OSError` when it cannot be written; a lock file that cannot be
    written is removed before the error is raised.
    """
    path = directory / LOCK_FILE
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        try:
            os.write(descriptor, json.dumps(record).encode("utf-8"))
        finally:
            os.close(descriptor)
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


def release_lock(path: Path) -> None:
    """Remove the lock file at `path`; raises `OSError` when it cannot be removed."""
    path.unlink(missing_ok=True)


def journal_holds(directory: Path, task_id: str) -> bool:
    """Tell whether the journal of `directory` has a line for `task_id`.

    False when there is no journal yet. Raises `OSError` when it cannot be read,
    `JournalError` when a line is not UTF-8 JSON or not an object.
    """
    path = directory / JOURNAL_FILE
    if not path.exists():
        return False
    with path.open("rb") as journal:
        for number, line in enumerate(journal, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line.decode("utf-8"))
            except ValueError as error:
                raise JournalError(
                    f"line {number} is not valid JSON: {error}"
                ) from None
            if not isinstance(entry, dict):
                raise JournalError(f"line {number} is not a JSON object.")
            if entry.get("task_id") == task_id:
                return True
    return False


def read_lock(directory: Path) -> dict[str, object] | None:
    """Return the record of the lock of `directory`, or `None` when no run holds it.

    The lock is read in one call, never through a handle kept open. A lock that is
    empty, unreadable, nested too deeply or not a JSON object gives `{}`: a run is in progress, details
    unknown. Never raises.
    """
    try:
        content = (directory / LOCK_FILE).read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    try:
        record = json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError):
        return {}
    return record if isinstance(record, dict) else {}


def read_entries(directory: Path) -> list[dict[str, object]]:
    """Return the journal entries of `directory`, oldest first.

    Lines that are not UTF-8 JSON objects, or are nested too deeply, are skipped; a missing or unreadable journal
    gives `[]`. Never raises.
    """
    try:
        content = (directory / JOURNAL_FILE).read_bytes()
    except OSError:
        return []
    entries: list[dict[str, object]] = []
    for line in content.splitlines():
        try:
            entry = json.loads(line.decode("utf-8"))
        except (ValueError, RecursionError):
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def torn_tail(content: bytes) -> bytes | None:
    """Return the unterminated last line of the journal `content`, `None` when it has none.

    A last line that lacks only its final newline, being a whole JSON object, is not torn.
    """
    if not content or content.endswith(b"\n"):
        return None
    tail = content[content.rfind(b"\n") + 1 :]
    try:
        entry = json.loads(tail.decode("utf-8"))
    except (ValueError, RecursionError):
        return tail
    return None if isinstance(entry, dict) else tail


def set_aside_torn_tail(directory: Path) -> None:
    """Cut the torn last line off the journal of `directory` into its torn file.

    The torn line is appended, with a final newline, to `journal.torn`; a last line that
    lacks only its final newline gets it instead. The journal is left as is when the
    torn file cannot be written. Raises `OSError` when a file cannot be read or written.
    """
    path = directory / JOURNAL_FILE
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return
    if not content or content.endswith(b"\n"):
        return
    tail = torn_tail(content)
    if tail is None:
        with path.open("ab") as journal:
            journal.write(b"\n")
        return
    with (directory / TORN_FILE).open("ab") as torn:
        torn.write(tail + b"\n")
    os.truncate(path, len(content) - len(tail))


def append_entry(directory: Path, entry: Mapping[str, object]) -> None:
    """Append `entry` as one JSON line to the journal of `directory`.

    Raises `OSError` when the journal cannot be written.
    """
    with (directory / JOURNAL_FILE).open(
        "a", encoding="utf-8", newline="\n"
    ) as journal:
        journal.write(json.dumps(entry) + "\n")
