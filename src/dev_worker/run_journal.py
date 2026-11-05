"""The run journal and the run lock of an instance, kept in the instance directory.

The instance directory is the instance file path without its suffix
(`instances/acme.yaml` gives `instances/acme/`). It holds `journal.jsonl`, one JSON
line per run, and `run.lock`, present while a run is in progress.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

JOURNAL_FILE = "journal.jsonl"
LOCK_FILE = "run.lock"


class JournalError(ValueError):
    """Raised when the journal holds a line that is not a JSON object."""


def instance_directory(instance_path: Path) -> Path:
    """Return the directory of the instance read from `instance_path`."""
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


def append_entry(directory: Path, entry: Mapping[str, object]) -> None:
    """Append `entry` as one JSON line to the journal of `directory`.

    Raises `OSError` when the journal cannot be written.
    """
    with (directory / JOURNAL_FILE).open(
        "a", encoding="utf-8", newline="\n"
    ) as journal:
        journal.write(json.dumps(entry) + "\n")
