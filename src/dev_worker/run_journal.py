"""The run journal and the run lock of an instance, kept in the instance directory.

The instance directory is the instance file path without its suffix
(`instances/acme.yaml` gives `instances/acme/`). It holds `journal.jsonl`, JSON lines
appended when a run starts and when it ends, the last line of a task giving its
outcome, `run.lock`, present while a run is in progress, `run.guard`, the file locked
while a run removes the lock, and `journal.torn`, the journal lines a run cut off while
writing them, once a later run has set them aside.
"""

import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import psutil

JOURNAL_FILE = "journal.jsonl"
LOCK_FILE = "run.lock"
GUARD_FILE = "run.guard"
TORN_FILE = "journal.torn"
CREATE_TIME_SLACK = 1.0
GUARD_POLL_SECONDS = 0.001
LOCK_REMOVAL_SECONDS = 1.0

if sys.platform == "win32":  # pragma: not on posix
    import msvcrt

    def lock_guard(descriptor: int) -> None:
        """Lock the first byte of the open guard file, waiting while it is locked.

        Raises `OSError` when the file cannot be locked for another reason.
        """
        while True:
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except PermissionError:
                time.sleep(GUARD_POLL_SECONDS)
            else:
                return

    def unlock_guard(descriptor: int) -> None:
        """Unlock the first byte of the open guard file."""
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

    def remove_lock_file(path: Path) -> None:
        """Remove the lock file at `path`, if present.

        While another process holds the file open, removal is retried every
        `GUARD_POLL_SECONDS` seconds for up to `LOCK_REMOVAL_SECONDS` seconds, then the
        `PermissionError` is raised. Raises another `OSError` at once.
        """
        deadline = time.perf_counter() + LOCK_REMOVAL_SECONDS
        while True:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                if time.perf_counter() >= deadline:
                    raise
                time.sleep(GUARD_POLL_SECONDS)
            else:
                return

else:  # pragma: not on Windows_NT
    import fcntl

    def lock_guard(descriptor: int) -> None:
        """Lock the open guard file, waiting while it is locked.

        Raises `OSError` when the file cannot be locked.
        """
        fcntl.flock(descriptor, fcntl.LOCK_EX)

    def unlock_guard(descriptor: int) -> None:
        """Unlock the open guard file."""
        fcntl.flock(descriptor, fcntl.LOCK_UN)

    def remove_lock_file(path: Path) -> None:
        """Remove the lock file at `path`, if present.

        Raises `OSError` when it cannot be removed.
        """
        path.unlink(missing_ok=True)


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
            os.write(descriptor, lock_content(record))
        finally:
            os.close(descriptor)
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


def lock_content(record: Mapping[str, object]) -> bytes:
    """Return the bytes a lock file holding `record` is written with."""
    return json.dumps(record).encode("utf-8")


def release_lock(path: Path, record: Mapping[str, object]) -> None:
    """Remove the lock file at `path` if it still holds `record`; leave it otherwise.

    The lock is read and removed while holding the guard of its directory (see
    `held_guard`), so that no run takes it over in between. A lock already gone, or
    taken over by another run, is left as it is. Raises `OSError` when the guard
    cannot be held or the lock cannot be read or removed.
    """
    if not path.exists():
        return
    with held_guard(path.parent):
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return
        if content == lock_content(record):
            remove_lock_file(path)


@contextmanager
def held_guard(directory: Path) -> Iterator[None]:
    """Hold the guard of `directory` for the duration of the `with` block.

    The guard is an OS lock on the guard file, created if missing; it waits while
    another run, or another thread, holds it, and the OS releases it when the process
    holding it dies. Raises `OSError` when the guard file cannot be opened or locked.
    """
    descriptor = os.open(directory / GUARD_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        lock_guard(descriptor)
        try:
            yield
        finally:
            unlock_guard(descriptor)
    finally:
        os.close(descriptor)


def journal_holds(directory: Path, task_id: str) -> bool:
    """Tell whether the journal of `directory` has a line for `task_id`.

    False when there is no journal yet. Raises `OSError` when it cannot be read,
    `JournalError` when a line is not UTF-8 JSON, is nested too deeply to parse, or
    is not an object.
    """
    path = directory / JOURNAL_FILE
    if not path.exists():
        return False
    with path.open("rb") as journal:
        for number, line in enumerate(journal, start=1):
            if not line.strip():
                continue
            if journal_entry(number, line).get("task_id") == task_id:
                return True
    return False


def journal_entry(number: int, line: bytes) -> dict[str, object]:
    """Return the entry held by `line`, the journal line numbered `number` from 1.

    Raises `JournalError` when `line` is not UTF-8 JSON, is nested too deeply to parse,
    or is not an object.
    """
    try:
        entry = json.loads(line.decode("utf-8"))
    except (ValueError, RecursionError) as error:
        raise JournalError(f"line {number} is not valid JSON: {error}") from None
    if not isinstance(entry, dict):
        raise JournalError(f"line {number} is not a JSON object.")
    return entry


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
    return parse_lock(content)


def parse_lock(content: bytes) -> dict[str, object]:
    """Return the lock record in `content`, `{}` when it is not a UTF-8 JSON object.

    A record nested too deeply gives `{}` too. Never raises.
    """
    try:
        record = json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError):
        return {}
    return record if isinstance(record, dict) else {}


def process_runs_since(pid: int, moment: float) -> bool:
    """Tell whether process `pid` runs and was started no later than `moment`.

    `pid` must be positive; `moment` is a POSIX timestamp in seconds, allowed
    `CREATE_TIME_SLACK` seconds of imprecision. A process started after `moment` holds
    a reused PID and does not count; a process this one cannot inspect, for lack of
    rights or otherwise, does.
    """
    try:
        created = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        return True
    return created <= moment + CREATE_TIME_SLACK


def remove_lock_if_unchanged(directory: Path, seen: bytes) -> bool:
    """Remove the lock of `directory` if it still holds `seen`; tell whether it did.

    The lock is read and removed while holding the guard of `directory` (see
    `held_guard`), so that of several runs removing the same lock only one succeeds,
    and a lock taken over since it was seen is left as it is. Raises `OSError` when
    the guard cannot be held or the lock cannot be read or removed.
    """
    path = directory / LOCK_FILE
    if not path.exists():
        return False
    with held_guard(directory):
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return False
        if content != seen:
            return False
        remove_lock_file(path)
    return True


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


def holds_torn_line(set_aside: bytes, tail: bytes) -> bool:
    """Tell whether the torn file content `set_aside` already ends with `tail` as a line."""
    return (b"\n" + set_aside).endswith(b"\n" + tail + b"\n")


def set_aside_torn_tail(directory: Path) -> None:
    """Cut the torn last line off the journal of `directory` into its torn file.

    The torn line is appended, with a final newline, to `journal.torn`, unless that file
    already ends with it as a whole line, as a cut-off that failed to truncate leaves it; a last line that
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
    torn_path = directory / TORN_FILE
    try:
        set_aside = torn_path.read_bytes()
    except FileNotFoundError:
        set_aside = b""
    if not holds_torn_line(set_aside, tail):
        with torn_path.open("ab") as torn:
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
