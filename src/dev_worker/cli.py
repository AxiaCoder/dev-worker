"""Command line entry point: the `run`, `state` and `health` commands.

Each command names the instance file with `--instance`; `run` runs a mandate read as JSON
on stdin against that instance and the `WORKER.md` shipped with the package.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from importlib.resources import as_file, files
from pathlib import Path

from pydantic import ValidationError

from dev_worker.contract import (
    CONTRACT_VERSION,
    Health,
    MandateInput,
    MandateOutput,
    State,
)
from dev_worker.header import Header, HeaderError, load_worker_file
from dev_worker.instance import Instance, InstanceError, load_instance
from dev_worker.run_journal import (
    JOURNAL_FILE,
    LOCK_FILE,
    TORN_FILE,
    JournalError,
    append_entry,
    instance_directory,
    journal_entry,
    journal_holds,
    parse_lock,
    process_runs_since,
    read_entries,
    read_lock,
    release_lock,
    set_aside_lock,
    set_aside_torn_tail,
    take_lock,
    torn_tail,
)
from dev_worker.validation import describe_validation_error

TICKET_KEY = re.compile(r"(.+)-([1-9][0-9]*)")
RECENT_RUNS = 10


def failed(task_id: str | None, *errors: str) -> MandateOutput:
    """Build a failed mandate output carrying the given error messages.

    A character that cannot be encoded as UTF-8, such as a lone surrogate from an
    undecodable path, is written as its backslash escape (`\\udce9`), so the output
    can always be serialized.
    """
    return MandateOutput(
        task_id=task_id,
        status="failed",
        summary="The mandate could not be run.",
        errors=[writable_text(error) for error in errors],
    )


def writable_text(text: str) -> str:
    """Return `text` with every character UTF-8 cannot encode written as its backslash escape."""
    return text.encode("utf-8", "backslashreplace").decode("utf-8")


def deliver_ticket(mandate: MandateInput, instance: Instance) -> MandateOutput:
    """Run the `deliver_ticket` mandate; its body is not built yet, it does nothing."""
    return MandateOutput(
        task_id=mandate.task_id,
        status="completed",
        summary="Nothing was done: the deliver_ticket mandate has no body yet.",
    )


MANDATES: dict[str, Callable[[MandateInput, Instance], MandateOutput]] = {
    "deliver_ticket": deliver_ticket,
}


def is_utf8_encodable(value: object) -> bool:
    """Tell whether a parsed JSON value can be written back as UTF-8.

    False when any string in it holds a lone surrogate.
    """
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def os_reason(error: OSError) -> str:
    """Return the reason of `error` as written in a message: its strerror, or its text.

    Copied from the Broker worker, with the error's text instead of its type as fallback.
    """
    return error.strerror or str(error)


def check_ticket_key(mandate: MandateInput, instance: Instance) -> str | None:
    """Return why `params.ticket_key` of `mandate` is refused, or None when it is valid.

    A valid key is a string `<PROJECT>-<n>`, where `<PROJECT>` is the tracker project
    of `instance` and `<n>` a positive integer written without a leading zero.
    """
    key = mandate.params.get("ticket_key")
    project = instance.tracker.project
    if key is None:
        return "params.ticket_key is missing."
    if not isinstance(key, str):
        return "params.ticket_key must be a string."
    match = TICKET_KEY.fullmatch(key)
    if match is None:
        return f"params.ticket_key {key!r} is not of the form {project}-<number>."
    if match.group(1) != project:
        return f"Ticket {key!r} is not in the tracker project {project!r}."
    return None


def run_journaled(
    handler: Callable[[MandateInput, Instance], MandateOutput],
    mandate: MandateInput,
    instance: Instance,
    instance_path: Path,
) -> MandateOutput:
    """Run `handler` under the lock of the instance directory and journal the run.

    The ticket key is checked first, then the lock is taken, or taken over from a run
    that is gone (see `take_over_lock`); a task_id already in the journal is refused.
    A refused run is not journaled. The lock is released only if no other run has
    taken it over meanwhile. Every file error becomes a failed output naming the
    reason, except a lock that cannot be removed: the run keeps its status and the
    reason is added to `errors`. An exception raised by `handler` propagates once the
    lock is released.
    """
    task_id = mandate.task_id
    refusal = check_ticket_key(mandate, instance)
    if refusal is not None:
        return failed(task_id, refusal)
    ticket_key = mandate.params["ticket_key"]
    try:
        directory = instance_directory(instance_path)
    except ValueError as error:
        return failed(task_id, str(error))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return failed(
            task_id,
            f"Instance directory '{directory}' cannot be created: {os_reason(error)}",
        )
    start = datetime.now(UTC).isoformat()
    record = {
        "task_id": task_id,
        "ticket_key": ticket_key,
        "start": start,
        "pid": os.getpid(),
    }
    try:
        taken: Path | str = take_lock(directory, record)
    except FileExistsError:
        taken = take_over_lock(
            directory, record, instance.limits["run_timeout_seconds"]
        )
    except OSError as error:
        taken = lock_unwritable(directory, error)
    if isinstance(taken, str):
        return failed(task_id, taken)
    lock = taken
    try:
        output = run_under_lock(handler, mandate, instance, directory, start)
    finally:
        try:
            release_lock(lock, record)
        except OSError as error:
            release_error = f"Lock file '{lock}' cannot be removed: {os_reason(error)}"
        else:
            release_error = None
    if release_error is not None:
        return output.model_copy(update={"errors": [*output.errors, release_error]})
    return output


def run_under_lock(
    handler: Callable[[MandateInput, Instance], MandateOutput],
    mandate: MandateInput,
    instance: Instance,
    directory: Path,
    start: str,
) -> MandateOutput:
    """Run `handler` unless its task_id is already journaled, journaling the run twice.

    A `running` line is appended before `handler` runs, so that the task is never run
    again even if its final line, appended once `handler` returns, cannot be written;
    the output is then kept, with the journal error added to `errors`. A run whose
    `running` line cannot be written fails without running `handler`. `start` is the
    ISO 8601 UTC start of the run, written as is in both lines. A torn last journal
    line is first set aside in the torn file.
    """
    task_id = mandate.task_id
    try:
        set_aside_torn_tail(directory)
        repeated = journal_holds(directory, task_id)
    except OSError as error:
        return failed(
            task_id,
            f"Journal in '{directory}' cannot be read or repaired: {os_reason(error)}",
        )
    except JournalError as error:
        return failed(task_id, f"Journal in '{directory}': {error}")
    if repeated:
        return failed(task_id, f"Task {task_id!r} has already been run.")
    entry: dict[str, object] = {
        "task_id": task_id,
        "ticket_key": mandate.params["ticket_key"],
        "status": "running",
        "start": start,
        "duration": None,
        "tokens": 0,
    }
    try:
        append_entry(directory, entry)
    except OSError as error:
        return failed(
            task_id, f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        )
    began = time.monotonic()
    output = handler(mandate, instance)
    entry |= {"status": output.status, "duration": time.monotonic() - began}
    try:
        append_entry(directory, entry)
    except OSError as error:
        journal_error = (
            f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        )
        return output.model_copy(
            update={"errors": [*output.errors, writable_text(journal_error)]}
        )
    return output


def load_header() -> Header:
    """Read and validate the header of the `WORKER.md` shipped with the package.

    Raises `HeaderError` when its header is invalid, `OSError` when it cannot be read.
    """
    with as_file(files("dev_worker") / "WORKER.md") as path:
        return load_worker_file(path).header


def run_mandate(
    mandate_name: str, raw_input: bytes, instance_path: Path
) -> MandateOutput:
    """Run the mandate named on the command line against the UTF-8 JSON input.

    The mandate gets the instance read from `instance_path` and checked against the
    header of the packaged `WORKER.md`, and runs under the lock of the instance
    directory, journaled there (see `run_journaled`). Never raises on bad input: bytes that are not
    UTF-8, JSON that cannot be parsed or written back as UTF-8, input that breaks the
    contract, an unknown mandate, and a `WORKER.md` or instance file that cannot be
    read or is invalid all become a failed output naming the reason.
    """
    try:
        text = raw_input.decode("utf-8")
    except UnicodeDecodeError as error:
        return failed(None, f"Input is not valid UTF-8: {error}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        return failed(None, f"Input is not valid JSON: {error}")
    except RecursionError:
        return failed(None, "Input JSON is nested too deeply.")
    if not isinstance(data, dict):
        return failed(None, "Input must be a JSON object.")

    raw_task_id = data.get("task_id")
    task_id = (
        raw_task_id
        if isinstance(raw_task_id, str) and is_utf8_encodable(raw_task_id)
        else None
    )
    if not is_utf8_encodable(data):
        return failed(task_id, "Input holds text that cannot be encoded as UTF-8.")

    version = data.get("contract_version")
    if version != CONTRACT_VERSION:
        return failed(
            task_id,
            f"Unsupported contract_version {version!r}, expected {CONTRACT_VERSION!r}.",
        )

    try:
        mandate = MandateInput.model_validate(data)
    except ValidationError as error:
        return failed(task_id, *describe_validation_error(error))

    if mandate.mandate != mandate_name:
        return failed(
            task_id,
            f"Mandate {mandate_name!r} on the command line differs from "
            f"{mandate.mandate!r} in the input.",
        )

    handler = MANDATES.get(mandate_name)
    if handler is None:
        return failed(task_id, f"Unknown mandate {mandate_name!r}.")

    instance = load_checked_instance(instance_path)
    if isinstance(instance, str):
        return failed(task_id, instance)
    return run_journaled(handler, mandate, instance, instance_path)


def load_checked_instance(instance_path: Path) -> Instance | str:
    """Read the instance from `instance_path`, checked against the packaged `WORKER.md`.

    Returns the reason as a sentence instead when `WORKER.md` or the instance file
    cannot be read or is invalid. Never raises on a file error.
    """
    try:
        header = load_header()
    except HeaderError as error:
        return f"WORKER.md does not load: {error}"
    except OSError as error:
        return f"WORKER.md cannot be read: {os_reason(error)}"
    try:
        return load_instance(instance_path, header)
    except InstanceError as error:
        return f"Instance file '{instance_path}': {error}"
    except OSError as error:
        return f"Instance file '{instance_path}' cannot be read: {os_reason(error)}"


def run_start(record: Mapping[str, object]) -> datetime | None:
    """Return the `start` of a journal entry or lock record as an aware datetime.

    `None` when `start` is missing, not an ISO 8601 string, or has no time zone.
    """
    start = record.get("start")
    if not isinstance(start, str):
        return None
    try:
        moment = datetime.fromisoformat(start)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment


def run_day(entry: Mapping[str, object]) -> date | None:
    """Return the UTC day the journaled run started, `None` when `start` is unusable."""
    moment = run_start(entry)
    return None if moment is None else moment.astimezone(UTC).date()


def run_tokens(entry: Mapping[str, object]) -> int:
    """Return the tokens the journaled run spent, 0 when `tokens` is not an integer."""
    tokens = entry.get("tokens")
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def answerable(record: Mapping[str, object]) -> bool:
    """Tell whether `record`, placed in the state, can be validated and written as JSON.

    False when a string in it holds a lone surrogate or it is nested deeper than the
    state model validates or serializes.
    """
    try:
        State.model_validate({"common": {"recent": [record]}}).model_dump_json()
    except ValueError:
        return False
    return True


def task_entries(
    entries: Sequence[Mapping[str, object]],
    locks: Sequence[Mapping[str, object] | None],
) -> list[Mapping[str, object]]:
    """Return one entry per task of the journal `entries`, its last line, oldest first.

    Tasks are ordered by their first line; a line whose `task_id` is not a string is a
    task of its own. `locks` are the lock records read before and after `entries`,
    `None` when no run held the lock. A task whose last line is `running` is shown as
    `failed`, interrupted, unless one of `locks` is held by that task or by a run
    whose task is unknown.
    """
    tasks: dict[object, Mapping[str, object]] = {}
    for entry in entries:
        task_id = entry.get("task_id")
        tasks[task_id if isinstance(task_id, str) else object()] = entry
    return [
        interrupted(entry)
        if entry.get("status") == "running"
        and not any(held_by(lock, entry) for lock in locks)
        else entry
        for entry in tasks.values()
    ]


def held_by(lock: Mapping[str, object] | None, entry: Mapping[str, object]) -> bool:
    """Tell whether the lock record `lock` may be held by the run of `entry`.

    False when no run holds the lock (`None`); true when the lock names no task.
    """
    if lock is None:
        return False
    return "task_id" not in lock or lock["task_id"] == entry.get("task_id")


def interrupted(entry: Mapping[str, object]) -> dict[str, object]:
    """Return the `running` journal entry `entry` as the failed run it became."""
    return {
        **entry,
        "status": "failed",
        "errors": ["The run stopped before its final journal line was written."],
    }


def read_state(instance_path: Path) -> State:
    """Build the state of the instance read from `instance_path`, writing nothing.

    `current` holds the lock record of the run in progress (`{}` when its details are
    unknown), `recent` the last `RECENT_RUNS` tasks of the journal (see
    `task_entries`), most recent first, and `cost` the number of those tasks and the
    tokens they spent on the current UTC day. The
    instance file itself is not read; an instance path without a suffix gives an empty
    state. A journal entry that `answerable` refuses is skipped, a lock record it
    refuses gives `{}`. Never raises.
    """
    try:
        directory = instance_directory(instance_path)
    except ValueError:
        directory = None
    lock = read_lock(directory) if directory else None
    if lock is not None and not answerable(lock):
        lock = {}
    entries = (
        task_entries(
            [entry for entry in read_entries(directory) if answerable(entry)],
            [lock, read_lock(directory)],
        )
        if directory
        else []
    )
    today = datetime.now(UTC).date()
    todays = [entry for entry in entries if run_day(entry) == today]
    tokens = sum(run_tokens(entry) for entry in todays)
    return State.model_validate(
        {
            "common": {
                "current": [] if lock is None else [lock],
                "recent": entries[::-1][:RECENT_RUNS],
                "pending": [],
                "cost": {
                    "day": today.isoformat(),
                    "runs": len(todays),
                    "tokens": tokens,
                },
            }
        }
    )


def unwritable_instance_directory(directory: Path) -> str | None:
    """Return why a run could not write in `directory`, or `None` when it could.

    Checks, without writing, the directory when it exists, else its nearest existing
    parent, in which a run would create it: it must be a directory the process may
    write in and enter. The journal, when it exists, must be writable too.
    """
    nearest = directory
    while not nearest.exists() and nearest.parent != nearest:
        nearest = nearest.parent
    if not (nearest.is_dir() and os.access(nearest, os.W_OK | os.X_OK)):
        return (
            f"Instance directory '{directory}' cannot be written: "
            f"'{nearest}' is not a writable directory."
        )
    journal = directory / JOURNAL_FILE
    if journal.exists() and not os.access(journal, os.W_OK):
        return f"Journal '{journal}' cannot be written."
    return None


def lock_age(directory: Path, lock: Mapping[str, object]) -> float | None:
    """Return for how many seconds the run holding the lock of `directory` has run.

    The age is counted from the `start` of the lock record, else from the time the
    lock file was last modified. `None` when the lock file is gone or cannot be read.
    """
    now = datetime.now(UTC)
    start = run_start(lock)
    if start is not None:
        return (now - start).total_seconds()
    try:
        modified = (directory / LOCK_FILE).stat().st_mtime
    except OSError:
        return None
    return now.timestamp() - modified


def gone_reason(
    directory: Path, lock: Mapping[str, object], timeout: int
) -> str | None:
    """Return why the run holding the lock of `directory` is gone, `None` if it is not.

    `lock` is the lock record. The run is gone when it has held the lock longer than
    `timeout` seconds (see `lock_age`), or when the record names its process by a
    positive integer `pid` and a usable `start`, and that process no longer runs or
    was started after `start`.
    """
    age = lock_age(directory, lock)
    if age is not None and age > timeout:
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"started {age:.0f} seconds ago, beyond the run timeout of "
            f"{timeout} seconds."
        )
    pid = lock.get("pid")
    start = run_start(lock)
    if (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and start is not None
        and not process_runs_since(pid, start.timestamp())
    ):
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"is gone: its process {pid} no longer runs."
        )
    return None


def take_over_lock(
    directory: Path, record: Mapping[str, object], timeout: int
) -> Path | str:
    """Take the lock of `directory` from a run that is gone, writing `record` in it.

    The gone run (see `gone_reason`) is journaled first, as a failed run with the
    `task_id`, `ticket_key` and `start` of its lock record and the reason it is gone
    in `errors`; a torn last journal line is set aside before. Returns the lock file
    path, or why the lock cannot be taken: its run is not gone, another run took it
    first, or a file cannot be read or written. Never raises on a file error.
    """
    in_progress = f"A run is already in progress in instance directory '{directory}'."
    try:
        seen = (directory / LOCK_FILE).read_bytes()
        held = parse_lock(seen)
        reason = gone_reason(directory, held, timeout)
        if reason is None or not set_aside_lock(directory, seen):
            return in_progress
    except FileNotFoundError:
        return in_progress
    except OSError as error:
        return lock_unwritable(directory, error)
    entry = {
        "task_id": held.get("task_id"),
        "ticket_key": held.get("ticket_key"),
        "status": "failed",
        "start": held.get("start"),
        "duration": None,
        "tokens": 0,
        "errors": [reason],
    }
    try:
        set_aside_torn_tail(directory)
        append_entry(directory, entry)
    except OSError as error:
        return f"Journal in '{directory}' cannot be written: {os_reason(error)}"
    try:
        return take_lock(directory, record)
    except FileExistsError:
        return in_progress
    except OSError as error:
        return lock_unwritable(directory, error)


def lock_unwritable(directory: Path, error: OSError) -> str:
    """Return the sentence saying the lock file in `directory` fails with `error`."""
    return f"Lock file in '{directory}' cannot be written: {os_reason(error)}"


def read_health(instance_path: Path) -> Health:
    """Tell whether the worker can run against the instance read from `instance_path`.

    `down` when `WORKER.md` or the instance file does not load, the instance path has
    no suffix, the instance directory or its existing journal cannot be written, or
    the journal cannot be read the way a run reads it (see `unreadable_journal`);
    `degraded` when the run holding the lock is gone (see `gone_reason`), which the
    next run mends by taking the lock over, or the journal holds a torn line (see
    `torn_journal_line`); `ok` otherwise. Writes nothing and never raises.
    """
    instance = load_checked_instance(instance_path)
    if isinstance(instance, str):
        return Health(status="down", reason=writable_text(instance), active=True)
    try:
        directory = instance_directory(instance_path)
    except ValueError as error:
        return Health(status="down", reason=writable_text(str(error)), active=True)
    unwritable = unwritable_instance_directory(directory)
    if unwritable is not None:
        return Health(status="down", reason=writable_text(unwritable), active=True)
    unreadable = unreadable_journal(directory)
    if unreadable is not None:
        return Health(status="down", reason=writable_text(unreadable), active=True)
    lock = read_lock(directory)
    timeout = instance.limits["run_timeout_seconds"]
    gone = None if lock is None else gone_reason(directory, lock, timeout)
    if gone is not None:
        return Health(status="degraded", reason=writable_text(gone), active=True)
    torn = torn_journal_line(directory)
    if torn is not None:
        return Health(status="degraded", reason=writable_text(torn), active=True)
    return Health(status="ok", reason=None, active=True)


def unreadable_journal(directory: Path) -> str | None:
    """Return why a run could not read the journal of `directory`, `None` when it could.

    A journal that does not exist yet can be read. Its torn last line, if any, is left
    out, since the next run sets it aside (see `torn_journal_line`). Never raises.
    """
    journal = directory / JOURNAL_FILE
    if not journal.exists():
        return None
    if not journal.is_file():
        return f"Journal '{journal}' is not a regular file."
    try:
        content = journal.read_bytes()
    except OSError as error:
        return f"Journal '{journal}' cannot be read: {os_reason(error)}"
    tail = torn_tail(content)
    whole = content if tail is None else content[: len(content) - len(tail)]
    for number, line in enumerate(io.BytesIO(whole), start=1):
        if not line.strip():
            continue
        try:
            journal_entry(number, line)
        except JournalError as error:
            return f"Journal '{journal}': {error}"
    return None


def torn_journal_line(directory: Path) -> str | None:
    """Return why the journal of `directory` holds a torn line, `None` when it does not.

    A torn line is one in the torn file, or a torn last line of the journal that no run
    has set aside yet. A journal that cannot be read is taken as whole. Never raises.
    """
    torn = directory / TORN_FILE
    if torn.exists():
        return f"Torn journal lines are set aside in '{torn}'."
    journal = directory / JOURNAL_FILE
    try:
        content = journal.read_bytes()
    except OSError:
        return None
    if torn_tail(content) is not None:
        return f"Journal '{journal}' ends with a torn line."
    return None


def add_instance_argument(command: argparse.ArgumentParser) -> None:
    """Add the required `--instance` option naming the YAML instance file."""
    command.add_argument(
        "--instance",
        required=True,
        type=Path,
        help="path of the YAML instance file naming the tracker project and repository",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the `run`, `state` and `health` subcommands.

    All three require `--instance`; `run` also takes the mandate name.
    """
    parser = argparse.ArgumentParser(prog="dev-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a mandate read as JSON on stdin")
    run.add_argument("mandate", help="name of the mandate to run")
    add_instance_argument(run)
    state = commands.add_parser("state", help="print the worker state")
    add_instance_argument(state)
    health = commands.add_parser("health", help="print the worker health")
    add_instance_argument(health)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, write the JSON answer on stdout and return the exit code.

    The `run` input is read from stdin as raw bytes and the answer is written as UTF-8
    bytes, whatever the encoding of the system. `state` reads the journal and lock of the
    instance directory; `health` reads the instance and its directory (see `read_health`). Returns 0 whenever a
    JSON document is written, even for a failed mandate; argparse exits with 2 on
    incorrect usage.
    """
    args = build_parser().parse_args(argv)
    if args.command == "run":
        answer = run_mandate(args.mandate, sys.stdin.buffer.read(), args.instance)
    elif args.command == "state":
        answer = read_state(args.instance)
    else:
        answer = read_health(args.instance)
    sys.stdout.flush()
    sys.stdout.buffer.write(answer.model_dump_json().encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0
