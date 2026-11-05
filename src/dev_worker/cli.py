"""Command line entry point: the `run`, `state` and `health` commands.

Each command names the instance file with `--instance`; `run` runs a mandate read as JSON
on stdin against that instance and the `WORKER.md` shipped with the package.
"""

import argparse
import json
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
    JournalError,
    append_entry,
    instance_directory,
    journal_holds,
    read_entries,
    read_lock,
    release_lock,
    take_lock,
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
        errors=[
            error.encode("utf-8", "backslashreplace").decode("utf-8")
            for error in errors
        ],
    )


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

    The ticket key is checked first, then the lock is taken; a task_id already in the
    journal is refused. A refused run is not journaled. Every file error becomes a
    failed output naming the reason, except a lock that cannot be removed: the run
    keeps its status and the reason is added to `errors`. An exception raised by
    `handler` propagates once the lock is released.
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
    try:
        lock = take_lock(
            directory, {"task_id": task_id, "ticket_key": ticket_key, "start": start}
        )
    except FileExistsError:
        return failed(
            task_id,
            f"A run is already in progress in instance directory '{directory}'.",
        )
    except OSError as error:
        return failed(
            task_id,
            f"Lock file in '{directory}' cannot be written: {os_reason(error)}",
        )
    try:
        output = run_under_lock(handler, mandate, instance, directory, start)
    finally:
        try:
            release_lock(lock)
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
    """Run `handler` unless its task_id is already journaled, then journal the run.

    `start` is the ISO 8601 UTC start of the run, written as is in the journal line.
    """
    task_id = mandate.task_id
    try:
        repeated = journal_holds(directory, task_id)
    except OSError as error:
        return failed(
            task_id, f"Journal in '{directory}' cannot be read: {os_reason(error)}"
        )
    except JournalError as error:
        return failed(task_id, f"Journal in '{directory}': {error}")
    if repeated:
        return failed(task_id, f"Task {task_id!r} has already been run.")
    began = time.monotonic()
    output = handler(mandate, instance)
    entry = {
        "task_id": task_id,
        "ticket_key": mandate.params["ticket_key"],
        "status": output.status,
        "start": start,
        "duration": time.monotonic() - began,
        "tokens": 0,
    }
    try:
        append_entry(directory, entry)
    except OSError as error:
        return failed(
            task_id, f"Journal in '{directory}' cannot be written: {os_reason(error)}"
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

    try:
        header = load_header()
    except HeaderError as error:
        return failed(task_id, f"WORKER.md does not load: {error}")
    except OSError as error:
        return failed(task_id, f"WORKER.md cannot be read: {os_reason(error)}")
    try:
        instance = load_instance(instance_path, header)
    except InstanceError as error:
        return failed(task_id, f"Instance file '{instance_path}': {error}")
    except OSError as error:
        return failed(
            task_id,
            f"Instance file '{instance_path}' cannot be read: {os_reason(error)}",
        )
    return run_journaled(handler, mandate, instance, instance_path)


def run_day(entry: Mapping[str, object]) -> date | None:
    """Return the UTC day the journaled run started, `None` when `start` is unusable."""
    start = entry.get("start")
    if not isinstance(start, str):
        return None
    try:
        moment = datetime.fromisoformat(start)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment.astimezone(UTC).date()


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


def read_state(instance_path: Path) -> State:
    """Build the state of the instance read from `instance_path`, writing nothing.

    `current` holds the lock record of the run in progress (`{}` when its details are
    unknown), `recent` the last `RECENT_RUNS` journal entries, most recent first, and
    `cost` the number of runs and the tokens they spent on the current UTC day. The
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
        [entry for entry in read_entries(directory) if answerable(entry)]
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
    instance directory; `health` does not read the instance yet. Returns 0 whenever a
    JSON document is written, even for a failed mandate; argparse exits with 2 on
    incorrect usage.
    """
    args = build_parser().parse_args(argv)
    if args.command == "run":
        answer = run_mandate(args.mandate, sys.stdin.buffer.read(), args.instance)
    elif args.command == "state":
        answer = read_state(args.instance)
    else:
        answer = Health(status="ok", reason=None, active=True)
    sys.stdout.flush()
    sys.stdout.buffer.write(answer.model_dump_json().encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0
