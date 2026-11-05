"""Command line entry point: the `run`, `state` and `health` commands.

Each command names the instance file with `--instance`; `run` runs a mandate read as JSON
on stdin against that instance and the `WORKER.md` shipped with the package.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
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
from dev_worker.validation import describe_validation_error


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
    """Run the `deliver_ticket` mandate; not built yet, it always fails."""
    return failed(mandate.task_id, "The deliver_ticket mandate is not built yet.")


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
    header of the packaged `WORKER.md`. Never raises on bad input: bytes that are not
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
    return handler(mandate, instance)


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
    bytes, whatever the encoding of the system. `state` and `health` do not read the
    instance file yet. Returns 0 whenever a JSON document is written, even for a failed
    mandate; argparse exits with 2 on incorrect usage.
    """
    args = build_parser().parse_args(argv)
    if args.command == "run":
        answer = run_mandate(args.mandate, sys.stdin.buffer.read(), args.instance)
    elif args.command == "state":
        answer = State()
    else:
        answer = Health(status="ok", reason=None, active=True)
    sys.stdout.flush()
    sys.stdout.buffer.write(answer.model_dump_json().encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0
