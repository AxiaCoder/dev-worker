"""The outcome a run of Claude Code ends on, read from the JSON result it writes to stdout.

Claude Code puts the outcome in the `structured_output` field of its result, and leaves
the field out when the run ends without a valid one (see `container.outcome_schema`).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dev_worker.container import ContainerExit
from dev_worker.text import writable_text

OUTCOME_FIELDS = ("outcome", "message")


@dataclass(frozen=True)
class Outcome:
    """The outcome a run ended on, and its message: the summary of the work on `done`, the reason otherwise."""

    outcome: str
    message: str


def read_outcome(exited: ContainerExit, outcomes: Sequence[str]) -> Outcome | str:
    """Return the outcome the container `exited` reports, or why it reports none.

    The outcome is usable when Claude Code exited 0 and wrote a JSON object whose
    `structured_output` holds exactly `outcome`, one of `outcomes`, and `message`, a
    string. Otherwise the reason is a clause saying why: the output is not a JSON object,
    Claude Code exited nonzero, gave no structured output, or one that does not match the
    schema. It quotes the `subtype` and `result` of the JSON result when they are
    strings, and the stderr of the container when its output is not JSON and it exited
    nonzero. The message and the reason hold no NUL and encode as UTF-8: each NUL and
    each character UTF-8 cannot encode is written as its backslash escape.
    """
    found = _read(exited, outcomes)
    if isinstance(found, Outcome):
        return Outcome(found.outcome, writable_text(found.message))
    return writable_text(found)


def _read(exited: ContainerExit, outcomes: Sequence[str]) -> Outcome | str:
    """Return the outcome the container `exited` reports, or why it reports none, its text as read."""
    try:
        answer = json.loads(exited.stdout)
    except (ValueError, RecursionError):
        answer = None
    if not isinstance(answer, dict):
        reason = "the output of Claude Code is not a JSON object"
        if exited.code == 0:
            return reason
        stderr = " ".join(
            line.strip() for line in exited.stderr.splitlines() if line.strip()
        )
        return f"{reason}; it exited with code {exited.code}" + (
            f": {stderr}" if stderr else ""
        )
    details = _details(answer)
    if exited.code != 0:
        return f"Claude Code exited with code {exited.code}{details}"
    if "structured_output" not in answer:
        return f"Claude Code gave no structured output{details}"
    found = answer["structured_output"]
    problem = _schema_problem(found, outcomes)
    if problem is not None:
        return (
            "the structured output of Claude Code does not match the outcome "
            f"schema: {problem}"
        )
    return Outcome(found["outcome"], found["message"])


def _details(answer: dict[Any, Any]) -> str:
    """Return the `subtype` and `result` of a Claude Code result as a parenthesis, empty when neither is a string."""
    parts: list[str] = []
    subtype = answer.get("subtype")
    if isinstance(subtype, str) and subtype.strip():
        parts.append(f"subtype {subtype.strip()}")
    result = answer.get("result")
    if isinstance(result, str) and result.strip():
        parts.append(f"result: {result.strip()}")
    return f" ({', '.join(parts)})" if parts else ""


def _schema_problem(found: object, outcomes: Sequence[str]) -> str | None:
    """Return why `found` is not an outcome among `outcomes`, or None when it is one."""
    if not isinstance(found, dict):
        return "it is not a JSON object"
    extra = sorted(str(name) for name in found if name not in OUTCOME_FIELDS)
    if extra:
        return f"it holds the unexpected field {', '.join(extra)}"
    outcome = found.get("outcome")
    if not isinstance(outcome, str) or outcome not in outcomes:
        return f"its outcome {outcome!r} is not one of {', '.join(outcomes)}"
    if not isinstance(found.get("message"), str):
        return "its message is missing or not a string"
    return None
