"""What a run of Claude Code ends on, read from the JSON result it writes to stdout.

Claude Code puts its answer in the `structured_output` field of its result, and leaves
the field out when the run ends without a valid one: the outcome of a run on a ticket
(see `container.outcome_schema`), or the verdict of a guard review (see
`container.verdict_schema`).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dev_worker.container import DEFECT_FIELDS, ContainerExit
from dev_worker.text import writable_text

OUTCOME_FIELDS = ("outcome", "message")
VERDICT_FIELDS = ("reviewed", "reason", "defects")


@dataclass(frozen=True)
class Outcome:
    """The outcome a run ended on, and its message: the summary of the work on `done`, the reason otherwise."""

    outcome: str
    message: str


@dataclass(frozen=True)
class Defect:
    """A reachable defect a guard review found: the path of its file, its line in the head commit, and its summary."""

    file: str
    line: int
    summary: str


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


def read_verdict(exited: ContainerExit) -> tuple[Defect, ...] | str:
    """Return the reachable defects the guard review in the container `exited` reports, or why it reports no verdict.

    The verdict is valid when Claude Code exited 0 and wrote a JSON object whose
    `structured_output` holds exactly `reviewed`, a boolean, `reason`, a string, and
    `defects`, a list of objects each holding exactly `file`, a string, `line`, an
    integer from 1, and `summary`, a string, and when `reviewed` is true: an empty list
    is then a review that found no reachable defect. Otherwise the reason is a clause
    saying why, as `read_outcome` gives it; a `reviewed` false gives "the review did not
    take place: " followed by the `reason` of the verdict. The file, the summary and the
    reason are made writable as `read_outcome` makes its message.
    """
    found = _structured_output(exited)
    if isinstance(found, str):
        return writable_text(found)
    verdict = found[0]
    problem = _verdict_problem(verdict)
    if problem is not None:
        return writable_text(
            "the structured output of Claude Code does not match the verdict "
            f"schema: {problem}"
        )
    if not verdict["reviewed"]:
        return writable_text(f"the review did not take place: {verdict['reason']}")
    return tuple(
        Defect(
            writable_text(defect["file"]),
            defect["line"],
            writable_text(defect["summary"]),
        )
        for defect in verdict["defects"]
    )


def _read(exited: ContainerExit, outcomes: Sequence[str]) -> Outcome | str:
    """Return the outcome the container `exited` reports, or why it reports none, its text as read."""
    found = _structured_output(exited)
    if isinstance(found, str):
        return found
    outcome = found[0]
    problem = _schema_problem(outcome, outcomes)
    if problem is not None:
        return (
            "the structured output of Claude Code does not match the outcome "
            f"schema: {problem}"
        )
    return Outcome(outcome["outcome"], outcome["message"])


def _structured_output(exited: ContainerExit) -> tuple[Any] | str:
    """Return, alone in a tuple, the structured output Claude Code wrote in the container `exited`, or why there is none, its text as read."""
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
    return (answer["structured_output"],)


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
    extra = _extra_fields(found, OUTCOME_FIELDS)
    if extra:
        return f"it holds the unexpected field {extra}"
    outcome = found.get("outcome")
    if not isinstance(outcome, str) or outcome not in outcomes:
        return f"its outcome {outcome!r} is not one of {', '.join(outcomes)}"
    if not isinstance(found.get("message"), str):
        return "its message is missing or not a string"
    return None


def _verdict_problem(found: object) -> str | None:
    """Return why `found` is not a verdict, or None when it is one."""
    if not isinstance(found, dict):
        return "it is not a JSON object"
    extra = _extra_fields(found, VERDICT_FIELDS)
    if extra:
        return f"it holds the unexpected field {extra}"
    if not isinstance(found.get("reviewed"), bool):
        return "its reviewed is missing or not a boolean"
    if not isinstance(found.get("reason"), str):
        return "its reason is missing or not a string"
    defects = found.get("defects")
    if not isinstance(defects, list):
        return "its defects are missing or not a list"
    for rank, defect in enumerate(defects, start=1):
        problem = _defect_problem(defect)
        if problem is not None:
            return f"its defect {rank} {problem}"
    return None


def _defect_problem(found: object) -> str | None:
    """Return, as a clause, why `found` is not a defect, or None when it is one."""
    if not isinstance(found, dict):
        return "is not a JSON object"
    extra = _extra_fields(found, DEFECT_FIELDS)
    if extra:
        return f"holds the unexpected field {extra}"
    if not isinstance(found.get("file"), str):
        return "has a file missing or not a string"
    line = found.get("line")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        return f"has the line {line!r}, not an integer from 1"
    if not isinstance(found.get("summary"), str):
        return "has a summary missing or not a string"
    return None


def _extra_fields(found: dict[Any, Any], fields: Sequence[str]) -> str:
    """Return the names of the keys of `found` not among `fields`, sorted and joined by commas, empty when there is none."""
    return ", ".join(sorted(str(name) for name in found if name not in fields))
