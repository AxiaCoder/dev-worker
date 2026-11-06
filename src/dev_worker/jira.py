"""The Jira ticket tracker, reached through Atlassian's `twg` command line."""

import json
import os
import re
import subprocess
from collections.abc import Sequence
from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

from dev_worker.header import _describe
from dev_worker.instance import Tracker
from dev_worker.tickets import Blocker, Ticket, TrackerError

TICKET_KEY = re.compile(r"[A-Z][A-Z0-9_]*-[0-9]+")
BLOCKS = "Blocks"
REDACTED = "[token]"


class _Status(BaseModel):
    """A status as `twg` writes it."""

    name: str


class _LinkedFields(BaseModel):
    """The fields of a linked ticket."""

    status: _Status


class _LinkedTicket(BaseModel):
    """A ticket at the other end of a link."""

    key: str
    fields: _LinkedFields


class _LinkType(BaseModel):
    """The type of a link."""

    name: str


class _Link(BaseModel):
    """A link of a ticket; `inwardIssue` is set when the ticket is on the inward side."""

    type: _LinkType
    inwardIssue: _LinkedTicket | None = None


class _Item(BaseModel):
    """A ticket as `twg jira workitem get` writes it."""

    key: str
    status: _Status
    labels: list[str]
    issuelinks: list[_Link]


class _GetAnswer(BaseModel):
    """The answer of `twg jira workitem get` for one key."""

    data: Annotated[list[_Item], Field(min_length=1, max_length=1)]


_UNREADABLE = object()


def _parse_json(text: str) -> object:
    """Return the JSON value of `text`, or `_UNREADABLE` when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _UNREADABLE


def _check_key(key: str) -> None:
    """Raise `TrackerError` when `key` is not a Jira ticket key such as `ABC-12`."""
    if not TICKET_KEY.fullmatch(key):
        raise TrackerError(f"{key!r} is not a Jira ticket key such as ABC-12")


class JiraTracker:
    """A `TicketTracker` for the Jira site of `tracker`, through the `twg` command line.

    `command` is the program and leading arguments that run `twg`. Each call passes
    `TWG_SITE`, `TWG_USER` and, as `TWG_TOKEN`, the value of the variable `tracker.token_env`
    names, read at call time.
    """

    def __init__(self, tracker: Tracker, command: Sequence[str] = ("twg",)) -> None:
        """Keep `tracker` and `command` for later calls; nothing is run or read yet."""
        self._tracker = tracker
        self._command = tuple(command)

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`, with its status, labels and blockers.

        A blocker is a ticket linked to `key` by a "Blocks" link that blocks it. Raises
        `TrackerError` when `key` is not a ticket key, the token variable is not set, `twg`
        fails, or its answer cannot be read.
        """
        _check_key(key)
        answer = self._run(
            f"reading {key}",
            ["jira", "workitem", "get", key, "--fields", "status,labels,issuelinks"],
        )
        try:
            item = _GetAnswer.model_validate(answer).data[0]
        except ValidationError as error:
            description = _describe(error, root="answer", echo=False)
            raise TrackerError(
                f"the answer of twg to reading {key} cannot be read: {description}"
            ) from None
        return Ticket(
            key=item.key,
            status=item.status.name,
            labels=tuple(item.labels),
            blockers=tuple(
                Blocker(
                    key=link.inwardIssue.key, status=link.inwardIssue.fields.status.name
                )
                for link in item.issuelinks
                if link.type.name == BLOCKS and link.inwardIssue is not None
            ),
        )

    def set_status(self, key: str, status: str) -> None:
        """Move the ticket `key` to the real status name `status`.

        Raises `TrackerError` when `key` is not a ticket key, the token variable is not set,
        `twg` fails, or its answer is not JSON.
        """
        _check_key(key)
        self._run(
            f"moving {key} to {status!r}",
            ["jira", "workitem", "transition", "--id", key, "--transition-id", status],
        )

    def _token(self) -> str:
        """Return the token held by the variable `token_env` names; raise `TrackerError` if unset."""
        name = self._tracker.token_env
        token = os.environ.get(name)
        if not token:
            raise TrackerError(
                f"the variable {name}, which should hold the Jira token, is not set"
            )
        return token

    def _run(self, action: str, arguments: list[str]) -> object:
        """Run `twg` with `arguments` and JSON output; return its decoded answer.

        `action` names the call in error sentences. Raises `TrackerError` when the
        token variable is not set, `twg` cannot be started, exits nonzero, answers
        `"ok": false`, or answers something that is not JSON.
        """
        token = self._token()
        environment = {
            **os.environ,
            "TWG_SITE": self._tracker.site,
            "TWG_USER": self._tracker.user,
            "TWG_TOKEN": token,
        }
        try:
            done = subprocess.run(
                [
                    *self._command,
                    *arguments,
                    "--output",
                    "json",
                    "--output-summary",
                    "none",
                ],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as error:
            reason = (error.strerror or type(error).__name__).replace(token, REDACTED)
            raise TrackerError(f"twg could not be started: {reason}") from None
        answer = _parse_json(done.stdout)
        failed = isinstance(answer, dict) and answer.get("ok") is False
        if done.returncode != 0 or failed:
            reason = _failure_reason(answer, done).replace(token, REDACTED)
            raise TrackerError(f"twg failed while {action}: {reason}")
        if answer is _UNREADABLE:
            raise TrackerError(f"the answer of twg to {action} is not JSON")
        return answer


def _failure_reason(answer: object, done: subprocess.CompletedProcess[str]) -> str:
    """Return why `twg` failed: its error message, else its last stderr line, else its exit code."""
    if isinstance(answer, dict):
        error = answer.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
    lines = done.stderr.strip().splitlines()
    if lines:
        return lines[-1].strip()
    return f"exit code {done.returncode}"
