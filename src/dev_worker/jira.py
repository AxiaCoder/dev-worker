"""The Jira ticket tracker, reached through Atlassian's `twg` command line.

Each `twg` call is bounded: one that has not answered within `TWG_TIMEOUT_SECONDS`
(60 seconds) is stopped and raises `TrackerError`.

A claim is a plain-text comment `CLAIM_MARKER` followed by the task_id, such as
`dev-worker claims this ticket for task task-1`; its id is the comment id.
"""

import json
import os
import re
import subprocess
import textwrap
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from itertools import count, repeat
from typing import Annotated, Any

from pydantic import AwareDatetime, BaseModel, Field, ValidationError

from dev_worker.header import _describe
from dev_worker.instance import Tracker
from dev_worker.tickets import Blocker, Claim, Ticket, TrackerError

TICKET_KEY = re.compile(r"[A-Z][A-Z0-9_]*-[0-9]+")
BLOCKS = "Blocks"
REDACTED = "[token]"
TWG_TIMEOUT_SECONDS = 60.0
CLAIM_MARKER = "dev-worker claims this ticket for task "
PAGE_SIZE = "100"
TEXT_BLOCKS = frozenset({"paragraph", "heading", "codeBlock"})
NESTED_INDENT = "  "
LISTS = frozenset({"bulletList", "orderedList", "taskList", "decisionList"})
DONE = "DONE"


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
    summary: str
    description: object = None


class _GetAnswer(BaseModel):
    """The answer of `twg jira workitem get` for one key."""

    data: Annotated[list[_Item], Field(min_length=1, max_length=1)]


class _CommentId(BaseModel):
    """The id of a comment, as Jira gives it back when the comment is created."""

    id: int


class _CreatedAnswer(BaseModel):
    """The answer of `twg jira workitem comment create`."""

    data: _CommentId


class _PageInfo(BaseModel):
    """Where the next page of a paged `twg` query starts, if there is one."""

    hasNextPage: bool
    nextCursor: str | None = None


class _Page(BaseModel):
    """A page of a paged `twg` query."""

    pageInfo: _PageInfo


class _Change(BaseModel):
    """A field change of a changelog entry; `toString` is the new value as text."""

    field: str
    toString: str | None = None


class _History(BaseModel):
    """A changelog entry: the changes made at one instant."""

    created: AwareDatetime
    items: list[_Change]


class _ChangelogPage(_Page):
    """A page of `twg jira workitem changelog query`."""

    data: list[_History]


class _Comment(BaseModel):
    """A comment; its body is text or an Atlassian document."""

    id: int
    created: AwareDatetime
    body: object = None


class _CommentPage(_Page):
    """A page of `twg jira workitem comment query`."""

    data: list[_Comment]


_UNREADABLE = object()


def _parse_json(text: str) -> object:
    """Return the JSON value of `text`, or `_UNREADABLE` when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _UNREADABLE


def _validate[M: BaseModel](model: type[M], answer: object, action: str) -> M:
    """Return `answer` read as `model`; raise `TrackerError` naming `action` when it cannot be."""
    try:
        return model.model_validate(answer)
    except ValidationError as error:
        description = _describe(error, root="answer", echo=False)
        raise TrackerError(
            f"the answer of twg to {action} cannot be read: {description}"
        ) from None


def _plain_text(body: object) -> str:
    """Return the text of a comment body: the string itself, or the text nodes of an Atlassian document."""
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return "".join(_plain_text(node) for node in body)
    if isinstance(body, dict):
        text = body.get("text")
        own = text if isinstance(text, str) else ""
        return own + _plain_text(body.get("content"))
    return ""


def _document_text(document: object) -> str:
    """Return a ticket description as plain text, empty when it is absent.

    The description is a string, kept as it is, or an Atlassian document, read as
    markdown-like text:

    - a line ends each paragraph, heading, code block, hard break and block or embed
      card holding a URL;
    - a link card reads as its URL, a text linked elsewhere as `text (url)`;
    - a mention, emoji or status reads as its text, a date as its timestamp in
      milliseconds;
    - a bullet list item starts with `- `, an ordered one with its number, counted from
      the list's start order or from 1; each nesting level is indented by two spaces;
    - a task starts with `- [x] ` when its state is `DONE`, with `- [ ] ` otherwise, and a
      decision with `- `; a task list placed among the tasks of another is indented by
      two spaces under them;
    - a node whose attributes are missing or malformed reads as nothing.

    Leading and trailing blank space is removed from a document's text.
    """
    if isinstance(document, str):
        return document
    return _blocks_text(document).strip()


def _blocks_text(node: object) -> str:
    """Return the text of an Atlassian document node, with a line ending each text block."""
    if isinstance(node, list):
        return "".join(_blocks_text(child) for child in node)
    if not isinstance(node, dict):
        return ""
    kind = node.get("type")
    render = _RENDERERS.get(kind, _node_text)
    own = render(node)
    return own + "\n" if kind in TEXT_BLOCKS else own


def _node_text(node: dict[Any, Any]) -> str:
    """Return the text of `node` followed by the text of its content; a linked text names its URL."""
    text = node.get("text")
    own = text if isinstance(text, str) else ""
    href = _link(node)
    if href and href != own:
        own += f" ({href})"
    return own + _blocks_text(node.get("content"))


def _link(node: dict[Any, Any]) -> str:
    """Return the URL of the link mark of `node`, empty when it has none."""
    marks = node.get("marks")
    for mark in marks if isinstance(marks, list) else ():
        if isinstance(mark, dict) and mark.get("type") == "link":
            return _attribute(mark, name="href")
    return ""


def _attribute(node: dict[Any, Any], name: str) -> str:
    """Return the string attribute `name` of `node`, empty when it is missing or not a string."""
    attrs = node.get("attrs")
    value = attrs.get(name) if isinstance(attrs, dict) else None
    return value if isinstance(value, str) else ""


def _block_card_text(node: dict[Any, Any]) -> str:
    """Return the URL of a block or embed card as a line, empty when it has none."""
    url = _attribute(node, name="url")
    return url + "\n" if url else ""


def _bullet_list_text(node: dict[Any, Any]) -> str:
    """Return the items of a bullet list, each starting with `- `."""
    return _list_text(node, repeat("- "))


def _ordered_list_text(node: dict[Any, Any]) -> str:
    """Return the items of an ordered list, numbered from its integer start order, or from 1."""
    attrs = node.get("attrs")
    order = attrs.get("order") if isinstance(attrs, dict) else None
    start = order if isinstance(order, int) else 1
    return _list_text(node, (f"{number}. " for number in count(start)))


def _list_text(node: dict[Any, Any], markers: Iterator[str]) -> str:
    """Return the items of a list, each starting with the next of `markers`.

    The lines after an item's first are indented by `NESTED_INDENT`.
    """
    content = node.get("content")
    items: Iterable[object] = content if isinstance(content, list) else ()
    return "".join(_item_text(item, marker) for item, marker in zip(items, markers))


def _item_text(item: object, marker: str) -> str:
    """Return the text of a list item, starting with `marker`, its later lines indented."""
    first, newline, rest = _blocks_text(item).partition("\n")
    return marker + first + newline + textwrap.indent(rest, NESTED_INDENT)


def _task_list_text(node: dict[Any, Any]) -> str:
    """Return the tasks of a task list, each starting with its checkbox; a nested list is indented."""
    content = node.get("content")
    entries: Iterable[object] = content if isinstance(content, list) else ()
    return "".join(_task_list_entry(entry) for entry in entries)


def _task_list_entry(entry: object) -> str:
    """Return a task with its checkbox, or any other entry of a task list indented."""
    if isinstance(entry, dict) and entry.get("type") == "taskItem":
        checked = _attribute(entry, name="state") == DONE
        return _item_text(entry, "- [x] " if checked else "- [ ] ")
    return textwrap.indent(_blocks_text(entry), NESTED_INDENT)


def _inline_item_text(node: dict[Any, Any]) -> str:
    """Return a task or decision as lines: its inline text on the first, nested lists below."""
    content = node.get("content")
    children: Iterable[object] = content if isinstance(content, list) else ()
    text = ""
    for child in children:
        nests = isinstance(child, dict) and child.get("type") in LISTS
        if nests and not text.endswith("\n"):
            text += "\n"
        text += _blocks_text(child)
    return text if text.endswith("\n") else text + "\n"


_RENDERERS: dict[object, Callable[[dict[Any, Any]], str]] = {
    "hardBreak": lambda _: "\n",
    "bulletList": _bullet_list_text,
    "orderedList": _ordered_list_text,
    "taskList": _task_list_text,
    "taskItem": _inline_item_text,
    "decisionList": _bullet_list_text,
    "decisionItem": _inline_item_text,
    "inlineCard": partial(_attribute, name="url"),
    "blockCard": _block_card_text,
    "embedCard": _block_card_text,
    "mention": partial(_attribute, name="text"),
    "emoji": partial(_attribute, name="text"),
    "status": partial(_attribute, name="text"),
    "date": partial(_attribute, name="timestamp"),
}


def _check_key(key: str) -> None:
    """Raise `TrackerError` when `key` is not a Jira ticket key such as `ABC-12`."""
    if not TICKET_KEY.fullmatch(key):
        raise TrackerError(f"{key!r} is not a Jira ticket key such as ABC-12")


class JiraTracker:
    """A `TicketTracker` for the Jira site of `tracker`, through the `twg` command line.

    `command` is the program and leading arguments that run `twg`. Each call passes
    `TWG_SITE`, `TWG_USER` and, as `TWG_TOKEN`, the value of the variable `tracker.token_env`
    names, read at call time. `timeout` is the bound of each call, in seconds.
    """

    def __init__(
        self,
        tracker: Tracker,
        command: Sequence[str] = ("twg",),
        timeout: float = TWG_TIMEOUT_SECONDS,
    ) -> None:
        """Keep `tracker`, `command` and `timeout` for later calls; nothing is run or read yet."""
        self._tracker = tracker
        self._command = tuple(command)
        self._timeout = timeout

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`, with its status, labels, blockers, title and description.

        A blocker is a ticket linked to `key` by a "Blocks" link that blocks it. The
        description is read as plain text (see `_document_text`). Raises `TrackerError`
        when `key` is not a ticket key, the token variable is not set, `twg` fails, or its
        answer cannot be read.
        """
        _check_key(key)
        answer = self._run(
            f"reading {key}",
            [
                "jira", "workitem", "get", key,
                "--fields", "status,labels,issuelinks,summary,description",
            ],
        )  # fmt: skip
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
            title=item.summary,
            description=_document_text(item.description),
        )

    def set_status(self, key: str, status: str) -> None:
        """Move the ticket `key` to the real status name `status`.

        Raises `TrackerError` when `key` is not a ticket key, the token variable is not set,
        `twg` fails, or its answer is not JSON. After any `TrackerError` the transition may
        still have happened: the status is to be read again, not assumed unchanged.
        """
        _check_key(key)
        self._run(
            f"moving {key} to {status!r}",
            ["jira", "workitem", "transition", "--id", key, "--transition-id", status],
        )

    def claim(self, key: str, task_id: str) -> int:
        """Post on the ticket `key` the claim comment naming `task_id`; return the comment id.

        Raises `TrackerError` when `key` is not a ticket key, the token variable is not set,
        `twg` fails, or its answer cannot be read. After any `TrackerError` the comment may
        still have been posted.
        """
        _check_key(key)
        action = f"claiming {key}"
        answer = self._run(
            action,
            [
                "jira", "workitem", "comment", "create", "--issue-id", key,
                "--body", f"{CLAIM_MARKER}{task_id}", "--body-format", "plain",
            ],
        )  # fmt: skip
        return _validate(_CreatedAnswer, answer, action).data.id

    def claims(self, key: str, status: str) -> tuple[Claim, ...]:
        """Return the claims on the ticket `key` since it last entered the status `status`, oldest first.

        The last entry is the latest changelog entry that sets the status to `status`; a
        claim counts when its comment was created at that instant or after, and every claim
        counts when there is no such entry. Each claim carries the creation time of its
        comment. Claims are ordered by comment id. Every page of the changelog and of the
        comments is read. Raises `TrackerError` when `key` is not a
        ticket key, the token variable is not set, `twg` fails, or an answer cannot be read.
        """
        _check_key(key)
        changelog = self._pages(
            f"reading the changelog of {key}",
            ["jira", "workitem", "changelog", "query", "--issue-id", key],
            _ChangelogPage,
        )
        entries = [
            history.created
            for page in changelog
            for history in page.data
            if any(
                change.field == "status" and change.toString == status
                for change in history.items
            )
        ]
        since = max(entries, default=None)
        comments = self._pages(
            f"reading the comments of {key}",
            [
                "jira", "workitem", "comment", "query", "--issue-id", key,
                "--order-by", "created",
            ],
            _CommentPage,
        )  # fmt: skip
        found: list[Claim] = []
        for page in comments:
            for comment in page.data:
                text = _plain_text(comment.body).strip()
                if text.startswith(CLAIM_MARKER) and (
                    since is None or comment.created >= since
                ):
                    found.append(
                        Claim(
                            comment.id,
                            text.removeprefix(CLAIM_MARKER),
                            comment.created,
                        )
                    )
        return tuple(sorted(found, key=lambda claim: claim.claim_id))

    def _pages[P: _Page](
        self, action: str, arguments: list[str], model: type[P]
    ) -> list[P]:
        """Run the paged `twg` query `arguments` page after page; return every page, in order.

        Raises what `_run` raises, and `TrackerError` when a page cannot be read or names
        a next page without a cursor not given before.
        """
        pages: list[P] = []
        cursors: set[str] = set()
        cursor: list[str] = []
        while True:
            answer = self._run(action, [*arguments, "--first", PAGE_SIZE, *cursor])
            page = _validate(model, answer, action)
            pages.append(page)
            if not page.pageInfo.hasNextPage:
                return pages
            following = page.pageInfo.nextCursor
            if following is None or following in cursors:
                raise TrackerError(
                    f"the answer of twg to {action} names a next page without a new cursor"
                )
            cursors.add(following)
            cursor = ["--after", following]

    def _token(self) -> str:
        """Return the token held by the variable `token_env` names; raise `TrackerError` if unset."""
        name = self._tracker.token_env
        token = os.environ.get(name)
        if not token:
            raise TrackerError(
                f"the variable {name}, which should hold the Jira token, is not set"
            )
        return token

    def check_access(self) -> None:
        """Raise `TrackerError` unless `twg doctor` reaches Jira with the instance credentials.

        Access holds only when the doctor reports its connectivity check passed, whatever
        its exit code. Raises when the token variable is not set, `twg` cannot be started,
        does not answer within the timeout, or reports no passing connectivity check; the
        sentence carries the doctor's own message when it gives one. Changes nothing on Jira.
        """
        answer, done, token = self._call("checking access to Jira", ["doctor"])
        connectivity = _connectivity(answer)
        if connectivity.get("ok") is True:
            return
        message = connectivity.get("message")
        if isinstance(message, str) and message.strip():
            reason = message.strip()
        else:
            reason = _failure_reason(answer, done)
        raise TrackerError(
            f"twg cannot reach Jira with the instance credentials: {reason}".replace(
                token, REDACTED
            )
        )

    def _run(self, action: str, arguments: list[str]) -> object:
        """Run `twg` with `arguments` and JSON output; return its decoded answer.

        `action` names the call in error sentences. Raises `TrackerError` when the
        token variable is not set, `twg` cannot be started, does not answer within the
        timeout, exits nonzero, answers `"ok": false`, or answers something that is not JSON.
        """
        answer, done, token = self._call(action, arguments)
        failed = isinstance(answer, dict) and answer.get("ok") is False
        if done.returncode != 0 or failed:
            reason = _failure_reason(answer, done).replace(token, REDACTED)
            raise TrackerError(f"twg failed while {action}: {reason}")
        if answer is _UNREADABLE:
            raise TrackerError(f"the answer of twg to {action} is not JSON")
        return answer

    def _call(
        self, action: str, arguments: list[str]
    ) -> tuple[object, subprocess.CompletedProcess[str], str]:
        """Run `twg` with `arguments` and JSON output, whatever its exit code.

        Returns its decoded answer (`_UNREADABLE` when not JSON), the finished process and
        the token passed. Raises `TrackerError` when the token variable is not set, `twg`
        cannot be started, or does not answer within the timeout.
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
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise TrackerError(
                f"twg did not answer within {self._timeout:g} seconds while {action}"
            ) from None
        except OSError as error:
            reason = (error.strerror or type(error).__name__).replace(token, REDACTED)
            raise TrackerError(f"twg could not be started: {reason}") from None
        return _parse_json(done.stdout), done, token


def _connectivity(answer: object) -> dict[str, object]:
    """Return the `data.connectivity` object of a `twg doctor` answer, empty when absent."""
    data = answer.get("data") if isinstance(answer, dict) else None
    connectivity = data.get("connectivity") if isinstance(data, dict) else None
    return connectivity if isinstance(connectivity, dict) else {}


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
