"""The ticket tracker as the rest of the worker sees it: read a ticket, list the tickets in a status, claim one, count its requeues, comment on it, change its status."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

REQUEUE_MARKER = "dev-worker returns this ticket to the queue: "


class TrackerError(Exception):
    """Raised when the tracker cannot be reached or its answer cannot be read.

    The message is a sentence saying why; it never holds a token.
    """


@dataclass(frozen=True)
class Blocker:
    """A ticket that blocks another one, with its real status name."""

    key: str
    status: str


@dataclass(frozen=True)
class Ticket:
    """A ticket as read from the tracker.

    `status` is the tracker's real status name; `blockers` are the tickets this one is
    blocked by, in the tracker's order. `description` is plain text, empty when the
    ticket has none.
    """

    key: str
    status: str
    labels: tuple[str, ...]
    blockers: tuple[Blocker, ...]
    title: str
    description: str


@dataclass(frozen=True)
class Claim:
    """A claim posted on a ticket by a run: the tracker id of the claim, the task_id it names and when it was created.

    Claim ids grow in the order the tracker recorded the claims: the lower, the older.
    `created` is the timezone-aware time the tracker server recorded the claim at.
    """

    claim_id: int
    task_id: str
    created: datetime


class TicketTracker(Protocol):
    """A ticket tracker: reads a ticket by key, lists the tickets in a status, claims one for a run, comments on it and changes its status."""

    def check_access(self) -> None:
        """Raise `TrackerError` when the tracker cannot be reached with the instance credentials; change nothing."""
        ...

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`; raise `TrackerError` when it cannot be read."""
        ...

    def tickets_in(self, status: str) -> tuple[Ticket, ...]:
        """Return the tickets of the instance project in the real status `status`, in the tracker's rank order.

        Raises `TrackerError` when they cannot be listed.
        """
        ...

    def claim(self, key: str, task_id: str) -> int:
        """Post on the ticket `key` a claim naming `task_id`; return the id of the claim.

        Raises `TrackerError` on failure. After any `TrackerError` the claim may still have
        been posted.
        """
        ...

    def claims(self, key: str, statuses: frozenset[str]) -> tuple[Claim, ...]:
        """Return the claims posted on the ticket `key` since it last entered any of the real statuses `statuses`, oldest first.

        Each claim carries the server time it was created at. A ticket that never entered any of `statuses` by a change of status gives all its claims.
        Raises `TrackerError` when they cannot be read.
        """
        ...

    def requeues(self, key: str, statuses: frozenset[str]) -> int:
        """Return how many requeue comments, whose text starts with `REQUEUE_MARKER`, were posted on the ticket `key` since it last entered any of the real statuses `statuses`.

        A ticket that never entered any of `statuses` by a change of status counts all of them.
        Raises `TrackerError` when they cannot be read.
        """
        ...

    def comment(self, key: str, body: str) -> None:
        """Post on the ticket `key` a plain-text comment holding `body`.

        Raises `TrackerError` on failure. After any `TrackerError` the comment may still
        have been posted.
        """
        ...

    def set_status(self, key: str, status: str) -> None:
        """Move the ticket `key` to the real status name `status`.

        Raises `TrackerError` on failure. After any `TrackerError` the transition may still
        have happened: the status is to be read again, not assumed unchanged.
        """
