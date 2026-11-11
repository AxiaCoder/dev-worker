"""The ticket tracker as the rest of the worker sees it: read a ticket, claim it, change its status."""

from dataclasses import dataclass
from typing import Protocol


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
    blocked by, in the tracker's order.
    """

    key: str
    status: str
    labels: tuple[str, ...]
    blockers: tuple[Blocker, ...]


@dataclass(frozen=True)
class Claim:
    """A claim posted on a ticket by a run: the tracker id of the claim and the task_id it names.

    Claim ids grow in the order the tracker recorded the claims: the lower, the older.
    """

    claim_id: int
    task_id: str


class TicketTracker(Protocol):
    """A ticket tracker: reads a ticket by key, claims it for a run and changes its status."""

    def check_access(self) -> None:
        """Raise `TrackerError` when the tracker cannot be reached with the instance credentials; change nothing."""
        ...

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`; raise `TrackerError` when it cannot be read."""
        ...

    def claim(self, key: str, task_id: str) -> int:
        """Post on the ticket `key` a claim naming `task_id`; return the id of the claim.

        Raises `TrackerError` on failure. After any `TrackerError` the claim may still have
        been posted.
        """
        ...

    def claims(self, key: str, status: str) -> tuple[Claim, ...]:
        """Return the claims posted on the ticket `key` since it last entered the real status `status`, oldest first.

        A ticket that never entered `status` by a change of status gives all its claims.
        Raises `TrackerError` when they cannot be read.
        """
        ...

    def set_status(self, key: str, status: str) -> None:
        """Move the ticket `key` to the real status name `status`.

        Raises `TrackerError` on failure. After any `TrackerError` the transition may still
        have happened: the status is to be read again, not assumed unchanged.
        """
