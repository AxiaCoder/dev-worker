"""The ticket tracker as the rest of the worker sees it: read a ticket, change its status."""

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


class TicketTracker(Protocol):
    """A ticket tracker: reads a ticket by key and changes its status."""

    def read(self, key: str) -> Ticket:
        """Return the ticket `key`; raise `TrackerError` when it cannot be read."""
        ...

    def set_status(self, key: str, status: str) -> None:
        """Move the ticket `key` to the real status name `status`; raise `TrackerError` on failure."""
