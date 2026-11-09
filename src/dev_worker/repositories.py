"""The code repository as the rest of the worker sees it: find a pull request, read its CI, merge it."""

from dataclasses import dataclass
from typing import Literal, Protocol

CiState = Literal["passed", "failed", "running"]
"""The state of the CI of a pull request: every check passed, one failed, or some still run."""


class RepositoryError(Exception):
    """Raised when the repository cannot be reached or its answer cannot be read.

    The message is a sentence saying why; it never holds a token.
    """


@dataclass(frozen=True)
class PullRequest:
    """An open pull request, by its number, its web address and the commit its head pointed to when found.

    `head` is the full SHA of that commit; the CI read and the merge are tied to it.
    """

    number: int
    url: str
    head: str


class CodeRepository(Protocol):
    """A code repository: finds the pull request of a branch, reads its CI and merges it."""

    def find_pull_request(self, branch: str) -> PullRequest | None:
        """Return the open pull request of `branch`, or `None` when it has none.

        Raises `RepositoryError` when the repository cannot be read.
        """
        ...

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Return the state of the CI of the commit `pull_request.head`.

        Raises `RepositoryError` when it cannot be read, no check is reported, or the head
        of the pull request has moved since it was found.
        """
        ...

    def merge(self, pull_request: PullRequest) -> None:
        """Merge `pull_request` as a single squashed commit, only while its head is still `pull_request.head`.

        Raises `RepositoryError` on failure, including when the head has moved.
        """
