"""The code repository as the rest of the worker sees it: find a pull request, read its CI, merge it."""

from dataclasses import dataclass
from typing import Literal, Protocol

CiState = Literal["passed", "failed", "running", "unreported"]
"""The state of the CI of a pull request: every check passed, one failed, some still run, or no check is reported yet while the repository has a CI to report one."""


class RepositoryError(Exception):
    """Raised when the repository cannot be reached or its answer cannot be read.

    The message is a sentence saying why; it never holds a token.
    """


class HeadMovedError(RepositoryError):
    """Raised when the head of a pull request is no longer the commit it pointed to when found.

    The pull request is to be found again and its CI read anew.
    """


@dataclass(frozen=True)
class PullRequest:
    """An open pull request, by its number, its web address, its title, the commit its head pointed to when found, and its base branch.

    `head` is the full SHA of that commit; the CI read, the review and the merge are tied
    to it. `title` is the subject of the squashed commit the merge makes. `base` is the
    name of the branch it merges into.
    """

    number: int
    url: str
    head: str
    title: str
    base: str


@dataclass(frozen=True)
class BaseCiFailure:
    """A failed CI on the default branch: the branch name, the full SHA of the commit and the names of its failed checks, in the order they were reported."""

    branch: str
    commit: str
    checks: tuple[str, ...]


@dataclass(frozen=True)
class ChangedFile:
    """A file a pull request changes: its path, the path it was renamed from or `None`, and its added and removed lines."""

    path: str
    previous_path: str | None
    additions: int
    deletions: int


@dataclass(frozen=True)
class OpenThread:
    """An unresolved review thread: the file it is on, its line or `None` when it has none, and the text of its comments from trusted authors, oldest first."""

    path: str
    line: int | None
    comments: tuple[str, ...]


@dataclass(frozen=True)
class PullRequestFeedback:
    """What a pull request still has against it: the names of its red checks and its open review threads."""

    red_checks: tuple[str, ...]
    threads: tuple[OpenThread, ...]


class CodeRepository(Protocol):
    """A code repository: finds the pull request of a branch, reads its CI and merges it."""

    def check_access(self) -> None:
        """Raise `RepositoryError` when the repository host cannot be reached with the instance credentials; change nothing."""
        ...

    def find_pull_request(self, branch: str) -> PullRequest | None:
        """Return the open pull request of `branch`, or `None` when it has none.

        Raises `RepositoryError` when the repository cannot be read.
        """
        ...

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Return the state of the CI of the commit `pull_request.head`.

        Raises `HeadMovedError` when the head of the pull request has moved since it was
        found, and `RepositoryError` when the CI cannot be read, or when no check is
        reported and the repository has no CI that could report one. No check reported
        while the repository has such a CI is `"unreported"`.
        """
        ...

    def base_ci_failure(self) -> BaseCiFailure | None:
        """Return the failure of the CI of the most recent commit of the default branch that reports one, or `None` when it did not fail.

        A failure the forge caused rather than the code may be rerun once; it then counts
        as not failed. `None` too when no recent commit reports a check. Raises
        `RepositoryError` when the CI cannot be read.
        """
        ...

    def changed_files(self, pull_request: PullRequest) -> tuple[ChangedFile, ...]:
        """Return the files `pull_request` changes, as many as the host lists.

        Raises `RepositoryError` when they cannot be read.
        """
        ...

    def feedback(self, pull_request: PullRequest) -> PullRequestFeedback:
        """Return the red checks of the latest commit of `pull_request` and its unresolved review threads.

        A thread holds only the comments of authors trusted on the repository; a thread
        left with none is not returned. Raises `RepositoryError` when they cannot be read.
        """
        ...

    def merge(self, pull_request: PullRequest) -> None:
        """Merge `pull_request` as a single squashed commit titled `pull_request.title`, only while its head is still `pull_request.head`.

        Returns only once the pull request is merged, not merely queued for a merge.
        Raises `HeadMovedError` when the head has moved: nothing was merged.
        Raises `RepositoryError` on any other failure, and when the pull request is still
        not merged once the host has answered; after it the merge may still have
        happened: the pull request is to be read again, not assumed open.
        """
