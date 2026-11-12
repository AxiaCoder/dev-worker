"""The GitHub code repository, reached through GitHub's `gh` command line.

Each `gh` call is bounded: one that has not answered within `GH_TIMEOUT_SECONDS`
(60 seconds) is stopped and raises `RepositoryError`.
"""

import json
import os
import subprocess
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from dev_worker.header import _describe
from dev_worker.instance import Repository
from dev_worker.repositories import (
    CiState,
    HeadMovedError,
    PullRequest,
    RepositoryError,
)

REDACTED = "[token]"
GH_TIMEOUT_SECONDS = 60.0
GITHUB_HOST = "github.com"
COMPLETED = "COMPLETED"
HEAD_MOVED = "Head branch was modified"
PASSING_CONCLUSIONS = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})
STATUS_STATES: dict[str, CiState] = {
    "SUCCESS": "passed",
    "PENDING": "running",
    "EXPECTED": "running",
    "FAILURE": "failed",
    "ERROR": "failed",
}


class _ListedPullRequest(BaseModel):
    """A pull request as `gh pr list --json number,url,headRefOid,title` writes it."""

    number: int
    url: str
    headRefOid: str
    title: str


_LIST_ANSWER = TypeAdapter(list[_ListedPullRequest])


class _CheckRun(BaseModel):
    """A check run of a commit, as `statusCheckRollup` writes it."""

    typename: Literal["CheckRun"] = Field(alias="__typename")
    status: str
    conclusion: str | None = None


class _StatusContext(BaseModel):
    """A commit status, as `statusCheckRollup` writes it."""

    typename: Literal["StatusContext"] = Field(alias="__typename")
    state: Literal["SUCCESS", "PENDING", "EXPECTED", "FAILURE", "ERROR"]


class _ViewAnswer(BaseModel):
    """The answer of `gh pr view --json headRefOid,statusCheckRollup`."""

    headRefOid: str
    statusCheckRollup: list[
        Annotated[_CheckRun | _StatusContext, Field(discriminator="typename")]
    ]


class _StateAnswer(BaseModel):
    """The answer of `gh pr view --json state`."""

    state: str


MERGED = "MERGED"

_UNREADABLE = object()


def _parse_json(text: str) -> object:
    """Return the JSON value of `text`, or `_UNREADABLE` when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _UNREADABLE


def _check_state(check: _CheckRun | _StatusContext) -> CiState:
    """Return whether one check passed, failed or is still running."""
    if isinstance(check, _StatusContext):
        return STATUS_STATES[check.state]
    if check.status != COMPLETED:
        return "running"
    return "passed" if check.conclusion in PASSING_CONCLUSIONS else "failed"


class GitHubRepository:
    """A `CodeRepository` for the GitHub repository `repository`, through the `gh` command line.

    `command` is the program and leading arguments that run `gh`. Each call passes
    `GH_PROMPT_DISABLED` and, as `GH_TOKEN`, the value of the variable `token_env`
    names, read at call time: `repository.worker_token_env` unless another is given. Each
    call on the repository also passes `--repo` with the repository name. `timeout` is the
    bound of each call, in seconds.
    """

    def __init__(
        self,
        repository: Repository,
        command: Sequence[str] = ("gh",),
        timeout: float = GH_TIMEOUT_SECONDS,
        token_env: str | None = None,
    ) -> None:
        """Keep `repository`, `command`, `timeout` and the token variable for later calls; nothing is run or read yet."""
        self._repository = repository
        self._command = tuple(command)
        self._timeout = timeout
        self._token_env = (
            repository.worker_token_env if token_env is None else token_env
        )

    def check_access(self) -> None:
        """Raise `RepositoryError` unless `gh` is signed in to github.com with the instance token.

        Raises when the token variable is not set, `gh` cannot be started, does not answer
        within the timeout, or reports the token cannot sign in. Changes nothing on GitHub.
        """
        self._run(
            f"checking access to {GITHUB_HOST}",
            ["auth", "status", "--active", "--hostname", GITHUB_HOST],
            on_repository=False,
        )

    def find_pull_request(self, branch: str) -> PullRequest | None:
        """Return the open pull request whose head is `branch`, or `None` when there is none.

        Raises `RepositoryError` when `branch` is blank, the token variable is not set, `gh`
        fails, its answer cannot be read, or more than one open pull request has that head.
        """
        if not branch.strip():
            raise RepositoryError("a blank branch name cannot have a pull request")
        action = f"finding the pull request of {branch}"
        answer = self._run_json(
            action,
            [
                "pr",
                "list",
                f"--head={branch}",
                "--state",
                "open",
                "--json",
                "number,url,headRefOid,title",
            ],
        )
        try:
            listed = _LIST_ANSWER.validate_python(answer)
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        if not listed:
            return None
        if len(listed) > 1:
            numbers = ", ".join(f"#{item.number}" for item in listed)
            raise RepositoryError(
                f"the branch {branch} has more than one open pull request: {numbers}"
            )
        found = listed[0]
        return PullRequest(
            number=found.number,
            url=found.url,
            head=found.headRefOid,
            title=found.title,
        )

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Return `"failed"` when a check failed, else `"running"` when one still runs, else `"passed"`.

        The checks are those of the head commit, read in the same answer as its SHA. Raises
        `HeadMovedError` when the head is no longer `pull_request.head`, and
        `RepositoryError` when the token variable is not set, `gh` fails, its answer cannot
        be read, or no check is reported.
        """
        number = pull_request.number
        action = f"reading the CI of pull request #{number}"
        answer = self._run_json(
            action,
            ["pr", "view", str(number), "--json", "headRefOid,statusCheckRollup"],
        )
        try:
            view = _ViewAnswer.model_validate(answer)
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        if view.headRefOid != pull_request.head:
            raise HeadMovedError(
                f"the head of pull request #{number} has moved from "
                f"{pull_request.head} to {view.headRefOid} since it was found"
            )
        checks = view.statusCheckRollup
        if not checks:
            raise RepositoryError(f"no CI check is reported on pull request #{number}")
        states = {_check_state(check) for check in checks}
        if "failed" in states:
            return "failed"
        if "running" in states:
            return "running"
        return "passed"

    def merge(self, pull_request: PullRequest) -> None:
        """Merge `pull_request` as one squashed commit whose subject is `pull_request.title`, which GitHub refuses once its head is not `pull_request.head`.

        The merge uses the worker's bypass of the base branch rules (`--admin`): a rule
        that blocks the merge for others does not block it for the worker.

        Once `gh` has answered, the state of the pull request is read again; this returns
        only when it is merged.

        Raises `RepositoryError` when the token variable is not set, `gh` fails, the state
        read again cannot be read or is not merged (as when a merge queue only enqueued the
        pull request); when the refusal is GitHub's for a moved head, it raises
        `HeadMovedError`, and nothing was merged. After any other `RepositoryError` the
        merge may still have happened: the pull request is to be read again, not assumed
        open.
        """
        number = pull_request.number
        try:
            self._run(
                f"merging pull request #{number}",
                [
                    "pr",
                    "merge",
                    str(number),
                    "--squash",
                    "--admin",
                    "--match-head-commit",
                    pull_request.head,
                    "--subject",
                    pull_request.title,
                ],
            )
        except RepositoryError as error:
            if HEAD_MOVED not in str(error):
                raise
            raise HeadMovedError(
                f"the head of pull request #{number} has moved since commit "
                f"{pull_request.head}; it was not merged"
            ) from None
        action = f"reading the state of pull request #{number} after its merge"
        answer = self._run_json(action, ["pr", "view", str(number), "--json", "state"])
        try:
            state = _StateAnswer.model_validate(answer).state
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        if state != MERGED:
            raise RepositoryError(
                f"pull request #{number} is not merged once gh answered: its state "
                f"is {state}; a merge queue may hold it."
            )

    def _token(self) -> str:
        """Return the token held by the variable `token_env` names; raise `RepositoryError` if unset."""
        name = self._token_env
        token = os.environ.get(name)
        if not token:
            raise RepositoryError(
                f"the variable {name}, which should hold the GitHub token, is not set"
            )
        return token

    def _unreadable(self, action: str, error: ValidationError) -> RepositoryError:
        """Return the error saying the answer of `gh` to `action` has the wrong shape."""
        description = _describe(error, root="answer", echo=False)
        return RepositoryError(
            f"the answer of gh to {action} cannot be read: {description}"
        )

    def _run_json(self, action: str, arguments: list[str]) -> object:
        """Run `gh` with `arguments` and return its decoded JSON answer.

        Raises `RepositoryError` as `_run` does, and when the answer is not JSON.
        """
        answer = _parse_json(self._run(action, arguments))
        if answer is _UNREADABLE:
            raise RepositoryError(f"the answer of gh to {action} is not JSON")
        return answer

    def _run(
        self,
        action: str,
        arguments: list[str],
        *,
        on_repository: bool = True,
    ) -> str:
        """Run `gh` with `arguments` and return what it wrote to stdout.

        `--repo` is added with the repository name when `on_repository` is true. `action`
        names the call in error sentences. Raises `RepositoryError` when the token variable
        is not set, `gh` cannot be started, does not answer within the timeout, or exits
        nonzero.
        """
        token = self._token()
        environment = {**os.environ, "GH_TOKEN": token, "GH_PROMPT_DISABLED": "1"}
        target = ["--repo", self._repository.name] if on_repository else []
        try:
            done = subprocess.run(
                [*self._command, *arguments, *target],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise RepositoryError(
                f"gh did not answer within {self._timeout:g} seconds while {action}"
            ) from None
        except OSError as error:
            reason = (error.strerror or type(error).__name__).replace(token, REDACTED)
            raise RepositoryError(f"gh could not be started: {reason}") from None
        if done.returncode != 0:
            reason = failure_reason(done).replace(token, REDACTED)
            raise RepositoryError(f"gh failed while {action}: {reason}")
        return done.stdout


def failure_reason(done: subprocess.CompletedProcess[str]) -> str:
    """Return why a command failed: its non-empty stderr lines, stripped and joined by one space, else its exit code."""
    lines = [line.strip() for line in done.stderr.splitlines() if line.strip()]
    if lines:
        return " ".join(lines)
    return f"exit code {done.returncode}"
