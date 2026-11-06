"""The GitHub code repository, reached through GitHub's `gh` command line."""

import json
import os
import subprocess
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from dev_worker.header import _describe
from dev_worker.instance import Repository
from dev_worker.repositories import CiState, PullRequest, RepositoryError

REDACTED = "[token]"
COMPLETED = "COMPLETED"
PASSING_CONCLUSIONS = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})
STATUS_STATES: dict[str, CiState] = {
    "SUCCESS": "passed",
    "PENDING": "running",
    "EXPECTED": "running",
    "FAILURE": "failed",
    "ERROR": "failed",
}


class _ListedPullRequest(BaseModel):
    """A pull request as `gh pr list --json number,url` writes it."""

    number: int
    url: str


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
    """The answer of `gh pr view --json statusCheckRollup`."""

    statusCheckRollup: list[
        Annotated[_CheckRun | _StatusContext, Field(discriminator="typename")]
    ]


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
    `--repo` with the repository name, `GH_PROMPT_DISABLED`, and, as `GH_TOKEN`, the value
    of the variable `repository.token_env` names, read at call time.
    """

    def __init__(
        self, repository: Repository, command: Sequence[str] = ("gh",)
    ) -> None:
        """Keep `repository` and `command` for later calls; nothing is run or read yet."""
        self._repository = repository
        self._command = tuple(command)

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
                "number,url",
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
        return PullRequest(number=listed[0].number, url=listed[0].url)

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Return `"failed"` when a check failed, else `"running"` when one still runs, else `"passed"`.

        Raises `RepositoryError` when the token variable is not set, `gh` fails, its answer
        cannot be read, or the pull request reports no check at all.
        """
        number = pull_request.number
        action = f"reading the CI of pull request #{number}"
        answer = self._run_json(
            action, ["pr", "view", str(number), "--json", "statusCheckRollup"]
        )
        try:
            checks = _ViewAnswer.model_validate(answer).statusCheckRollup
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        if not checks:
            raise RepositoryError(f"no CI check is reported on pull request #{number}")
        states = {_check_state(check) for check in checks}
        if "failed" in states:
            return "failed"
        if "running" in states:
            return "running"
        return "passed"

    def merge(self, pull_request: PullRequest) -> None:
        """Merge `pull_request` as one squashed commit.

        Raises `RepositoryError` when the token variable is not set or `gh` fails.
        """
        number = pull_request.number
        self._run(
            f"merging pull request #{number}",
            ["pr", "merge", str(number), "--squash"],
        )

    def _token(self) -> str:
        """Return the token held by the variable `token_env` names; raise `RepositoryError` if unset."""
        name = self._repository.token_env
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

    def _run(self, action: str, arguments: list[str]) -> str:
        """Run `gh` with `arguments` on the repository and return what it wrote to stdout.

        `action` names the call in error sentences. Raises `RepositoryError` when the token
        variable is not set, `gh` cannot be started, or it exits nonzero.
        """
        token = self._token()
        environment = {**os.environ, "GH_TOKEN": token, "GH_PROMPT_DISABLED": "1"}
        try:
            done = subprocess.run(
                [*self._command, *arguments, "--repo", self._repository.name],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except OSError as error:
            reason = (error.strerror or type(error).__name__).replace(token, REDACTED)
            raise RepositoryError(f"gh could not be started: {reason}") from None
        if done.returncode != 0:
            reason = _failure_reason(done).replace(token, REDACTED)
            raise RepositoryError(f"gh failed while {action}: {reason}")
        return done.stdout


def _failure_reason(done: subprocess.CompletedProcess[str]) -> str:
    """Return why `gh` failed: its last stderr line, else its exit code."""
    lines = done.stderr.strip().splitlines()
    if lines:
        return lines[-1].strip()
    return f"exit code {done.returncode}"
