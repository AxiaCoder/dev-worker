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
    BaseCiFailure,
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
    """A pull request as `gh pr list --json number,url,headRefOid,title,baseRefName` writes it."""

    number: int
    url: str
    headRefOid: str
    title: str
    baseRefName: str


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


class _Workflow(BaseModel):
    """A GitHub Actions workflow, as `gh api repos/{name}/actions/workflows` writes it."""

    path: str
    state: str


class _WorkflowsAnswer(BaseModel):
    """The answer of `gh api repos/{name}/actions/workflows`."""

    workflows: list[_Workflow]


class _StateAnswer(BaseModel):
    """The answer of `gh pr view --json state`."""

    state: str


class _WorkflowRunReference(BaseModel):
    """The GitHub Actions workflow run of a check suite, by its id."""

    databaseId: int


class _CheckSuite(BaseModel):
    """The check suite of a check run, with its workflow run when GitHub Actions made it."""

    workflowRun: _WorkflowRunReference | None = None


class _NamedCheckRun(_CheckRun):
    """A check run with its name and check suite, as the default branch query writes it."""

    name: str
    checkSuite: _CheckSuite | None = None


class _NamedStatusContext(_StatusContext):
    """A commit status with its name, as the default branch query writes it."""

    context: str


class _Contexts(BaseModel):
    """The checks of a commit, as `statusCheckRollup.contexts` writes them."""

    nodes: list[
        Annotated[_NamedCheckRun | _NamedStatusContext, Field(discriminator="typename")]
    ]


class _Rollup(BaseModel):
    """The `statusCheckRollup` of a commit."""

    contexts: _Contexts


class _HistoryCommit(BaseModel):
    """A commit of the default branch with its checks; `statusCheckRollup` is `None` when it has none."""

    oid: str
    statusCheckRollup: _Rollup | None = None


class _History(BaseModel):
    """The most recent commits of the default branch."""

    nodes: list[_HistoryCommit]


class _BranchTarget(BaseModel):
    """The commit the default branch points to, by its history."""

    history: _History


class _BranchReference(BaseModel):
    """The default branch, by its name and the commit it points to."""

    name: str
    target: _BranchTarget


class _BaseRepository(BaseModel):
    """The repository of the default branch query; `defaultBranchRef` is `None` when it has no commit."""

    defaultBranchRef: _BranchReference | None = None


class _BaseData(BaseModel):
    """The `data` of the default branch query."""

    repository: _BaseRepository


class _BaseAnswer(BaseModel):
    """The answer of `gh api graphql` to the default branch query."""

    data: _BaseData


class _Step(BaseModel):
    """A step of a job, as `gh run view --json jobs` writes it."""

    name: str
    conclusion: str


class _Job(BaseModel):
    """A job of a workflow run, as `gh run view --json jobs` writes it."""

    conclusion: str
    steps: list[_Step]


class _RunAnswer(BaseModel):
    """The answer of `gh run view --json attempt,conclusion,jobs`."""

    attempt: int
    conclusion: str
    jobs: list[_Job]


BASE_HISTORY_DEPTH = 10
BASE_CONTEXTS_PAGE_SIZE = 100
BASE_CI_QUERY = f"""
query($owner: String!, $name: String!) {{
  repository(owner: $owner, name: $name) {{
    defaultBranchRef {{
      name
      target {{
        ... on Commit {{
          history(first: {BASE_HISTORY_DEPTH}) {{
            nodes {{
              oid
              statusCheckRollup {{
                contexts(first: {BASE_CONTEXTS_PAGE_SIZE}) {{
                  nodes {{
                    __typename
                    ... on CheckRun {{
                      name
                      status
                      conclusion
                      checkSuite {{ workflowRun {{ databaseId }} }}
                    }}
                    ... on StatusContext {{ context state }}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""
FORGE_RUN_CONCLUSIONS = frozenset({"cancelled", "startup_failure"})
FINE_JOB_CONCLUSIONS = frozenset({"success", "neutral", "skipped", "cancelled"})
FAILURE = "failure"
SETUP_STEP = "Set up job"
FIRST_ATTEMPT = 1

MERGED = "MERGED"
ACTIVE = "active"
WORKFLOW_FILES_DIRECTORY = ".github/workflows/"
WORKFLOWS_PAGE_SIZE = 100

_UNREADABLE = object()


def _parse_json(text: str) -> object:
    """Return the JSON value of `text`, or `_UNREADABLE` when it is not JSON."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return _UNREADABLE


def _forge_caused(run: _RunAnswer) -> bool:
    """Return whether the failure of the workflow run `run` is the forge's rather than the code's.

    It is when the run was cancelled or failed to start, or when each of its jobs that
    did not pass was cancelled or failed with no failed step but the setup one.
    """
    if run.conclusion in FORGE_RUN_CONCLUSIONS:
        return True
    return all(
        job.conclusion in FINE_JOB_CONCLUSIONS
        or (
            job.conclusion == FAILURE
            and all(
                step.name == SETUP_STEP
                for step in job.steps
                if step.conclusion == FAILURE
            )
        )
        for job in run.jobs
    )


def _workflow_runs(
    checks: Sequence[_NamedCheckRun | _NamedStatusContext],
) -> tuple[int, ...] | None:
    """Return the ids of the GitHub Actions workflow runs of `checks`, each once in the order met, or `None` when one of them belongs to none."""
    runs: dict[int, None] = {}
    for check in checks:
        if (
            not isinstance(check, _NamedCheckRun)
            or check.checkSuite is None
            or check.checkSuite.workflowRun is None
        ):
            return None
        runs[check.checkSuite.workflowRun.databaseId] = None
    return tuple(runs)


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
                "number,url,headRefOid,title,baseRefName",
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
            base=found.baseRefName,
        )

    def read_ci(self, pull_request: PullRequest) -> CiState:
        """Return `"failed"` when a check failed, else `"running"` when one still runs, else `"passed"`.

        The checks are those of the head commit, read in the same answer as its SHA. When
        none is reported, the GitHub Actions workflows of the repository are read: this
        returns `"unreported"` when at least one is active and defined by a workflow file
        under `.github/workflows/`; GitHub's dynamic workflows, such as Dependabot's, do
        not count. Raises `HeadMovedError` when the
        head is no longer `pull_request.head`, and `RepositoryError` when the token
        variable is not set, `gh` fails, its answer cannot be read, or no check is reported
        and no workflow file is active.
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
            if self._has_active_workflow():
                return "unreported"
            raise RepositoryError(
                f"the repository {self._repository.name} has no active GitHub Actions "
                f"workflow file and reports no CI check on pull request #{number}"
            )
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

    def base_ci_failure(self) -> BaseCiFailure | None:
        """Return the failure of the CI of the most recent of the last 10 commits of the default branch that reports a check, or `None` when it did not fail.

        Its first 100 checks are read; it failed when one failed, as `read_ci` reads a
        check, even while others still run. When every failed check belongs to a GitHub
        Actions workflow run on its first attempt whose failure the forge caused (see
        `_forge_caused`), those runs are rerun — their failed jobs, or the whole run when
        it was cancelled or failed to start — and this returns `None`; when a rerun
        fails, the failure is returned. `None` too when the repository has no commit or
        none of those commits reports a check.

        Raises `RepositoryError` when the token variable is not set, or `gh` fails or
        its answer cannot be read while reading the commits or their workflow runs.
        """
        action = "reading the CI of the default branch"
        owner, _, name = self._repository.name.partition("/")
        answer = self._run_json(
            action,
            [
                "api",
                "graphql",
                "-f",
                f"query={BASE_CI_QUERY}",
                "-f",
                f"owner={owner}",
                "-f",
                f"name={name}",
            ],
            on_repository=False,
        )
        try:
            branch = _BaseAnswer.model_validate(answer).data.repository.defaultBranchRef
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        if branch is None:
            return None
        for commit in branch.target.history.nodes:
            rollup = commit.statusCheckRollup
            if rollup is not None and rollup.contexts.nodes:
                return self._settle_failure(branch.name, commit.oid, rollup.contexts)
        return None

    def _settle_failure(
        self, branch: str, commit: str, contexts: _Contexts
    ) -> BaseCiFailure | None:
        """Return the failure of the checks `contexts` of `commit` on `branch`, or `None` when none failed or the forge's failure was rerun, as `base_ci_failure` says.

        Raises `RepositoryError` as `_read_run` does.
        """
        failed = [check for check in contexts.nodes if _check_state(check) == "failed"]
        if not failed:
            return None
        failure = BaseCiFailure(
            branch=branch,
            commit=commit,
            checks=tuple(
                dict.fromkeys(
                    check.name if isinstance(check, _NamedCheckRun) else check.context
                    for check in failed
                )
            ),
        )
        runs = _workflow_runs(failed)
        if runs is None:
            return failure
        answers = {run: self._read_run(run) for run in runs}
        if not all(
            answer.attempt == FIRST_ATTEMPT and _forge_caused(answer)
            for answer in answers.values()
        ):
            return failure
        try:
            for run, answer in answers.items():
                whole = answer.conclusion in FORGE_RUN_CONCLUSIONS
                self._run(
                    f"rerunning workflow run {run}",
                    ["run", "rerun", str(run), *([] if whole else ["--failed"])],
                )
        except RepositoryError:
            return failure
        return None

    def _read_run(self, run: int) -> _RunAnswer:
        """Return the attempt, conclusion and jobs of the GitHub Actions workflow run `run`.

        Raises `RepositoryError` as `_run_json` does, and when the answer cannot be read.
        """
        action = f"reading workflow run {run}"
        answer = self._run_json(
            action, ["run", "view", str(run), "--json", "attempt,conclusion,jobs"]
        )
        try:
            return _RunAnswer.model_validate(answer)
        except ValidationError as error:
            raise self._unreadable(action, error) from None

    def _has_active_workflow(self) -> bool:
        """Return whether one of the first 100 GitHub Actions workflows of the repository is active and defined by a workflow file.

        A workflow file is one under `.github/workflows/`; GitHub's dynamic workflows,
        always active, do not count.

        Raises `RepositoryError` as `_run_json` does, and when the answer cannot be read.
        """
        action = "reading the GitHub Actions workflows of the repository"
        path = (
            f"repos/{self._repository.name}/actions/workflows"
            f"?per_page={WORKFLOWS_PAGE_SIZE}"
        )
        answer = self._run_json(action, ["api", path], on_repository=False)
        try:
            workflows = _WorkflowsAnswer.model_validate(answer).workflows
        except ValidationError as error:
            raise self._unreadable(action, error) from None
        return any(
            workflow.state == ACTIVE
            and workflow.path.startswith(WORKFLOW_FILES_DIRECTORY)
            for workflow in workflows
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

    def _run_json(
        self, action: str, arguments: list[str], *, on_repository: bool = True
    ) -> object:
        """Run `gh` with `arguments` and return its decoded JSON answer.

        `on_repository` is passed to `_run`. Raises `RepositoryError` as `_run` does, and
        when the answer is not JSON.
        """
        answer = _parse_json(self._run(action, arguments, on_repository=on_repository))
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
