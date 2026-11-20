import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dev_worker.github import GitHubRepository
from dev_worker.instance import Repository
from dev_worker.repositories import (
    BaseCiFailure,
    HeadMovedError,
    PullRequest,
    RepositoryError,
)

FAKE_GH = Path(__file__).with_name("fake_gh.py")
TOKEN = "s3cret-token-value"
AGENT_TOKEN = "agent-token-value"
REPOSITORY = Repository(
    type="github",
    name="octo/widgets",
    agent_token_env="AGENT_TOKEN",
    worker_token_env="GITHUB_TOKEN",
)
HEAD = "0123456789abcdef0123456789abcdef01234567"
MOVED = "fedcba9876543210fedcba9876543210fedcba98"
TITLE = "feat: the widgets page lists the widgets"
PULL = PullRequest(
    number=42,
    url="https://github.com/octo/widgets/pull/42",
    head=HEAD,
    title=TITLE,
    base="main",
)
LISTED = {
    "number": 42,
    "url": "https://github.com/octo/widgets/pull/42",
    "headRefOid": HEAD,
    "title": TITLE,
    "baseRefName": "main",
}


def run(status: str, conclusion: str | None = None) -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": "build",
        "status": status,
        "conclusion": conclusion,
    }


def context(state: str) -> dict[str, Any]:
    return {"__typename": "StatusContext", "context": "ci/legacy", "state": state}


def rollup(*checks: dict[str, Any], head: str = HEAD) -> dict[str, Any]:
    return {"headRefOid": head, "statusCheckRollup": list(checks)}


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_GH_RECORD", str(path))
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    return path


def answer(
    monkeypatch: pytest.MonkeyPatch, stdout: object, stderr: str = "", code: int = 0
) -> None:
    text = stdout if isinstance(stdout, str) else json.dumps(stdout)
    monkeypatch.setenv("FAKE_GH_STDOUT", text)
    monkeypatch.setenv("FAKE_GH_STDERR", stderr)
    monkeypatch.setenv("FAKE_GH_EXIT", str(code))


def answer_in_turn(
    monkeypatch: pytest.MonkeyPatch, record: Path, *answers: object
) -> Path:
    """Have the fake answer each call with the next of `answers`, and return the log of calls.

    An answer is a `GhFailure`, or else the stdout, as text or as a value given as JSON,
    of a call that exits 0.
    """
    calls = record.with_name("calls.jsonl")
    monkeypatch.setenv("FAKE_GH_CALLS", str(calls))
    monkeypatch.setenv(
        "FAKE_GH_ANSWERS", json.dumps([_fake_answer(item) for item in answers])
    )
    return calls


@dataclass(frozen=True)
class GhFailure:
    """A call of the fake that writes `stderr` and exits with `exit`."""

    stderr: str
    exit: int = 1


def _fake_answer(item: object) -> dict[str, object]:
    """Return the answer of the fake for one item given to `answer_in_turn`."""
    if isinstance(item, GhFailure):
        return {"stdout": "", "stderr": item.stderr, "exit": item.exit}
    stdout = item if isinstance(item, str) else json.dumps(item)
    return {"stdout": stdout, "stderr": "", "exit": 0}


def every_call(calls: Path) -> list[list[str]]:
    """Return the arguments of each call the fake logged, in order."""
    return [
        json.loads(line)["argv"]
        for line in calls.read_text(encoding="utf-8").splitlines()
    ]


def repository() -> GitHubRepository:
    return GitHubRepository(REPOSITORY, command=[sys.executable, str(FAKE_GH)])


def called(record: Path) -> dict[str, Any]:
    return json.loads(record.read_text(encoding="utf-8"))


def test_find_pull_request_returns_the_open_pull_request_of_the_branch(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [LISTED])

    found = repository().find_pull_request("feat/widgets")

    assert found == PULL
    assert called(record)["argv"] == [
        "pr", "list", "--head=feat/widgets", "--state", "open",
        "--json", "number,url,headRefOid,title,baseRefName", "--repo", "octo/widgets",
    ]  # fmt: skip
    assert called(record)["env"] == {"GH_TOKEN": TOKEN, "GH_PROMPT_DISABLED": "1"}


def test_a_branch_without_an_open_pull_request_gives_none(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [])

    assert repository().find_pull_request("feat/widgets") is None


def test_a_branch_with_two_open_pull_requests_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [LISTED, {**LISTED, "number": 43}])

    with pytest.raises(
        RepositoryError,
        match="^the branch feat/widgets has more than one open pull request: #42, #43$",
    ):
        repository().find_pull_request("feat/widgets")


@pytest.mark.parametrize("branch", ["", "  "])
def test_a_blank_branch_is_refused_and_gh_not_run(
    record: Path, monkeypatch: pytest.MonkeyPatch, branch: str
) -> None:
    answer(monkeypatch, [LISTED])

    with pytest.raises(RepositoryError, match="blank branch name"):
        repository().find_pull_request(branch)
    assert not record.exists()


def test_a_branch_that_looks_like_a_flag_stays_the_head_value(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [])

    repository().find_pull_request("--help")

    assert "--head=--help" in called(record)["argv"]


@pytest.mark.parametrize(
    ("stdout", "where"),
    [
        ({"number": 42}, "answer"),
        ([{"number": "x", "url": "u"}], "0.number"),
        ([{"number": 42}], "0.url"),
        ([{"number": 42, "url": "u"}], "0.headRefOid"),
        ([{"number": 42, "url": "u", "headRefOid": 7}], "0.headRefOid"),
        ([{"number": 42, "url": "u", "headRefOid": HEAD}], "0.title"),
        ([{"number": 42, "url": "u", "headRefOid": HEAD, "title": 7}], "0.title"),
        (
            [{"number": 42, "url": "u", "headRefOid": HEAD, "title": TITLE}],
            "0.baseRefName",
        ),
    ],
)
def test_a_listing_of_the_wrong_shape_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch, stdout: object, where: str
) -> None:
    answer(monkeypatch, stdout)

    with pytest.raises(RepositoryError) as raised:
        repository().find_pull_request("feat/widgets")

    message = str(raised.value)
    assert message.startswith(
        "the answer of gh to finding the pull request of feat/widgets cannot be read: "
    )
    assert where in message


def test_read_ci_asks_for_the_check_rollup(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, rollup(run("COMPLETED", "SUCCESS")))

    assert repository().read_ci(PULL) == "passed"
    assert called(record)["argv"] == [
        "pr", "view", "42", "--json", "headRefOid,statusCheckRollup",
        "--repo", "octo/widgets",
    ]  # fmt: skip
    assert called(record)["env"]["GH_TOKEN"] == TOKEN


@pytest.mark.parametrize(
    ("checks", "state"),
    [
        ([run("COMPLETED", "SUCCESS")], "passed"),
        ([run("COMPLETED", "NEUTRAL"), run("COMPLETED", "SKIPPED")], "passed"),
        ([context("SUCCESS")], "passed"),
        ([run("COMPLETED", "FAILURE")], "failed"),
        ([run("COMPLETED", "CANCELLED")], "failed"),
        ([run("COMPLETED", "TIMED_OUT")], "failed"),
        ([run("COMPLETED", None)], "failed"),
        ([context("FAILURE")], "failed"),
        ([context("ERROR")], "failed"),
        ([run("IN_PROGRESS")], "running"),
        ([run("QUEUED")], "running"),
        ([context("PENDING")], "running"),
        ([context("EXPECTED")], "running"),
        ([run("COMPLETED", "SUCCESS"), run("IN_PROGRESS")], "running"),
        ([run("IN_PROGRESS"), context("FAILURE")], "failed"),
        ([context("PENDING"), run("COMPLETED", "FAILURE")], "failed"),
        ([run("COMPLETED", "FAILURE"), run("IN_PROGRESS")], "failed"),
        ([run("COMPLETED", "FAILURE"), run("COMPLETED", "SUCCESS")], "failed"),
        ([run("IN_PROGRESS"), context("SUCCESS")], "running"),
    ],
)
def test_read_ci_aggregates_the_checks(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    checks: list[dict[str, Any]],
    state: str,
) -> None:
    answer(monkeypatch, rollup(*checks))

    assert repository().read_ci(PULL) == state


def workflows(*states: str) -> dict[str, Any]:
    return workflows_at(
        *((f".github/workflows/ci-{i}.yml", s) for i, s in enumerate(states))
    )


def workflows_at(*entries: tuple[str, str]) -> dict[str, Any]:
    return {
        "total_count": len(entries),
        "workflows": [
            {"name": f"ci-{i}", "path": path, "state": state}
            for i, (path, state) in enumerate(entries)
        ],
    }


DEPENDABOT = "dynamic/dependabot/update-graph"
COPILOT = "dynamic/copilot-pull-request-reviewer/copilot-pull-request-reviewer"


@pytest.mark.parametrize(
    "entries",
    [
        ((DEPENDABOT, "active"),),
        ((COPILOT, "active"),),
        ((DEPENDABOT, "active"), (COPILOT, "active")),
        ((".github/workflows/ci.yml", "disabled_manually"), (DEPENDABOT, "active")),
    ],
)
def test_a_pull_request_without_checks_in_a_repository_with_only_dynamic_active_workflows_raises(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    entries: tuple[tuple[str, str], ...],
) -> None:
    answer_in_turn(monkeypatch, record, rollup(), workflows_at(*entries))

    with pytest.raises(
        RepositoryError,
        match=(
            "^the repository octo/widgets has no active GitHub Actions workflow file and "
            "reports no CI check on pull request #42$"
        ),
    ):
        repository().read_ci(PULL)


def test_a_pull_request_without_checks_in_a_repository_with_a_dynamic_and_a_file_workflow_is_unreported(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_in_turn(
        monkeypatch,
        record,
        rollup(),
        workflows_at((".github/workflows/check.yml", "active"), (DEPENDABOT, "active")),
    )

    assert repository().read_ci(PULL) == "unreported"


@pytest.mark.parametrize(
    "states",
    [("active",), ("disabled_manually", "active")],
)
def test_a_pull_request_without_checks_in_a_repository_with_an_active_workflow_is_unreported(
    record: Path, monkeypatch: pytest.MonkeyPatch, states: tuple[str, ...]
) -> None:
    calls = answer_in_turn(monkeypatch, record, rollup(), workflows(*states))

    assert repository().read_ci(PULL) == "unreported"
    assert every_call(calls)[1] == [
        "api", "repos/octo/widgets/actions/workflows?per_page=100",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "states",
    [(), ("disabled_manually",), ("disabled_inactivity", "deleted")],
)
def test_a_pull_request_without_checks_in_a_repository_without_an_active_workflow_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch, states: tuple[str, ...]
) -> None:
    answer_in_turn(monkeypatch, record, rollup(), workflows(*states))

    with pytest.raises(
        RepositoryError,
        match=(
            "^the repository octo/widgets has no active GitHub Actions workflow file and "
            "reports no CI check on pull request #42$"
        ),
    ):
        repository().read_ci(PULL)


def test_a_pull_request_with_checks_does_not_read_the_workflows(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(monkeypatch, record, rollup(run("COMPLETED", "SUCCESS")))

    repository().read_ci(PULL)

    assert len(every_call(calls)) == 1


@pytest.mark.parametrize(
    ("workflows_answer", "ending"),
    [
        ("not json", "is not JSON"),
        ({"total_count": 0}, "cannot be read: "),
        ({"workflows": [{"name": "ci"}]}, "cannot be read: "),
    ],
)
def test_an_unreadable_workflows_answer_raises(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflows_answer: object,
    ending: str,
) -> None:
    answer_in_turn(monkeypatch, record, rollup(), workflows_answer)

    with pytest.raises(RepositoryError) as raised:
        repository().read_ci(PULL)

    assert str(raised.value).startswith(
        "the answer of gh to reading the GitHub Actions workflows of the repository "
    )
    assert ending in str(raised.value)


def test_a_failed_workflows_read_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_in_turn(monkeypatch, record, rollup(), GhFailure("HTTP 404: Not Found"))

    with pytest.raises(
        RepositoryError,
        match=(
            "^gh failed while reading the GitHub Actions workflows of the repository: "
            "HTTP 404: Not Found$"
        ),
    ):
        repository().read_ci(PULL)


def test_read_ci_refuses_a_pull_request_whose_head_has_moved(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, rollup(run("COMPLETED", "SUCCESS"), head=MOVED))

    with pytest.raises(HeadMovedError) as raised:
        repository().read_ci(PULL)

    assert str(raised.value) == (
        f"the head of pull request #42 has moved from {HEAD} to {MOVED} "
        "since it was found"
    )


@pytest.mark.parametrize(
    ("stdout", "where"),
    [
        ({"headRefOid": HEAD}, "statusCheckRollup"),
        ({"headRefOid": HEAD, "statusCheckRollup": None}, "statusCheckRollup"),
        ({"statusCheckRollup": [run("COMPLETED", "SUCCESS")]}, "headRefOid"),
        ({**rollup(run("COMPLETED", "SUCCESS")), "headRefOid": None}, "headRefOid"),
        (rollup({"__typename": "Other"}), "statusCheckRollup.0"),
        (rollup({"state": "SUCCESS"}), "statusCheckRollup.0"),
        (rollup(context("WEIRD")), "statusCheckRollup.0"),
        (rollup({"__typename": "CheckRun"}), "statusCheckRollup.0"),
    ],
)
def test_a_rollup_of_the_wrong_shape_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch, stdout: object, where: str
) -> None:
    answer(monkeypatch, stdout)

    with pytest.raises(RepositoryError) as raised:
        repository().read_ci(PULL)

    message = str(raised.value)
    assert message.startswith(
        "the answer of gh to reading the CI of pull request #42 cannot be read: "
    )
    assert where in message


def test_merge_squashes_the_pull_request_under_its_title_bypassing_the_base_branch_rules(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        "✓ Squashed and merged pull request #42\n",
        {"state": "MERGED"},
    )

    repository().merge(PULL)

    assert every_call(calls)[0] == [
        "pr", "merge", "42", "--squash", "--admin", "--match-head-commit", HEAD,
        "--subject", TITLE, "--repo", "octo/widgets",
    ]  # fmt: skip
    assert called(record)["env"]["GH_TOKEN"] == TOKEN


def test_merge_reads_the_pull_request_again_and_returns_once_it_is_merged(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        "✓ Squashed and merged pull request #42\n",
        {"state": "MERGED"},
    )

    repository().merge(PULL)

    assert every_call(calls)[1:] == [
        ["pr", "view", "42", "--json", "state", "--repo", "octo/widgets"],
    ]


def test_a_pull_request_left_open_by_a_merge_queue_is_not_told_as_merged(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_in_turn(
        monkeypatch,
        record,
        "✓ Pull request octo/widgets#42 will be added to the merge queue\n",
        {"state": "OPEN"},
    )

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert type(raised.value) is RepositoryError
    assert "#42" in str(raised.value)
    assert "OPEN" in str(raised.value)


@pytest.mark.parametrize(
    ("state", "said"),
    [
        ({"merged": True}, "cannot be read"),
        ({"state": None}, "cannot be read"),
        ("<html>", "is not JSON"),
    ],
)
def test_a_state_read_again_that_cannot_be_read_is_not_told_as_merged(
    record: Path, monkeypatch: pytest.MonkeyPatch, state: object, said: str
) -> None:
    answer_in_turn(
        monkeypatch, record, "✓ Squashed and merged pull request #42\n", state
    )

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert type(raised.value) is RepositoryError
    assert "#42" in str(raised.value)
    assert said in str(raised.value)


def test_a_state_that_gh_fails_to_read_again_is_not_told_as_merged(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_in_turn(
        monkeypatch, record, "✓ Squashed and merged\n", GhFailure("HTTP 502\n")
    )

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert type(raised.value) is RepositoryError
    assert "HTTP 502" in str(raised.value)


def test_a_merge_refused_because_the_head_moved_says_so(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        "",
        stderr=(
            "GraphQL: Head branch was modified. Review and try the merge again. "
            "(mergePullRequest)\n"
        ),
        code=1,
    )

    with pytest.raises(HeadMovedError) as raised:
        repository().merge(PULL)

    assert str(raised.value) == (
        f"the head of pull request #42 has moved since commit {HEAD}; it was not merged"
    )


@pytest.mark.parametrize(
    ("stdout", "stderr", "code"),
    [
        (rollup(), "", 0),
        ({"headRefOid": HEAD}, "", 0),
        ("<html>", "", 0),
        (rollup(run("COMPLETED", "SUCCESS")), "HTTP 502\n", 1),
        ("", "", 4),
    ],
)
def test_read_ci_failures_other_than_a_moved_head_are_not_told_as_one(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: object,
    stderr: str,
    code: int,
) -> None:
    answer(monkeypatch, stdout, stderr=stderr, code=code)

    with pytest.raises(RepositoryError) as raised:
        repository().read_ci(PULL)

    assert type(raised.value) is RepositoryError


@pytest.mark.parametrize(
    ("stderr", "code"),
    [
        ("X Pull request octo/widgets#42 is not mergeable\n", 1),
        ("HTTP 502\n", 1),
        ("", 4),
    ],
)
def test_merge_failures_other_than_a_moved_head_are_not_told_as_one(
    record: Path, monkeypatch: pytest.MonkeyPatch, stderr: str, code: int
) -> None:
    answer(monkeypatch, "", stderr=stderr, code=code)

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert type(raised.value) is RepositoryError


def test_merge_failure_is_given_without_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        "",
        stderr=f"  first  \n\n   \nPull request is not mergeable {TOKEN}\n",
        code=1,
    )

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert str(raised.value) == (
        "gh failed while merging pull request #42: "
        "first Pull request is not mergeable [token]"
    )


def test_a_refused_merge_keeps_the_reason_gh_gave_before_its_hints(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        "",
        stderr=(
            "X Pull request octo/widgets#42 is not mergeable: "
            "the base branch policy prohibits the merge.\n"
            "To have the pull request merged after all the requirements have been "
            "met, add the `--auto` flag.\n"
            "To use administrator privileges to immediately merge the pull request, "
            "add the `--admin` flag.\n"
        ),
        code=1,
    )

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert "is not mergeable: the base branch policy prohibits the merge" in str(
        raised.value
    )


@pytest.mark.parametrize("value", [None, ""])
def test_unset_token_variable_is_named_and_gh_not_run(
    record: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("GITHUB_TOKEN")
    else:
        monkeypatch.setenv("GITHUB_TOKEN", value)
    answer(monkeypatch, [LISTED])

    with pytest.raises(RepositoryError, match="GITHUB_TOKEN.*not set"):
        repository().find_pull_request("feat/widgets")
    with pytest.raises(RepositoryError, match="GITHUB_TOKEN.*not set"):
        repository().merge(PULL)
    assert not record.exists()


def test_gh_that_cannot_be_started_raises(record: Path) -> None:
    missing = GitHubRepository(REPOSITORY, command=[str(record.parent / "no-such-gh")])

    with pytest.raises(RepositoryError, match="gh could not be started"):
        missing.read_ci(PULL)


def test_without_stderr_the_exit_code_is_given(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "", code=4)

    with pytest.raises(
        RepositoryError, match="reading the CI of pull request #42: exit code 4$"
    ):
        repository().read_ci(PULL)


def test_a_nonzero_exit_is_a_failure_even_with_a_readable_answer(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [LISTED], stderr="HTTP 502\n", code=1)

    with pytest.raises(
        RepositoryError,
        match="^gh failed while finding the pull request of feat/widgets: HTTP 502$",
    ):
        repository().find_pull_request("feat/widgets")


def test_an_answer_that_is_not_json_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "<html>")

    with pytest.raises(
        RepositoryError,
        match="answer of gh to reading the CI of pull request #42 is not JSON",
    ):
        repository().read_ci(PULL)


def test_an_answer_of_the_wrong_shape_does_not_echo_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, [{"number": TOKEN, "url": TOKEN}])

    with pytest.raises(RepositoryError) as raised:
        repository().find_pull_request("feat/widgets")

    message = str(raised.value)
    assert "0.number" in message
    assert TOKEN not in message


def test_gh_variables_already_set_are_replaced(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GH_TOKEN", "stale-token")
    monkeypatch.setenv("GH_PROMPT_DISABLED", "")
    answer(monkeypatch, [])

    repository().find_pull_request("feat/widgets")

    assert called(record)["env"] == {"GH_TOKEN": TOKEN, "GH_PROMPT_DISABLED": "1"}


def test_the_token_is_read_at_each_call_not_at_construction(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")
    built = repository()
    monkeypatch.setenv("GITHUB_TOKEN", "first-token")
    merged = {"state": "MERGED"}
    answer_in_turn(monkeypatch, record, "", merged, "", merged)

    built.merge(PULL)
    first = called(record)["env"]["GH_TOKEN"]
    monkeypatch.setenv("GITHUB_TOKEN", "second-token")
    built.merge(PULL)
    second = called(record)["env"]["GH_TOKEN"]

    assert (first, second) == ("first-token", "second-token")


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            OSError(13, f"Permission denied for {TOKEN}"),
            "Permission denied for [token]",
        ),
        (OSError(), "OSError"),
    ],
)
def test_gh_that_cannot_be_started_is_explained_without_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch, error: OSError, reason: str
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("dev_worker.github.subprocess.run", refuse)

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert str(raised.value) == f"gh could not be started: {reason}"


def test_check_access_asks_gh_for_the_active_github_account_only(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "github.com\n  ✓ Logged in to github.com\n")

    repository().check_access()

    assert called(record)["argv"] == [
        "auth", "status", "--active", "--hostname", "github.com",
    ]  # fmt: skip
    assert called(record)["env"] == {"GH_TOKEN": TOKEN, "GH_PROMPT_DISABLED": "1"}


def test_calls_use_the_worker_token_not_the_agent_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_TOKEN", AGENT_TOKEN)
    merged = {"state": "MERGED"}
    answer_in_turn(monkeypatch, record, "", merged)

    repository().merge(PULL)

    assert called(record)["env"]["GH_TOKEN"] == TOKEN


def test_check_access_uses_the_token_variable_it_is_given(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_TOKEN", AGENT_TOKEN)
    answer(monkeypatch, "github.com\n  ✓ Logged in to github.com\n")
    agent = GitHubRepository(
        REPOSITORY, command=[sys.executable, str(FAKE_GH)], token_env="AGENT_TOKEN"
    )

    agent.check_access()

    assert called(record)["env"]["GH_TOKEN"] == AGENT_TOKEN


def test_a_token_gh_cannot_sign_in_with_refuses_access_without_echoing_it(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        "",
        stderr=(
            "github.com\n"
            "  X Failed to log in to github.com using token (GH_TOKEN)\n"
            "  - Active account: true\n"
            f"  - The token in GH_TOKEN is invalid: {TOKEN}\n"
        ),
        code=1,
    )

    with pytest.raises(RepositoryError) as raised:
        repository().check_access()

    assert str(raised.value) == (
        "gh failed while checking access to github.com: github.com "
        "X Failed to log in to github.com using token (GH_TOKEN) "
        "- Active account: true - The token in GH_TOKEN is invalid: [token]"
    )


def test_check_access_without_the_token_variable_names_it_and_runs_nothing(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")

    with pytest.raises(RepositoryError, match="GITHUB_TOKEN.*not set"):
        repository().check_access()
    assert not record.exists()


def test_check_access_with_gh_missing_raises(record: Path) -> None:
    missing = GitHubRepository(REPOSITORY, command=[str(record.parent / "no-such-gh")])

    with pytest.raises(RepositoryError, match="^gh could not be started: "):
        missing.check_access()


def test_check_access_is_bounded_by_the_timeout(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds: list[object] = []

    def hang(*args: object, **kwargs: object) -> None:
        bounds.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(cmd="gh", timeout=0.5)

    monkeypatch.setattr("dev_worker.github.subprocess.run", hang)
    bounded = GitHubRepository(
        REPOSITORY, command=[sys.executable, str(FAKE_GH)], timeout=0.5
    )

    with pytest.raises(RepositoryError) as raised:
        bounded.check_access()

    assert str(raised.value) == (
        "gh did not answer within 0.5 seconds while checking access to github.com"
    )
    assert bounds == [0.5]


def test_check_access_is_bounded_by_default(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bounds: list[object] = []

    def finish(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        bounds.append(kwargs["timeout"])
        merged = json.dumps({"state": "MERGED"})
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=merged, stderr=""
        )

    monkeypatch.setattr("dev_worker.github.subprocess.run", finish)

    repository().check_access()
    repository().merge(PULL)

    assert bounds == [60.0, 60.0, 60.0]


@pytest.mark.parametrize(
    ("call", "stdout", "action"),
    [
        (
            lambda github: github.find_pull_request("feature/x"),
            [LISTED],
            "finding the pull request of feature/x",
        ),
        (
            lambda github: github.read_ci(PULL),
            rollup(run("COMPLETED", "SUCCESS")),
            "reading the CI of pull request #42",
        ),
        (lambda github: github.merge(PULL), "", "merging pull request #42"),
        (lambda github: github.check_access(), "", "checking access to github.com"),
    ],
    ids=["find_pull_request", "read_ci", "merge", "check_access"],
)
def test_gh_that_does_not_answer_is_stopped_at_the_timeout(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[GitHubRepository], object],
    stdout: object,
    action: str,
) -> None:
    answer(monkeypatch, stdout)
    monkeypatch.setenv("FAKE_GH_SLEEP", "30")
    github = GitHubRepository(
        REPOSITORY, command=[sys.executable, str(FAKE_GH)], timeout=0.5
    )

    started = time.monotonic()
    with pytest.raises(RepositoryError) as raised:
        call(github)
    elapsed = time.monotonic() - started

    assert str(raised.value) == f"gh did not answer within 0.5 seconds while {action}"
    assert TOKEN not in str(raised.value)
    assert elapsed < 10


BASE_COMMIT = "abc1234def5678abc1234def5678abc1234def56"
OLDER_COMMIT = "9999999def5678abc1234def5678abc1234def56"


def action_run(
    name: str, conclusion: str | None, run_id: int | None = 501, *, suite: bool = True
) -> dict[str, Any]:
    """Return a check run of the default branch query, of the workflow run `run_id`."""
    check: dict[str, Any] = {
        "__typename": "CheckRun",
        "name": name,
        "status": "COMPLETED" if conclusion else "IN_PROGRESS",
        "conclusion": conclusion,
    }
    if suite:
        check["checkSuite"] = {
            "workflowRun": None if run_id is None else {"databaseId": run_id}
        }
    return check


def base_commit(oid: str, *checks: dict[str, Any] | None) -> dict[str, Any]:
    """Return a commit of the default branch history; `None` alone gives it no rollup."""
    if checks == (None,):
        return {"oid": oid, "statusCheckRollup": None}
    return {"oid": oid, "statusCheckRollup": {"contexts": {"nodes": list(checks)}}}


def base_history(*commits: dict[str, Any]) -> dict[str, Any]:
    """Return the answer of the default branch query for a `main` branch of `commits`."""
    return {
        "data": {
            "repository": {
                "defaultBranchRef": {
                    "name": "main",
                    "target": {"history": {"nodes": list(commits)}},
                }
            }
        }
    }


def job(conclusion: str, *steps: tuple[str, str]) -> dict[str, Any]:
    """Return a job of `gh run view`, with its steps as (name, conclusion) pairs."""
    return {
        "conclusion": conclusion,
        "steps": [{"name": name, "conclusion": state} for name, state in steps],
    }


def workflow_run(
    conclusion: str, *jobs: dict[str, Any], attempt: int = 1
) -> dict[str, Any]:
    """Return the answer of `gh run view --json attempt,conclusion,jobs`."""
    return {"attempt": attempt, "conclusion": conclusion, "jobs": list(jobs)}


SETUP_FAILED = job("failure", ("Set up job", "failure"))
TESTS_FAILED = job("failure", ("Set up job", "success"), ("Run tests", "failure"))
RED = BaseCiFailure(branch="main", commit=BASE_COMMIT, checks=("tests",))


def test_base_ci_failure_queries_the_default_branch_history_of_the_repository(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch, record, base_history(base_commit(BASE_COMMIT, None))
    )

    assert repository().base_ci_failure() is None

    ((api, graphql, query_flag, query, *variables),) = every_call(calls)
    assert (api, graphql, query_flag) == ("api", "graphql", "-f")
    assert query.startswith("query=")
    assert "defaultBranchRef" in query and "history(first: 10)" in query
    assert variables == ["-f", "owner=octo", "-f", "name=widgets"]


def test_a_repository_without_a_default_branch_has_no_base_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"data": {"repository": {"defaultBranchRef": None}}})

    assert repository().base_ci_failure() is None


@pytest.mark.parametrize(
    "checks",
    [
        pytest.param([action_run("tests", "SUCCESS")], id="passed"),
        pytest.param([action_run("tests", None)], id="running"),
        pytest.param(
            [action_run("tests", "SUCCESS"), context("SUCCESS")], id="passed-status"
        ),
    ],
)
def test_a_base_ci_that_passed_or_still_runs_is_not_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch, checks: list[dict[str, Any]]
) -> None:
    calls = answer_in_turn(
        monkeypatch, record, base_history(base_commit(BASE_COMMIT, *checks))
    )

    assert repository().base_ci_failure() is None
    assert len(every_call(calls)) == 1


def test_the_base_ci_is_read_on_the_most_recent_commit_that_reports_a_check(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        base_history(
            base_commit(BASE_COMMIT, None),
            base_commit("5555555def5678abc1234def5678abc1234def56"),
            base_commit(OLDER_COMMIT, context("FAILURE")),
            base_commit("7777777def5678abc1234def5678abc1234def56", context("SUCCESS")),
        ),
    )

    assert repository().base_ci_failure() == BaseCiFailure(
        branch="main", commit=OLDER_COMMIT, checks=("ci/legacy",)
    )


def test_a_history_without_checks_has_no_base_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        base_history(base_commit(BASE_COMMIT), base_commit(OLDER_COMMIT, None)),
    )

    assert repository().base_ci_failure() is None


@pytest.mark.parametrize(
    "checks",
    [
        pytest.param([context("ERROR")], id="status"),
        pytest.param([action_run("tests", "FAILURE"), context("ERROR")], id="mixed"),
        pytest.param([action_run("tests", "FAILURE", suite=False)], id="no-suite"),
        pytest.param([action_run("tests", "FAILURE", None)], id="no-workflow-run"),
    ],
)
def test_a_base_failure_outside_github_actions_is_not_rerun(
    record: Path, monkeypatch: pytest.MonkeyPatch, checks: list[dict[str, Any]]
) -> None:
    calls = answer_in_turn(
        monkeypatch, record, base_history(base_commit(BASE_COMMIT, *checks))
    )

    failure = repository().base_ci_failure()

    assert failure is not None
    assert failure.commit == BASE_COMMIT
    assert len(every_call(calls)) == 1


def test_a_base_failure_names_each_failed_check_once_in_order(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(
                BASE_COMMIT,
                action_run("lint", "FAILURE"),
                action_run("build", "SUCCESS"),
                action_run("tests", "TIMED_OUT"),
                action_run("lint", "FAILURE"),
            )
        ),
        workflow_run("failure", TESTS_FAILED),
    )

    assert repository().base_ci_failure() == BaseCiFailure(
        branch="main", commit=BASE_COMMIT, checks=("lint", "tests")
    )
    assert every_call(calls)[1] == [
        "run", "view", "501", "--json", "attempt,conclusion,jobs",
        "--repo", "octo/widgets",
    ]  # fmt: skip
    assert len(every_call(calls)) == 2


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(workflow_run("failure", TESTS_FAILED), id="test-step"),
        pytest.param(
            workflow_run("failure", SETUP_FAILED, TESTS_FAILED), id="setup-and-test"
        ),
        pytest.param(workflow_run("timed_out", job("timed_out")), id="timed-out"),
        pytest.param(workflow_run("failure", SETUP_FAILED, attempt=2), id="rerun"),
        pytest.param(workflow_run("cancelled", attempt=2), id="cancelled-rerun"),
    ],
)
def test_a_base_failure_of_the_code_or_already_rerun_is_not_rerun(
    record: Path, monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(base_commit(BASE_COMMIT, action_run("tests", "FAILURE"))),
        run,
    )

    assert repository().base_ci_failure() == RED
    assert len(every_call(calls)) == 2


@pytest.mark.parametrize(
    ("run", "rerun"),
    [
        pytest.param(
            workflow_run("failure", SETUP_FAILED, job("success"), job("cancelled")),
            ["--failed"],
            id="setup-step",
        ),
        pytest.param(
            workflow_run("failure", job("failure")), ["--failed"], id="no-step"
        ),
        pytest.param(workflow_run("cancelled", TESTS_FAILED), [], id="cancelled"),
        pytest.param(workflow_run("startup_failure"), [], id="startup-failure"),
    ],
)
def test_a_base_failure_the_forge_caused_on_a_first_attempt_is_rerun_once(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    run: dict[str, Any],
    rerun: list[str],
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(
                BASE_COMMIT,
                action_run("tests", "FAILURE"),
                action_run("lint", "CANCELLED"),
            )
        ),
        run,
        "",
    )

    assert repository().base_ci_failure() is None
    assert every_call(calls)[2] == [
        "run", "rerun", "501", *rerun, "--repo", "octo/widgets",
    ]  # fmt: skip


def test_base_failures_of_two_runs_the_forge_caused_rerun_both(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(
                BASE_COMMIT,
                action_run("tests", "FAILURE", 501),
                action_run("lint", "FAILURE", 502),
            )
        ),
        workflow_run("failure", SETUP_FAILED),
        workflow_run("cancelled"),
        "",
        "",
    )

    assert repository().base_ci_failure() is None
    assert [call[:3] for call in every_call(calls)[1:]] == [
        ["run", "view", "501"],
        ["run", "view", "502"],
        ["run", "rerun", "501"],
        ["run", "rerun", "502"],
    ]


def test_a_base_failure_whose_rerun_fails_stays_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_in_turn(
        monkeypatch,
        record,
        base_history(base_commit(BASE_COMMIT, action_run("tests", "FAILURE"))),
        workflow_run("failure", SETUP_FAILED),
        GhFailure("HTTP 403: Resource not accessible by integration"),
    )

    assert repository().base_ci_failure() == RED


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(GhFailure("HTTP 404: Not Found"), id="gh-fails"),
        pytest.param({"attempt": "first"}, id="wrong-shape"),
    ],
)
def test_a_base_failure_whose_workflow_run_cannot_be_read_stays_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch, run: object
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(base_commit(BASE_COMMIT, action_run("tests", "FAILURE"))),
        run,
    )

    assert repository().base_ci_failure() == RED
    assert len(every_call(calls)) == 2


def test_a_check_cancelled_on_a_commit_older_than_the_head_is_not_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(BASE_COMMIT, None),
            base_commit(
                OLDER_COMMIT,
                action_run("tests", "CANCELLED"),
                action_run("lint", "SUCCESS"),
            ),
        ),
    )

    assert repository().base_ci_failure() is None
    assert len(every_call(calls)) == 1


def test_a_cancelled_run_on_a_commit_older_than_the_head_is_not_rerun(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(BASE_COMMIT),
            base_commit(
                OLDER_COMMIT,
                action_run("tests", "FAILURE"),
                action_run("lint", "CANCELLED"),
            ),
        ),
        workflow_run("cancelled", SETUP_FAILED),
    )

    assert repository().base_ci_failure() == BaseCiFailure(
        branch="main", commit=OLDER_COMMIT, checks=("tests",)
    )
    assert [call[:2] for call in every_call(calls)[1:]] == [["run", "view"]]


def test_a_forge_failure_on_a_commit_older_than_the_head_is_rerun(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(BASE_COMMIT),
            base_commit(OLDER_COMMIT, action_run("tests", "FAILURE")),
        ),
        workflow_run("failure", SETUP_FAILED),
        "",
    )

    assert repository().base_ci_failure() is None
    assert every_call(calls)[2][:3] == ["run", "rerun", "501"]


@pytest.mark.parametrize(
    "stdout",
    [
        {"data": {}},
        {"data": {"repository": {"defaultBranchRef": {"name": "main"}}}},
        base_history(base_commit(BASE_COMMIT, {"__typename": "CheckRun"})),
    ],
)
def test_a_default_branch_answer_of_the_wrong_shape_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch, stdout: object
) -> None:
    answer(monkeypatch, stdout)

    with pytest.raises(
        RepositoryError,
        match="^the answer of gh to reading the CI of the default branch cannot be",
    ):
        repository().base_ci_failure()


def test_base_failures_of_a_forge_run_and_a_code_run_rerun_neither(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = answer_in_turn(
        monkeypatch,
        record,
        base_history(
            base_commit(
                BASE_COMMIT,
                action_run("lint", "FAILURE", 501),
                action_run("tests", "FAILURE", 502),
            )
        ),
        workflow_run("failure", SETUP_FAILED),
        workflow_run("failure", TESTS_FAILED),
    )

    assert repository().base_ci_failure() == BaseCiFailure(
        branch="main", commit=BASE_COMMIT, checks=("lint", "tests")
    )
    assert [call[:2] for call in every_call(calls)[1:]] == [
        ["run", "view"],
        ["run", "view"],
    ]


def test_a_base_check_that_failed_while_another_still_runs_is_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(
        monkeypatch,
        base_history(
            base_commit(BASE_COMMIT, action_run("build", None), context("FAILURE"))
        ),
    )

    assert repository().base_ci_failure() == BaseCiFailure(
        branch="main", commit=BASE_COMMIT, checks=("ci/legacy",)
    )
