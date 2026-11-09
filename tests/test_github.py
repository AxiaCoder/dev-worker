import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dev_worker.github import GitHubRepository
from dev_worker.instance import Repository
from dev_worker.repositories import PullRequest, RepositoryError

FAKE_GH = Path(__file__).with_name("fake_gh.py")
TOKEN = "s3cret-token-value"
REPOSITORY = Repository(type="github", name="octo/widgets", token_env="GITHUB_TOKEN")
HEAD = "0123456789abcdef0123456789abcdef01234567"
MOVED = "fedcba9876543210fedcba9876543210fedcba98"
PULL = PullRequest(number=42, url="https://github.com/octo/widgets/pull/42", head=HEAD)
LISTED = {
    "number": 42,
    "url": "https://github.com/octo/widgets/pull/42",
    "headRefOid": HEAD,
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
        "--json", "number,url,headRefOid", "--repo", "octo/widgets",
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


def test_a_pull_request_without_checks_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, rollup())

    with pytest.raises(
        RepositoryError, match="^no CI check is reported on pull request #42$"
    ):
        repository().read_ci(PULL)


def test_read_ci_refuses_a_pull_request_whose_head_has_moved(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, rollup(run("COMPLETED", "SUCCESS"), head=MOVED))

    with pytest.raises(RepositoryError) as raised:
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


def test_merge_squashes_the_pull_request(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "✓ Squashed and merged pull request #42\n")

    repository().merge(PULL)

    assert called(record)["argv"] == [
        "pr", "merge", "42", "--squash", "--match-head-commit", HEAD,
        "--repo", "octo/widgets",
    ]  # fmt: skip
    assert called(record)["env"]["GH_TOKEN"] == TOKEN


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

    with pytest.raises(RepositoryError) as raised:
        repository().merge(PULL)

    assert str(raised.value) == (
        f"the head of pull request #42 has moved since commit {HEAD}; it was not merged"
    )


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
    answer(monkeypatch, "")

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
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("dev_worker.github.subprocess.run", finish)

    repository().check_access()
    repository().merge(PULL)

    assert bounds == [60.0, 60.0]


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
    assert record.exists()
    assert elapsed < 10
