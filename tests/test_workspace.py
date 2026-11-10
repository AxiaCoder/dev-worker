import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dev_worker import workspace as workspace_module
from dev_worker.instance import Repository
from dev_worker.workspace import Workspace, WorkspaceError

TOKEN = "s3cret-token-value"
REPOSITORY = Repository(type="github", name="octo/widgets", token_env="GITHUB_TOKEN")
BRANCH = "agent/DEMO-1"


def git(*arguments: str | Path) -> str:
    """Run git with `arguments` as the test's own user and return its stripped stdout."""
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            *map(str, arguments),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def commit(work: Path, message: str) -> str:
    """Commit nothing in `work` with `message` and return the new commit."""
    git("-C", work, "commit", "--quiet", "--allow-empty", "-m", message)
    return git("-C", work, "rev-parse", "HEAD")


def remote_refs(mirror: Path) -> str:
    """Return every `refs/remotes` ref of `mirror` with the object it points to."""
    return git("-C", mirror, "for-each-ref", "refs/remotes")


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep the user's and the system's git configuration out, and set the token variable."""
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """Return a bare repository whose default branch `main` holds one commit."""
    bare = tmp_path / "remote.git"
    git("init", "--quiet", "--bare", "--initial-branch=main", bare)
    return bare


@pytest.fixture
def work(tmp_path: Path, remote: Path) -> Path:
    """Return a clone of `remote` with its first commit pushed to `main`."""
    clone = tmp_path / "work"
    git("init", "--quiet", "--initial-branch=main", clone)
    git("-C", clone, "remote", "add", "origin", remote)
    commit(clone, "first")
    git("-C", clone, "push", "--quiet", "origin", "main")
    return clone


@pytest.fixture
def space(tmp_path: Path, remote: Path, work: Path) -> Workspace:
    """Return a workspace in an instance directory, fetching from `remote`."""
    directory = tmp_path / "instance"
    directory.mkdir()
    return Workspace(REPOSITORY, directory, url=remote.as_uri())


def test_first_refresh_creates_the_mirror_following_the_remote(
    space: Workspace, remote: Path, work: Path
) -> None:
    space.refresh()

    mirror = space.mirror
    assert git("-C", mirror, "rev-parse", "--is-bare-repository") == "true"
    assert git("-C", mirror, "rev-parse", "refs/remotes/origin/main") == git(
        "-C", work, "rev-parse", "HEAD"
    )
    assert (
        git("-C", mirror, "symbolic-ref", "refs/remotes/origin/HEAD")
        == "refs/remotes/origin/main"
    )
    assert git("-C", mirror, "config", "remote.origin.url") == remote.as_uri()
    assert (
        git("-C", mirror, "config", "--get-all", "remote.origin.fetch")
        == "+refs/heads/*:refs/remotes/origin/*"
    )
    assert git("-C", mirror, "for-each-ref", "refs/heads") == ""
    assert TOKEN not in (mirror / "config").read_text(encoding="utf-8")


def test_next_refresh_sees_new_commits_and_prunes_gone_branches(
    space: Workspace, work: Path
) -> None:
    git("-C", work, "push", "--quiet", "origin", "main:gone")
    space.refresh()
    newer = commit(work, "second")
    git("-C", work, "push", "--quiet", "origin", "main", ":gone")

    space.refresh()

    assert git("-C", space.mirror, "rev-parse", "refs/remotes/origin/main") == newer
    assert "refs/remotes/origin/gone" not in remote_refs(space.mirror)


def test_open_on_a_branch_the_remote_has_starts_from_it_untracked(
    space: Workspace, work: Path
) -> None:
    tip = commit(work, "on the branch")
    git("-C", work, "push", "--quiet", "origin", f"main:{BRANCH}")
    git("-C", work, "reset", "--quiet", "--hard", "HEAD~1")
    space.refresh()

    path = space.open(BRANCH)

    assert path == space.worktree
    assert git("-C", path, "rev-parse", "HEAD") == tip
    assert git("-C", path, "symbolic-ref", "--short", "HEAD") == BRANCH
    with pytest.raises(subprocess.CalledProcessError):
        git("-C", path, "config", f"branch.{BRANCH}.merge")


def test_open_on_a_branch_the_remote_lacks_starts_from_its_default_branch(
    space: Workspace, work: Path
) -> None:
    git("-C", work, "push", "--quiet", "origin", f"main:{BRANCH}/nested")
    main = git("-C", work, "rev-parse", "HEAD")
    other = commit(work, "elsewhere")
    git("-C", work, "push", "--quiet", "origin", "HEAD:other")
    space.refresh()
    git("-C", space.mirror, "branch", "main", other)

    path = space.open(BRANCH)

    assert git("-C", path, "rev-parse", "HEAD") == main


def test_close_removes_the_worktree_and_its_branch_leaving_the_remote_refs(
    space: Workspace, work: Path
) -> None:
    space.refresh()
    before = remote_refs(space.mirror)
    path = space.open(BRANCH)
    (path / "unsaved.txt").write_text("work in progress", encoding="utf-8")

    space.close(BRANCH)

    assert not path.exists()
    assert git("-C", space.mirror, "for-each-ref", "refs/heads") == ""
    assert (
        git("-C", space.mirror, "worktree", "list", "--porcelain").count("worktree ")
        == 1
    )
    assert remote_refs(space.mirror) == before


def test_open_fails_naming_why_when_a_worktree_is_left(space: Workspace) -> None:
    space.refresh()
    space.open(BRANCH)

    with pytest.raises(
        WorkspaceError,
        match=f"git failed while creating the worktree of the branch {BRANCH}: ",
    ):
        space.open(BRANCH)


def test_close_fails_naming_why_when_no_worktree_is_open(space: Workspace) -> None:
    space.refresh()

    with pytest.raises(WorkspaceError, match="git failed while removing the worktree"):
        space.close(BRANCH)


def test_refresh_that_cannot_fetch_fails_with_the_token_masked(
    tmp_path: Path,
) -> None:
    missing = tmp_path / TOKEN
    space = Workspace(REPOSITORY, tmp_path / "instance", url=missing.as_uri())

    with pytest.raises(WorkspaceError) as caught:
        space.refresh()

    message = str(caught.value)
    assert message.startswith("git failed while fetching the mirror: ")
    assert TOKEN not in message
    assert "[token]" in message


def test_unset_token_variable_fails_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")

    with pytest.raises(WorkspaceError, match="the variable GITHUB_TOKEN, which"):
        Workspace(REPOSITORY, tmp_path).refresh()


def test_empty_token_variable_fails_naming_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "")

    with pytest.raises(WorkspaceError, match="the variable GITHUB_TOKEN, which"):
        Workspace(REPOSITORY, tmp_path).refresh()


def test_git_that_cannot_be_started_fails_with_the_token_masked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(arguments: list[str], **options: Any) -> None:
        raise OSError(13, f"denied near {TOKEN}")

    monkeypatch.setattr(workspace_module.subprocess, "run", refuse)

    with pytest.raises(WorkspaceError) as caught:
        Workspace(REPOSITORY, tmp_path).refresh()

    assert str(caught.value) == "git could not be started: denied near [token]"


def test_git_that_does_not_answer_in_time_fails_naming_the_action(
    tmp_path: Path,
) -> None:
    sleeper = [sys.executable, "-c", "import time; time.sleep(30)"]
    space = Workspace(REPOSITORY, tmp_path, command=sleeper, timeout=0.5)

    with pytest.raises(
        WorkspaceError,
        match="git did not answer within 0.5 seconds while creating the mirror",
    ):
        space.refresh()


def test_git_that_cannot_be_started_fails_naming_why(tmp_path: Path) -> None:
    space = Workspace(REPOSITORY, tmp_path, command=[str(tmp_path / "no-git")])

    with pytest.raises(WorkspaceError, match="git could not be started: "):
        space.refresh()


def test_every_git_call_signs_in_through_gh_with_the_token_in_its_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def record(
        arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, options["env"]))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(workspace_module.subprocess, "run", record)

    Workspace(REPOSITORY, tmp_path).refresh()

    assert len(calls) == 5
    for arguments, environment in calls:
        assert arguments[:5] == [
            "git",
            "-c",
            "credential.helper=",
            "-c",
            "credential.helper=!gh auth git-credential",
        ]
        assert not any(TOKEN in argument for argument in arguments)
        assert environment["GH_TOKEN"] == TOKEN
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        assert environment["GH_PROMPT_DISABLED"] == "1"
    assert calls[1][0][-2:] == [
        "remote.origin.url",
        "https://github.com/octo/widgets.git",
    ]
