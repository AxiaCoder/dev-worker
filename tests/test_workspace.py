import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dev_worker import workspace as workspace_module
from dev_worker.instance import GitIdentity, Repository
from dev_worker.workspace import Workspace, WorkspaceError

TOKEN = "s3cret-token-value"
REPOSITORY = Repository(type="github", name="octo/widgets", token_env="GITHUB_TOKEN")
BRANCH = "agent/DEMO-1"
BRANCH_REF = f"refs/heads/{BRANCH}"
IDENTITY = GitIdentity(name="Rescue Bot", email="bot@example.com")


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


@pytest.mark.parametrize(
    "rewritten", ["gitdir: /elsewhere\n", "not a gitdir line at all\n"]
)
def test_close_removes_the_worktree_and_its_branch_when_its_git_file_was_rewritten(
    space: Workspace, rewritten: str
) -> None:
    space.refresh()
    before = remote_refs(space.mirror)
    path = space.open(BRANCH)
    (path / ".git").unlink()
    (path / ".git").write_text(rewritten, encoding="utf-8")

    space.close(BRANCH)

    assert not path.exists()
    assert git("-C", space.mirror, "for-each-ref", "refs/heads") == ""
    assert (
        git("-C", space.mirror, "worktree", "list", "--porcelain").count("worktree ")
        == 1
    )
    assert remote_refs(space.mirror) == before


def test_close_leaves_the_target_of_a_link_out_of_the_worktree_untouched(
    space: Workspace, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    kept = outside / "kept.txt"
    kept.write_text("not the worktree's", encoding="utf-8")
    space.refresh()
    path = space.open(BRANCH)
    try:
        (path / "link").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links cannot be created here: {error}")

    space.close(BRANCH)

    assert not path.exists()
    assert kept.read_text(encoding="utf-8") == "not the worktree's"


def test_open_fails_naming_why_when_a_worktree_is_left(space: Workspace) -> None:
    space.refresh()
    space.open(BRANCH)

    with pytest.raises(
        WorkspaceError,
        match=f"git failed while creating the worktree of the branch {BRANCH}: ",
    ):
        space.open(BRANCH)


def test_open_that_fails_leaves_no_branch_it_created(space: Workspace) -> None:
    space.refresh()
    space.worktree.mkdir(parents=True)
    (space.worktree / "occupied.txt").write_text("left behind", encoding="utf-8")

    with pytest.raises(WorkspaceError):
        space.open("agent/Y")

    assert git("-C", space.mirror, "for-each-ref", "refs/heads/agent/Y") == ""


def test_open_that_fails_keeps_a_branch_that_existed_before(
    space: Workspace, work: Path
) -> None:
    space.refresh()
    tip = git("-C", work, "rev-parse", "HEAD")
    git("-C", space.mirror, "branch", "agent/Z", tip)

    with pytest.raises(WorkspaceError):
        space.open("agent/Z")

    assert git("-C", space.mirror, "rev-parse", "refs/heads/agent/Z") == tip


def test_open_that_fails_then_cannot_delete_its_branch_says_both(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def answer(
        arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        if "worktree" in arguments:
            return subprocess.CompletedProcess(arguments, 1, "", "path is busy\n")
        if "update-ref" in arguments:
            return subprocess.CompletedProcess(arguments, 1, "", "ref is locked\n")
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(workspace_module.subprocess, "run", answer)

    with pytest.raises(WorkspaceError) as caught:
        Workspace(REPOSITORY, tmp_path).open("agent/Y")

    assert str(caught.value) == (
        "git failed while creating the worktree of the branch agent/Y: path is busy;"
        " then git failed while deleting refs/heads/agent/Y: ref is locked"
    )


def test_close_fails_naming_why_when_no_worktree_is_open(space: Workspace) -> None:
    space.refresh()

    with pytest.raises(WorkspaceError, match="^the mirror records no worktree at "):
        space.close(BRANCH)


def test_close_that_deletes_the_worktree_partly_leaves_no_branch_to_rescue(
    space: Workspace, work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (work / "tracked.txt").write_text("committed", encoding="utf-8")
    git("-C", work, "add", "tracked.txt")
    commit(work, "tracked")
    git("-C", work, "push", "--quiet", "origin", "main")
    space.refresh()
    path = space.open(BRANCH)
    tracked = path / "tracked.txt"
    assert tracked.exists()
    rmtree = shutil.rmtree

    def delete_partly(target: Any, *arguments: Any, **options: Any) -> None:
        if Path(target) == path:
            tracked.unlink()
            raise OSError("permission denied")
        rmtree(target, *arguments, **options)

    monkeypatch.setattr(workspace_module.shutil, "rmtree", delete_partly)

    with pytest.raises(
        WorkspaceError, match="^the worktree cannot be deleted: permission denied$"
    ):
        space.close(BRANCH)

    with pytest.raises(WorkspaceError, match="the mirror records no worktree"):
        space.left_branch()


def failing_git(monkeypatch: pytest.MonkeyPatch, failing: str) -> list[list[str]]:
    """Make the git call holding `failing` fail, let the others run; return every call."""
    calls: list[list[str]] = []
    run = subprocess.run

    def answer(
        arguments: list[str], **options: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if failing in arguments:
            return subprocess.CompletedProcess(arguments, 1, "", "refused\n")
        return run(arguments, **options)

    monkeypatch.setattr(workspace_module.subprocess, "run", answer)
    return calls


def test_close_that_cannot_prune_fails_naming_why_and_keeps_the_branch(
    space: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    calls = failing_git(monkeypatch, "prune")

    with pytest.raises(WorkspaceError) as caught:
        space.close(BRANCH)

    assert str(caught.value) == (
        "git failed while forgetting the removed worktree: refused"
    )
    assert not path.exists()
    assert not any("-D" in call for call in calls)
    assert git("-C", space.mirror, "for-each-ref", "refs/heads") != ""


def test_close_that_cannot_delete_the_branch_fails_naming_it(
    space: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    failing_git(monkeypatch, "-D")

    with pytest.raises(WorkspaceError) as caught:
        space.close(BRANCH)

    assert str(caught.value) == (
        f"git failed while deleting the local branch {BRANCH}: refused"
    )
    assert not path.exists()
    assert (
        git("-C", space.mirror, "worktree", "list", "--porcelain").count("worktree ")
        == 1
    )


def remote_branch(remote: Path) -> str:
    """Return the commit `BRANCH` points to on `remote`, empty when it lacks the branch."""
    return git("-C", remote, "for-each-ref", "--format=%(objectname)", BRANCH_REF)


def test_rescue_with_nothing_to_commit_nor_beyond_the_default_branch_pushes_nothing(
    space: Workspace, remote: Path, work: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)

    space.rescue(BRANCH, IDENTITY)

    assert git("-C", path, "rev-parse", "HEAD") == git("-C", work, "rev-parse", "HEAD")
    assert remote_branch(remote) == ""


def test_rescue_commits_what_is_left_as_the_instance_then_pushes_the_branch(
    space: Workspace, remote: Path, work: Path, tmp_path: Path
) -> None:
    (tmp_path / "gitconfig").write_text(
        "[commit]\n\tgpgsign = true\n[gpg]\n\tprogram = no-such-signer\n",
        encoding="utf-8",
    )
    space.refresh()
    path = space.open(BRANCH)
    (path / "untracked.txt").write_text("new", encoding="utf-8")
    git("-C", path, "commit", "--quiet", "--no-gpg-sign", "--allow-empty", "-m", "own")

    space.rescue(BRANCH, IDENTITY)

    head = git("-C", path, "rev-parse", "HEAD")
    assert git("-C", path, "log", "-1", "--format=%an|%ae|%cn|%ce|%s") == (
        "Rescue Bot|bot@example.com|Rescue Bot|bot@example.com|"
        "chore: rescue of the work the run left uncommitted"
    )
    assert git("-C", path, "show", "--format=", "--name-only", head) == "untracked.txt"
    assert git("-C", path, "status", "--porcelain") == ""
    assert remote_branch(remote) == head


def test_rescue_commits_past_a_hook_that_refuses_every_commit(
    space: Workspace, remote: Path
) -> None:
    space.refresh()
    hook = space.mirror / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    path = space.open(BRANCH)
    (path / "left.txt").write_text("left", encoding="utf-8")

    space.rescue(BRANCH, IDENTITY)

    assert git("-C", path, "status", "--porcelain") == ""
    assert remote_branch(remote) == git("-C", path, "rev-parse", "HEAD")


def test_rescue_ignores_a_git_file_the_worktree_was_given(
    space: Workspace, tmp_path: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    trap = tmp_path / "trap"
    git("init", "--quiet", trap)
    witness = tmp_path / "witness"
    monitor = tmp_path / "monitor.py"
    monitor.write_text(f"open({str(witness)!r}, 'w').close()\n", encoding="utf-8")
    python = Path(sys.executable).as_posix()
    git("-C", trap, "config", "core.fsmonitor", f'"{python}" "{monitor.as_posix()}"')
    (path / ".git").unlink()
    (path / ".git").write_text(
        f"gitdir: {(trap / '.git').as_posix()}\n", encoding="utf-8"
    )
    (path / "left.txt").write_text("left", encoding="utf-8")

    space.rescue(BRANCH, IDENTITY)

    assert not witness.exists()
    assert git("-C", space.mirror, "log", "-1", "--format=%s", BRANCH_REF) == (
        "chore: rescue of the work the run left uncommitted"
    )
    assert (
        git("-C", space.mirror, "show", "--format=", "--name-only", BRANCH_REF)
        == "left.txt"
    )
    assert git("-C", trap, "for-each-ref") == ""
    subprocess.run(
        ["git", "-C", path, "add", "--all"], capture_output=True, check=False
    )
    assert witness.exists()


def test_rescue_refuses_a_worktree_holding_a_nested_repository(
    space: Workspace, remote: Path, tmp_path: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    head = git("-C", path, "rev-parse", "HEAD")
    nested = path / "sub"
    git("init", "--quiet", nested)
    witness = tmp_path / "witness"
    monitor = tmp_path / "monitor.py"
    monitor.write_text(f"open({str(witness)!r}, 'w').close()\n", encoding="utf-8")
    python = Path(sys.executable).as_posix()
    git("-C", nested, "config", "core.fsmonitor", f'"{python}" "{monitor.as_posix()}"')
    (path / "left.txt").write_text("left", encoding="utf-8")

    with pytest.raises(WorkspaceError) as caught:
        space.rescue(BRANCH, IDENTITY)

    assert str(caught.value) == (
        "the worktree holds a nested git entry at 'sub/.git'; "
        "no git command is run on it"
    )
    assert not witness.exists()
    assert git("-C", space.mirror, "rev-parse", BRANCH_REF) == head
    assert remote_branch(remote) == ""


@pytest.mark.parametrize("name", [".git", ".GIT"])
def test_rescue_refuses_a_nested_git_file_in_any_case(
    space: Workspace, name: str
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    (path / "a" / "b").mkdir(parents=True)
    (path / "a" / "b" / name).write_text("gitdir: elsewhere\n", encoding="utf-8")

    with pytest.raises(WorkspaceError, match=f"nested git entry at 'a/b/{name}'"):
        space.rescue(BRANCH, IDENTITY)


def test_rescue_refuses_a_root_git_entry_in_another_letter_case(
    space: Workspace, remote: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    (path / ".git").unlink()
    (path / ".GIT").mkdir()

    with pytest.raises(WorkspaceError, match="nested git entry at '.GIT'"):
        space.rescue(BRANCH, IDENTITY)

    assert remote_branch(remote) == ""


def test_rescue_refuses_a_worktree_with_a_directory_it_cannot_read(
    monkeypatch: pytest.MonkeyPatch, space: Workspace, remote: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    head = git("-C", path, "rev-parse", "HEAD")
    locked = path / "a" / "locked"
    locked.mkdir(parents=True)
    (path / "left.txt").write_text("left", encoding="utf-8")
    scandir = os.scandir

    def refuse_locked(directory: Any) -> Any:
        if Path(directory) == locked:
            raise PermissionError(13, "Permission denied", str(locked))
        return scandir(directory)

    monkeypatch.setattr(workspace_module.os, "scandir", refuse_locked)

    with pytest.raises(WorkspaceError) as caught:
        space.rescue(BRANCH, IDENTITY)

    monkeypatch.undo()
    assert str(caught.value) == (
        "the worktree cannot be searched for nested git entries: "
        "'a/locked' cannot be read: Permission denied"
    )
    assert git("-C", space.mirror, "rev-parse", BRANCH_REF) == head
    assert remote_branch(remote) == ""


def test_rescue_runs_on_subdirectories_holding_no_git_entry(
    space: Workspace, remote: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    (path / "docs" / "deep").mkdir(parents=True)
    (path / "docs" / "deep" / ".gitignore").write_text("*.tmp\n", encoding="utf-8")

    space.rescue(BRANCH, IDENTITY)

    assert remote_branch(remote) == git("-C", path, "rev-parse", "HEAD")


def test_rescue_does_not_follow_a_link_to_a_repository_outside(
    space: Workspace, remote: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    git("init", "--quiet", outside)
    space.refresh()
    path = space.open(BRANCH)
    (path / "docs").mkdir()
    try:
        (path / "docs" / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links cannot be created on this machine")

    space.rescue(BRANCH, IDENTITY)

    assert remote_branch(remote) == git("-C", path, "rev-parse", "HEAD")


def test_worktree_the_mirror_does_not_record_is_refused(space: Workspace) -> None:
    space.worktree.mkdir(parents=True)

    with pytest.raises(
        WorkspaceError, match="the mirror records no worktree at '.*'; it was not"
    ):
        space.left_branch()


def test_rescue_with_commits_but_nothing_left_pushes_without_committing(
    space: Workspace, remote: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    own = commit(path, "own")

    space.rescue(BRANCH, IDENTITY)

    assert git("-C", path, "rev-parse", "HEAD") == own
    assert remote_branch(remote) == own


def test_rescue_of_a_branch_the_remote_has_pushes_it_forward(
    space: Workspace, remote: Path, work: Path
) -> None:
    commit(work, "earlier run")
    git("-C", work, "push", "--quiet", "origin", f"HEAD:{BRANCH}")
    space.refresh()
    path = space.open(BRANCH)
    (path / "more.txt").write_text("more", encoding="utf-8")

    space.rescue(BRANCH, IDENTITY)

    assert remote_branch(remote) == git("-C", path, "rev-parse", "HEAD")


def test_rescue_never_forces_a_remote_branch_that_moved_on(
    space: Workspace, remote: Path, work: Path
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    commit(path, "own")
    elsewhere = commit(work, "pushed meanwhile")
    git("-C", work, "push", "--quiet", "origin", f"HEAD:{BRANCH}")

    with pytest.raises(
        WorkspaceError, match=f"git failed while pushing the branch {BRANCH}: "
    ):
        space.rescue(BRANCH, IDENTITY)

    assert remote_branch(remote) == elsewhere


def test_drop_left_branches_without_a_mirror_does_nothing(space: Workspace) -> None:
    space.drop_left_branches()

    assert not space.mirror.exists()


def test_drop_left_branches_without_any_leaves_the_mirror_as_it_is(
    space: Workspace,
) -> None:
    space.refresh()
    before = remote_refs(space.mirror)

    space.drop_left_branches()

    assert git("-C", space.mirror, "for-each-ref", "refs/heads") == ""
    assert remote_refs(space.mirror) == before


def test_branch_left_without_its_worktree_is_dropped_so_the_next_open_works(
    space: Workspace,
) -> None:
    space.refresh()
    space.open(BRANCH)
    git("-C", space.mirror, "worktree", "remove", "--force", space.worktree)
    git("-C", space.mirror, "branch", "agent/OTHER-2", "refs/remotes/origin/main")

    space.drop_left_branches()

    assert git("-C", space.mirror, "for-each-ref", "refs/heads") == ""
    assert space.open(BRANCH) == space.worktree


def test_branch_holding_a_commit_the_remote_lacks_is_kept(
    space: Workspace,
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    own = commit(path, "rescued but not pushed")
    git("-C", space.mirror, "worktree", "remove", "--force", path)
    git("-C", space.mirror, "branch", "agent/OTHER-2", "refs/remotes/origin/main")

    space.drop_left_branches()

    assert git(
        "-C",
        space.mirror,
        "for-each-ref",
        "--format=%(objectname) %(refname)",
        "refs/heads",
    ) == (f"{own} {BRANCH_REF}")


def test_drop_left_branches_forgets_a_worktree_deleted_from_the_host(
    space: Workspace,
) -> None:
    space.refresh()
    space.open(BRANCH)
    shutil.rmtree(space.worktree)

    space.drop_left_branches()

    assert git("-C", space.mirror, "for-each-ref", "refs/heads") == ""
    assert space.open(BRANCH) == space.worktree


def test_left_branch_is_none_without_a_worktree(space: Workspace) -> None:
    space.refresh()

    assert space.left_branch() is None


def test_left_branch_is_read_from_the_mirror_record_of_the_worktree(
    space: Workspace,
) -> None:
    space.refresh()
    (space.mirror / "worktrees" / "stale").mkdir(parents=True)
    elsewhere = space.mirror / "worktrees" / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "gitdir").write_text(
        (space.mirror.parent / "gone" / ".git").as_posix(), encoding="utf-8"
    )
    space.open("agent/OLD-7")

    assert space.left_branch() == "agent/OLD-7"


def test_left_branch_takes_a_relative_gitdir_from_its_admin_directory(
    space: Workspace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    space.refresh()
    space.open(BRANCH)
    [admin] = (space.mirror / "worktrees").iterdir()
    relative = os.path.relpath(space.worktree / ".git", admin)
    (admin / "gitdir").write_text(Path(relative).as_posix(), encoding="utf-8")
    monkeypatch.chdir(tmp_path / "work")

    assert space.left_branch() == BRANCH


def test_left_branch_of_a_detached_worktree_fails_naming_why(
    space: Workspace,
) -> None:
    space.refresh()
    path = space.open(BRANCH)
    git("-C", path, "checkout", "--quiet", "--detach")

    with pytest.raises(
        WorkspaceError, match="git failed while reading the branch of the worktree"
    ):
        space.left_branch()


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
