"""The workspace of a run: a git mirror of the repository and a fresh worktree drawn from it.

Both live in the instance directory. The mirror is a bare repository whose
`refs/remotes/origin/*` follow the remote; a run only adds, then deletes, the local
branch of its worktree, and before deleting it rescues its work to the remote (see
`Workspace.rescue`). Every git call signs in through `gh` with the instance token,
which is never written in an argument or in the mirror's configuration, and is bounded:
one that has not answered within `GIT_TIMEOUT_SECONDS` (600 seconds) is stopped and
raises `WorkspaceError`.
"""

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from dev_worker.github import REDACTED, failure_reason
from dev_worker.instance import GitIdentity, Repository

GIT_TIMEOUT_SECONDS = 600.0
MIRROR_DIRECTORY = "mirror.git"
WORKTREE_DIRECTORY = "worktree"
REMOTE = "origin"
FETCH_REFSPEC = f"+refs/heads/*:refs/remotes/{REMOTE}/*"
DEFAULT_BRANCH_REF = f"refs/remotes/{REMOTE}/HEAD"
CREDENTIAL_HELPER = "!gh auth git-credential"
RESCUE_MESSAGE = "chore: rescue of the work the run left uncommitted"


class WorkspaceError(Exception):
    """Raised when the mirror cannot be refreshed or a worktree cannot be read, opened, rescued or removed.

    The message is a sentence saying why; it never holds a token.
    """


class Workspace:
    """The mirror and the worktree of an instance, kept in its `directory`."""

    def __init__(
        self,
        repository: Repository,
        directory: Path,
        command: Sequence[str] = ("git",),
        url: str | None = None,
        timeout: float = GIT_TIMEOUT_SECONDS,
    ) -> None:
        """Keep what later calls need; nothing is run or read yet.

        `url` is the address the mirror fetches from, by default the GitHub address of
        `repository`; `timeout` bounds each git call, in seconds.
        """
        self._repository = repository
        self._command = tuple(command)
        self._url = url or f"https://github.com/{repository.name}.git"
        self._timeout = timeout
        self.mirror = directory / MIRROR_DIRECTORY
        self.worktree = directory / WORKTREE_DIRECTORY

    def refresh(self) -> None:
        """Create the mirror when it is missing, then fetch every branch of the remote into it.

        Branches gone from the remote are pruned, and `refs/remotes/origin/HEAD` is set to
        the remote's default branch. Raises `WorkspaceError` when a git call fails.
        """
        if not self.mirror.exists():
            self._git(
                "creating the mirror", "init", "--bare", "--quiet", str(self.mirror)
            )
        self._git_mirror(
            "setting the mirror remote", "config", f"remote.{REMOTE}.url", self._url
        )
        self._git_mirror(
            "setting the mirror refspec",
            "config",
            "--replace-all",
            f"remote.{REMOTE}.fetch",
            FETCH_REFSPEC,
        )
        self._git_mirror("fetching the mirror", "fetch", "--prune", "--quiet", REMOTE)
        self._git_mirror(
            "reading the default branch", "remote", "set-head", REMOTE, "--auto"
        )

    def open(self, branch: str) -> Path:
        """Create the worktree on a new local `branch` and return its path.

        The branch starts from `refs/remotes/origin/<branch>` when the remote has it, else
        from the remote's default branch; never from a local branch. Raises
        `WorkspaceError` when a git call fails, as when a worktree or a local `branch` is
        left from an earlier run.

        An `open` that fails leaves no local `branch` it created: one that did not exist
        before is deleted, and a deletion that fails is added to the error message. A
        local `branch` that existed before, and whatever occupies the worktree path, are
        left as they are.
        """
        remote_ref = f"refs/remotes/{REMOTE}/{branch}"
        local_ref = f"refs/heads/{branch}"
        listed = self._git_mirror(
            f"looking for the branch {branch}",
            "for-each-ref",
            "--format=%(refname)",
            remote_ref,
            local_ref,
        ).splitlines()
        start = remote_ref if remote_ref in listed else DEFAULT_BRANCH_REF
        try:
            self._git_mirror(
                f"creating the worktree of the branch {branch}",
                "worktree",
                "add",
                "--quiet",
                "--no-track",
                "-b",
                branch,
                str(self.worktree),
                start,
            )
        except WorkspaceError as error:
            if local_ref not in listed:
                self._delete_ref_after(error, local_ref)
            raise
        return self.worktree

    def close(self, branch: str) -> None:
        """Remove the worktree, whatever it holds, then delete the local `branch` from the mirror.

        The mirror's record of the worktree is deleted first, so a worktree only partly
        deleted can no longer be rescued; then its directory is deleted without reading
        its `.git` and without following any symbolic link in it. Raises
        `WorkspaceError` when the mirror records no worktree there, when either
        directory cannot be deleted, or when a git call fails.
        """
        admin = self._worktree_admin()
        try:
            shutil.rmtree(admin)
            shutil.rmtree(self.worktree)
        except OSError as error:
            raise WorkspaceError(f"the worktree cannot be deleted: {error}") from error
        self._git_mirror("forgetting the removed worktree", "worktree", "prune")
        self._git_mirror(
            f"deleting the local branch {branch}", "branch", "--quiet", "-D", branch
        )

    def left_branch(self) -> str | None:
        """Return the branch of the worktree left on the host, or None when there is no worktree.

        The branch is read from the mirror's record of the worktree, not from the
        template. Raises `WorkspaceError` when the worktree path exists but its branch
        cannot be read, as when the mirror records no worktree there or its HEAD is
        detached.
        """
        if not self.worktree.exists():
            return None
        return self._git_worktree(
            "reading the branch of the worktree", "symbolic-ref", "--short", "HEAD"
        ).strip()

    def drop_left_branches(self) -> None:
        """Forget worktrees gone from the host, then delete the local branches the remote holds.

        A local branch is deleted only when its tip is in a `refs/remotes/origin/*` ref
        of the mirror; one holding a commit the remote lacks is kept. Nothing is done
        when the mirror does not exist yet. Meant for when no worktree is left: the
        local branches are then all from earlier runs. Raises `WorkspaceError` when a
        git call fails.
        """
        if not self.mirror.exists():
            return
        self._git_mirror("forgetting the removed worktrees", "worktree", "prune")
        branches = self._git_mirror(
            "listing the local branches",
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/heads",
        ).split()
        left = [
            branch
            for branch in branches
            if self._git_mirror(
                f"looking for the branch {branch} on the remote",
                "for-each-ref",
                "--count=1",
                "--contains",
                f"refs/heads/{branch}",
                f"refs/remotes/{REMOTE}",
            ).strip()
        ]
        if left:
            self._git_mirror(
                "deleting the local branches left by earlier runs",
                "branch",
                "--quiet",
                "-D",
                *left,
            )

    def rescue(self, branch: str, identity: GitIdentity) -> None:
        """Commit what the worktree holds uncommitted, then push `branch` when it is ahead.

        Every change, untracked files included, goes into one commit with
        `RESCUE_MESSAGE`, authored and committed by `identity`, unsigned and without
        hooks; nothing is committed when there is no change. `branch` is then pushed to
        `refs/heads/<branch>` on the remote, never forced, only when the worktree HEAD
        holds commits the remote's default branch lacks. Raises `WorkspaceError` when a
        git call fails, as when the remote branch has moved on.

        Before any git call, the worktree is searched, without following symbolic
        links, for an entry named `.git` in any letter case other than the root `.git`
        itself; when one is
        found, or a directory cannot be read, no git call is made and `WorkspaceError`
        names its path relative to the worktree.
        """
        nested = self._nested_git_entry()
        if nested is not None:
            raise WorkspaceError(
                f"the worktree holds a nested git entry at '{nested}'; "
                "no git command is run on it"
            )
        self._git_worktree("staging the work left in the worktree", "add", "--all")
        staged = self._git_worktree(
            "listing the staged changes", "diff", "--cached", "--name-only"
        )
        if staged.strip():
            self._git_worktree(
                "committing the work left in the worktree",
                "-c",
                f"user.name={identity.name}",
                "-c",
                f"user.email={identity.email}",
                "commit",
                "--quiet",
                "--no-verify",
                "--no-gpg-sign",
                "--message",
                RESCUE_MESSAGE,
            )
        ahead = self._git_worktree(
            "counting the commits beyond the default branch",
            "rev-list",
            "--count",
            f"{DEFAULT_BRANCH_REF}..HEAD",
        )
        if int(ahead) > 0:
            self._git_worktree(
                f"pushing the branch {branch}",
                "push",
                "--quiet",
                REMOTE,
                f"HEAD:refs/heads/{branch}",
            )

    def _delete_ref_after(self, error: WorkspaceError, ref: str) -> None:
        """Delete `ref` from the mirror, if it exists, after `error` made an `open` fail.

        Raises a `WorkspaceError` holding the message of `error` followed by why the
        deletion failed, when it does; returns otherwise.
        """
        try:
            self._git_mirror(f"deleting {ref}", "update-ref", "-d", ref)
        except WorkspaceError as cleanup_error:
            raise WorkspaceError(f"{error}; then {cleanup_error}") from error

    def _token(self) -> str:
        """Return the token held by the variable `worker_token_env` names; raise `WorkspaceError` if unset."""
        name = self._repository.worker_token_env
        token = os.environ.get(name)
        if not token:
            raise WorkspaceError(
                f"the variable {name}, which should hold the GitHub token, is not set"
            )
        return token

    def _git_mirror(self, action: str, *arguments: str) -> str:
        """Run git with `arguments` in the mirror and return what it wrote to stdout; see `_git`."""
        return self._git(action, "-C", str(self.mirror), *arguments)

    def _nested_git_entry(self) -> str | None:
        """Return the path, relative to the worktree and with `/`, of the first entry named
        `.git` in any letter case, or None when there is none.

        The entry named exactly `.git` at the root is neither reported nor entered; any
        other letter case of it at the root is reported. Symbolic links are listed but
        never followed; directories that cannot be read stop the search. Raises
        `WorkspaceError` naming the relative path of a directory that cannot be read,
        and why.
        """

        def refuse(error: OSError) -> None:
            unread = Path(error.filename).relative_to(self.worktree).as_posix()
            reason = error.strerror or str(error)
            raise WorkspaceError(
                f"the worktree cannot be searched for nested git entries: "
                f"'{unread}' cannot be read: {reason}"
            )

        for directory, subdirectories, files in os.walk(self.worktree, onerror=refuse):
            here = Path(directory)
            if here == self.worktree:
                subdirectories[:] = [name for name in subdirectories if name != ".git"]
            subdirectories.sort()
            for name in sorted([*subdirectories, *files]):
                if name.casefold() == ".git" and (here, name) != (
                    self.worktree,
                    ".git",
                ):
                    return (here / name).relative_to(self.worktree).as_posix()
        return None

    def _worktree_admin(self) -> Path:
        """Return the directory where the mirror keeps the worktree's git data.

        It is found among the mirror's `worktrees/*` by the worktree path each records in
        its `gitdir` file, a relative one being taken from that admin directory; the
        worktree's own `.git` is never read. Raises `WorkspaceError` when the mirror
        records no such directory.
        """
        worktree = self.worktree.resolve()
        admins = self.mirror / "worktrees"
        for admin in sorted(admins.iterdir()) if admins.is_dir() else []:
            try:
                recorded = (admin / "gitdir").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if (admin / recorded).parent.resolve() == worktree:
                return admin
        raise WorkspaceError(
            f"the mirror records no worktree at '{self.worktree}'; "
            "delete that directory by hand"
        )

    def _git_worktree(self, action: str, *arguments: str) -> str:
        """Run git with `arguments` on the worktree and return what it wrote to stdout; see `_git`.

        Git reads the repository from the mirror (see `_worktree_admin`), never through
        the worktree's `.git`, which the container may have rewritten.
        """
        admin = self._worktree_admin()
        return self._git(
            action,
            f"--git-dir={admin}",
            f"--work-tree={self.worktree}",
            *arguments,
        )

    def _git(self, action: str, *arguments: str) -> str:
        """Run git with `arguments` and return what it wrote to stdout.

        Inherited credential helpers are cleared and `gh` is the only one, reading the
        token from `GH_TOKEN`; git and `gh` never prompt. `action` names the call in error
        sentences. Raises `WorkspaceError` when the token variable is not set, git cannot
        be started, does not answer within the timeout, or exits nonzero.
        """
        token = self._token()
        environment = {
            **os.environ,
            "GH_TOKEN": token,
            "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
        helpers = [
            "-c",
            "credential.helper=",
            "-c",
            f"credential.helper={CREDENTIAL_HELPER}",
        ]
        try:
            done = subprocess.run(
                [*self._command, *helpers, *arguments],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise WorkspaceError(
                f"git did not answer within {self._timeout:g} seconds while {action}"
            ) from None
        except OSError as error:
            reason = (error.strerror or type(error).__name__).replace(token, REDACTED)
            raise WorkspaceError(f"git could not be started: {reason}") from None
        if done.returncode != 0:
            reason = failure_reason(done).replace(token, REDACTED)
            raise WorkspaceError(f"git failed while {action}: {reason}")
        return done.stdout
