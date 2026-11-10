"""The workspace of a run: a git mirror of the repository and a fresh worktree drawn from it.

Both live in the instance directory. The mirror is a bare repository whose
`refs/remotes/origin/*` follow the remote; a run only adds, then deletes, the local
branch of its worktree. Every git call signs in through `gh` with the instance token,
which is never written in an argument or in the mirror's configuration, and is bounded:
one that has not answered within `GIT_TIMEOUT_SECONDS` (600 seconds) is stopped and
raises `WorkspaceError`.
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from dev_worker.github import REDACTED, _failure_reason
from dev_worker.instance import Repository

GIT_TIMEOUT_SECONDS = 600.0
MIRROR_DIRECTORY = "mirror.git"
WORKTREE_DIRECTORY = "worktree"
REMOTE = "origin"
FETCH_REFSPEC = f"+refs/heads/*:refs/remotes/{REMOTE}/*"
DEFAULT_BRANCH_REF = f"refs/remotes/{REMOTE}/HEAD"
CREDENTIAL_HELPER = "!gh auth git-credential"


class WorkspaceError(Exception):
    """Raised when the mirror cannot be refreshed or a worktree cannot be opened or removed.

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
        """
        remote_ref = f"refs/remotes/{REMOTE}/{branch}"
        listed = self._git_mirror(
            f"looking for the branch {branch}",
            "for-each-ref",
            "--format=%(refname)",
            remote_ref,
        )
        start = remote_ref if remote_ref in listed.splitlines() else DEFAULT_BRANCH_REF
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
        return self.worktree

    def close(self, branch: str) -> None:
        """Remove the worktree, whatever it holds, then delete the local `branch` from the mirror.

        Raises `WorkspaceError` when a git call fails.
        """
        self._git_mirror(
            "removing the worktree", "worktree", "remove", "--force", str(self.worktree)
        )
        self._git_mirror(
            f"deleting the local branch {branch}", "branch", "--quiet", "-D", branch
        )

    def _token(self) -> str:
        """Return the token held by the variable `token_env` names; raise `WorkspaceError` if unset."""
        name = self._repository.token_env
        token = os.environ.get(name)
        if not token:
            raise WorkspaceError(
                f"the variable {name}, which should hold the GitHub token, is not set"
            )
        return token

    def _git_mirror(self, action: str, *arguments: str) -> str:
        """Run git with `arguments` in the mirror and return what it wrote to stdout; see `_git`."""
        return self._git(action, "-C", str(self.mirror), *arguments)

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
            reason = _failure_reason(done).replace(token, REDACTED)
            raise WorkspaceError(f"git failed while {action}: {reason}")
        return done.stdout
