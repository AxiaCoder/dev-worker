"""The checkout of the Claude Code plugin a run loads, kept in the instance directory.

The checkout is a shallow clone of the tag the `WORKER.md` header names, fetched
anonymously: inherited credential helpers are cleared and git never prompts. It is
fetched only when it is missing or is not at that tag of that repository, as read from
the checkout itself. A new clone is made beside it, in `STAGING_DIRECTORY`, and replaces
it only once complete, so a fetch that fails leaves the previous checkout as it was.
Every git call is bounded by `GIT_TIMEOUT_SECONDS` (600 seconds).
"""

import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

from dev_worker.github import failure_reason
from dev_worker.header import Plugin

GIT_TIMEOUT_SECONDS = 600.0
CHECKOUT_DIRECTORY = "plugin"
STAGING_DIRECTORY = "plugin.new"
DISCARD_DIRECTORY = "plugin.old"
REMOTE = "origin"


class PluginError(Exception):
    """Raised when the plugin cannot be fetched or its checkout cannot be replaced.

    The message is a sentence saying why.
    """


class PluginCheckout:
    """The checkout of the plugin of an instance, kept in its `directory`."""

    def __init__(
        self,
        directory: Path,
        command: Sequence[str] = ("git",),
        timeout: float = GIT_TIMEOUT_SECONDS,
    ) -> None:
        """Keep what later calls need; nothing is run or read yet.

        `timeout` bounds each git call, in seconds.
        """
        self._command = tuple(command)
        self._timeout = timeout
        self.checkout = directory / CHECKOUT_DIRECTORY
        self._staging = directory / STAGING_DIRECTORY
        self._discard = directory / DISCARD_DIRECTORY

    def prepare(self, plugin: Plugin) -> Path:
        """Return the checkout of `plugin`, fetching it first when it is not there.

        A checkout at the tag of `plugin`, cloned from its repository, is used as it is,
        without reaching the network. Otherwise the tag is cloned into
        `STAGING_DIRECTORY`, which then replaces the checkout. What an earlier fetch
        left in `STAGING_DIRECTORY` or `DISCARD_DIRECTORY` is deleted first. Raises
        `PluginError` when the clone fails, is not at a tag of that name, or a
        directory cannot be deleted or moved.
        """
        if self._holds(self.checkout, plugin):
            return self.checkout
        self._delete(self._staging)
        self._delete(self._discard)
        self._git(
            f"cloning the tag {plugin.tag} of {plugin.repository}",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--branch",
            plugin.tag,
            "--",
            plugin.repository,
            str(self._staging),
        )
        if not self._holds(self._staging, plugin):
            raise PluginError(f"{plugin.repository} has no tag {plugin.tag}")
        try:
            if self.checkout.exists():
                self.checkout.rename(self._discard)
            self._staging.rename(self.checkout)
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise PluginError(
                f"the checkout '{self.checkout}' cannot be replaced: {reason}"
            ) from None
        self._delete(self._discard)
        return self.checkout

    def _holds(self, checkout: Path, plugin: Plugin) -> bool:
        """Tell whether `checkout` is a clone of the repository of `plugin` at its tag.

        False when `checkout` is not a directory or git cannot read it.
        """
        if not checkout.is_dir():
            return False
        git_dir = f"--git-dir={checkout / '.git'}"
        try:
            url = self._git(
                "reading the plugin remote",
                git_dir,
                "config",
                "--get",
                f"remote.{REMOTE}.url",
            )
            head, tagged = self._git(
                "reading the plugin tag",
                git_dir,
                "rev-parse",
                "HEAD",
                f"refs/tags/{plugin.tag}^{{commit}}",
            ).split()
        except PluginError:
            return False
        return url.strip() == plugin.repository and head == tagged

    def _delete(self, directory: Path) -> None:
        """Delete `directory` and everything in it, if it exists.

        The read-only files git keeps its objects in are made writable first. Raises
        `PluginError` when it cannot be deleted.
        """
        if not directory.exists():
            return
        try:
            for folder, _, files in os.walk(directory / ".git" / "objects"):
                for name in files:
                    os.chmod(Path(folder, name), stat.S_IREAD | stat.S_IWRITE)
            shutil.rmtree(directory)
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise PluginError(f"'{directory}' cannot be deleted: {reason}") from None

    def _git(self, action: str, *arguments: str) -> str:
        """Run git with `arguments` and return what it wrote to stdout.

        Inherited credential helpers are cleared and git never prompts. `action` names
        the call in error sentences. Raises `PluginError` when git cannot be started,
        does not answer within the timeout, or exits nonzero.
        """
        environment = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        try:
            done = subprocess.run(
                [*self._command, "-c", "credential.helper=", *arguments],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise PluginError(
                f"git did not answer within {self._timeout:g} seconds while {action}"
            ) from None
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise PluginError(f"git could not be started: {reason}") from None
        if done.returncode != 0:
            raise PluginError(f"git failed while {action}: {failure_reason(done)}")
        return done.stdout
