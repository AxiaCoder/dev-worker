"""The disposable container of a run, driven through the `docker` command line.

A run creates one container with `docker create --rm`, the worktree bind-mounted at
`WORKDIR` and nothing else mounted, then starts it; `--rm` has Docker remove it once it
ends. The container is named, so that a run reaching its duration ceiling, or
interrupted, can remove it by force: created before it is started, it exists by then,
and once removed it can no longer start. It is labelled
`OWNER_LABEL` with the instance that started it, so that the running containers of an
instance can be listed and killed. Every other docker call is bounded by
`DOCKER_TIMEOUT_SECONDS` (60 seconds).
"""

import csv
import io
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from dev_worker.github import failure_reason

DOCKER_TIMEOUT_SECONDS = 60.0
WORKDIR = "/workspace"
RUN_COMMAND = ("true",)
NAME_PREFIX = "dev-worker-"
OWNER_LABEL = "dev-worker.instance"


class ContainerError(Exception):
    """Raised when Docker cannot be reached, or a container cannot be run or stopped.

    The message is a sentence saying why.
    """


def mount_option(source: Path) -> str:
    """Return the `--mount` value binding the directory `source` at `WORKDIR`.

    Each field is quoted as CSV when it needs to be, so a path holding a comma or a
    quote is passed whole.
    """
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(
        ["type=bind", f"source={source}", f"target={WORKDIR}"]
    )
    return buffer.getvalue()


class Container:
    """The container a run of the instance `owner` starts from `image`."""

    def __init__(
        self,
        image: str,
        owner: str,
        command: Sequence[str] = ("docker",),
        timeout: float = DOCKER_TIMEOUT_SECONDS,
    ) -> None:
        """Keep what later calls need; nothing is run yet.

        `owner` names the instance, and is the value of `OWNER_LABEL` on the containers
        it runs. `timeout` bounds, in seconds, each docker call other than the run itself.
        """
        self._image = image
        self._owner = owner
        self._command = tuple(command)
        self._timeout = timeout

    def check_access(self) -> None:
        """Ask the Docker daemon for its version; raise `ContainerError` when it cannot answer."""
        self._docker(
            "reaching the Docker daemon", "version", "--format", "{{.Server.Version}}"
        )

    def run(self, worktree: Path, time_left: float) -> None:
        """Run `RUN_COMMAND` in a new container with `worktree` mounted, and wait for it to end.

        The container is created, then started. `time_left` is the time left in the run,
        in seconds, counted from the call: a container still running then is removed by
        force. Raises `ContainerError` when the container cannot be created or started,
        exits nonzero, or outlasts `time_left`, whether it could then be removed or not.
        Any other exception raised meanwhile, such as `KeyboardInterrupt`, has the
        container removed by force and is raised again; a removal that fails is added to
        it as a note.
        """
        name = f"{NAME_PREFIX}{uuid.uuid4().hex}"
        deadline = time.monotonic() + time_left
        try:
            self._docker(
                "creating the container",
                "create",
                "--rm",
                "--name",
                name,
                "--label",
                f"{OWNER_LABEL}={self._owner}",
                "--mount",
                mount_option(worktree),
                "--workdir",
                WORKDIR,
                self._image,
                *RUN_COMMAND,
            )
            self._docker(
                "running the container",
                "start",
                "--attach",
                name,
                timeout=max(deadline - time.monotonic(), 0),
            )
        except subprocess.TimeoutExpired:
            reached = (
                f"the container {name} was still running when the run reached "
                "its ceiling"
            )
            try:
                self._remove(name)
            except ContainerError as removal_error:
                raise ContainerError(
                    f"{reached} and could not be removed: {removal_error}"
                ) from None
            raise ContainerError(f"{reached} and was removed") from None
        except ContainerError:
            raise
        except BaseException as error:
            try:
                self._remove(name)
            except ContainerError as removal_error:
                error.add_note(
                    f"the container {name} could not be removed: {removal_error}"
                )
            raise

    def running(self) -> list[str]:
        """Return the names of the containers of the owner still running, in Docker's order.

        Raises `ContainerError` when docker cannot list them.
        """
        listed = self._docker(
            "listing the containers of the instance",
            "ps",
            "--filter",
            f"label={OWNER_LABEL}={self._owner}",
            "--format",
            "{{.Names}}",
        )
        return listed.split()

    def check_none_running(self) -> None:
        """Raise `ContainerError` naming them when containers of the owner are running.

        Also raises `ContainerError` when docker cannot list them.
        """
        names = self.running()
        if names:
            raise ContainerError(
                f"the container {', '.join(names)} of the instance is still running"
                if len(names) == 1
                else f"the containers {', '.join(names)} of the instance are still running"
            )

    def kill_running(self) -> None:
        """Kill each container of the owner still running.

        Raises `ContainerError` when docker cannot list them or kill one; the
        containers listed after it are then left running.
        """
        for name in self.running():
            self._kill(name)

    def _kill(self, name: str) -> None:
        """Kill the container `name`; raise `ContainerError` when docker cannot."""
        self._docker(f"killing the container {name}", "kill", name)

    def _remove(self, name: str) -> None:
        """Remove the container `name`, killing it first when it runs.

        Raises `ContainerError` when docker cannot, such as when no container has that name.
        """
        self._docker(f"removing the container {name}", "rm", "--force", name)

    def _docker(
        self, action: str, *arguments: str, timeout: float | None = None
    ) -> str:
        """Run docker with `arguments` and return what it wrote to stdout.

        `action` names the call in error sentences. `timeout` bounds the call, in
        seconds, `self._timeout` when None; a call reaching it is stopped and raises
        `subprocess.TimeoutExpired` when `timeout` was given, `ContainerError` otherwise.
        Raises `ContainerError` when docker cannot be started or exits nonzero.
        """
        try:
            done = subprocess.run(
                [*self._command, *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self._timeout if timeout is None else timeout,
            )
        except subprocess.TimeoutExpired:
            if timeout is not None:
                raise
            raise ContainerError(
                f"docker did not answer within {self._timeout:g} seconds while {action}"
            ) from None
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise ContainerError(f"docker could not be started: {reason}") from None
        if done.returncode != 0:
            raise ContainerError(
                f"docker failed while {action}: {failure_reason(done)}"
            )
        return done.stdout
