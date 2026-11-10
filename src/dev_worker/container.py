"""The disposable container of a run, driven through the `docker` command line.

A run starts one container with `docker run --rm`, the worktree bind-mounted at
`WORKDIR` and nothing else mounted. The container is named, so that a run reaching its
duration ceiling can kill it; `--rm` then has Docker remove it. Every other docker call
is bounded by `DOCKER_TIMEOUT_SECONDS` (60 seconds).
"""

import csv
import io
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path

from dev_worker.github import failure_reason

DOCKER_TIMEOUT_SECONDS = 60.0
WORKDIR = "/workspace"
RUN_COMMAND = ("true",)
NAME_PREFIX = "dev-worker-"


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
    """The container a run starts from `image`."""

    def __init__(
        self,
        image: str,
        command: Sequence[str] = ("docker",),
        timeout: float = DOCKER_TIMEOUT_SECONDS,
    ) -> None:
        """Keep what later calls need; nothing is run yet.

        `timeout` bounds, in seconds, each docker call other than the run itself.
        """
        self._image = image
        self._command = tuple(command)
        self._timeout = timeout

    def check_access(self) -> None:
        """Ask the Docker daemon for its version; raise `ContainerError` when it cannot answer."""
        self._docker(
            "reaching the Docker daemon", "version", "--format", "{{.Server.Version}}"
        )

    def run(self, worktree: Path, time_left: float) -> None:
        """Run `RUN_COMMAND` in a new container with `worktree` mounted, and wait for it to end.

        `time_left` is the time left in the run, in seconds: a container still running
        then is killed, which `--rm` has Docker remove. Raises `ContainerError` when the
        container cannot be started, exits nonzero, or outlasts `time_left`, whether it
        could then be killed or not. Any other exception raised while waiting, such as
        `KeyboardInterrupt`, has the container killed and is raised again; a kill that
        fails is added to it as a note.
        """
        name = f"{NAME_PREFIX}{uuid.uuid4().hex}"
        try:
            self._docker(
                "running the container",
                "run",
                "--rm",
                "--name",
                name,
                "--mount",
                mount_option(worktree),
                "--workdir",
                WORKDIR,
                self._image,
                *RUN_COMMAND,
                timeout=time_left,
            )
        except subprocess.TimeoutExpired:
            reached = (
                f"the container {name} was still running when the run reached "
                "its ceiling"
            )
            try:
                self._kill(name)
            except ContainerError as kill_error:
                raise ContainerError(
                    f"{reached} and could not be killed: {kill_error}"
                ) from None
            raise ContainerError(f"{reached} and was killed") from None
        except ContainerError:
            raise
        except BaseException as error:
            try:
                self._kill(name)
            except ContainerError as kill_error:
                error.add_note(
                    f"the container {name} could not be killed: {kill_error}"
                )
            raise

    def _kill(self, name: str) -> None:
        """Kill the container `name`; raise `ContainerError` when docker cannot."""
        self._docker(f"killing the container {name}", "kill", name)

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
