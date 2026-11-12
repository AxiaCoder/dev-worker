"""The disposable container of a run, driven through the `docker` command line.

A run pulls its image when the daemon does not have it, within the time left in the
run, then creates one container with `docker create --rm`, the worktree bind-mounted at
`WORKDIR` and the plugin checkout bind-mounted read only at `PLUGIN_DIR`, nothing else
mounted, then starts it; `--rm` has Docker remove it once it
ends. The container runs Claude Code headless on the ticket (see `agent_command`), and
receives no environment variable but those the run names, whose values never appear on
the docker command line. The container is named, so that a run reaching its duration ceiling, or
interrupted, can remove it by force: created before it is started, it exists by then,
and once removed it can no longer start. It is labelled
`OWNER_LABEL` with the instance that started it, so that the running containers of an
instance can be listed and killed. Every docker call other than the pull and the run of
the container is bounded by
`DOCKER_TIMEOUT_SECONDS` (60 seconds).
"""

import csv
import io
import json
import os
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from dev_worker.github import failure_reason
from dev_worker.tickets import Ticket

DOCKER_TIMEOUT_SECONDS = 60.0
WORKDIR = "/workspace"
PLUGIN_DIR = "/plugin"
AGENT_PROGRAM = "claude"
NAME_PREFIX = "dev-worker-"
OWNER_LABEL = "dev-worker.instance"


class ContainerError(Exception):
    """Raised when Docker cannot be reached, or a container cannot be run or stopped.

    The message is a sentence saying why.
    """


def ticket_prompt(ticket: Ticket) -> str:
    """Return the prompt of a run on `ticket`: its key, title and description."""
    return (
        f"Ticket: {ticket.key}\n"
        f"Title: {ticket.title}\n"
        "\n"
        f"Description:\n{ticket.description}\n"
    )


def outcome_schema(outcomes: Sequence[str]) -> dict[str, object]:
    """Return the JSON schema of the outcome a run ends on.

    The outcome is an object holding exactly `outcome`, one of `outcomes`, and `message`,
    a string: the summary of the work on `done`, the reason otherwise.
    """
    return {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": list(outcomes)},
            "message": {"type": "string"},
        },
        "required": ["outcome", "message"],
        "additionalProperties": False,
    }


def agent_command(
    ticket: Ticket, rules: str, max_turns: int, outcomes: Sequence[str]
) -> tuple[str, ...]:
    """Return the command that runs Claude Code headless on `ticket` in the container.

    `rules` is appended to its system prompt, `max_turns` caps its turns and `outcomes`
    are the outcomes it may end on (see `outcome_schema`); it loads the plugin mounted at
    `PLUGIN_DIR` and answers in JSON.
    """
    return (
        AGENT_PROGRAM,
        "-p",
        ticket_prompt(ticket),
        "--append-system-prompt",
        rules,
        "--plugin-dir",
        PLUGIN_DIR,
        "--max-turns",
        str(max_turns),
        "--json-schema",
        json.dumps(outcome_schema(outcomes)),
        "--output-format",
        "json",
    )


def mount_option(source: Path, target: str = WORKDIR, readonly: bool = False) -> str:
    """Return the `--mount` value binding the directory `source` at `target`, read only if `readonly`.

    Each field is quoted as CSV when it needs to be, so a path holding a comma or a
    quote is passed whole.
    """
    fields = ["type=bind", f"source={source}", f"target={target}"]
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(
        [*fields, "readonly"] if readonly else fields
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
        it runs. `timeout` bounds, in seconds, each docker call other than the pull of the
        image and the run of the container, which the time left in the run bounds.
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

    def run(
        self,
        worktree: Path,
        time_left: float,
        plugin: Path,
        command: Sequence[str],
        environment: Mapping[str, str],
    ) -> None:
        """Run `command` in a new container with `worktree` mounted, and wait for it to end.

        The plugin checkout `plugin` is mounted read only at `PLUGIN_DIR`. The container
        receives the variables of `environment`, by name and value, and no other; the
        values are handed to docker through its environment, never on its command line.

        The image is pulled when it is missing, then the container is created and
        started. `time_left` is the time left in the run, in seconds, counted from the
        call; it bounds the pull and the run of the container, and a container still
        there when it runs out is removed by force. Raises `ContainerError` when the
        image cannot be pulled, the container cannot be created or started, exits
        nonzero, or `time_left` runs out, whether the container could then be removed
        or not. Any other exception raised meanwhile, such as `KeyboardInterrupt`, has
        the container removed by force and is raised again; a removal that fails is
        added to it as a note.
        """
        name = f"{NAME_PREFIX}{uuid.uuid4().hex}"
        deadline = time.monotonic() + time_left
        self._pull_missing_image(deadline)
        try:
            self._docker(
                "creating the container",
                "create",
                "--pull",
                "never",
                "--rm",
                "--name",
                name,
                "--label",
                f"{OWNER_LABEL}={self._owner}",
                "--mount",
                mount_option(worktree),
                "--mount",
                mount_option(plugin, PLUGIN_DIR, readonly=True),
                "--workdir",
                WORKDIR,
                *(option for name in environment for option in ("--env", name)),
                self._image,
                *command,
                environment=environment,
            )
            start_time_left = deadline - time.monotonic()
            if start_time_left <= 0:
                self._remove_at_ceiling(
                    name,
                    f"the run reached its ceiling before the container {name} "
                    "was started",
                )
            self._docker(
                "running the container",
                "start",
                "--attach",
                name,
                timeout=start_time_left,
            )
        except subprocess.TimeoutExpired:
            self._remove_at_ceiling(
                name,
                f"the container {name} was still running when the run reached "
                "its ceiling",
            )
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

    def _pull_missing_image(self, deadline: float) -> None:
        """Pull the image when the daemon does not have it, within `deadline`.

        `deadline` is a `time.monotonic()` reading. Raises `ContainerError` when the pull
        fails or is still going on at `deadline`.
        """
        try:
            self._docker(
                "inspecting the image",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                self._image,
            )
            return
        except ContainerError:
            pass
        try:
            self._docker(
                f"pulling the image {self._image}",
                "pull",
                self._image,
                timeout=max(deadline - time.monotonic(), 0),
            )
        except subprocess.TimeoutExpired:
            raise ContainerError(
                f"the image {self._image} was still being pulled when the run "
                "reached its ceiling"
            ) from None

    def _remove_at_ceiling(self, name: str, reached: str) -> None:
        """Remove the container `name` by force, then raise `ContainerError` from `reached`.

        `reached` is the sentence saying where the run was at its ceiling; the error
        adds whether the container was removed.
        """
        try:
            self._remove(name)
        except ContainerError as removal_error:
            raise ContainerError(
                f"{reached} and could not be removed: {removal_error}"
            ) from None
        raise ContainerError(f"{reached} and was removed") from None

    def _remove(self, name: str) -> None:
        """Remove the container `name`, killing it first when it runs.

        Raises `ContainerError` when docker cannot, such as when no container has that name.
        """
        self._docker(f"removing the container {name}", "rm", "--force", name)

    def _docker(
        self,
        action: str,
        *arguments: str,
        timeout: float | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> str:
        """Run docker with `arguments` and return what it wrote to stdout.

        `action` names the call in error sentences. `environment` is added to the
        environment docker inherits. `timeout` bounds the call, in
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
                env=None if environment is None else {**os.environ, **environment},
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
