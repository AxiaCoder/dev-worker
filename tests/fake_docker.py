"""A stand-in for `docker`: records each call and answers what the test asked for its subcommand.

`FAKE_DOCKER_RECORD` names the file it appends its arguments to, one JSON line per call.
For a subcommand such as `run`, `FAKE_DOCKER_RUN_STDOUT`, `FAKE_DOCKER_RUN_STDERR` and
`FAKE_DOCKER_RUN_EXIT` set its answer and exit code, and `FAKE_DOCKER_RUN_SLEEP`, in
seconds, delays the answer after the call is recorded. Any of them suffixed with `_<n>`,
such as `FAKE_DOCKER_PS_STDOUT_1`, overrides it for the call of that subcommand that
`n` earlier calls of it preceded.

When `FAKE_DOCKER_STATE` names a file, the fake also plays the daemon's containers, kept
in that file as a JSON object by name. `run` and `create` make the container named by
their `--name`; `run` starts it, `start` starts an existing one, and either removes it
when its sleep ends, as `--rm` does. `kill` removes a running container and `rm --force`
any existing one. `FAKE_DOCKER_CREATION_DELAY`, in seconds, is how long after `run` is
called the daemon takes to make its container: until then the container does not exist,
yet it will. A call on a container that does not exist fails with `No such container`.
The state is written before the call is recorded.

When `FAKE_DOCKER_ENV_PROBE` names a file, `create` writes to it, as a JSON object, the
value it inherited of each variable named by an `--env` option.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

NO_SUCH_CONTAINER = "No such container"


def load(state: Path) -> dict[str, dict[str, Any]]:
    """Return the containers held in `state`, by name."""
    return json.loads(state.read_text("utf-8")) if state.exists() else {}


def save(state: Path, containers: dict[str, dict[str, Any]]) -> None:
    """Write `containers` to `state`."""
    state.write_text(json.dumps(containers), "utf-8")


def exists(container: dict[str, Any] | None) -> bool:
    """Tell whether `container` was made by the daemon and not removed yet."""
    return (
        container is not None
        and not container["removed"]
        and container["created_at"] <= time.time()
    )


def play(state: Path, arguments: list[str]) -> str | None:
    """Apply the call to the containers in `state`; return why it fails, or None."""
    containers = load(state)
    subcommand = arguments[0]
    if subcommand in ("run", "create"):
        name = arguments[arguments.index("--name") + 1]
        delay = float(os.environ.get("FAKE_DOCKER_CREATION_DELAY", "0"))
        containers[name] = {
            "created_at": time.time() + (delay if subcommand == "run" else 0),
            "started": subcommand == "run",
            "removed": False,
        }
    elif subcommand in ("start", "kill", "rm"):
        name = arguments[-1]
        container = containers.get(name)
        if container is None or not exists(container):
            return f"{NO_SUCH_CONTAINER}: {name}"
        if subcommand == "start":
            container["started"] = True
        elif subcommand == "kill" and not container["started"]:
            return f"Container {name} is not running"
        else:
            container["removed"] = True
    save(state, containers)
    return None


def finish(state: Path, arguments: list[str]) -> None:
    """Remove the container of a `run` or `start` whose sleep ended, as `--rm` does."""
    if arguments[0] not in ("run", "start"):
        return
    name = (
        arguments[arguments.index("--name") + 1]
        if arguments[0] == "run"
        else arguments[-1]
    )
    containers = load(state)
    containers[name]["removed"] = True
    save(state, containers)


def main() -> int:
    """Record the call, write the canned answer of its subcommand, and return its exit code."""
    arguments = sys.argv[1:]
    record = Path(os.environ["FAKE_DOCKER_RECORD"])
    probe = os.environ.get("FAKE_DOCKER_ENV_PROBE")
    if probe and arguments[0] == "create":
        names = [
            arguments[index + 1]
            for index, argument in enumerate(arguments)
            if argument == "--env"
        ]
        values = {name: os.environ.get(name) for name in names}
        Path(probe).write_text(json.dumps(values), "utf-8")
    state_setting = os.environ.get("FAKE_DOCKER_STATE")
    state = Path(state_setting) if state_setting else None
    failure = play(state, arguments) if state else None
    earlier = (
        [json.loads(line) for line in record.read_text("utf-8").splitlines()]
        if record.exists()
        else []
    )
    with record.open("a", encoding="utf-8") as file:
        file.write(json.dumps(arguments) + "\n")
    if failure:
        sys.stderr.buffer.write(f"Error response from daemon: {failure}\n".encode())
        return 1
    rank = sum(call[0] == arguments[0] for call in earlier)
    prefix = f"FAKE_DOCKER_{arguments[0].upper()}_"

    def setting(name: str, default: str) -> str:
        """Return the setting `name` for this call, its unranked value, or `default`."""
        return os.environ.get(
            f"{prefix}{name}_{rank}", os.environ.get(f"{prefix}{name}", default)
        )

    time.sleep(float(setting("SLEEP", "0")))
    if state:
        finish(state, arguments)
    sys.stdout.buffer.write(setting("STDOUT", "").encode("utf-8"))
    sys.stderr.buffer.write(setting("STDERR", "").encode("utf-8"))
    return int(setting("EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
