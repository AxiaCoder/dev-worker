import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from dev_worker.container import (
    Container,
    ContainerError,
    agent_command,
    mount_option,
    outcome_schema,
    ticket_prompt,
)
from dev_worker.tickets import Ticket

FAKE_DOCKER = Path(__file__).with_name("fake_docker.py")
IMAGE = "example/agent:1.0.0"
NAME = re.compile(r"dev-worker-[0-9a-f]{32}")
OWNER = "/srv/instances/demo"
PLUGIN = Path("instance", "plugin")
COMMAND = ("claude", "-p", "Ticket: ABC-1")
OUTCOMES = ("done", "blocked", "needs_human", "abandoned")
TICKET = Ticket(
    key="ABC-1",
    status="To Do",
    labels=(),
    blockers=(),
    title="Add a login page",
    description="The page asks for an email.\nThen a password.",
)


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_DOCKER_RECORD", str(path))
    return path


def answer(
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    *,
    stdout: str = "",
    stderr: str = "",
    code: int = 0,
    sleep: float = 0,
) -> None:
    prefix = f"FAKE_DOCKER_{subcommand.upper()}_"
    monkeypatch.setenv(f"{prefix}STDOUT", stdout)
    monkeypatch.setenv(f"{prefix}STDERR", stderr)
    monkeypatch.setenv(f"{prefix}EXIT", str(code))
    monkeypatch.setenv(f"{prefix}SLEEP", str(sleep))


def container(timeout: float = 60) -> Container:
    return Container(
        IMAGE, OWNER, command=[sys.executable, str(FAKE_DOCKER)], timeout=timeout
    )


def calls(record: Path) -> list[list[str]]:
    return [json.loads(line) for line in record.read_text("utf-8").splitlines()]


def test_check_access_asks_the_daemon_for_its_version(record: Path) -> None:
    container().check_access()

    assert calls(record) == [["version", "--format", "{{.Server.Version}}"]]


def test_a_daemon_that_cannot_be_reached_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "version", stderr="Cannot connect to the daemon\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().check_access()

    assert str(raised.value) == (
        "docker failed while reaching the Docker daemon: Cannot connect to the daemon"
    )


def test_docker_that_cannot_be_started_raises(tmp_path: Path) -> None:
    absent = Container(IMAGE, OWNER, command=[str(tmp_path / "no-docker")])

    with pytest.raises(ContainerError) as raised:
        absent.check_access()

    assert str(raised.value).startswith("docker could not be started: ")


def test_docker_that_does_not_answer_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "version", sleep=5)

    with pytest.raises(ContainerError) as raised:
        container(timeout=0.5).check_access()

    assert str(raised.value) == (
        "docker did not answer within 0.5 seconds while reaching the Docker daemon"
    )


INSPECT = ["image", "inspect", "--format", "{{.Id}}", IMAGE]


def created(record: Path) -> str:
    """Return the name given to the one `docker create` call `record` holds."""
    (create,) = [call for call in calls(record) if call[0] == "create"]
    return create[create.index("--name") + 1]


def test_run_creates_one_named_disposable_container_of_its_owner_with_only_the_worktree_and_the_plugin_mounted_then_starts_it(
    record: Path, tmp_path: Path
) -> None:
    worktree = tmp_path / "worktree"

    container().run(worktree, 60, PLUGIN, COMMAND, {})

    _, create, start = calls(record)
    assert create[:5] == ["create", "--pull", "never", "--rm", "--name"]
    name = create[5]
    assert NAME.fullmatch(name)
    assert create[6:] == [
        "--label",
        f"dev-worker.instance={OWNER}",
        "--mount",
        f"type=bind,source={worktree},target=/workspace",
        "--mount",
        f"type=bind,source={PLUGIN},target=/plugin,readonly",
        "--workdir",
        "/workspace",
        IMAGE,
        *COMMAND,
    ]
    assert start == ["start", "--attach", name]


def test_run_passes_each_variable_by_name_only_and_its_value_through_docker_environment(
    record: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "environment.json"
    monkeypatch.setenv("FAKE_DOCKER_ENV_PROBE", str(probe))
    environment = {"GH_TOKEN": "s3cret-gh", "TWG_SITE": "example.atlassian.net"}

    container().run(tmp_path, 60, PLUGIN, COMMAND, environment)

    _, create, _ = calls(record)
    image_at = create.index(IMAGE)
    assert create[image_at - 4 : image_at] == [
        "--env",
        "GH_TOKEN",
        "--env",
        "TWG_SITE",
    ]
    assert not any("s3cret-gh" in argument for argument in create)
    assert json.loads(probe.read_text("utf-8")) == environment


def test_ticket_prompt_holds_the_key_title_and_description() -> None:
    assert ticket_prompt(TICKET) == (
        "Ticket: ABC-1\n"
        "Title: Add a login page\n"
        "\n"
        "Description:\n"
        "The page asks for an email.\nThen a password.\n"
    )


def test_outcome_schema_requires_one_of_the_outcomes_and_a_message() -> None:
    assert outcome_schema(OUTCOMES) == {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": list(OUTCOMES)},
            "message": {"type": "string"},
        },
        "required": ["outcome", "message"],
        "additionalProperties": False,
    }


def test_agent_command_runs_claude_headless_with_the_rules_plugin_turns_and_schema() -> (
    None
):
    command = agent_command(TICKET, "# Worker\nRules.\n", 40, OUTCOMES)

    assert command == (
        "claude",
        "-p",
        ticket_prompt(TICKET),
        "--append-system-prompt",
        "# Worker\nRules.\n",
        "--plugin-dir",
        "/plugin",
        "--max-turns",
        "40",
        "--json-schema",
        json.dumps(outcome_schema(OUTCOMES)),
        "--output-format",
        "json",
    )


def test_an_image_the_daemon_has_is_not_pulled(record: Path, tmp_path: Path) -> None:
    container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert calls(record)[0] == INSPECT
    assert [call[0] for call in calls(record)] == ["image", "create", "start"]


def test_an_image_the_daemon_lacks_is_pulled_before_the_container_is_created(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "image", stderr="No such image\n", code=1)

    container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    inspect, pull, create, start = calls(record)
    assert inspect == INSPECT
    assert pull == ["pull", IMAGE]
    assert create[0] == "create"
    assert start[0] == "start"


def test_an_image_that_cannot_be_pulled_raises_naming_why_and_creates_no_container(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "image", stderr="No such image\n", code=1)
    answer(monkeypatch, "pull", stderr="manifest unknown\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert str(raised.value) == (
        f"docker failed while pulling the image {IMAGE}: manifest unknown"
    )
    assert [call[0] for call in calls(record)] == ["image", "pull"]


def test_an_image_still_being_pulled_when_the_time_is_up_raises_and_creates_no_container(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "image", stderr="No such image\n", code=1)
    answer(monkeypatch, "pull", sleep=10)
    began = time.monotonic()

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5, PLUGIN, COMMAND, {})

    assert time.monotonic() - began < 5
    assert str(raised.value) == (
        f"the image {IMAGE} was still being pulled when the run reached its ceiling"
    )
    assert [call[0] for call in calls(record)] == ["image", "pull"]


def test_an_interrupted_pull_raises_again_without_removing_any_container(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "image", stderr="No such image\n", code=1)
    interrupt(monkeypatch, "pull")

    with pytest.raises(KeyboardInterrupt):
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert [call[0] for call in calls(record)] == ["image"]


def test_each_run_names_its_container_anew(record: Path, tmp_path: Path) -> None:
    container().run(tmp_path, 60, PLUGIN, COMMAND, {})
    first = created(record)
    record.unlink()
    container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert first != created(record)


def test_a_worktree_path_holding_a_comma_or_a_quote_is_mounted_whole() -> None:
    assert mount_option(Path("a,b")) == 'type=bind,"source=a,b",target=/workspace'
    assert mount_option(Path('a"b')) == 'type=bind,"source=a""b",target=/workspace'
    assert (
        mount_option(Path("a,b"), "/plugin", readonly=True)
        == 'type=bind,"source=a,b",target=/plugin,readonly'
    )


def test_a_container_that_cannot_be_created_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "create", stderr="Unable to find image\n", code=125)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert str(raised.value) == (
        "docker failed while creating the container: Unable to find image"
    )
    assert [call[0] for call in calls(record)] == ["image", "create"]


def test_a_container_that_cannot_be_started_or_exits_nonzero_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "start", stderr="exec failed\n", code=126)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert str(raised.value) == (
        "docker failed while running the container: exec failed"
    )
    assert [call[0] for call in calls(record)] == ["image", "create", "start"]


def test_a_container_still_running_when_the_time_is_up_is_removed_by_its_name(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "start", sleep=5)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 1.5, PLUGIN, COMMAND, {})

    name = created(record)
    assert [call[0] for call in calls(record)] == ["image", "create", "start", "rm"]
    assert calls(record)[-1] == ["rm", "--force", name]
    assert str(raised.value) == (
        f"the container {name} was still running when the run reached its ceiling "
        "and was removed"
    )


def test_the_time_spent_creating_the_container_counts_against_the_ceiling(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "create", sleep=2)
    answer(monkeypatch, "start", sleep=10)
    began = time.monotonic()

    with pytest.raises(ContainerError):
        container().run(tmp_path, 4, PLUGIN, COMMAND, {})

    assert time.monotonic() - began < 5.5
    assert "start" in [call[0] for call in calls(record)]


def test_a_ceiling_reached_while_the_container_is_created_removes_it_unstarted(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "create", sleep=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5, PLUGIN, COMMAND, {})

    name = created(record)
    assert [call[0] for call in calls(record)] == ["image", "create", "rm"]
    assert calls(record)[-1] == ["rm", "--force", name]
    assert str(raised.value) == (
        f"the run reached its ceiling before the container {name} was started "
        "and was removed"
    )


def test_a_container_that_cannot_be_removed_when_the_ceiling_comes_before_its_start_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "create", sleep=1)
    answer(monkeypatch, "rm", stderr="permission denied\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5, PLUGIN, COMMAND, {})

    name = created(record)
    assert str(raised.value) == (
        f"the run reached its ceiling before the container {name} was started "
        f"and could not be removed: docker failed while removing the container {name}: "
        "permission denied"
    )


def test_a_container_that_cannot_be_removed_when_the_time_is_up_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "start", sleep=5)
    answer(monkeypatch, "rm", stderr="permission denied\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 1.5, PLUGIN, COMMAND, {})

    name = created(record)
    assert str(raised.value) == (
        f"the container {name} was still running when the run reached its ceiling "
        f"and could not be removed: docker failed while removing the container {name}: "
        "permission denied"
    )


def interrupt(monkeypatch: pytest.MonkeyPatch, subcommand: str) -> None:
    """Make `docker <subcommand>` raise `KeyboardInterrupt`; every other docker call runs as usual."""
    real_run = subprocess.run

    def run(arguments: list[str], **options: Any) -> Any:
        if arguments[2] == subcommand:
            raise KeyboardInterrupt
        return real_run(arguments, **options)

    monkeypatch.setattr(subprocess, "run", run)


@pytest.mark.parametrize("subcommand", ["create", "start"])
def test_an_interrupted_run_removes_its_container_and_raises_again(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, subcommand: str
) -> None:
    interrupt(monkeypatch, subcommand)

    with pytest.raises(KeyboardInterrupt) as raised:
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    remove = calls(record)[-1]
    assert remove[:2] == ["rm", "--force"]
    assert NAME.fullmatch(remove[2])
    assert not hasattr(raised.value, "__notes__")


def test_an_interrupted_run_whose_container_cannot_be_removed_notes_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interrupt(monkeypatch, "start")
    answer(monkeypatch, "rm", stderr="permission denied\n", code=1)

    with pytest.raises(KeyboardInterrupt) as raised:
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    name = created(record)
    note = (
        f"the container {name} could not be removed: docker failed while removing "
        f"the container {name}: permission denied"
    )
    assert raised.value.__notes__ == [note]


PS = ["ps", "--filter", f"label=dev-worker.instance={OWNER}", "--format", "{{.Names}}"]


def test_running_lists_the_containers_of_the_owner_by_name(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "ps", stdout="dev-worker-a\ndev-worker-b\n")

    assert container().running() == ["dev-worker-a", "dev-worker-b"]
    assert calls(record) == [PS]


def test_check_none_running_passes_when_no_container_of_the_owner_runs(
    record: Path,
) -> None:
    container().check_none_running()

    assert calls(record) == [PS]


@pytest.mark.parametrize(
    ("listed", "message"),
    [
        (
            "dev-worker-a\n",
            "the container dev-worker-a of the instance is still running",
        ),
        (
            "dev-worker-a\ndev-worker-b\n",
            "the containers dev-worker-a, dev-worker-b of the instance are still running",
        ),
    ],
)
def test_check_none_running_raises_naming_the_containers_still_running(
    record: Path, monkeypatch: pytest.MonkeyPatch, listed: str, message: str
) -> None:
    answer(monkeypatch, "ps", stdout=listed)

    with pytest.raises(ContainerError) as raised:
        container().check_none_running()

    assert str(raised.value) == message


def test_containers_that_cannot_be_listed_raise_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "ps", stderr="Cannot connect to the daemon\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().check_none_running()

    assert str(raised.value) == (
        "docker failed while listing the containers of the instance: "
        "Cannot connect to the daemon"
    )


def test_kill_running_kills_each_container_of_the_owner_still_running(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "ps", stdout="dev-worker-a\ndev-worker-b\n")

    container().kill_running()

    assert calls(record) == [PS, ["kill", "dev-worker-a"], ["kill", "dev-worker-b"]]


def test_kill_running_raises_on_the_first_container_it_cannot_kill(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "ps", stdout="dev-worker-a\ndev-worker-b\n")
    answer(monkeypatch, "kill", stderr="permission denied\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().kill_running()

    assert str(raised.value) == (
        "docker failed while killing the container dev-worker-a: permission denied"
    )
    assert calls(record) == [PS, ["kill", "dev-worker-a"]]


def test_kill_running_kills_nothing_when_no_container_of_the_owner_runs(
    record: Path,
) -> None:
    container().kill_running()

    assert calls(record) == [PS]


@pytest.fixture
def daemon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the fake docker keep its containers, made 30 seconds after `run` asks."""
    path = tmp_path / "containers.json"
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(path))
    monkeypatch.setenv("FAKE_DOCKER_CREATION_DELAY", "30")
    return path


def left_over(daemon: Path) -> list[str]:
    """Name the containers the fake daemon made, or will make, and did not remove."""
    containers = json.loads(daemon.read_text("utf-8"))
    return [name for name, state in containers.items() if not state["removed"]]


def test_a_ceiling_reached_before_the_daemon_made_the_container_leaves_no_container(
    record: Path, daemon: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=10)
    answer(monkeypatch, "start", sleep=10)

    with pytest.raises(ContainerError):
        container().run(tmp_path, 0.5, PLUGIN, COMMAND, {})

    assert left_over(daemon) == []


def interrupt_once_the_container_is_asked_to_start(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make `docker run` or `docker start` raise `KeyboardInterrupt` once it reached the fake.

    The docker client is stopped with the interruption; every other docker call runs as usual.
    """
    real_run = subprocess.run

    def run(arguments: list[str], **options: Any) -> Any:
        if arguments[2] not in ("run", "start"):
            return real_run(arguments, **options)
        before = len(calls(record)) if record.exists() else 0
        client = subprocess.Popen(
            arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        while not record.exists() or len(calls(record)) == before:
            time.sleep(0.05)
        client.kill()
        client.wait()
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", run)


def test_an_interruption_before_the_daemon_made_the_container_leaves_no_container(
    record: Path, daemon: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=10)
    answer(monkeypatch, "start", sleep=10)
    interrupt_once_the_container_is_asked_to_start(record, monkeypatch)

    with pytest.raises(KeyboardInterrupt):
        container().run(tmp_path, 60, PLUGIN, COMMAND, {})

    assert left_over(daemon) == []
