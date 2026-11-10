import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dev_worker.container import Container, ContainerError, mount_option

FAKE_DOCKER = Path(__file__).with_name("fake_docker.py")
IMAGE = "example/agent:1.0.0"
NAME = re.compile(r"dev-worker-[0-9a-f]{32}")
OWNER = "/srv/instances/demo"


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


def test_run_starts_one_named_disposable_container_of_its_owner_with_only_the_worktree_mounted(
    record: Path, tmp_path: Path
) -> None:
    worktree = tmp_path / "worktree"

    container().run(worktree, 60)

    [call] = calls(record)
    assert call[:3] == ["run", "--rm", "--name"]
    assert NAME.fullmatch(call[3])
    assert call[4:] == [
        "--label",
        f"dev-worker.instance={OWNER}",
        "--mount",
        f"type=bind,source={worktree},target=/workspace",
        "--workdir",
        "/workspace",
        IMAGE,
        "true",
    ]


def test_each_run_names_its_container_anew(record: Path, tmp_path: Path) -> None:
    container().run(tmp_path, 60)
    container().run(tmp_path, 60)

    first, second = calls(record)
    assert first[3] != second[3]


def test_a_worktree_path_holding_a_comma_or_a_quote_is_mounted_whole() -> None:
    assert mount_option(Path("a,b")) == 'type=bind,"source=a,b",target=/workspace'
    assert mount_option(Path('a"b')) == 'type=bind,"source=a""b",target=/workspace'


def test_a_container_that_cannot_be_started_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", stderr="Unable to find image\n", code=125)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 60)

    assert str(raised.value) == (
        "docker failed while running the container: Unable to find image"
    )
    assert [call[0] for call in calls(record)] == ["run"]


def test_a_container_still_running_when_the_time_is_up_is_killed_by_its_name(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=5)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5)

    run, kill = calls(record)
    name = run[3]
    assert kill == ["kill", name]
    assert str(raised.value) == (
        f"the container {name} was still running when the run reached its ceiling "
        "and was killed"
    )


def test_a_container_that_cannot_be_killed_when_the_time_is_up_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=5)
    answer(monkeypatch, "kill", stderr="No such container\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5)

    name = calls(record)[0][3]
    assert str(raised.value) == (
        f"the container {name} was still running when the run reached its ceiling "
        f"and could not be killed: docker failed while killing the container {name}: "
        "No such container"
    )


def interrupt_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `docker run` raise `KeyboardInterrupt`; every other docker call runs as usual."""
    real_run = subprocess.run

    def run(arguments: list[str], **options: Any) -> Any:
        if arguments[2] == "run":
            raise KeyboardInterrupt
        return real_run(arguments, **options)

    monkeypatch.setattr(subprocess, "run", run)


def test_an_interrupted_run_kills_its_container_and_raises_again(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interrupt_the_run(monkeypatch)

    with pytest.raises(KeyboardInterrupt) as raised:
        container().run(tmp_path, 60)

    [kill] = calls(record)
    assert kill[0] == "kill"
    assert NAME.fullmatch(kill[1])
    assert not hasattr(raised.value, "__notes__")


def test_an_interrupted_run_whose_container_cannot_be_killed_notes_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interrupt_the_run(monkeypatch)
    answer(monkeypatch, "kill", stderr="No such container\n", code=1)

    with pytest.raises(KeyboardInterrupt) as raised:
        container().run(tmp_path, 60)

    [[_, name]] = calls(record)
    note = (
        f"the container {name} could not be killed: docker failed while killing "
        f"the container {name}: No such container"
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
