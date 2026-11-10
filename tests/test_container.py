import json
import re
import sys
from pathlib import Path

import pytest

from dev_worker.container import Container, ContainerError, mount_option

FAKE_DOCKER = Path(__file__).with_name("fake_docker.py")
IMAGE = "example/agent:1.0.0"
NAME = re.compile(r"dev-worker-[0-9a-f]{32}")


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_DOCKER_RECORD", str(path))
    return path


def answer(
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    *,
    stderr: str = "",
    code: int = 0,
    sleep: float = 0,
) -> None:
    prefix = f"FAKE_DOCKER_{subcommand.upper()}_"
    monkeypatch.setenv(f"{prefix}STDERR", stderr)
    monkeypatch.setenv(f"{prefix}EXIT", str(code))
    monkeypatch.setenv(f"{prefix}SLEEP", str(sleep))


def container(timeout: float = 60) -> Container:
    return Container(IMAGE, command=[sys.executable, str(FAKE_DOCKER)], timeout=timeout)


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
    absent = Container(IMAGE, command=[str(tmp_path / "no-docker")])

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


def test_run_starts_one_named_disposable_container_with_only_the_worktree_mounted(
    record: Path, tmp_path: Path
) -> None:
    worktree = tmp_path / "worktree"

    container().run(worktree, 60)

    [call] = calls(record)
    assert call[:3] == ["run", "--rm", "--name"]
    assert NAME.fullmatch(call[3])
    assert call[4:] == [
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


def test_a_container_still_running_at_the_ceiling_is_killed_by_its_name(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=5)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5)

    run, kill = calls(record)
    name = run[3]
    assert kill == ["kill", name]
    assert str(raised.value) == (
        f"the container {name} was still running at the run ceiling "
        "of 0.5 seconds and was killed"
    )


def test_a_container_that_cannot_be_killed_at_the_ceiling_raises_naming_why(
    record: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answer(monkeypatch, "run", sleep=5)
    answer(monkeypatch, "kill", stderr="No such container\n", code=1)

    with pytest.raises(ContainerError) as raised:
        container().run(tmp_path, 0.5)

    name = calls(record)[0][3]
    assert str(raised.value) == (
        f"the container {name} was still running at the run ceiling of 0.5 seconds "
        f"and could not be killed: docker failed while killing the container {name}: "
        "No such container"
    )
