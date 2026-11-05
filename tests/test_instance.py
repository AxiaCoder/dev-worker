from pathlib import Path

import pytest
from pydantic import ValidationError

from dev_worker.header import Header
from dev_worker.instance import Instance, InstanceError, load_instance

HEADER = Header.model_validate(
    {
        "status_keys": ["ready", "in_progress", "blocked", "done"],
        "outcomes": {
            "done": "done",
            "blocked": "blocked",
            "needs_human": "blocked",
            "abandoned": "ready",
        },
        "ceilings": {"run_timeout_seconds": 1800, "max_turns": 60},
    }
)

VALID = """\
tracker:
  type: jira
  project: ABC
  token_env: JIRA_TOKEN
repository:
  type: github
  name: example/project
  token_env: GITHUB_TOKEN
statuses:
  ready: To Do
  in_progress: In Progress
  blocked: Blocked
  done: Done
limits:
  max_turns: 40
"""


def write(tmp_path: Path, text: str) -> Path:
    """Write `text` as `instance.yaml` under `tmp_path`, UTF-8, and return its path."""
    path = tmp_path / "instance.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path: Path, text: str) -> Instance:
    """Write `text` as an instance file and load it against `HEADER`."""
    return load_instance(write(tmp_path, text), HEADER)


def test_valid_instance_loads(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID)

    assert instance.tracker.project == "ABC"
    assert instance.tracker.token_env == "JIRA_TOKEN"
    assert instance.repository.name == "example/project"
    assert instance.repository.token_env == "GITHUB_TOKEN"
    assert instance.statuses == {
        "ready": "To Do",
        "in_progress": "In Progress",
        "blocked": "Blocked",
        "done": "Done",
    }


def test_unset_limit_comes_from_the_header(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID)

    assert instance.limits == {"run_timeout_seconds": 1800, "max_turns": 40}


def test_missing_limits_come_from_the_header(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID.split("limits:")[0])

    assert instance.limits == {"run_timeout_seconds": 1800, "max_turns": 60}


def test_limit_equal_to_its_ceiling_is_accepted(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID.replace("max_turns: 40", "max_turns: 60"))

    assert instance.limits["max_turns"] == 60


def test_looser_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError,
        match="limits.max_turns: 80 is looser than the header ceiling 60",
    ):
        load(tmp_path, VALID.replace("max_turns: 40", "max_turns: 80"))


def test_unknown_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError,
        match="limits.max_cost: unknown limit, the header sets run_timeout_seconds, max_turns",
    ):
        load(tmp_path, VALID + "  max_cost: 5\n")


@pytest.mark.parametrize("value", ["0", "-5", "1.5", "'40'", "true"])
def test_limit_not_a_positive_int_is_refused(tmp_path: Path, value: str) -> None:
    with pytest.raises(InstanceError, match="invalid instance file: limits.max_turns"):
        load(tmp_path, VALID.replace("max_turns: 40", f"max_turns: {value}"))


def test_unknown_status_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError,
        match="statuses.shelved: unknown status key, the header declares ready, "
        "in_progress, blocked, done",
    ):
        load(tmp_path, VALID.replace("  done: Done\n", "  done: Done\n  shelved: X\n"))


def test_missing_status_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="statuses.blocked: missing"):
        load(tmp_path, VALID.replace("  blocked: Blocked\n", ""))


def test_empty_status_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="statuses.blocked"):
        load(tmp_path, VALID.replace("blocked: Blocked", "blocked: ''"))


def test_every_problem_is_reported(tmp_path: Path) -> None:
    text = VALID.replace("  blocked: Blocked\n", "").replace(
        "max_turns: 40", "max_turns: 80"
    )

    with pytest.raises(InstanceError) as caught:
        load(tmp_path, text)

    assert "statuses.blocked" in str(caught.value)
    assert "limits.max_turns" in str(caught.value)


@pytest.mark.parametrize(
    ("section", "variable"), [("tracker", "JIRA_TOKEN"), ("repository", "GITHUB_TOKEN")]
)
@pytest.mark.parametrize(
    "value", ["ghp_abcdef0123456789", "jira_token", "1TOKEN", "JIRA-TOKEN", "''"]
)
def test_token_env_not_a_variable_name_is_refused_without_echo(
    tmp_path: Path, section: str, variable: str, value: str
) -> None:
    text = VALID.replace(f"token_env: {variable}", f"token_env: {value}")

    with pytest.raises(
        InstanceError, match=f"{section}.token_env: .*name of an environment variable"
    ) as caught:
        load(tmp_path, text)

    assert value not in str(caught.value)
    assert "(got" not in str(caught.value)


@pytest.mark.parametrize(
    ("old", "new", "location"),
    [
        ("type: jira", "type: linear", "tracker.type"),
        ("type: github", "type: gitlab", "repository.type"),
        ("project: ABC", "project: ''", "tracker.project"),
        ("name: example/project", "name: ''", "repository.name"),
    ],
)
def test_invalid_field_is_refused(
    tmp_path: Path, old: str, new: str, location: str
) -> None:
    with pytest.raises(InstanceError, match=f"invalid instance file: {location}"):
        load(tmp_path, VALID.replace(old, new))


@pytest.mark.parametrize(
    ("text", "location"),
    [
        (VALID + "owner: someone\n", "owner: Extra inputs"),
        (VALID.replace("  project: ABC\n", "  project: ABC\n  site: x\n"), "site"),
        (VALID.replace("  project: ABC\n", ""), "tracker.project: Field required"),
        (VALID.split("statuses:")[0], "statuses: Field required"),
    ],
)
def test_extra_or_missing_key_is_refused(
    tmp_path: Path, text: str, location: str
) -> None:
    with pytest.raises(InstanceError, match=location):
        load(tmp_path, text)


def test_instance_not_a_mapping_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError, match=r"invalid instance file: instance: .*\(got \['jira'\]\)"
    ):
        load(tmp_path, "- jira\n")


def test_invalid_yaml_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="not valid YAML"):
        load(tmp_path, "tracker: [jira\n")


def test_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="duplicate key 'max_turns'"):
        load(tmp_path, VALID + "  max_turns: 10\n")


def test_byte_order_mark_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "instance.yaml"
    path.write_bytes(VALID.encode("utf-8-sig"))

    assert load_instance(path, HEADER).limits["max_turns"] == 40


def test_loaded_instance_is_immutable(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID)

    with pytest.raises(ValidationError, match="frozen"):
        instance.tracker.project = "XYZ"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        instance.limits["max_turns"] = 10**9  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        instance.statuses["done"] = "X"  # pyright: ignore[reportIndexIssue]
