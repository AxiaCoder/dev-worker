import traceback
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
  site: example.atlassian.net
  user: bot@example.com
  project: ABC
  token_env: JIRA_TOKEN
repository:
  type: github
  name: example/project
  token_env: GITHUB_TOKEN
branch: agent/{key}
container:
  image: python:3.12-slim
git:
  name: Worker
  email: worker@users.noreply.github.com
hold_merge_label: hold-merge
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

    assert instance.tracker.site == "example.atlassian.net"
    assert instance.tracker.user == "bot@example.com"
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
    assert instance.hold_merge_label == "hold-merge"


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


def test_status_name_shared_by_two_keys_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError,
        match="statuses: ready, blocked share the real status 'To Do', "
        "each status key needs its own",
    ):
        load(tmp_path, VALID.replace("blocked: Blocked", "blocked: To Do"))


def test_status_name_shared_by_three_keys_is_refused(tmp_path: Path) -> None:
    text = VALID.replace("blocked: Blocked", "blocked: Done").replace(
        "in_progress: In Progress", "in_progress: Done"
    )

    with pytest.raises(
        InstanceError,
        match="statuses: in_progress, blocked, done share the real status 'Done'",
    ):
        load(tmp_path, text)


def test_each_shared_status_name_is_reported(tmp_path: Path) -> None:
    text = VALID.replace("blocked: Blocked", "blocked: To Do").replace(
        "in_progress: In Progress", "in_progress: Done"
    )

    with pytest.raises(InstanceError) as caught:
        load(tmp_path, text)

    message = str(caught.value)
    assert "statuses: ready, blocked share the real status 'To Do'" in message
    assert "statuses: in_progress, done share the real status 'Done'" in message


def test_keys_sharing_a_status_name_are_listed_in_header_order(tmp_path: Path) -> None:
    statuses = (
        "statuses:\n"
        "  done: Done\n"
        "  blocked: To Do\n"
        "  in_progress: In Progress\n"
        "  ready: To Do\n"
    )
    text = VALID.replace(
        VALID[VALID.index("statuses:") : VALID.index("limits:")], statuses
    )

    with pytest.raises(
        InstanceError, match="statuses: ready, blocked share the real status 'To Do'"
    ):
        load(tmp_path, text)


def test_unknown_status_key_does_not_share_a_status_name(tmp_path: Path) -> None:
    text = VALID.replace("  done: Done\n", "  done: Done\n  shelved: To Do\n")

    with pytest.raises(InstanceError, match="statuses.shelved: unknown") as caught:
        load(tmp_path, text)

    assert "share the real status" not in str(caught.value)


def test_status_names_differing_in_case_or_spacing_are_distinct(
    tmp_path: Path,
) -> None:
    text = VALID.replace("blocked: Blocked", "blocked: to do").replace(
        "done: Done", 'done: "To Do "'
    )

    instance = load(tmp_path, text)

    assert instance.statuses["ready"] == "To Do"
    assert instance.statuses["blocked"] == "to do"
    assert instance.statuses["done"] == "To Do "


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
        (
            "site: example.atlassian.net",
            "site: ''",
            "tracker.site: .*must not be blank",
        ),
        ("user: bot@example.com", "user: ''", "tracker.user: .*must not be blank"),
        (
            "hold_merge_label: hold-merge",
            "hold_merge_label: ''",
            "hold_merge_label: .*must not be blank",
        ),
        ("name: example/project", "name: ''", "repository.name"),
    ],
)
def test_invalid_field_is_refused(
    tmp_path: Path, old: str, new: str, location: str
) -> None:
    with pytest.raises(InstanceError, match=f"invalid instance file: {location}"):
        load(tmp_path, VALID.replace(old, new))


@pytest.mark.parametrize(
    ("old", "location"),
    [
        ("site: example.atlassian.net", "tracker.site"),
        ("user: bot@example.com", "tracker.user"),
        ("project: ABC", "tracker.project"),
        ("hold_merge_label: hold-merge", "hold_merge_label"),
        ("name: example/project", "repository.name"),
        ("blocked: Blocked", "statuses.blocked"),
    ],
)
@pytest.mark.parametrize("blank", ['" "', '"   "', '"\\t"', '" \\t "'])
def test_blank_name_is_refused(
    tmp_path: Path, old: str, location: str, blank: str
) -> None:
    key = old.split(":")[0]
    text = VALID.replace(old, f"{key}: {blank}")

    with pytest.raises(
        InstanceError, match=f"invalid instance file: {location}: .*blank"
    ) as caught:
        load(tmp_path, text)

    assert "(got" not in str(caught.value)
    assert "input_value" not in str(caught.value)


def test_status_name_with_surrounding_blanks_is_kept_as_written(
    tmp_path: Path,
) -> None:
    instance = load(tmp_path, VALID.replace("ready: To Do", 'ready: "  To Do "'))

    assert instance.statuses["ready"] == "  To Do "


@pytest.mark.parametrize(
    ("text", "location"),
    [
        (VALID + "owner: someone\n", "owner: Extra inputs"),
        (VALID.replace("  project: ABC\n", "  project: ABC\n  region: x\n"), "region"),
        (VALID.replace("  project: ABC\n", ""), "tracker.project: Field required"),
        (
            VALID.replace("  site: example.atlassian.net\n", ""),
            "tracker.site: Field required",
        ),
        (
            VALID.replace("  user: bot@example.com\n", ""),
            "tracker.user: Field required",
        ),
        (
            VALID.replace("hold_merge_label: hold-merge\n", ""),
            "hold_merge_label: Field required",
        ),
        (VALID.split("statuses:")[0], "statuses: Field required"),
    ],
)
def test_extra_or_missing_key_is_refused(
    tmp_path: Path, text: str, location: str
) -> None:
    with pytest.raises(InstanceError, match=location):
        load(tmp_path, text)


def test_instance_not_a_mapping_is_refused_without_echo(tmp_path: Path) -> None:
    with pytest.raises(
        InstanceError, match=r"invalid instance file: instance: "
    ) as caught:
        load(tmp_path, f"- {SECRET}\n")

    assert SECRET not in str(caught.value)
    assert "(got" not in str(caught.value)


def test_invalid_yaml_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="not valid YAML"):
        load(tmp_path, "tracker: [jira\n")


SECRET = "ghp_secret0123456789"


@pytest.mark.parametrize(
    ("text", "location"),
    [
        (
            VALID.replace("  token_env: JIRA_TOKEN\n", f"  token: {SECRET}\n"),
            "tracker.token",
        ),
        (
            VALID.replace("  token_env: GITHUB_TOKEN\n", f"  token: {SECRET}\n"),
            "repository.token",
        ),
        (VALID + f"token: {SECRET}\n", "token"),
        (VALID.replace("type: jira", f"type: {SECRET}"), "tracker.type"),
        (VALID.replace("type: github", f"type: {SECRET}"), "repository.type"),
        (VALID.replace("max_turns: 40", f"max_turns: {SECRET}"), "limits.max_turns"),
        (VALID.replace("blocked: Blocked", f"blocked: [{SECRET}]"), "statuses.blocked"),
    ],
)
def test_invalid_value_is_refused_without_echo(
    tmp_path: Path, text: str, location: str
) -> None:
    with pytest.raises(
        InstanceError, match=f"invalid instance file: .*{location}"
    ) as caught:
        load(tmp_path, text)

    assert SECRET not in str(caught.value)
    assert "(got" not in str(caught.value)


def test_invalid_yaml_is_refused_without_echo(tmp_path: Path) -> None:
    text = f'tracker:\n  token_env: "{SECRET}\n'

    with pytest.raises(InstanceError, match="not valid YAML") as caught:
        load(tmp_path, text)

    assert SECRET not in str(caught.value)
    assert "ghp_" not in str(caught.value)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f'tracker:\n  token_env: "{SECRET}\n', id="malformed-yaml"),
        pytest.param(
            VALID.replace("  token_env: JIRA_TOKEN\n", f"  token: {SECRET}\n"),
            id="extra-field",
        ),
        pytest.param(VALID.replace("type: jira", f"type: {SECRET}"), id="bad-literal"),
    ],
)
def test_refusal_traceback_does_not_carry_the_pasted_value(
    tmp_path: Path, text: str
) -> None:
    with pytest.raises(InstanceError) as caught:
        load(tmp_path, text)

    error = caught.value
    assert error.__cause__ is None
    assert SECRET not in "".join(traceback.format_exception(error))


def test_invalid_yaml_reports_each_reason_with_line_and_column(tmp_path: Path) -> None:
    text = f'tracker:\n  token_env: "{SECRET}\n'

    with pytest.raises(InstanceError) as caught:
        load(tmp_path, text)

    assert str(caught.value) == (
        "instance file is not valid YAML"
        ": while scanning a quoted scalar (line 2, column 14)"
        ": found unexpected end of stream (line 3, column 1)"
    )


def test_unreadable_character_is_refused_without_echo(tmp_path: Path) -> None:
    with pytest.raises(InstanceError, match="not valid YAML") as caught:
        load(tmp_path, "a: \x07\n")

    message = str(caught.value)
    assert message == "instance file is not valid YAML"
    assert "x07" not in message
    assert "#x0007" not in message


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


def test_instance_not_in_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "instance.yaml"
    path.write_bytes(VALID.replace("ABC", "Café").encode("latin-1"))

    with pytest.raises(InstanceError):
        load_instance(path, HEADER)


def test_instance_nested_too_deeply_is_refused(tmp_path: Path) -> None:
    with pytest.raises(InstanceError):
        load(tmp_path, "tracker: " + "[" * 5000 + "\n")


def test_run_keys_are_exposed(tmp_path: Path) -> None:
    instance = load(tmp_path, VALID)

    assert instance.branch == "agent/{key}"
    assert instance.container.image == "python:3.12-slim"
    assert instance.git.name == "Worker"
    assert instance.git.email == "worker@users.noreply.github.com"


@pytest.mark.parametrize("branch", ["agent/{key}", "{key}", "feat/{key}-run"])
def test_branch_containing_the_key_is_accepted(tmp_path: Path, branch: str) -> None:
    instance = load(
        tmp_path, VALID.replace("branch: agent/{key}", f"branch: '{branch}'")
    )

    assert instance.branch == branch


@pytest.mark.parametrize(
    "branch",
    [
        "agent/main",
        "agent/{KEY}",
        "agent/key",
        "",
        "agent/{key}/{other}",
        "agent/{{key}}",
        "agent/{key",
        "agent/}{key}",
    ],
)
def test_branch_without_the_key_is_refused_with_its_reason(
    tmp_path: Path, branch: str
) -> None:
    with pytest.raises(
        InstanceError,
        match=r"invalid instance file: branch: .*must contain \{key\}, which the "
        r"ticket key replaces, and no other brace, e\.g\. agent/\{key\}",
    ):
        load(tmp_path, VALID.replace("branch: agent/{key}", f"branch: '{branch}'"))


DIGEST = "sha256:" + "0123456789abcdef" * 4


@pytest.mark.parametrize(
    "image",
    [
        "python:3.12-slim",
        "python:3",
        "ghcr.io/example-org/app:1.0.0",
        "localhost:5000/app:1.0",
        f"python@{DIGEST}",
        f"ghcr.io/example-org/app@{DIGEST}",
        f"python:latest@{DIGEST}",
        f"registry.example.com:443/a/b@{DIGEST}",
        "a__b:1",
        "a--b:1",
        "a_b.c-d:1",
        "localhost/app:1",
        "my-registry.example.com/app:1",
        "myregistry:5000/app:1",
        "127.0.0.1:5000/app:1",
        "My.Registry.example.com/app:1",
        "a-b.example.com/team/app:1",
        "localhostx/app:1",
        "a/b/c/d:1",
    ],
)
def test_pinned_image_is_accepted(tmp_path: Path, image: str) -> None:
    text = VALID.replace("image: python:3.12-slim", f"image: '{image}'")

    assert load(tmp_path, text).container.image == image


@pytest.mark.parametrize(
    "image",
    [
        "python",
        "python:latest",
        "ghcr.io/example-org/app",
        "ghcr.io/example-org/app:latest",
        "localhost:5000/app",
        "registry:5000/app",
        "localhost/app:latest",
    ],
)
def test_unpinned_image_is_refused_with_its_reason(tmp_path: Path, image: str) -> None:
    text = VALID.replace("image: python:3.12-slim", f"image: '{image}'")

    with pytest.raises(
        InstanceError,
        match=r"invalid instance file: container\.image: .*must pin a version, with a "
        r"tag other than latest or a digest, e\.g\. python:3\.12-slim",
    ):
        load(tmp_path, text)


@pytest.mark.parametrize(
    "image",
    [
        "MyRegistry/app:1",
        "./app:1",
        "-/app:1",
        "a.-b:1",
        "a...b:1",
        "a._b:1",
        "a___b:1",
        "a-_b:1",
        ".app:1",
        "app-:1",
        "a//b:1",
        "a-.b/app:1",
        "-a.com/app:1",
        "a..b/app:1",
        ".registry.com/app:1",
        "registry.com./app:1",
        "registry.example.com:abc/app:1",
        "registry.example.com:/app:1",
        "localhost:/app:1",
        "Localhostx/app:1",
        "my_registry.com/app:1",
        "my_registry:5000/app:1",
        "python:",
        ":1.0",
        "python@",
        "@sha256:abc",
        "python:latest@",
        "python: 3",
        "python:1.0 ",
        "python@sha256:xyz",
        "python:latest:1",
        "a:b:c",
        "/:3",
        "python/:3",
        "python:é",
        "python:１",
    ],
)
def test_malformed_image_is_refused_as_invalid(tmp_path: Path, image: str) -> None:
    text = VALID.replace("image: python:3.12-slim", f"image: '{image}'")

    with pytest.raises(
        InstanceError,
        match=r"invalid instance file: container\.image: .*is not a valid image "
        r"reference, e\.g\. python:3\.12-slim",
    ):
        load(tmp_path, text)


@pytest.mark.parametrize(
    ("old", "location"),
    [
        ("image: python:3.12-slim", "container.image"),
        ("name: Worker", "git.name"),
        ("email: worker@users.noreply.github.com", "git.email"),
    ],
)
@pytest.mark.parametrize("blank", ["''", '" "', '"\t"'])
def test_blank_run_key_is_refused(
    tmp_path: Path, old: str, location: str, blank: str
) -> None:
    key = old.split(":")[0]

    with pytest.raises(
        InstanceError, match=f"invalid instance file: {location}: .*blank"
    ):
        load(tmp_path, VALID.replace(old, f"{key}: {blank}"))


@pytest.mark.parametrize(
    ("text", "location"),
    [
        (VALID.replace("branch: agent/{key}\n", ""), "branch: Field required"),
        (
            VALID.replace("container:\n  image: python:3.12-slim\n", ""),
            "container: Field required",
        ),
        (
            VALID.replace("  image: python:3.12-slim\n", "  tag: x\n"),
            "container.image: Field required",
        ),
        (
            VALID.replace(
                "  image: python:3.12-slim\n", "  image: python:3\n  pull: x\n"
            ),
            "container.pull: Extra inputs",
        ),
        (
            VALID.replace(
                "git:\n  name: Worker\n  email: worker@users.noreply.github.com\n", ""
            ),
            "git: Field required",
        ),
        (VALID.replace("  name: Worker\n", ""), "git.name: Field required"),
        (
            VALID.replace("  email: worker@users.noreply.github.com\n", ""),
            "git.email: Field required",
        ),
        (
            VALID.replace("  name: Worker\n", "  name: Worker\n  signing: x\n"),
            "git.signing: Extra inputs",
        ),
    ],
)
def test_missing_or_extra_run_key_is_refused(
    tmp_path: Path, text: str, location: str
) -> None:
    with pytest.raises(InstanceError, match=f"invalid instance file: .*{location}"):
        load(tmp_path, text)


def test_unpinned_image_is_refused_without_echo(tmp_path: Path) -> None:
    text = VALID.replace("image: python:3.12-slim", f"image: {SECRET}")

    with pytest.raises(InstanceError, match="container.image") as caught:
        load(tmp_path, text)

    assert SECRET not in str(caught.value)
    assert "(got" not in str(caught.value)
