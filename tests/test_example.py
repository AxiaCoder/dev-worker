from pathlib import Path

from dev_worker.header import load_worker_file
from dev_worker.instance import load_instance

ROOT = Path(__file__).resolve().parent.parent


def test_example_instance_loads_against_the_worker_header() -> None:
    header = load_worker_file(ROOT / "src" / "dev_worker" / "WORKER.md").header

    instance = load_instance(ROOT / "examples" / "instance.yaml", header)

    assert set(instance.statuses) == set(header.status_keys)
    assert instance.limits["max_turns"] == 40
    assert instance.tracker.site == "example.atlassian.net"
    assert instance.tracker.user == "worker@example.com"
    assert instance.hold_merge_label == "hold-merge"


def test_example_instance_sets_the_run_keys() -> None:
    header = load_worker_file(ROOT / "src" / "dev_worker" / "WORKER.md").header

    instance = load_instance(ROOT / "examples" / "instance.yaml", header)

    assert instance.branch == "agent/{key}"
    assert instance.container.image == "ghcr.io/example-org/example-app-agent:1.0.0"
    assert instance.git.name == "Example Worker"
    assert instance.git.email.endswith("@users.noreply.github.com")
    assert instance.agent.model == "opus"
    assert instance.agent.review_model == "opus"


def test_example_instance_declares_no_handover() -> None:
    header = load_worker_file(ROOT / "src" / "dev_worker" / "WORKER.md").header

    instance = load_instance(ROOT / "examples" / "instance.yaml", header)

    assert instance.handover is None


def test_example_handover_block_loads_once_uncommented(tmp_path: Path) -> None:
    header = load_worker_file(ROOT / "src" / "dev_worker" / "WORKER.md").header
    text = (ROOT / "examples" / "instance.yaml").read_text(encoding="utf-8")
    lines = text.splitlines()
    start = lines.index("# handover:")
    end = lines.index("", start)
    uncommented = [line.removeprefix("# ") for line in lines[start:end]]
    path = tmp_path / "instance.yaml"
    path.write_text("\n".join([*lines, *uncommented, ""]), encoding="utf-8")

    handover = load_instance(path, header).handover

    assert handover is not None
    assert handover.sensitive_paths == (".github/workflows/**", "src/auth/**")
    assert handover.max_changed_lines == 400
    assert handover.test_paths == ("tests/**",)
