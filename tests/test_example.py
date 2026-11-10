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
