import json
import subprocess
import sys
from pathlib import Path

import pytest

from dev_worker import plugin as plugin_module
from dev_worker.header import Plugin
from dev_worker.plugin import PluginCheckout, PluginError


def git(*arguments: str | Path) -> str:
    """Run git with `arguments` as the test's own user and return its stripped stdout."""
    done = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            *map(str, arguments),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def source(path: Path, *versions: str) -> str:
    """Make a repository at `path` with one commit tagged per version; return its URL.

    Each commit writes the version into `VERSION`; the tags are annotated.
    """
    git("init", "--quiet", path)
    for version in versions:
        (path / "VERSION").write_text(version, encoding="utf-8")
        git("-C", path, "add", "VERSION")
        git("-C", path, "commit", "--quiet", "-m", version)
        git("-C", path, "tag", "-a", version, "-m", version)
    return path.as_uri()


@pytest.fixture
def url(tmp_path: Path) -> str:
    """Return the URL of a repository tagged `v1` then `v2`."""
    return source(tmp_path / "source", "v1", "v2")


@pytest.fixture
def instance(tmp_path: Path) -> Path:
    """Return an empty instance directory."""
    path = tmp_path / "instance"
    path.mkdir()
    return path


def version(checkout: Path) -> str:
    """Return what the checkout holds in `VERSION`."""
    return (checkout / "VERSION").read_text(encoding="utf-8")


def test_first_prepare_clones_the_tag_into_the_instance_directory(
    url: str, instance: Path
) -> None:
    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert checkout == instance / "plugin"
    assert version(checkout) == "v1"
    assert sorted(path.name for path in instance.iterdir()) == ["plugin"]


def test_a_checkout_at_the_tag_is_used_without_reaching_the_repository(
    url: str, instance: Path, tmp_path: Path
) -> None:
    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))
    (tmp_path / "source").rename(tmp_path / "gone")

    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert version(checkout) == "v1"


def test_another_tag_replaces_the_checkout(url: str, instance: Path) -> None:
    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v2"))

    assert version(checkout) == "v2"
    assert sorted(path.name for path in instance.iterdir()) == ["plugin"]


def test_the_same_tag_of_another_repository_replaces_the_checkout(
    url: str, instance: Path, tmp_path: Path
) -> None:
    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))
    other = source(tmp_path / "other", "v0", "v1")
    (tmp_path / "other" / "VERSION").write_text("unused", encoding="utf-8")

    checkout = PluginCheckout(instance).prepare(Plugin(repository=other, tag="v1"))

    assert version(checkout) == "v1"
    assert git("-C", checkout, "config", "--get", "remote.origin.url") == other


def test_a_checkout_git_cannot_read_is_fetched_again(url: str, instance: Path) -> None:
    (instance / "plugin").mkdir()
    (instance / "plugin" / "VERSION").write_text("stale", encoding="utf-8")

    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert version(checkout) == "v1"


def test_what_an_earlier_fetch_left_is_deleted_before_fetching(
    url: str, instance: Path
) -> None:
    for name in ("plugin.new", "plugin.old"):
        (instance / name).mkdir()
        (instance / name / "left").write_text("left", encoding="utf-8")

    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert sorted(path.name for path in instance.iterdir()) == ["plugin"]


def test_a_tag_that_cannot_be_cloned_raises_and_keeps_the_checkout(
    url: str, instance: Path
) -> None:
    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    with pytest.raises(PluginError) as raised:
        PluginCheckout(instance).prepare(Plugin(repository=url, tag="v9"))

    assert str(raised.value).startswith(
        f"git failed while cloning the tag v9 of {url}: "
    )
    assert version(instance / "plugin") == "v1"


def test_a_branch_named_like_the_tag_is_refused(
    url: str, instance: Path, tmp_path: Path
) -> None:
    git("-C", tmp_path / "source", "branch", "edge")

    with pytest.raises(PluginError) as raised:
        PluginCheckout(instance).prepare(Plugin(repository=url, tag="edge"))

    assert str(raised.value) == f"{url} has no tag edge"
    assert not (instance / "plugin").exists()


def test_a_directory_that_cannot_be_deleted_raises_naming_it(
    url: str, instance: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (instance / "plugin.new").mkdir()

    def refuse(path: Path) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(plugin_module.shutil, "rmtree", refuse)

    with pytest.raises(PluginError) as raised:
        PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert str(raised.value) == (
        f"'{instance / 'plugin.new'}' cannot be deleted: Permission denied"
    )


def test_a_replaced_checkout_that_cannot_be_deleted_leaves_the_new_one_in_place(
    url: str, instance: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))
    rmtree = plugin_module.shutil.rmtree

    def refuse_the_discard(path: Path) -> None:
        if Path(path) == instance / "plugin.old":
            raise PermissionError(13, "Permission denied")
        rmtree(path)

    monkeypatch.setattr(plugin_module.shutil, "rmtree", refuse_the_discard)

    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v2"))

    assert checkout == instance / "plugin"
    assert version(checkout) == "v2"


def test_a_checkout_that_cannot_be_replaced_raises_naming_it(
    url: str, instance: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path, target: Path) -> Path:
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "rename", refuse)

    with pytest.raises(PluginError) as raised:
        PluginCheckout(instance).prepare(Plugin(repository=url, tag="v1"))

    assert str(raised.value) == (
        f"the checkout '{instance / 'plugin'}' cannot be replaced: Access is denied"
    )


def test_git_that_cannot_be_started_raises(url: str, instance: Path) -> None:
    absent = PluginCheckout(instance, command=[str(instance / "no-git")])

    with pytest.raises(PluginError) as raised:
        absent.prepare(Plugin(repository=url, tag="v1"))

    assert str(raised.value).startswith("git could not be started: ")


def test_git_that_does_not_answer_raises(url: str, instance: Path) -> None:
    slow = PluginCheckout(
        instance,
        command=[sys.executable, "-c", "import time; time.sleep(5)"],
        timeout=0.5,
    )

    with pytest.raises(PluginError) as raised:
        slow.prepare(Plugin(repository=url, tag="v1"))

    assert str(raised.value) == (
        f"git did not answer within 0.5 seconds while cloning the tag v1 of {url}"
    )


def test_the_tag_is_cloned_shallow(url: str, instance: Path) -> None:
    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="v2"))

    assert git("-C", checkout, "rev-parse", "--is-shallow-repository") == "true"
    assert git("-C", checkout, "rev-list", "--count", "HEAD") == "1"


def test_a_lightweight_tag_is_cloned(url: str, instance: Path, tmp_path: Path) -> None:
    git("-C", tmp_path / "source", "tag", "light", "v1")

    checkout = PluginCheckout(instance).prepare(Plugin(repository=url, tag="light"))

    assert version(checkout) == "v1"


def test_git_is_run_without_credential_helpers_nor_prompts(
    url: str, instance: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = tmp_path / "recorded.json"
    fake = tmp_path / "fake_git.py"
    fake.write_text(
        "import json, os, sys\n"
        "with open(sys.argv[1], 'w', encoding='utf-8') as out:\n"
        "    json.dump({'argv': sys.argv[2:], 'prompt': os.environ.get("
        "'GIT_TERMINAL_PROMPT')}, out)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    recording = PluginCheckout(
        instance, command=[sys.executable, str(fake), str(recorded)]
    )

    with pytest.raises(PluginError):
        recording.prepare(Plugin(repository=url, tag="v1"))

    call = json.loads(recorded.read_text(encoding="utf-8"))
    assert call["argv"][:3] == ["-c", "credential.helper=", "clone"]
    assert call["prompt"] == "0"
