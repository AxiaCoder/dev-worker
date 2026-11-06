import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dev_worker.instance import Tracker
from dev_worker.jira import JiraTracker
from dev_worker.tickets import Blocker, Ticket, TrackerError

FAKE_TWG = Path(__file__).with_name("fake_twg.py")
TOKEN = "s3cret-token-value"
TRACKER = Tracker(
    type="jira",
    site="example.atlassian.net",
    user="bot@example.com",
    project="ABC",
    token_env="JIRA_TOKEN",
)


def link(side: str, key: str, status: str, name: str = "Blocks") -> dict[str, Any]:
    return {
        "type": {"name": name, "inward": "is blocked by", "outward": "blocks"},
        side: {"key": key, "fields": {"status": {"name": status}}},
    }


ITEM: dict[str, Any] = {
    "key": "ABC-218",
    "issuelinks": [
        link("inwardIssue", "ABC-217", "Terminé"),
        link("outwardIssue", "ABC-220", "Annulé"),
        link("inwardIssue", "ABC-300", "Done", name="Relates"),
        link("inwardIssue", "ABC-219", "En cours"),
    ],
    "status": {"name": "À faire", "statusCategory": {"key": "new"}},
    "labels": ["hold-merge", "backend"],
    "url": "https://example.atlassian.net/browse/ABC-218",
}
TICKET = {
    "apiVersion": "v2",
    "command": "jira.workitem.get",
    "data": [ITEM],
    "meta": {},
}


@pytest.fixture
def record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "record.json"
    monkeypatch.setenv("FAKE_TWG_RECORD", str(path))
    monkeypatch.setenv("JIRA_TOKEN", TOKEN)
    return path


def answer(
    monkeypatch: pytest.MonkeyPatch, stdout: object, stderr: str = "", code: int = 0
) -> None:
    text = stdout if isinstance(stdout, str) else json.dumps(stdout)
    monkeypatch.setenv("FAKE_TWG_STDOUT", text)
    monkeypatch.setenv("FAKE_TWG_STDERR", stderr)
    monkeypatch.setenv("FAKE_TWG_EXIT", str(code))


def tracker() -> JiraTracker:
    return JiraTracker(TRACKER, command=[sys.executable, str(FAKE_TWG)])


def test_read_returns_status_labels_and_blockers(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, TICKET)

    ticket = tracker().read("ABC-218")

    assert ticket == Ticket(
        key="ABC-218",
        status="À faire",
        labels=("hold-merge", "backend"),
        blockers=(Blocker("ABC-217", "Terminé"), Blocker("ABC-219", "En cours")),
    )
    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["argv"] == [
        "jira", "workitem", "get", "ABC-218",
        "--fields", "status,labels,issuelinks",
        "--output", "json", "--output-summary", "none",
    ]  # fmt: skip
    assert called["env"] == {
        "TWG_SITE": "example.atlassian.net",
        "TWG_USER": "bot@example.com",
        "TWG_TOKEN": TOKEN,
    }


def test_set_status_transitions_by_status_name(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"ok": True, "data": {}})

    tracker().set_status("ABC-218", "En cours")

    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["argv"] == [
        "jira", "workitem", "transition", "--id", "ABC-218",
        "--transition-id", "En cours",
        "--output", "json", "--output-summary", "none",
    ]  # fmt: skip
    assert called["env"]["TWG_TOKEN"] == TOKEN


@pytest.mark.parametrize("value", [None, ""])
def test_unset_token_variable_is_named_and_twg_not_run(
    record: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("JIRA_TOKEN")
    else:
        monkeypatch.setenv("JIRA_TOKEN", value)
    answer(monkeypatch, TICKET)

    with pytest.raises(TrackerError, match="JIRA_TOKEN.*not set"):
        tracker().read("ABC-218")
    assert not record.exists()


@pytest.mark.parametrize("key", ["--help", "abc-1", "ABC", "ABC-1 ", ""])
def test_a_key_that_is_not_a_ticket_key_is_refused(
    record: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    answer(monkeypatch, TICKET)

    with pytest.raises(TrackerError, match="not a Jira ticket key"):
        tracker().read(key)
    with pytest.raises(TrackerError, match="not a Jira ticket key"):
        tracker().set_status(key, "Done")
    assert not record.exists()


def test_twg_that_cannot_be_started_raises(record: Path) -> None:
    missing = JiraTracker(TRACKER, command=[str(record.parent / "no-such-twg")])

    with pytest.raises(TrackerError, match="twg could not be started"):
        missing.read("ABC-218")


def test_twg_error_message_is_given_without_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = {
        "ok": False,
        "error": {"code": "X", "message": f"Le ticket n'existe pas ({TOKEN})."},
    }
    answer(monkeypatch, failure, code=1)

    with pytest.raises(TrackerError) as raised:
        tracker().read("ABC-218")

    message = str(raised.value)
    assert message.startswith(
        "twg failed while reading ABC-218: Le ticket n'existe pas"
    )
    assert TOKEN not in message


def test_ok_false_with_a_zero_exit_is_a_failure(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"ok": False, "error": {"message": "Transition invalide"}})

    with pytest.raises(
        TrackerError,
        match="twg failed while moving ABC-218 to 'Nope': Transition invalide",
    ):
        tracker().set_status("ABC-218", "Nope")


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        {"ok": False, "error": "boom"},
        {"ok": False, "error": {"message": " "}},
    ],
)
def test_without_an_error_message_the_last_stderr_line_is_given(
    record: Path, monkeypatch: pytest.MonkeyPatch, stdout: object
) -> None:
    answer(monkeypatch, stdout, stderr=f"first line\nauth {TOKEN} refused\n", code=2)

    with pytest.raises(TrackerError) as raised:
        tracker().read("ABC-218")

    assert str(raised.value) == "twg failed while reading ABC-218: auth [token] refused"


def test_without_message_or_stderr_the_exit_code_is_given(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "", code=3)

    with pytest.raises(TrackerError, match="reading ABC-218: exit code 3$"):
        tracker().read("ABC-218")


def test_an_answer_that_is_not_json_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "<html>")

    with pytest.raises(
        TrackerError, match="answer of twg to reading ABC-218 is not JSON"
    ):
        tracker().read("ABC-218")


@pytest.mark.parametrize(
    ("stdout", "where"),
    [
        ({"data": []}, "data"),
        ({"data": [ITEM, ITEM]}, "data"),
        ({"items": []}, "data"),
        ({"data": [{**ITEM, "status": {"name": 7}}]}, "data.0.status.name"),
        ({"data": [{**ITEM, "labels": None}]}, "data.0.labels"),
        ([1, 2], "answer"),
    ],
)
def test_an_answer_of_the_wrong_shape_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch, stdout: object, where: str
) -> None:
    answer(monkeypatch, stdout)

    with pytest.raises(TrackerError) as raised:
        tracker().read("ABC-218")

    message = str(raised.value)
    assert message.startswith("the answer of twg to reading ABC-218 cannot be read: ")
    assert where in message


def test_twg_variables_already_set_are_replaced_by_the_tracker_ones(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TWG_SITE", "stale.atlassian.net")
    monkeypatch.setenv("TWG_USER", "stale@example.com")
    monkeypatch.setenv("TWG_TOKEN", "stale-token")
    answer(monkeypatch, TICKET)

    tracker().read("ABC-218")

    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["env"] == {
        "TWG_SITE": "example.atlassian.net",
        "TWG_USER": "bot@example.com",
        "TWG_TOKEN": TOKEN,
    }


def test_the_token_is_read_at_each_call_not_at_construction(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JIRA_TOKEN")
    built = tracker()
    monkeypatch.setenv("JIRA_TOKEN", "first-token")
    answer(monkeypatch, {"ok": True, "data": {}})

    built.set_status("ABC-218", "Done")
    first = json.loads(record.read_text(encoding="utf-8"))["env"]["TWG_TOKEN"]
    monkeypatch.setenv("JIRA_TOKEN", "second-token")
    built.set_status("ABC-218", "Done")
    second = json.loads(record.read_text(encoding="utf-8"))["env"]["TWG_TOKEN"]

    assert (first, second) == ("first-token", "second-token")


def test_a_ticket_without_links_or_labels_has_no_blockers(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"data": [{**ITEM, "labels": [], "issuelinks": []}]})

    ticket = tracker().read("ABC-218")

    assert (ticket.labels, ticket.blockers) == ((), ())


def test_a_ticket_that_only_blocks_others_has_no_blockers(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    links = [link("outwardIssue", "ABC-220", "À faire")]
    answer(monkeypatch, {"data": [{**ITEM, "issuelinks": links}]})

    assert tracker().read("ABC-218").blockers == ()


def test_a_nonzero_exit_is_a_failure_even_with_a_readable_answer(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, TICKET, stderr="rate limited\n", code=1)

    with pytest.raises(
        TrackerError, match="^twg failed while reading ABC-218: rate limited$"
    ):
        tracker().read("ABC-218")


def test_an_answer_of_the_wrong_shape_does_not_echo_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"data": [{**ITEM, "labels": TOKEN, "status": TOKEN}]})

    with pytest.raises(TrackerError) as raised:
        tracker().read("ABC-218")

    message = str(raised.value)
    assert "data.0.labels" in message
    assert TOKEN not in message


def test_set_status_with_a_twg_failure_does_not_give_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "", stderr=f"token {TOKEN} expired\n", code=1)

    with pytest.raises(TrackerError) as raised:
        tracker().set_status("ABC-218", "Done")

    assert str(raised.value) == (
        "twg failed while moving ABC-218 to 'Done': token [token] expired"
    )


def test_set_status_with_an_answer_that_is_not_json_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, "Transitioned.")

    with pytest.raises(
        TrackerError, match="answer of twg to moving ABC-218 to 'Done' is not JSON"
    ):
        tracker().set_status("ABC-218", "Done")
