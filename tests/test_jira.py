import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from dev_worker.instance import Tracker
from dev_worker.jira import JiraTracker
from dev_worker.tickets import Blocker, Claim, Ticket, TrackerError

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
    "summary": "Add a login page",
    "description": "The page asks for an email.",
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
        title="Add a login page",
        description="The page asks for an email.",
    )
    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["argv"] == [
        "jira", "workitem", "get", "ABC-218",
        "--fields", "status,labels,issuelinks,summary,description",
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


def test_a_description_written_as_an_atlassian_document_is_read_as_plain_text(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "heading",
                "attrs": {"level": 2},
                "content": [{"type": "text", "text": "Goal"}],
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Run "},
                    {"type": "text", "text": "main", "marks": [{"type": "code"}]},
                    {"type": "hardBreak"},
                    {"type": "text", "text": "again."},
                ],
            },
            {
                "type": "bulletList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": "one"}],
                            }
                        ],
                    }
                ],
            },
            {"type": "codeBlock", "content": [{"type": "text", "text": "x = 1"}]},
            {"type": "rule"},
        ],
    }
    answer(monkeypatch, {"data": [{**ITEM, "description": document}]})

    ticket = tracker().read("ABC-218")

    assert ticket.description == "Goal\nRun main\nagain.\none\nx = 1"


@pytest.mark.parametrize("present", [True, False])
def test_a_ticket_without_description_has_an_empty_one(
    record: Path, monkeypatch: pytest.MonkeyPatch, present: bool
) -> None:
    item = {key: value for key, value in ITEM.items() if key != "description"}
    if present:
        item["description"] = None
    answer(monkeypatch, {"data": [item]})

    assert tracker().read("ABC-218").description == ""


def test_a_description_written_as_a_string_is_kept_as_it_is(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"data": [{**ITEM, "description": "  Line one\n\nLine two\n"}]})

    assert tracker().read("ABC-218").description == "  Line one\n\nLine two\n"


def test_a_description_ending_on_a_hard_break_or_holding_unknown_nodes_keeps_only_its_text(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = {
        "type": "doc",
        "content": [
            {
                "type": "orderedList",
                "content": [
                    {
                        "type": "listItem",
                        "content": [
                            {
                                "type": "bulletList",
                                "content": [
                                    {
                                        "type": "listItem",
                                        "content": [
                                            {
                                                "type": "paragraph",
                                                "content": [
                                                    {"type": "text", "text": "deep"}
                                                ],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "type": "paragraph",
                "content": [
                    {"type": "mention", "attrs": {"id": "1"}},
                    {"type": "text", "text": 42},
                    {"type": "text", "text": "end"},
                    "stray",
                    {"type": "hardBreak"},
                ],
            },
        ],
    }
    answer(monkeypatch, {"data": [{**ITEM, "description": document}]})

    assert tracker().read("ABC-218").description == "deep\nend"


@pytest.mark.parametrize("description", [42, ["a"], True])
def test_a_description_neither_text_nor_a_document_is_read_as_empty(
    record: Path, monkeypatch: pytest.MonkeyPatch, description: object
) -> None:
    answer(monkeypatch, {"data": [{**ITEM, "description": description}]})

    assert tracker().read("ABC-218").description == ""


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


@pytest.mark.parametrize(
    ("call", "action"),
    [
        (lambda jira: jira.read("ABC-218"), "reading ABC-218"),
        (lambda jira: jira.set_status("ABC-218", "Done"), "moving ABC-218 to 'Done'"),
        (lambda jira: jira.check_access(), "checking access to Jira"),
    ],
    ids=["read", "set_status", "check_access"],
)
def test_twg_that_does_not_answer_is_stopped_at_the_timeout(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    call: Callable[[JiraTracker], object],
    action: str,
) -> None:
    answer(monkeypatch, TICKET)
    monkeypatch.setenv("FAKE_TWG_SLEEP", "30")
    jira = JiraTracker(TRACKER, command=[sys.executable, str(FAKE_TWG)], timeout=0.5)

    started = time.monotonic()
    with pytest.raises(TrackerError) as raised:
        call(jira)
    elapsed = time.monotonic() - started

    assert str(raised.value) == f"twg did not answer within 0.5 seconds while {action}"
    assert TOKEN not in str(raised.value)
    assert elapsed < 10


def doctor(connectivity: object) -> dict[str, Any]:
    return {"ok": None, "data": {"connectivity": connectivity}}


REFUSED = {
    "attempted": True,
    "ok": False,
    "message": "GraphQL request failed with status 401 (HTTP 401)",
    "source": "TWG_TOKEN + TWG_USER",
    "statusCode": 401,
}


def test_check_access_runs_the_doctor_with_the_instance_credentials(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, doctor({"attempted": True, "ok": True}))

    tracker().check_access()

    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["argv"] == [
        "doctor", "--output", "json", "--output-summary", "none",
    ]  # fmt: skip
    assert called["env"] == {
        "TWG_SITE": "example.atlassian.net",
        "TWG_USER": "bot@example.com",
        "TWG_TOKEN": TOKEN,
    }


def test_check_access_passes_on_connectivity_whatever_the_exit_code(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, doctor({"ok": True}), code=1)

    tracker().check_access()


def test_refused_credentials_are_explained_by_the_doctor_message(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, doctor(REFUSED), code=1)

    with pytest.raises(TrackerError) as raised:
        tracker().check_access()

    assert str(raised.value) == (
        "twg cannot reach Jira with the instance credentials: "
        "GraphQL request failed with status 401 (HTTP 401)"
    )


@pytest.mark.parametrize(
    ("stdout", "stderr", "reason"),
    [
        (doctor({"ok": False, "message": "  "}), "offline\n", "offline"),
        (doctor({"ok": "yes"}), "", "exit code 1"),
        (doctor("broken"), "", "exit code 1"),
        ({"data": "broken"}, "", "exit code 1"),
        ("not json", "", "exit code 1"),
    ],
)
def test_a_doctor_without_a_passing_connectivity_check_refuses_access(
    record: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: object,
    stderr: str,
    reason: str,
) -> None:
    answer(monkeypatch, stdout, stderr=stderr, code=1)

    with pytest.raises(TrackerError) as raised:
        tracker().check_access()

    assert str(raised.value) == (
        f"twg cannot reach Jira with the instance credentials: {reason}"
    )


def test_a_refused_access_does_not_give_the_token(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, doctor({**REFUSED, "message": f"bad token {TOKEN}"}), code=1)

    with pytest.raises(TrackerError) as raised:
        tracker().check_access()

    assert TOKEN not in str(raised.value)
    assert str(raised.value).endswith("bad token [token]")


def test_check_access_without_the_token_variable_names_it_and_runs_nothing(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JIRA_TOKEN")

    with pytest.raises(TrackerError, match="JIRA_TOKEN.*not set"):
        tracker().check_access()
    assert not record.exists()


def test_check_access_with_twg_missing_raises(record: Path) -> None:
    missing = JiraTracker(TRACKER, command=[str(record.parent / "no-such-twg")])

    with pytest.raises(TrackerError, match="^twg could not be started: "):
        missing.check_access()


def test_claim_posts_the_claim_comment_and_returns_its_id(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"ok": True, "data": {"id": "10042", "body": {}}})

    claim_id = tracker().claim("ABC-218", "task-1")

    assert claim_id == 10042
    called = json.loads(record.read_text(encoding="utf-8"))
    assert called["argv"] == [
        "jira", "workitem", "comment", "create", "--issue-id", "ABC-218",
        "--body", "dev-worker claims this ticket for task task-1",
        "--body-format", "plain",
        "--output", "json", "--output-summary", "none",
    ]  # fmt: skip


def test_claim_whose_answer_has_no_comment_id_raises(
    record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer(monkeypatch, {"ok": True, "data": {}})

    with pytest.raises(TrackerError, match="the answer of twg to claiming ABC-218"):
        tracker().claim("ABC-218", "task-1")


def test_claim_on_a_key_that_is_not_a_ticket_key_is_refused(record: Path) -> None:
    with pytest.raises(TrackerError, match="is not a Jira ticket key"):
        tracker().claim("abc-218", "task-1")
    assert not record.exists()


def page(data: list[Any], cursor: str | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "data": data,
        "pageInfo": {"hasNextPage": cursor is not None, "nextCursor": cursor},
    }


def history(created: str, *changes: tuple[str, str]) -> dict[str, Any]:
    return {
        "id": "1",
        "created": created,
        "items": [
            {"field": field, "fromString": "x", "toString": value}
            for field, value in changes
        ],
    }


def adf(text: str) -> dict[str, Any]:
    return {
        "type": "doc",
        "version": 1,
        "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
    }


def comment(comment_id: str, created: str, body: object) -> dict[str, Any]:
    return {"id": comment_id, "created": created, "body": body}


def serve_pages(monkeypatch: pytest.MonkeyPatch, *answers: object) -> list[list[str]]:
    calls: list[list[str]] = []
    queue = list(answers)

    def run(self: JiraTracker, action: str, arguments: list[str]) -> object:
        calls.append(arguments)
        return queue.pop(0)

    monkeypatch.setattr(JiraTracker, "_run", run)
    return calls


def test_claims_count_only_those_since_the_last_entry_in_the_status_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = serve_pages(
        monkeypatch,
        page(
            [
                history("2026-11-10T09:00:00.000+0100", ("status", "To Do")),
                history("2026-11-10T10:00:00.000+0100", ("status", "In Progress")),
            ],
            cursor="2",
        ),
        page(
            [
                history(
                    "2026-11-11T09:00:00.000+0100",
                    ("labels", "To Do"),
                    ("status", "To Do"),
                ),
                history("2026-11-11T09:30:00.000+0100", ("labels", "x")),
            ]
        ),
        page(
            [
                comment(
                    "90",
                    "2026-11-10T09:30:00.000+0100",
                    adf("dev-worker claims this ticket for task old"),
                ),
                comment(
                    "120",
                    "2026-11-11T09:00:00.000+0100",
                    "dev-worker claims this ticket for task task-2",
                ),
                comment(
                    "100",
                    "2026-11-11T08:00:00.000Z",
                    [adf("dev-worker claims this ticket for task task-1")],
                ),
            ],
            cursor="3",
        ),
        page(
            [
                comment("130", "2026-11-11T10:00:00.000+0100", adf("Looks good")),
                comment("140", "2026-11-11T10:00:00.000+0100", None),
                comment("150", "2026-11-11T10:00:00.000+0100", {"text": 3}),
            ]
        ),
    )

    claims = tracker().claims("ABC-218", "To Do")

    assert claims == (
        Claim(100, "task-1", datetime(2026, 11, 11, 8, 0, tzinfo=UTC)),
        Claim(120, "task-2", datetime(2026, 11, 11, 8, 0, tzinfo=UTC)),
    )
    assert calls == [
        ["jira", "workitem", "changelog", "query", "--issue-id", "ABC-218",
         "--first", "100"],
        ["jira", "workitem", "changelog", "query", "--issue-id", "ABC-218",
         "--first", "100", "--after", "2"],
        ["jira", "workitem", "comment", "query", "--issue-id", "ABC-218",
         "--order-by", "created", "--first", "100"],
        ["jira", "workitem", "comment", "query", "--issue-id", "ABC-218",
         "--order-by", "created", "--first", "100", "--after", "3"],
    ]  # fmt: skip


def test_claims_of_a_ticket_that_never_entered_the_status_all_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve_pages(
        monkeypatch,
        page([]),
        page(
            [
                comment(
                    "90",
                    "2020-01-01T00:00:00.000Z",
                    "dev-worker claims this ticket for task task-1",
                ),
            ]
        ),
    )

    assert tracker().claims("ABC-218", "To Do") == (
        Claim(90, "task-1", datetime(2020, 1, 1, tzinfo=UTC)),
    )


def test_claims_carry_the_creation_time_of_their_own_comment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve_pages(
        monkeypatch,
        page([history("2026-11-11T08:00:00.000Z", ("status", "To Do"))]),
        page(
            [
                comment(
                    "100",
                    "2026-11-11T08:05:00.000Z",
                    "dev-worker claims this ticket for task task-1",
                ),
                comment(
                    "110",
                    "2026-11-11T09:10:30.000+0100",
                    "dev-worker claims this ticket for task task-2",
                ),
            ]
        ),
    )

    assert tracker().claims("ABC-218", "To Do") == (
        Claim(100, "task-1", datetime(2026, 11, 11, 8, 5, tzinfo=UTC)),
        Claim(110, "task-2", datetime(2026, 11, 11, 8, 10, 30, tzinfo=UTC)),
    )


@pytest.mark.parametrize("cursor", [None, "2"])
def test_claims_with_a_next_page_and_no_new_cursor_raise(
    monkeypatch: pytest.MonkeyPatch, cursor: str | None
) -> None:
    looping = page([], cursor="2")
    stuck = {
        "ok": True,
        "data": [],
        "pageInfo": {"hasNextPage": True, "nextCursor": cursor},
    }
    serve_pages(monkeypatch, looping, stuck)

    with pytest.raises(TrackerError, match="names a next page without a new cursor"):
        tracker().claims("ABC-218", "To Do")


def test_claims_page_that_cannot_be_read_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve_pages(monkeypatch, {"ok": True, "data": []})

    with pytest.raises(
        TrackerError,
        match="the answer of twg to reading the changelog of ABC-218 cannot be read",
    ):
        tracker().claims("ABC-218", "To Do")


def test_claims_on_a_key_that_is_not_a_ticket_key_are_refused(record: Path) -> None:
    with pytest.raises(TrackerError, match="is not a Jira ticket key"):
        tracker().claims("abc-218", "To Do")
    assert not record.exists()
