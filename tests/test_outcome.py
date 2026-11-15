import json

import pytest

from dev_worker.container import ContainerExit
from dev_worker.outcome import Defect, Outcome, read_outcome, read_verdict

OUTCOMES = ("done", "blocked", "needs_human", "abandoned")
SCHEMA_MISMATCH = (
    "the structured output of Claude Code does not match the outcome schema: "
)


def result(**fields: object) -> str:
    """Return a Claude Code JSON result holding `fields`."""
    return json.dumps({"type": "result", **fields})


def test_a_structured_output_matching_the_schema_is_the_outcome() -> None:
    exited = ContainerExit(
        0,
        result(
            subtype="success",
            structured_output={"outcome": "blocked", "message": "No spec."},
        ),
        "",
    )

    assert read_outcome(exited, OUTCOMES) == Outcome("blocked", "No spec.")


@pytest.mark.parametrize(
    ("exited", "reason"),
    [
        pytest.param(
            ContainerExit(0, "Error: boom", ""),
            "the output of Claude Code is not a JSON object",
            id="not-json",
        ),
        pytest.param(
            ContainerExit(0, "[1, 2]", ""),
            "the output of Claude Code is not a JSON object",
            id="json-not-object",
        ),
        pytest.param(
            ContainerExit(0, "[" * 100_000 + "]" * 100_000, ""),
            "the output of Claude Code is not a JSON object",
            id="nested-too-deeply",
        ),
        pytest.param(
            ContainerExit(1, "", "OCI runtime failed\n\nfor good\n"),
            "the output of Claude Code is not a JSON object; it exited with code 1: "
            "OCI runtime failed for good",
            id="not-json-nonzero-with-stderr",
        ),
        pytest.param(
            ContainerExit(2, "", ""),
            "the output of Claude Code is not a JSON object; it exited with code 2",
            id="not-json-nonzero-without-stderr",
        ),
        pytest.param(
            ContainerExit(
                1,
                result(subtype="error_max_turns", is_error=True, result=" Max turns. "),
                "",
            ),
            "Claude Code exited with code 1 "
            "(subtype error_max_turns, result: Max turns.)",
            id="nonzero-with-subtype-and-result",
        ),
        pytest.param(
            ContainerExit(
                1,
                result(
                    subtype=3,
                    result="  ",
                    structured_output={"outcome": "done", "message": "Done."},
                ),
                "",
            ),
            "Claude Code exited with code 1",
            id="nonzero-without-readable-details",
        ),
        pytest.param(
            ContainerExit(0, result(subtype="error_during_execution"), ""),
            "Claude Code gave no structured output (subtype error_during_execution)",
            id="no-structured-output",
        ),
        pytest.param(
            ContainerExit(0, result(structured_output="done"), ""),
            f"{SCHEMA_MISMATCH}it is not a JSON object",
            id="structured-output-not-object",
        ),
        pytest.param(
            ContainerExit(
                0,
                result(
                    structured_output={
                        "outcome": "done",
                        "message": "Done.",
                        "pr": 4,
                        "branch": "x",
                    }
                ),
                "",
            ),
            f"{SCHEMA_MISMATCH}it holds the unexpected field branch, pr",
            id="extra-field",
        ),
        pytest.param(
            ContainerExit(
                0, result(structured_output={"outcome": "merged", "message": ""}), ""
            ),
            f"{SCHEMA_MISMATCH}its outcome 'merged' is not one of "
            "done, blocked, needs_human, abandoned",
            id="unknown-outcome",
        ),
        pytest.param(
            ContainerExit(0, result(structured_output={"message": "Done."}), ""),
            f"{SCHEMA_MISMATCH}its outcome None is not one of "
            "done, blocked, needs_human, abandoned",
            id="missing-outcome",
        ),
        pytest.param(
            ContainerExit(
                0, result(structured_output={"outcome": 1, "message": "Done."}), ""
            ),
            f"{SCHEMA_MISMATCH}its outcome 1 is not one of "
            "done, blocked, needs_human, abandoned",
            id="outcome-not-string",
        ),
        pytest.param(
            ContainerExit(0, result(structured_output={"outcome": "done"}), ""),
            f"{SCHEMA_MISMATCH}its message is missing or not a string",
            id="missing-message",
        ),
        pytest.param(
            ContainerExit(
                0, result(structured_output={"outcome": "done", "message": 7}), ""
            ),
            f"{SCHEMA_MISMATCH}its message is missing or not a string",
            id="message-not-string",
        ),
    ],
)
def test_an_output_without_a_usable_outcome_gives_why(
    exited: ContainerExit, reason: str
) -> None:
    assert read_outcome(exited, OUTCOMES) == reason


def assert_transmissible(text: str) -> None:
    """Check that `text` holds no NUL and encodes as UTF-8."""
    assert "\x00" not in text
    text.encode("utf-8")


@pytest.mark.parametrize("message", ["x\x00y", "x\ud800y"], ids=["nul", "surrogate"])
def test_a_message_holding_a_nul_or_a_lone_surrogate_is_made_transmissible(
    message: str,
) -> None:
    exited = ContainerExit(
        0, result(structured_output={"outcome": "done", "message": message}), ""
    )

    found = read_outcome(exited, OUTCOMES)

    assert isinstance(found, Outcome)
    assert found.outcome == "done"
    assert_transmissible(found.message)


@pytest.mark.parametrize(
    "exited",
    [
        pytest.param(ContainerExit(0, result(result="x\x00y"), ""), id="result-nul"),
        pytest.param(
            ContainerExit(0, result(result="x\ud800y"), ""), id="result-surrogate"
        ),
        pytest.param(ContainerExit(0, result(subtype="x\x00y"), ""), id="subtype-nul"),
        pytest.param(
            ContainerExit(0, result(subtype="x\ud800y"), ""), id="subtype-surrogate"
        ),
        pytest.param(ContainerExit(1, "boom", "x\x00y"), id="stderr-nul"),
    ],
)
def test_a_reason_quoting_a_nul_or_a_lone_surrogate_is_made_transmissible(
    exited: ContainerExit,
) -> None:
    found = read_outcome(exited, OUTCOMES)

    assert isinstance(found, str)
    assert_transmissible(found)


VERDICT_MISMATCH = (
    "the structured output of Claude Code does not match the verdict schema: "
)


def verdict(**fields: object) -> ContainerExit:
    """Return a container that exited 0 with `fields` as its structured output."""
    return ContainerExit(0, result(structured_output=fields), "")


def test_a_verdict_without_defects_is_an_empty_list() -> None:
    assert read_verdict(verdict(defects=[])) == ()


def test_a_verdict_lists_its_defects_in_order() -> None:
    found = read_verdict(
        verdict(
            defects=[
                {"file": "src/a.py", "line": 3, "summary": "Divides by zero."},
                {"file": "src/b.py", "line": 1, "summary": "x\x00y"},
            ]
        )
    )

    assert isinstance(found, tuple)
    assert found[0] == Defect("src/a.py", 3, "Divides by zero.")
    assert (found[1].file, found[1].line) == ("src/b.py", 1)
    assert_transmissible(found[1].summary)


GOOD = {"file": "a.py", "line": 1, "summary": "Bug."}


@pytest.mark.parametrize(
    ("exited", "reason"),
    [
        pytest.param(
            ContainerExit(0, "Error: boom", ""),
            "the output of Claude Code is not a JSON object",
            id="not-json",
        ),
        pytest.param(
            ContainerExit(
                1,
                result(subtype="error_max_turns", is_error=True, result="Max turns."),
                "",
            ),
            "Claude Code exited with code 1 "
            "(subtype error_max_turns, result: Max turns.)",
            id="ceiling",
        ),
        pytest.param(
            ContainerExit(0, result(subtype="error_during_execution"), ""),
            "Claude Code gave no structured output (subtype error_during_execution)",
            id="no-structured-output",
        ),
        pytest.param(
            ContainerExit(0, result(structured_output=[]), ""),
            f"{VERDICT_MISMATCH}it is not a JSON object",
            id="not-object",
        ),
        pytest.param(
            verdict(defects=[], verdict="ok"),
            f"{VERDICT_MISMATCH}it holds the unexpected field verdict",
            id="extra-field",
        ),
        pytest.param(
            verdict(),
            f"{VERDICT_MISMATCH}its defects are missing or not a list",
            id="missing-defects",
        ),
        pytest.param(
            verdict(defects={}),
            f"{VERDICT_MISMATCH}its defects are missing or not a list",
            id="defects-not-list",
        ),
        pytest.param(
            verdict(defects=[GOOD, "bug"]),
            f"{VERDICT_MISMATCH}its defect 2 is not a JSON object",
            id="defect-not-object",
        ),
        pytest.param(
            verdict(defects=[{**GOOD, "severity": "high"}]),
            f"{VERDICT_MISMATCH}its defect 1 holds the unexpected field severity",
            id="defect-extra-field",
        ),
        pytest.param(
            verdict(defects=[{**GOOD, "file": 3}]),
            f"{VERDICT_MISMATCH}its defect 1 has a file missing or not a string",
            id="file-not-string",
        ),
        pytest.param(
            verdict(defects=[{**GOOD, "line": "3"}]),
            f"{VERDICT_MISMATCH}its defect 1 has the line '3', not an integer from 1",
            id="line-not-integer",
        ),
        pytest.param(
            verdict(defects=[{**GOOD, "line": True}]),
            f"{VERDICT_MISMATCH}its defect 1 has the line True, not an integer from 1",
            id="line-boolean",
        ),
        pytest.param(
            verdict(defects=[{**GOOD, "line": 0}]),
            f"{VERDICT_MISMATCH}its defect 1 has the line 0, not an integer from 1",
            id="line-zero",
        ),
        pytest.param(
            verdict(defects=[{"file": "a.py", "line": 1}]),
            f"{VERDICT_MISMATCH}its defect 1 has a summary missing or not a string",
            id="missing-summary",
        ),
    ],
)
def test_an_output_without_a_valid_verdict_gives_why(
    exited: ContainerExit, reason: str
) -> None:
    assert read_verdict(exited) == reason
