import json

import pytest

from dev_worker.container import ContainerExit
from dev_worker.outcome import Outcome, read_outcome

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
