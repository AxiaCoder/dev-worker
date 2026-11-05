from pathlib import Path

import pytest
from pydantic import ValidationError

from dev_worker.header import HeaderError, load_worker_file

SHIPPED = Path(__file__).resolve().parent.parent / "src" / "dev_worker" / "WORKER.md"

VALID = """\
status_keys: [ready, in_progress, in_review, blocked, done]
outcomes:
  done: done
  blocked: blocked
  needs_human: blocked
  abandoned: ready
ceilings:
  run_timeout_seconds: 1800
  max_turns: 60
"""


def write(tmp_path: Path, text: str) -> Path:
    """Write `text` as `WORKER.md` under `tmp_path`, UTF-8, and return its path."""
    path = tmp_path / "WORKER.md"
    path.write_text(text, encoding="utf-8")
    return path


def with_header(front_matter: str, body: str = "Body.\n") -> str:
    """Return `front_matter` between fence lines, followed by `body`."""
    return f"---\n{front_matter}---\n{body}"


def test_shipped_worker_file_loads() -> None:
    worker = load_worker_file(SHIPPED)

    assert worker.header.status_keys == (
        "ready",
        "in_progress",
        "in_review",
        "blocked",
        "done",
    )
    assert worker.header.outcomes.model_dump() == {
        "done": "in_review",
        "blocked": "blocked",
        "needs_human": "blocked",
        "abandoned": "blocked",
    }
    assert worker.header.ceilings == {"run_timeout_seconds": 1800, "max_turns": 60}
    assert worker.body.strip()


def test_valid_file_splits_header_and_body(tmp_path: Path) -> None:
    worker = load_worker_file(write(tmp_path, with_header(VALID, "\nRules.\n")))

    assert worker.body == "\nRules.\n"
    assert worker.header.outcomes.abandoned == "ready"


def test_crlf_line_endings_are_read(tmp_path: Path) -> None:
    path = tmp_path / "WORKER.md"
    path.write_bytes(with_header(VALID).replace("\n", "\r\n").encode("utf-8"))

    assert load_worker_file(path).header.ceilings["max_turns"] == 60


@pytest.mark.parametrize("text", ["", "status_keys: [ready]\n", "Body only.\n"])
def test_missing_front_matter_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(HeaderError, match="front matter missing"):
        load_worker_file(write(tmp_path, text))


def test_unterminated_front_matter_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HeaderError, match="unterminated"):
        load_worker_file(write(tmp_path, f"---\n{VALID}Body.\n"))


def test_invalid_yaml_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HeaderError, match="not valid YAML"):
        load_worker_file(write(tmp_path, with_header("outcomes: [done\n")))


def test_outcome_tied_to_unknown_status_key_is_refused(tmp_path: Path) -> None:
    text = with_header(VALID.replace("abandoned: ready", "abandoned: shelved"))

    with pytest.raises(HeaderError, match="'abandoned'.*unknown status key 'shelved'"):
        load_worker_file(write(tmp_path, text))


def test_duplicate_status_key_is_refused(tmp_path: Path) -> None:
    text = with_header(VALID.replace("in_review,", "ready,"))

    with pytest.raises(HeaderError, match="duplicates"):
        load_worker_file(write(tmp_path, text))


def test_unknown_outcome_is_refused(tmp_path: Path) -> None:
    text = with_header(
        VALID.replace("abandoned: ready", "abandoned: ready\n  merged: done")
    )

    with pytest.raises(HeaderError, match="outcomes.merged"):
        load_worker_file(write(tmp_path, text))


def test_missing_outcome_is_refused(tmp_path: Path) -> None:
    text = with_header(VALID.replace("  needs_human: blocked\n", ""))

    with pytest.raises(HeaderError, match="outcomes.needs_human: Field required"):
        load_worker_file(write(tmp_path, text))


@pytest.mark.parametrize("value", ["0", "-5", "1.5", "'60'", "true"])
def test_ceiling_not_a_positive_int_is_refused(tmp_path: Path, value: str) -> None:
    text = with_header(VALID.replace("max_turns: 60", f"max_turns: {value}"))

    with pytest.raises(HeaderError, match="ceilings.max_turns"):
        load_worker_file(write(tmp_path, text))


def test_extra_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HeaderError, match="owner: Extra inputs"):
        load_worker_file(write(tmp_path, with_header(VALID + "owner: someone\n")))


def test_front_matter_not_a_mapping_is_refused_naming_the_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        HeaderError, match=r"invalid header: header: .*dictionary.*\(got \['ready'\]\)"
    ):
        load_worker_file(write(tmp_path, with_header("- ready\n")))


def test_header_wide_error_does_not_dump_the_header(tmp_path: Path) -> None:
    text = with_header(VALID.replace("in_review,", "ready,"))

    with pytest.raises(HeaderError) as caught:
        load_worker_file(write(tmp_path, text))

    assert "(got" not in str(caught.value)
    assert "ceilings" not in str(caught.value)


def test_empty_status_key_list_is_refused(tmp_path: Path) -> None:
    text = with_header(
        VALID.replace("[ready, in_progress, in_review, blocked, done]", "[]")
    )

    with pytest.raises(HeaderError, match="invalid header: status_keys: "):
        load_worker_file(write(tmp_path, text))


def test_empty_status_key_is_refused(tmp_path: Path) -> None:
    text = with_header(VALID.replace("in_progress,", "'',"))

    with pytest.raises(HeaderError, match="status_keys.1: "):
        load_worker_file(write(tmp_path, text))


def test_loaded_header_is_immutable(tmp_path: Path) -> None:
    header = load_worker_file(write(tmp_path, with_header(VALID))).header

    with pytest.raises(ValidationError, match="frozen"):
        header.ceilings = {}  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        header.ceilings["max_turns"] = 10**9  # pyright: ignore[reportIndexIssue]
    assert header.ceilings == {"run_timeout_seconds": 1800, "max_turns": 60}


def test_empty_ceilings_are_refused(tmp_path: Path) -> None:
    text = with_header(VALID.split("ceilings:")[0] + "ceilings: {}\n")

    with pytest.raises(HeaderError, match="invalid header: ceilings: "):
        load_worker_file(write(tmp_path, text))


def test_empty_ceiling_name_is_refused(tmp_path: Path) -> None:
    text = with_header(VALID.replace("max_turns: 60", "'': 60"))

    with pytest.raises(HeaderError, match="invalid header: ceilings"):
        load_worker_file(write(tmp_path, text))


@pytest.mark.parametrize(
    ("front_matter", "key"),
    [
        (VALID + "  max_turns: 10\n", "max_turns"),
        (VALID.replace("outcomes:\n", "outcomes:\n  done: done\n"), "done"),
        (VALID + "status_keys: [done]\n", "status_keys"),
    ],
)
def test_duplicate_yaml_key_is_refused(
    tmp_path: Path, front_matter: str, key: str
) -> None:
    with pytest.raises(HeaderError, match=f"duplicate key '{key}'"):
        load_worker_file(write(tmp_path, with_header(front_matter)))


def test_unhashable_yaml_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HeaderError, match="not valid YAML"):
        load_worker_file(write(tmp_path, with_header(VALID + "? [a]\n: 1\n")))


def test_byte_order_mark_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "WORKER.md"
    path.write_bytes(with_header(VALID).encode("utf-8-sig"))

    assert load_worker_file(path).header.ceilings["max_turns"] == 60


def test_worker_file_not_in_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "WORKER.md"
    path.write_bytes(with_header(VALID, "Café.\n").encode("latin-1"))

    with pytest.raises(HeaderError):
        load_worker_file(path)


def test_header_nested_too_deeply_is_refused(tmp_path: Path) -> None:
    with pytest.raises(HeaderError):
        load_worker_file(write(tmp_path, with_header("outcomes: " + "[" * 5000 + "\n")))
