"""Reading and validation of the front matter of a `WORKER.md` file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

FENCE = "---"

StatusKey = Annotated[str, Field(min_length=1)]
Ceiling = Annotated[int, Field(strict=True, gt=0)]


class HeaderError(ValueError):
    """Raised when a `WORKER.md` file has no valid header."""


class Outcomes(BaseModel):
    """The status key each possible outcome of a run is tied to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    done: StatusKey
    blocked: StatusKey
    needs_human: StatusKey
    abandoned: StatusKey


class Header(BaseModel):
    """The validated front matter of a `WORKER.md` file.

    `ceilings` maps a limit name to its absolute ceiling, a positive integer; a lower value
    is always tighter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status_keys: tuple[StatusKey, ...] = Field(min_length=1)
    outcomes: Outcomes
    ceilings: dict[str, Ceiling]

    @model_validator(mode="after")
    def _check_status_keys(self) -> Self:
        """Refuse a duplicated status key, and an outcome tied to an undeclared one."""
        if len(set(self.status_keys)) != len(self.status_keys):
            raise ValueError(f"status_keys has duplicates: {list(self.status_keys)}")
        for outcome, key in self.outcomes.model_dump().items():
            if key not in self.status_keys:
                raise ValueError(
                    f"outcome {outcome!r} is tied to unknown status key {key!r}"
                )
        return self


@dataclass(frozen=True)
class WorkerFile:
    """A `WORKER.md` file split into its validated header and its prose body."""

    header: Header
    body: str


def _split(text: str) -> tuple[str, str]:
    """Return the front matter and the body of `text`.

    Raises `HeaderError` when the text does not open on a fence line or the fence is not closed.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != FENCE:
        raise HeaderError(
            f"front matter missing: the file must open on a {FENCE!r} line"
        )
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == FENCE:
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise HeaderError(f"front matter unterminated: no closing {FENCE!r} line")


def _describe(error: ValidationError) -> str:
    """Return one line per validation failure, naming its location and offending value."""
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(item) for item in detail["loc"]) or "header"
        parts.append(f"{location}: {detail['msg']} (got {detail['input']!r})")
    return "; ".join(parts)


def load_worker_file(path: Path) -> WorkerFile:
    """Read the `WORKER.md` file at `path`, UTF-8, and validate its header.

    Raises `HeaderError` when the front matter is missing, unterminated, not valid YAML,
    or does not match `Header`.
    """
    front_matter, body = _split(path.read_text(encoding="utf-8"))
    try:
        data = yaml.safe_load(front_matter)
    except yaml.YAMLError as error:
        raise HeaderError(f"front matter is not valid YAML: {error}") from error
    try:
        header = Header.model_validate(data)
    except ValidationError as error:
        raise HeaderError(f"invalid header: {_describe(error)}") from error
    return WorkerFile(header=header, body=body)
