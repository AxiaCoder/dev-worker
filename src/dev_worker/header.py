"""Reading and validation of the front matter of a `WORKER.md` file."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

FENCE = "---"

StatusKey = Annotated[str, Field(min_length=1)]
CeilingName = Annotated[str, Field(min_length=1)]
Ceiling = Annotated[int, Field(strict=True, gt=0)]


class HeaderError(ValueError):
    """Raised when a `WORKER.md` file has no valid header."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a key repeated within one mapping."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        """Build the mapping of `node`; raise `ConstructorError` on a repeated key."""
        seen: set[Hashable] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, Hashable):
                if key in seen:
                    raise ConstructorError(
                        None, None, f"duplicate key {key!r}", key_node.start_mark
                    )
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


class Outcomes(BaseModel):
    """The status key each possible outcome of a run is tied to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    done: StatusKey
    blocked: StatusKey
    needs_human: StatusKey
    abandoned: StatusKey


class Plugin(BaseModel):
    """The Claude Code plugin a run loads: the git `repository` it comes from, at `tag`.

    Neither value is empty nor starts with `-`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    tag: str = Field(min_length=1)

    @field_validator("repository", "tag", mode="after")
    @classmethod
    def _refuse_option(cls, value: str) -> str:
        """Refuse a value git would read as an option."""
        if value.startswith("-"):
            raise ValueError("must not start with '-'")
        return value


class Header(BaseModel):
    """The validated front matter of a `WORKER.md` file.

    `ceilings` maps a limit name to its absolute ceiling, a positive integer; a lower value
    is always tighter. It holds at least one entry and is read-only. `plugin` names the
    plugin every run loads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status_keys: tuple[StatusKey, ...] = Field(min_length=1)
    outcomes: Outcomes
    ceilings: Mapping[CeilingName, Ceiling] = Field(min_length=1)
    plugin: Plugin

    @field_validator("ceilings", mode="after")
    @classmethod
    def _freeze_ceilings(cls, value: Mapping[str, int]) -> Mapping[str, int]:
        """Return `value` as a read-only mapping."""
        return MappingProxyType(dict(value))

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


def _describe(
    error: ValidationError,
    root: str = "header",
    echo: bool = True,
) -> str:
    """Return one line per validation failure, naming its location and, if `echo`, its value.

    A failure on the whole document is located at `root`. The value is left out when `echo`
    is false, or when it is the whole document mapping."""
    parts: list[str] = []
    for detail in error.errors():
        location = ".".join(str(item) for item in detail["loc"]) or root
        message = f"{location}: {detail['msg']}"
        if echo and (detail["loc"] or not isinstance(detail["input"], dict)):
            message += f" (got {detail['input']!r})"
        parts.append(message)
    return "; ".join(parts)


def load_worker_file(path: Path) -> WorkerFile:
    """Read the `WORKER.md` file at `path`, UTF-8 with an optional BOM, and validate its header.

    Raises `HeaderError` when the file is not valid UTF-8, or when the front matter is
    missing, unterminated, not valid YAML (a key repeated within one mapping included), nests
    too deeply to be read, or does not match `Header`.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise HeaderError("WORKER.md is not valid UTF-8") from None
    front_matter, body = _split(text)
    try:
        data = yaml.load(front_matter, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise HeaderError(f"front matter is not valid YAML: {error}") from error
    except RecursionError:
        raise HeaderError("front matter nests too deeply to be read") from None
    try:
        header = Header.model_validate(data)
    except ValidationError as error:
        raise HeaderError(f"invalid header: {_describe(error)}") from error
    return WorkerFile(header=header, body=body)
