"""Reading and validation of an instance file, checked against a `WORKER.md` header."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from dev_worker.header import (
    Ceiling,
    CeilingName,
    Header,
    StatusKey,
    _describe,
    _UniqueKeyLoader,
)

ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")

Name = Annotated[str, Field(min_length=1)]


class InstanceError(ValueError):
    """Raised when an instance file is invalid or loosens its header."""


def _check_env_name(value: str) -> str:
    """Return `value` when it is an environment variable name; raise `ValueError` otherwise."""
    if not ENV_NAME.fullmatch(value):
        raise ValueError(
            "must be the name of an environment variable (capitals, digits and"
            " underscores, e.g. GITHUB_TOKEN), not the token itself"
        )
    return value


EnvName = Annotated[str, AfterValidator(_check_env_name)]


class Tracker(BaseModel):
    """The ticket tracker of an instance and the environment variable holding its token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["jira"]
    project: Name
    token_env: EnvName


class Repository(BaseModel):
    """The code repository of an instance and the environment variable holding its token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["github"]
    name: Name
    token_env: EnvName


class _InstanceData(BaseModel):
    """An instance file as written, before it is checked against a header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracker: Tracker
    repository: Repository
    statuses: Mapping[StatusKey, Name]
    limits: Mapping[CeilingName, Ceiling] = Field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """A validated instance, resolved against its header.

    `statuses` maps every status key of the header to the tracker's real status name.
    `limits` maps every ceiling name of the header to its effective value: the instance's
    own when it sets one, the header ceiling otherwise. Both mappings are read-only.
    """

    tracker: Tracker
    repository: Repository
    statuses: Mapping[str, str]
    limits: Mapping[str, int]


def _status_problems(statuses: Mapping[str, str], header: Header) -> list[str]:
    """Return one sentence per status key of `statuses` unknown to `header`, or missing."""
    declared = ", ".join(header.status_keys)
    problems = [
        f"statuses.{key}: unknown status key, the header declares {declared}"
        for key in statuses
        if key not in header.status_keys
    ]
    problems += [
        f"statuses.{key}: missing, every status key of the header needs a real status name"
        for key in header.status_keys
        if key not in statuses
    ]
    return problems


def _limit_problems(limits: Mapping[str, int], header: Header) -> list[str]:
    """Return one sentence per limit of `limits` unknown to `header`, or looser than its ceiling."""
    problems: list[str] = []
    for name, value in limits.items():
        ceiling = header.ceilings.get(name)
        if ceiling is None:
            known = ", ".join(header.ceilings)
            problems.append(f"limits.{name}: unknown limit, the header sets {known}")
        elif value > ceiling:
            problems.append(
                f"limits.{name}: {value} is looser than the header ceiling {ceiling}"
            )
    return problems


def _yaml_reasons(error: yaml.YAMLError) -> str:
    """Return the reasons of a YAML failure, each led by `": "`, without the failing text.

    A reason is followed by its line and column, both counted from 1. The result is empty
    for an error that carries no reason, such as an unreadable character.
    """
    reasons = (
        (getattr(error, "context", None), getattr(error, "context_mark", None)),
        (getattr(error, "problem", None), getattr(error, "problem_mark", None)),
    )
    return "".join(
        f": {reason} (line {mark.line + 1}, column {mark.column + 1})"
        if mark
        else f": {reason}"
        for reason, mark in reasons
        if reason
    )


def load_instance(path: Path, header: Header) -> Instance:
    """Read the YAML instance file at `path`, UTF-8 with an optional BOM, against `header`.

    Raises `InstanceError` when the file is not valid YAML (a key repeated within one mapping
    included), does not match the instance shape, maps a status key the header does not
    declare or leaves one unmapped, sets a limit the header has no ceiling for, or sets a
    limit above its ceiling. The message names key paths, lines and columns, never a value
    written in the file.
    """
    try:
        raw = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise InstanceError(
            f"instance file is not valid YAML{_yaml_reasons(error)}"
        ) from error
    try:
        data = _InstanceData.model_validate(raw)
    except ValidationError as error:
        description = _describe(error, root="instance", echo=False)
        raise InstanceError(f"invalid instance file: {description}") from error
    problems = _status_problems(data.statuses, header) + _limit_problems(
        data.limits, header
    )
    if problems:
        raise InstanceError(f"invalid instance file: {'; '.join(problems)}")
    return Instance(
        tracker=data.tracker,
        repository=data.repository,
        statuses=MappingProxyType(
            {key: data.statuses[key] for key in header.status_keys}
        ),
        limits=MappingProxyType({**header.ceilings, **data.limits}),
    )
