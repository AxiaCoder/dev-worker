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
_PATH_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_DOMAIN_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_DOMAIN = rf"{_DOMAIN_COMPONENT}(?:\.{_DOMAIN_COMPONENT})*"
_HOST_LIKE = r"(?:[^/]*[.:A-Z][^/]*|localhost)/"
IMAGE_REFERENCE = re.compile(
    rf"(?:(?={_HOST_LIKE})(?P<host>{_DOMAIN}(?::[0-9]+)?)/|(?!{_HOST_LIKE}))"
    rf"(?P<path>{_PATH_COMPONENT}(?:/{_PATH_COMPONENT})*)"
    r"(?::(?P<tag>[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}))?"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}|sha384:[0-9a-f]{96}|sha512:[0-9a-f]{128}))?"
)
PATH_LENGTH_MAX = 255
_DOCKER_HUB_HOSTS = (None, "docker.io", "index.docker.io")
_OFFICIAL_PREFIX = "library/"


def _normalized_path(host: str | None, path: str) -> str:
    """Return `path` as Docker normalizes it: under `library/` when it is a lone Docker Hub component."""
    if host in _DOCKER_HUB_HOSTS and "/" not in path:
        return _OFFICIAL_PREFIX + path
    return path


class InstanceError(ValueError):
    """Raised when an instance file is invalid or loosens its header."""


def _check_not_blank(value: str) -> str:
    """Return `value` unchanged when it holds a non-blank character; raise `ValueError` otherwise."""
    if not value.strip():
        raise ValueError("must not be blank")
    return value


Name = Annotated[str, AfterValidator(_check_not_blank)]


def _check_env_name(value: str) -> str:
    """Return `value` when it is an environment variable name; raise `ValueError` otherwise."""
    if not ENV_NAME.fullmatch(value):
        raise ValueError(
            "must be the name of an environment variable (capitals, digits and"
            " underscores, e.g. GITHUB_TOKEN), not the token itself"
        )
    return value


EnvName = Annotated[str, AfterValidator(_check_env_name)]


def _check_branch_template(value: str) -> str:
    """Return `value` when it contains `{key}` and no other brace; raise `ValueError` otherwise."""
    rest = value.replace("{key}", "")
    if rest == value or "{" in rest or "}" in rest:
        raise ValueError(
            "must contain {key}, which the ticket key replaces, and no other brace,"
            " e.g. agent/{key}"
        )
    return value


BranchTemplate = Annotated[str, AfterValidator(_check_branch_template)]


def _check_pinned_image(value: str) -> str:
    """Return `value` when it names an image by digest or by a tag other than `latest`.

    The reference reads `[host[:port]/]path[:tag][@digest]`. The first component is a host
    when it holds a `.`, a `:` or an uppercase letter, or is `localhost`, as Docker splits
    it, and must then be a valid domain with an optional numeric port; path components are
    lowercase ASCII joined by `.`, `_`, `__` or a run of `-`; the path, without host, tag or
    digest and prefixed with `library/` when it is a single component on Docker Hub, holds
    at most 255 characters; a tag is ASCII; a digest is `sha256`, `sha384` or `sha512`
    followed by `:` and 64, 96 or 128 lowercase hex digits. Raises `ValueError` when the
    reference is malformed, or when it is well formed but pins no version.
    """
    match = IMAGE_REFERENCE.fullmatch(value)
    if (
        match is None
        or len(_normalized_path(match["host"], match["path"])) > PATH_LENGTH_MAX
    ):
        raise ValueError("is not a valid image reference, e.g. python:3.12-slim")
    if not match["digest"] and match["tag"] in (None, "latest"):
        raise ValueError(
            "must pin a version, with a tag other than latest or a digest,"
            " e.g. python:3.12-slim"
        )
    return value


Image = Annotated[Name, AfterValidator(_check_pinned_image)]


class Tracker(BaseModel):
    """The ticket tracker of an instance, its account and the variable holding its token.

    `site` is the Jira site address and `user` the email of the account the token belongs to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["jira"]
    site: Name
    user: Name
    project: Name
    token_env: EnvName


class Repository(BaseModel):
    """The code repository of an instance and the environment variable holding its token."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["github"]
    name: Name
    token_env: EnvName


class Agent(BaseModel):
    """The AI agent a run starts in its container, and the environment variable holding its token.

    `token_env` names the variable holding the Claude Code OAuth token.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_env: EnvName


class Container(BaseModel):
    """The container a run starts: `image` is its image, pinned by a tag or a digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image: Image


class GitIdentity(BaseModel):
    """The author name and email of the commits the worker's code makes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Name
    email: Name


class _InstanceData(BaseModel):
    """An instance file as written, before it is checked against a header."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tracker: Tracker
    repository: Repository
    agent: Agent
    branch: BranchTemplate
    container: Container
    git: GitIdentity
    statuses: Mapping[StatusKey, Name]
    hold_merge_label: Name
    limits: Mapping[CeilingName, Ceiling] = Field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """A validated instance, resolved against its header.

    `branch` is the branch name template, in which `{key}` stands for the ticket key.
    `statuses` maps every status key of the header to the tracker's real status name.
    `hold_merge_label` is the ticket label that holds a merge back.
    `limits` maps every ceiling name of the header to its effective value: the instance's
    own when it sets one, the header ceiling otherwise. Both mappings are read-only.
    """

    tracker: Tracker
    repository: Repository
    agent: Agent
    branch: str
    container: Container
    git: GitIdentity
    statuses: Mapping[str, str]
    hold_merge_label: str
    limits: Mapping[str, int]


def _status_problems(statuses: Mapping[str, str], header: Header) -> list[str]:
    """Return one sentence per status key of `statuses` unknown to `header`, or missing.

    Adds one sentence per real status name shared by several header keys, which it quotes
    and follows with those keys in header order. Names are compared exactly, case and
    whitespace included.
    """
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
    keys_by_name: dict[str, list[str]] = {}
    for key in header.status_keys:
        if key in statuses:
            keys_by_name.setdefault(statuses[key], []).append(key)
    problems += [
        f"statuses: {', '.join(keys)} share the real status {name!r},"
        " each status key needs its own"
        for name, keys in keys_by_name.items()
        if len(keys) > 1
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

    Raises `InstanceError` when the file is not valid UTF-8, is not valid YAML (a key repeated
    within one mapping included), nests too deeply to be read, does not match the instance shape, maps a status key the header does not
    declare or leaves one unmapped, maps two status keys to the same real status name, sets
    a limit the header has no ceiling for, or sets a limit above its ceiling. The message
    names key paths, lines and columns, and quotes no value written in the file but a
    shared real status name.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise InstanceError("instance file is not valid UTF-8") from None
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise InstanceError(
            f"instance file is not valid YAML{_yaml_reasons(error)}"
        ) from None
    except RecursionError:
        raise InstanceError("instance file nests too deeply to be read") from None
    try:
        data = _InstanceData.model_validate(raw)
    except ValidationError as error:
        description = _describe(error, root="instance", echo=False)
        raise InstanceError(f"invalid instance file: {description}") from None
    problems = _status_problems(data.statuses, header) + _limit_problems(
        data.limits, header
    )
    if problems:
        raise InstanceError(f"invalid instance file: {'; '.join(problems)}")
    return Instance(
        tracker=data.tracker,
        repository=data.repository,
        agent=data.agent,
        branch=data.branch,
        container=data.container,
        git=data.git,
        statuses=MappingProxyType(
            {key: data.statuses[key] for key in header.status_keys}
        ),
        hold_merge_label=data.hold_merge_label,
        limits=MappingProxyType({**header.ceilings, **data.limits}),
    )
