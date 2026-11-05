"""Models of the worker contract, version 0.1: mandate input and output, state, health.

Copied as is from the Broker worker, without its Broker-specific `wake` result; to be
extracted into a shared library once a third worker needs it.
"""

from typing import Literal

from pydantic import BaseModel, Field, JsonValue

CONTRACT_VERSION = "0.1"

MandateStatus = Literal["running", "waiting", "completed", "failed", "stopped"]
HealthStatus = Literal["ok", "degraded", "down"]


class MandateInput(BaseModel):
    """A mandate as received on stdin by `run`."""

    contract_version: str
    task_id: str
    mandate: str
    caller: str
    params: dict[str, JsonValue]
    options: dict[str, JsonValue] = Field(default_factory=dict)


class Deliverable(BaseModel):
    """Something the mandate produced, and where to find it."""

    kind: str
    location: str


class ApprovalRequest(BaseModel):
    """An action the worker will not take without the caller's approval."""

    action: str
    reason: str
    reversible: bool


class MandateOutput(BaseModel):
    """The result of a mandate, written on stdout by `run`.

    `task_id` is `None` only when the input could not be read far enough to find one.
    """

    contract_version: str = CONTRACT_VERSION
    task_id: str | None
    status: MandateStatus
    summary: str
    deliverables: list[Deliverable] = Field(default_factory=list)
    approval_request: ApprovalRequest | None = None
    errors: list[str] = Field(default_factory=list)


class CommonState(BaseModel):
    """The part of the state every worker reports the same way."""

    current: list[JsonValue] = Field(default_factory=list)
    recent: list[JsonValue] = Field(default_factory=list)
    pending: list[JsonValue] = Field(default_factory=list)
    cost: dict[str, JsonValue] = Field(default_factory=dict)


class State(BaseModel):
    """The worker state, written on stdout by `state`."""

    common: CommonState = Field(default_factory=CommonState)
    domain: dict[str, JsonValue] = Field(default_factory=dict)


class Health(BaseModel):
    """The worker health, written on stdout by `health`.

    `reason` is one sentence when `status` is not `"ok"`, `None` otherwise.
    """

    status: HealthStatus
    reason: str | None
    active: bool
