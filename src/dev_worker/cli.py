"""Command line entry point: the `run`, `wake`, `state` and `health` commands.

Each command names the instance file with `--instance`; `run` runs a mandate read as JSON
on stdin against that instance and the `WORKER.md` shipped with the package, and `wake`
runs the `deliver_ticket` mandate on the first ticket ready to be taken.
"""

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.resources import as_file, files
from pathlib import Path
from typing import assert_never

from pydantic import ValidationError

from dev_worker.container import (
    WORKDIR,
    Container,
    ContainerError,
    agent_command,
    review_command,
)
from dev_worker.contract import (
    CONTRACT_VERSION,
    Health,
    MandateInput,
    MandateOutput,
    State,
)
from dev_worker.github import GitHubRepository
from dev_worker.handover import handover_reason
from dev_worker.header import HeaderError, WorkerFile, load_worker_file
from dev_worker.instance import (
    CI_WAIT_MARGIN_SECONDS,
    Instance,
    InstanceError,
    Provider,
    Repository,
    Tracker,
    load_instance,
)
from dev_worker.jira import TWG_TIMEOUT_SECONDS, JiraTracker
from dev_worker.outcome import Outcome, UsageLimit, read_outcome, read_verdict
from dev_worker.plugin import PluginCheckout, PluginError
from dev_worker.repositories import (
    CodeRepository,
    HeadMovedError,
    PullRequest,
    RepositoryError,
)
from dev_worker.run_journal import (
    GUARD_FILE,
    JOURNAL_FILE,
    LOCK_FILE,
    TORN_FILE,
    JournalError,
    append_entry,
    holds_torn_line,
    instance_directory,
    journal_entry,
    journal_holds,
    parse_lock,
    process_runs_since,
    read_entries,
    read_lock,
    release_lock,
    remove_lock_if_unchanged,
    set_aside_torn_tail,
    take_lock,
    torn_tail,
)
from dev_worker.text import writable_text
from dev_worker.tickets import (
    REQUEUE_MARKER,
    Claim,
    Ticket,
    TicketTracker,
    TrackerError,
)
from dev_worker.validation import describe_validation_error
from dev_worker.workspace import Workspace, WorkspaceError

TICKET_KEY = re.compile(r"(.+)-([1-9][0-9]*)")
RECENT_RUNS = 10
CLAIM_TAKE_LIMIT_SECONDS = 300
CLAIM_EXPIRY_SECONDS = 600
TAKE_DATING_LIMIT_SECONDS = CLAIM_TAKE_LIMIT_SECONDS + TWG_TIMEOUT_SECONDS
USAGE_LIMIT_FILE = "usage-limit.json"
SHORT_SHA_LENGTH = 7


class Benign(str):
    """Why a run did not take its ticket when no part of the worker failed.

    The lock is held by a run that is not gone, another run claimed the ticket first,
    the ticket is not ready to be taken, or it was blocked for an empty required field. A `done` run is also left blocked for a
    `Benign` reason when its pull request is missing, its CI has not passed or has
    reported no check yet on its head commit, its guard review found a reachable defect,
    or its head moved before the merge.
    """


@dataclass(frozen=True)
class RunReport:
    """The output of a run, and whether the worker failed during it.

    `worker_failed` is false when the run stopped short for a `Benign` reason, or took
    its ticket and settled it by the outcome its container ended on with no error of the
    worker's own added, a `done` run left blocked only for a `Benign` reason; true
    otherwise.
    """

    output: MandateOutput
    worker_failed: bool


def failed(task_id: str | None, *errors: str) -> MandateOutput:
    """Build a failed mandate output carrying the given error messages.

    A character that cannot be encoded as UTF-8, such as a lone surrogate from an
    undecodable path, is written as its backslash escape (`\\udce9`), so the output
    can always be serialized.
    """
    return MandateOutput(
        task_id=task_id,
        status="failed",
        summary="The mandate could not be run.",
        errors=[writable_text(error) for error in errors],
    )


def deliver_ticket(
    mandate: MandateInput, instance: Instance, outcome: Outcome
) -> MandateOutput:
    """Return the output of the `deliver_ticket` mandate whose run ended on `outcome`.

    `done` completes it, its message as the summary; any other outcome fails it, the
    outcome and its message as the error.
    """
    if outcome.outcome == "done":
        return MandateOutput(
            task_id=mandate.task_id, status="completed", summary=outcome.message
        )
    return failed(
        mandate.task_id,
        f"The run ended on the outcome {outcome.outcome!r}: {outcome.message}",
    )


type Handler = Callable[[MandateInput, Instance, Outcome], MandateOutput]

MANDATES: dict[str, Handler] = {
    "deliver_ticket": deliver_ticket,
}


def is_utf8_encodable(value: object) -> bool:
    """Tell whether a parsed JSON value can be written back as UTF-8.

    False when any string in it holds a lone surrogate.
    """
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def os_reason(error: OSError) -> str:
    """Return the reason of `error` as written in a message: its strerror, or its text.

    Copied from the Broker worker, with the error's text instead of its type as fallback.
    """
    return error.strerror or str(error)


def check_ticket_key(mandate: MandateInput, instance: Instance) -> str | None:
    """Return why `params.ticket_key` of `mandate` is refused, or None when it is valid.

    A valid key is a string `<PROJECT>-<n>`, where `<PROJECT>` is the tracker project
    of `instance` and `<n>` a positive integer written without a leading zero.
    """
    key = mandate.params.get("ticket_key")
    project = instance.tracker.project
    if key is None:
        return "params.ticket_key is missing."
    if not isinstance(key, str):
        return "params.ticket_key must be a string."
    match = TICKET_KEY.fullmatch(key)
    if match is None:
        return f"params.ticket_key {key!r} is not of the form {project}-<number>."
    if match.group(1) != project:
        return f"Ticket {key!r} is not in the tracker project {project!r}."
    return None


def make_tracker(tracker: Tracker) -> TicketTracker:
    """Return the ticket tracker that `tracker` of an instance describes."""
    return JiraTracker(tracker)


def make_repository(
    repository: Repository, token_env: str | None = None
) -> CodeRepository:
    """Return the code repository that `repository` of an instance describes.

    It calls the repository with the token held by the variable `token_env` names, the
    worker's token variable when `token_env` is `None`.
    """
    return GitHubRepository(repository, token_env=token_env)


def make_workspace(repository: Repository, directory: Path) -> Workspace:
    """Return the workspace of `repository` kept in the instance directory `directory`."""
    return Workspace(repository, directory)


def make_plugin(directory: Path) -> PluginCheckout:
    """Return the plugin checkout kept in the instance directory `directory`."""
    return PluginCheckout(directory)


def make_container(image: str, directory: Path) -> Container:
    """Return the container a run of the instance kept in `directory` starts from `image`.

    The instance is named by the absolute path of `directory`, with its links and `..`
    steps resolved, so that every spelling of it gives the same name (see `Container`).
    """
    return Container(image, str(directory.resolve()))


def unreachable_docker(instance: Instance, directory: Path) -> str | None:
    """Return why the Docker daemon cannot be reached, `None` when it can. Never raises.

    `directory` is the instance directory of `instance`.
    """
    try:
        make_container(instance.container.image, directory).check_access()
    except ContainerError as error:
        return f"Docker cannot be reached: {error}"
    return None


def unreachable_connector(instance: Instance) -> str | None:
    """Return why the tracker or then the repository of `instance` cannot be reached, `None` when both can.

    Each is asked to check its access with the instance credentials, the repository with
    the worker's token and then the agent's, and the sentence names the token that
    failed; nothing is changed on either. Never raises.
    """
    try:
        make_tracker(instance.tracker).check_access()
    except TrackerError as error:
        return f"The tracker cannot be reached: {error}"
    tokens = (
        ("worker's", instance.repository.worker_token_env),
        ("agent's", instance.repository.agent_token_env),
    )
    for owner, token_env in tokens:
        try:
            make_repository(instance.repository, token_env).check_access()
        except RepositoryError as error:
            return f"The repository cannot be reached with the {owner} token: {error}"
    return None


def read_ready_ticket(
    key: str, instance: Instance, tracker: TicketTracker
) -> Ticket | str:
    """Return the ticket `key` read from `tracker`, or why it cannot be run.

    It is refused when the tracker cannot read it,
    when its status is not the real status of the `ready` key of `instance`, or when a
    ticket that blocks it is not in the real status of the `done` key; the last two
    reasons are `Benign`.
    """
    try:
        ticket = tracker.read(key)
    except TrackerError as error:
        return f"Ticket {key!r} cannot be read from the tracker: {error}"
    ready = instance.statuses["ready"]
    if ticket.status != ready:
        return Benign(f"Ticket {key!r} is in status {ticket.status!r}, not {ready!r}.")
    done = instance.statuses["done"]
    open_blockers = [
        f"{blocker.key} ({blocker.status!r})"
        for blocker in ticket.blockers
        if blocker.status != done
    ]
    if open_blockers:
        return Benign(
            f"Ticket {key!r} is blocked by {', '.join(open_blockers)}, "
            f"not in status {done!r}."
        )
    return ticket


@dataclass(frozen=True)
class ClaimStart:
    """The `time.monotonic()` and `time.time()` readings taken when a run began posting its claim."""

    monotonic: float
    wall: float


def start_claim() -> ClaimStart:
    """Read both clocks now, as the start of a claim."""
    return ClaimStart(monotonic=time.monotonic(), wall=time.time())


def claim_overdue(start: ClaimStart) -> bool:
    """Return whether more than `CLAIM_TAKE_LIMIT_SECONDS` have passed since `start` on either clock.

    The wall clock counts time the machine spent suspended; the monotonic clock counts
    time while the wall clock was set back.
    """
    return (
        time.monotonic() - start.monotonic > CLAIM_TAKE_LIMIT_SECONDS
        or time.time() - start.wall > CLAIM_TAKE_LIMIT_SECONDS
    )


def claim_ticket(
    key: str,
    task_id: str,
    instance: Instance,
    tracker: TicketTracker,
    start: ClaimStart,
) -> Ticket | str:
    """Claim the ticket `key` for `task_id` through `tracker`; return the ticket read again once the claim wins, or why the run may not take it.

    The claims posted since the ticket last entered the real status of the `ready` key
    are judged in the order of their ids: a claim wins unless an older winning claim was
    created at most `CLAIM_EXPIRY_SECONDS` before it, by the tracker's clock. An older
    claim is otherwise left by a run that is gone, and a losing claim never blocks
    another. The run may take the ticket only when its own claim wins: of runs that both
    read the ticket ready, one only goes on. A run whose claim cannot be posted, whose claims
    cannot be read, whose claims are read back too late after `start` (see
    `claim_overdue`), or whose own claim is not found among them may not take it; the
    reason says whether the claim may have been posted. A run whose claim wins reads the
    ticket again and may not take it unless it is still in the `ready` status; that reading
    is the ticket returned. `start` is taken right before the claim is posted. A claim that loses, and a ticket read again
    out of `ready`, give a `Benign` reason.
    """
    ready = instance.statuses["ready"]
    try:
        own = tracker.claim(key, task_id)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be claimed for task {task_id!r}: {error} "
            "The claim may have been posted."
        )
    try:
        claims = tracker.claims(key, frozenset({ready}))
    except TrackerError as error:
        return (
            f"The claims on ticket {key!r} cannot be read: {error} "
            f"The claim of task {task_id!r} was posted."
        )
    if claim_overdue(start):
        return (
            f"The claims on ticket {key!r} were read back more than "
            f"{CLAIM_TAKE_LIMIT_SECONDS} seconds after the claim of task {task_id!r} "
            "began to be posted. The claim was posted."
        )
    own_claims = [claim for claim in claims if claim.claim_id == own]
    if not own_claims:
        return (
            f"The claim of task {task_id!r} is not found among the claims on "
            f"ticket {key!r} since it last entered {ready!r}. "
            f"The claim of task {task_id!r} was posted."
        )
    expiry = timedelta(seconds=CLAIM_EXPIRY_SECONDS)
    winners: list[Claim] = []
    for claim in sorted(claims, key=lambda claim: claim.claim_id):
        if not any(winner.created >= claim.created - expiry for winner in winners):
            winners.append(claim)
    if all(winner.claim_id != own for winner in winners):
        oldest_live = own_claims[0].created - expiry
        rival = next(
            winner
            for winner in winners
            if winner.claim_id < own and winner.created >= oldest_live
        )
        return Benign(
            f"Ticket {key!r} is claimed by task {rival.task_id!r}. "
            f"The claim of task {task_id!r} was posted."
        )
    try:
        ticket = tracker.read(key)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be read again from the tracker: {error} "
            f"The claim of task {task_id!r} was posted."
        )
    if ticket.status != ready:
        return Benign(
            f"Ticket {key!r} is no longer in {ready!r}: it is in {ticket.status!r}. "
            f"The claim of task {task_id!r} was posted."
        )
    return ticket


def block_unfilled_ticket(
    ticket: Ticket, instance: Instance, tracker: TicketTracker
) -> str | None:
    """Block `ticket` when a field the instance requires is empty; return why the run may not take it, or None.

    The comment posted on the ticket names the empty fields by their shown name, then the
    ticket is moved to the real status of the `blocked` key, and the reason is `Benign`.
    A ticket whose comment fails is not moved. After a tracker failure the reason is not
    `Benign` and says whether the comment may have been posted, or that the status is
    unknown.
    """
    empty = ticket.empty_fields()
    if not empty:
        return None
    key = ticket.key
    named = ", ".join(empty)
    body = (
        f"This ticket cannot be taken, these required fields are empty: {named}. "
        f"Fill them, then move it back to {instance.statuses['ready']!r}."
    )
    try:
        tracker.comment(key, body)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be commented on: {error} The comment may have "
            "been posted; the ticket was not moved."
        )
    blocked = instance.statuses["blocked"]
    try:
        tracker.set_status(key, blocked)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be moved to {blocked!r}: {error} "
            "Its status is unknown and is to be read again in the tracker."
        )
    return Benign(f"Ticket {key!r} has empty required fields: {named}.")


def take_ticket(
    key: str,
    task_id: str,
    instance: Instance,
    tracker: TicketTracker,
    start: ClaimStart,
) -> str | None:
    """Move the ticket `key` to the real status of the `in_progress` key; return why it failed, or None.

    The ticket is not moved when the claim of `task_id`, begun at `start`, is overdue
    (see `claim_overdue`). After a tracker failure the reason says the status is
    unknown: the move may still have happened.
    """
    if claim_overdue(start):
        return (
            f"Ticket {key!r} was about to be moved more than "
            f"{CLAIM_TAKE_LIMIT_SECONDS} seconds after the claim of task {task_id!r} "
            "began to be posted. The claim was posted; the ticket was not moved."
        )
    in_progress = instance.statuses["in_progress"]
    try:
        tracker.set_status(key, in_progress)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be moved to {in_progress!r}: {error} "
            "Its status is unknown and is to be read again in the tracker."
        )
    return None


def run_branch(instance: Instance, key: str) -> str:
    """Return the branch the `branch` template of `instance` names for the ticket `key`."""
    return instance.branch.replace("{key}", key)


def land_done(
    key: str,
    summary: str,
    instance: Instance,
    worker: WorkerFile,
    tracker: TicketTracker,
    directory: Path,
    began: float,
) -> tuple[str, str, str | None, list[str]]:
    """Merge the pull request of a run on the ticket `key` that ended on `done`; return the status key the ticket goes to, the body of its comment, why it is blocked, and the errors that did not block it.

    The open pull request whose head is the run branch (see `run_branch`) is found, its
    CI read once, the ticket read again through `tracker`, then the pull request given
    its guard review (see `review_pull_request`), in the instance directory `directory`
    within the time the run started at `began` has left, the ticket read once more, and
    the pull request merged as one squashed commit under its title, tied to the reviewed
    head commit. The status key is `done` once merged; the one the header of `worker`
    ties the `done` outcome to when the ticket carries the `hold_merge_label` of
    `instance` at either read, the pull request left open, and not reviewed when it
    carries it at the first read. When `instance` declares a `handover`, the files of the
    pull request are read after that first read: when `handover_reason` gives a reason,
    the status key is that same `done` outcome's, the pull request left open, neither
    reviewed nor merged, and the body says why. The status key is the one the header ties
    `blocked` to when there is no pull request, its CI has
    not passed, the review found a reachable defect or gave no verdict, or a step fails,
    the files of the pull request unreadable included,
    the pull request left open and nothing retried. The body is `summary` followed by
    what happened, the defects found listed; after a merge that failed for a reason
    other than a moved head, it says the merge may have happened. The reason the ticket
    is blocked is `None` unless it goes to the `blocked` status key, and `Benign` when no
    pull request is open, its CI has not passed or has reported no check yet on the head
    commit, the review found a reachable defect or the head moved before the merge; any
    other reason, such as a repository that reports no check and has no CI to report
    one, is a failure of the worker's own chain. The errors hold why
    the review checkout could not be deleted, which neither blocks the ticket nor holds
    the merge back.
    """
    outcomes = worker.header.outcomes
    errors: list[str] = []

    def blocked(reason: str) -> tuple[str, str, str, list[str]]:
        """Return the landing of a ticket blocked for `reason`."""
        return (
            outcomes.blocked,
            f"{summary}\n\nThe ticket is blocked: {reason}",
            reason,
            errors,
        )

    branch = run_branch(instance, key)
    repository = make_repository(instance.repository)
    try:
        pull_request = repository.find_pull_request(branch)
        if pull_request is None:
            return blocked(
                Benign(f"no open pull request has the branch {branch} as its head.")
            )
        ci = repository.read_ci(pull_request)
        if ci == "unreported":
            return blocked(
                Benign(
                    "no CI check is reported yet on the head commit "
                    f"{pull_request.head} of the pull request {pull_request.url}; it "
                    "was not merged."
                )
            )
        if ci != "passed":
            state = "failed" if ci == "failed" else "has not finished"
            return blocked(
                Benign(
                    f"the CI of the pull request {pull_request.url} {state}; it was "
                    "not merged."
                )
            )
        held = instance.hold_merge_label in tracker.read(key).labels
    except (RepositoryError, TrackerError) as error:
        return blocked(str(error))
    left_open = (
        f"{summary}\n\nThe pull request {pull_request.url} is left open: the "
        f"ticket carries the label {instance.hold_merge_label!r}, which holds its "
        "merge back."
    )
    if held:
        return outcomes.done, left_open, None, errors
    if instance.handover is not None:
        try:
            files = repository.changed_files(pull_request)
        except RepositoryError as error:
            return blocked(str(error))
        handed_over = handover_reason(files, instance.handover)
        if handed_over is not None:
            left_for_human = (
                f"{summary}\n\nThe pull request {pull_request.url} is left open for "
                f"a human: {handed_over}"
            )
            return outcomes.done, left_for_human, None, errors
    reason, removal = review_pull_request(
        pull_request, instance, worker, directory, began
    )
    if removal is not None:
        errors.append(removal)
    if reason is not None:
        return blocked(reason)
    try:
        held = instance.hold_merge_label in tracker.read(key).labels
    except TrackerError as error:
        return blocked(str(error))
    if held:
        return outcomes.done, left_open, None, errors
    try:
        repository.merge(pull_request)
    except HeadMovedError as error:
        return blocked(Benign(error))
    except RepositoryError as error:
        return blocked(
            f"the pull request {pull_request.url} cannot be merged: {error} It may "
            "have been merged and is to be read again."
        )
    return "done", f"{summary}\n\nMerged: {pull_request.url}", None, errors


def settle_ticket(
    key: str,
    ending: Outcome | UsageLimit | str,
    instance: Instance,
    worker: WorkerFile,
    tracker: TicketTracker,
    directory: Path,
    began: float,
) -> tuple[str | None, list[str]]:
    """Comment on the ticket `key` then move it as the run's `ending` says; return why a `done` run left it blocked, and why either step failed.

    `ending` is the outcome the run ended on, usable, or why it ended without one. The
    `done` outcome first has its pull request reviewed and merged, in the instance
    directory `directory` within the time the run started at `began` has left, and the
    ticket is commented on and moved as `land_done` says. The comment is made writable as UTF-8 whatever it
    carries. Any other outcome has its message posted and the ticket
    moved to the real status of the status key the header of `worker` ties it to; a
    reason is posted and the ticket moved as the `blocked` outcome is. The ticket is
    moved even when the comment fails, and never to the real status of the `ready` key.
    After a tracker failure the sentence says whether the comment may have been posted,
    or that the status is unknown. The reason a `done` run left the ticket blocked is the
    one `land_done` gives, `None` for any other ending; the errors `land_done` gives
    that did not block it come first among the errors.

    A run that reached the usage limit of the agent's account is the one exception: its
    ticket is moved back to the real status of the `ready` key, as `settle_usage_limit`
    says, without counting as an attempt.
    """
    if isinstance(ending, UsageLimit):
        return None, settle_usage_limit(key, ending, instance, tracker, directory)
    tied = worker.header.outcomes.model_dump()
    blocking: str | None = None
    errors: list[str] = []
    if isinstance(ending, Outcome) and ending.outcome == "done":
        status_key, body, blocking, errors = land_done(
            key, ending.message, instance, worker, tracker, directory, began
        )
    elif isinstance(ending, Outcome):
        status_key, body = tied[ending.outcome], ending.message
    else:
        status_key, body = tied["blocked"], ending
    try:
        tracker.comment(key, writable_text(body))
    except TrackerError as error:
        errors.append(
            f"Ticket {key!r} cannot be commented on: {error} "
            "The comment may have been posted."
        )
    status = instance.statuses[status_key]
    if status == instance.statuses["ready"]:
        errors.append(
            f"Ticket {key!r} was not moved: the run ended tied to {status!r}, and a "
            "ticket is never moved back to the ready status."
        )
        return blocking, errors
    try:
        tracker.set_status(key, status)
    except TrackerError as error:
        errors.append(
            f"Ticket {key!r} cannot be moved to {status!r}: {error} "
            "Its status is unknown and is to be read again in the tracker."
        )
    return blocking, errors


def settle_usage_limit(
    key: str,
    limit: UsageLimit,
    instance: Instance,
    tracker: TicketTracker,
    directory: Path,
) -> list[str]:
    """Record when `limit` resets, comment on the ticket `key` then move it back to the ready status; return why a step failed.

    The UTC moment `limit` resets is written in ISO 8601 as `resets_at` in the
    `USAGE_LIMIT_FILE` of the instance directory `directory`, where `wake` reads it.
    The comment says the run reached the usage limit, why it gave no outcome, and when
    the worker takes a ticket again, in UTC; it does not start with `REQUEUE_MARKER`,
    so the run is not counted as an attempt. The ticket is moved to the real status of
    the `ready` key even when the file cannot be written or the comment fails. After a
    tracker failure the sentence says whether the comment may have been posted, or that
    the status is unknown.
    """
    errors: list[str] = []
    resets_at = limit.resets_at.astimezone(UTC)
    usage_file = directory / USAGE_LIMIT_FILE
    try:
        usage_file.write_text(
            json.dumps({"resets_at": resets_at.isoformat()}) + "\n", encoding="utf-8"
        )
    except OSError as error:
        errors.append(
            f"Usage limit file '{usage_file}' cannot be written: {os_reason(error)} "
            "The next wake may take a ticket before the limit resets."
        )
    ready = instance.statuses["ready"]
    body = (
        f"The run reached the usage limit of the agent's account. {limit.reason} "
        f"The ticket goes back to {ready!r} without counting as an attempt; the worker "
        f"takes no ticket before {resets_at:%Y-%m-%d %H:%M:%S} UTC."
    )
    try:
        tracker.comment(key, writable_text(body))
    except TrackerError as error:
        errors.append(
            f"Ticket {key!r} cannot be commented on: {error} "
            "The comment may have been posted."
        )
    try:
        tracker.set_status(key, ready)
    except TrackerError as error:
        errors.append(
            f"Ticket {key!r} cannot be moved to {ready!r}: {error} "
            "Its status is unknown and is to be read again in the tracker."
        )
    return errors


def usage_limit_reset(instance_path: Path) -> datetime | None:
    """Return when the usage limit a run of the instance at `instance_path` reached resets, while that moment is still to come.

    The moment is read from the `USAGE_LIMIT_FILE` of the instance directory (see
    `settle_usage_limit`). `None` when it is past, or when the file is missing, cannot
    be read or holds no timezone-aware ISO 8601 `resets_at`. Never raises on a file
    error.
    """
    try:
        recorded = json.loads(
            (instance_directory(instance_path) / USAGE_LIMIT_FILE).read_text(
                encoding="utf-8"
            )
        )
        resets_at = datetime.fromisoformat(recorded["resets_at"])
        return resets_at if resets_at > datetime.now(UTC) else None
    except (OSError, ValueError, RecursionError, KeyError, TypeError):
        return None


def with_errors(output: MandateOutput, errors: Sequence[str]) -> MandateOutput:
    """Return `output` with `errors` added after its own, each made writable as UTF-8."""
    if not errors:
        return output
    return output.model_copy(
        update={"errors": [*output.errors, *map(writable_text, errors)]}
    )


def run_journaled(
    handler: Handler,
    mandate: MandateInput,
    instance: Instance,
    worker: WorkerFile,
    instance_path: Path,
) -> RunReport:
    """Run `handler` under the lock of the instance directory and journal the run; return its output and whether the worker failed.

    `worker` is the `WORKER.md` the run follows (see `run_in_worktree`).
    The ticket key is checked first, then the ticket is read from the tracker and
    refused unless it is ready (see `read_ready_ticket`), then the lock is taken, or
    taken over from a run that is gone (see `take_over_lock`); a task_id already in the
    journal is refused, otherwise the ticket is claimed for the run (see `claim_ticket`)
    and read again, and that reading, not the first, is
    blocked when a field the instance requires is empty (see `block_unfilled_ticket`),
    and otherwise moved to the real status of the `in_progress` key (see `take_ticket`)
    before the run is journaled. A refused run, a run whose ticket is blocked, and one
    whose ticket cannot be claimed, blocked or moved, are not journaled. A run that took its ticket settles it once it ends (see
    `run_under_lock`). The lock is
    released only if no other run has taken it over meanwhile. Every file error
    becomes a failed output naming the reason, except a lock that cannot be removed:
    the run keeps its status and the reason is added to `errors`. An exception raised
    by `handler` propagates once the lock is released, the ticket left as it is.

    The worker failed unless the run was refused for a `Benign` reason, or took its
    ticket and settled it with no error of the worker's own (see `run_under_lock`), and
    its lock was released.
    """
    task_id = mandate.task_id
    refusal = check_ticket_key(mandate, instance)
    if refusal is not None:
        return worker_failure(task_id, refusal)
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    tracker = make_tracker(instance.tracker)
    ticket = read_ready_ticket(ticket_key, instance, tracker)
    if isinstance(ticket, str):
        return refused(task_id, ticket)
    directory = made_instance_directory(instance_path)
    if isinstance(directory, str):
        return worker_failure(task_id, directory)
    start = datetime.now(UTC).isoformat()
    began = time.monotonic()
    record = {
        "task_id": task_id,
        "ticket_key": ticket_key,
        "model": instance.agent.model,
        "review_model": instance.agent.review_model,
        "start": start,
        "pid": os.getpid(),
    }
    taken = take_instance_lock(directory, record, instance)
    if isinstance(taken, str):
        return refused(task_id, taken)
    lock = taken
    try:
        report = run_under_lock(
            handler, mandate, instance, worker, ticket, tracker, directory, start, began
        )
    finally:
        try:
            release_lock(lock, record)
        except OSError as error:
            release_error = f"Lock file '{lock}' cannot be removed: {os_reason(error)}"
        else:
            release_error = None
    if release_error is not None:
        output = report.output
        return RunReport(
            output.model_copy(update={"errors": [*output.errors, release_error]}),
            worker_failed=True,
        )
    return report


def made_instance_directory(instance_path: Path) -> Path | str:
    """Return the instance directory of `instance_path`, created if missing, or why it cannot be.

    Never raises on a file error.
    """
    try:
        directory = instance_directory(instance_path)
    except ValueError as error:
        return str(error)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return f"Instance directory '{directory}' cannot be created: {os_reason(error)}"
    return directory


def take_instance_lock(
    directory: Path,
    record: Mapping[str, object],
    instance: Instance,
) -> Path | str:
    """Take the lock of `directory`, writing `record` in it, or take it over from a run that is gone.

    A run whose lock names its process is gone once that process is gone; one whose
    lock names none, once it has held the lock beyond the `run_timeout_seconds` limit
    of `instance` (see `gone_reason`). Returns the lock file path, or why the lock
    cannot be taken; the reason is `Benign` when a run that is not gone holds it.
    Never raises on a file error.
    """
    try:
        return take_lock(directory, record)
    except FileExistsError:
        return take_over_lock(directory, record, instance.limits["run_timeout_seconds"])
    except OSError as error:
        return lock_unwritable(directory, error)


def worker_failure(task_id: str | None, *errors: str) -> RunReport:
    """Return the failed output carrying `errors`, as a failure of the worker."""
    return RunReport(failed(task_id, *errors), worker_failed=True)


def refused(task_id: str, reason: str) -> RunReport:
    """Return the failed output carrying `reason`; the worker failed unless `reason` is `Benign`."""
    return RunReport(
        failed(task_id, reason), worker_failed=not isinstance(reason, Benign)
    )


def run_under_lock(
    handler: Handler,
    mandate: MandateInput,
    instance: Instance,
    worker: WorkerFile,
    ticket: Ticket,
    tracker: TicketTracker,
    directory: Path,
    start: str,
    began: float,
) -> RunReport:
    """Run `handler` unless its task_id is already journaled, journaling the run twice; return its output and whether the worker failed.

    `worker` is the `WORKER.md` the run follows and `ticket` the ticket as read before
    the lock was taken; once the claim wins, the ticket read again by `claim_ticket`
    replaces it, both for the required fields check and for the run (see
    `run_in_worktree`).
    Before anything is journaled, a worktree an earlier run left in `directory` is
    rescued and removed (see `rescue_left_worktree`); a run that cannot do so fails
    without taking its ticket nor being journaled. Then the ticket is claimed for the run
    through `tracker` (see `claim_ticket`): the claim must be read back in time, win
    over the live claims and find the ticket still ready. A ticket that leaves a field the
    instance requires empty is then commented on and blocked, and the run refused for a
    `Benign` reason (see `block_unfilled_ticket`). The ticket is otherwise moved to the
    real status of the `in_progress` key; a run whose ticket cannot be claimed, blocked or
    moved fails without being journaled. A `running` line is appended before `handler` runs, so that the
    task is never run again even if its final line, appended once `handler` returns,
    cannot be written; the output is then kept, with the journal error added to
    `errors`. A run whose `running` line cannot be written fails without running
    `handler`. Once the `running` line is written, `handler` runs in a fresh worktree,
    after a disposable container (see `run_in_worktree`). `start` is the ISO 8601 UTC
    start of the run, written as is in both lines, and `began` the `time.monotonic()`
    reading taken with it. A torn last journal line is first set aside in the torn file.

    Once its ticket is taken, the run settles it before the final line is written (see
    `settle_ticket`): by the outcome the run ended on when it is usable and its work
    rescued, back to ready when the run reached the usage limit of the agent's account
    and its work was rescued, as `blocked` with the reason otherwise, a `running` line
    that cannot be written included. A `done` run has its pull request given its guard review, after
    its worktree is removed and within the time left since `began`, before it is merged. A run that ended on `done` but whose ticket is left blocked fails,
    the reason as its first error ahead of the errors its output already carried, and
    is journaled `failed`. A tracker failure while
    settling adds its reason to `errors`. An exception raised by `handler` leaves the
    ticket as it is.

    The worker did not fail when the run was refused for a `Benign` reason, or when it
    ended on an outcome of its container or on the usage limit, its worktree removed, its ticket commented on
    and moved and its final line written; a `done` run left blocked for a `Benign`
    reason included (see `land_done`). It failed otherwise, a run that ended without a
    usable outcome, and a `done` run left blocked for any other reason, included.
    """
    task_id = mandate.task_id
    try:
        set_aside_torn_tail(directory)
        repeated = journal_holds(directory, task_id)
    except OSError as error:
        return worker_failure(
            task_id,
            f"Journal in '{directory}' cannot be read or repaired: {os_reason(error)}",
        )
    except JournalError as error:
        return worker_failure(task_id, f"Journal in '{directory}': {error}")
    if repeated:
        return worker_failure(task_id, f"Task {task_id!r} has already been run.")
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    refusal = rescue_left_worktree(instance, directory)
    if refusal is None:
        claim_start = start_claim()
        claimed = claim_ticket(ticket_key, task_id, instance, tracker, claim_start)
        if isinstance(claimed, str):
            refusal = claimed
        else:
            ticket = claimed
            refusal = block_unfilled_ticket(ticket, instance, tracker) or take_ticket(
                ticket_key, task_id, instance, tracker, claim_start
            )
    if refusal is not None:
        return refused(task_id, refusal)
    entry: dict[str, object] = {
        "task_id": task_id,
        "ticket_key": mandate.params["ticket_key"],
        "model": instance.agent.model,
        "review_model": instance.agent.review_model,
        "status": "running",
        "start": start,
        "duration": None,
        "tokens": 0,
    }
    try:
        append_entry(directory, entry)
    except OSError as error:
        reason = f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        _, errors = settle_ticket(
            ticket_key, reason, instance, worker, tracker, directory, began
        )
        return RunReport(with_errors(failed(task_id, reason), errors), True)
    mandate_began = time.monotonic()
    output, ending, closed = run_in_worktree(
        handler, mandate, instance, worker, ticket, directory, began
    )
    blocking, errors = settle_ticket(
        ticket_key, ending, instance, worker, tracker, directory, began
    )
    if blocking is not None:
        output = failed(
            task_id,
            f"The run ended on 'done' but its ticket is blocked: {blocking}",
            *output.errors,
        )
    output = with_errors(output, errors)
    settled = (
        isinstance(ending, Outcome | UsageLimit)
        and closed
        and not errors
        and (blocking is None or isinstance(blocking, Benign))
    )
    entry |= {"status": output.status, "duration": time.monotonic() - mandate_began}
    try:
        append_entry(directory, entry)
    except OSError as error:
        journal_error = (
            f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        )
        return RunReport(
            output.model_copy(
                update={"errors": [*output.errors, writable_text(journal_error)]}
            ),
            worker_failed=True,
        )
    return RunReport(output, worker_failed=not settled)


def end_worktree(
    workspace: Workspace, branch: str, instance: Instance, directory: Path
) -> str | None:
    """Rescue then close the worktree of `workspace` on `branch`; return why it failed, or None.

    A worktree whose rescue fails is not closed: it stays on the host, with its local
    branch, for the next run to rescue (see `rescue_worktree`).
    """
    return rescue_worktree(workspace, branch, instance, directory) or close_worktree(
        workspace, branch
    )


def rescue_worktree(
    workspace: Workspace, branch: str, instance: Instance, directory: Path
) -> str | None:
    """Rescue the work in the worktree of `workspace` on `branch`; return why it failed, or None.

    The rescue commits with the git identity of `instance` (see `Workspace.rescue`). It
    fails while a container of the instance kept in `directory` is running, or when
    Docker cannot tell (see `Container.check_none_running`).
    """
    try:
        make_container(instance.container.image, directory).check_none_running()
        workspace.rescue(branch, instance.git)
    except (ContainerError, WorkspaceError) as error:
        return f"The work of the run cannot be rescued, its worktree is kept: {error}"
    return None


def close_worktree(workspace: Workspace, branch: str) -> str | None:
    """Remove the worktree of `workspace` on `branch` and its local branch; return why it failed, or None."""
    try:
        workspace.close(branch)
    except WorkspaceError as error:
        return f"The worktree cannot be removed: {error}"
    return None


def rescue_left_worktree(instance: Instance, directory: Path) -> str | None:
    """Rescue then remove a worktree an earlier run left in `directory`; return why it failed, or None.

    When no worktree is left, the local branches earlier runs left in the mirror are
    deleted when the remote holds their tip, and kept otherwise (see
    `Workspace.drop_left_branches`). Otherwise the branch of the worktree is read from
    the mirror's record of it (see `Workspace.left_branch`), the containers of the
    instance still running are killed (see `Container.kill_running`), the mirror is
    refreshed, and the worktree is ended as a run ends its own (see `end_worktree`).
    """
    workspace = make_workspace(instance.repository, directory)
    try:
        branch = workspace.left_branch()
        if branch is None:
            workspace.drop_left_branches()
            return None
        make_container(instance.container.image, directory).kill_running()
        workspace.refresh()
    except (ContainerError, WorkspaceError) as error:
        return f"The worktree left by an earlier run cannot be rescued: {error}"
    reason = end_worktree(workspace, branch, instance, directory)
    if reason is None:
        return None
    return f"The worktree left by an earlier run on the branch {branch}: {reason}"


def agent_credential(provider: Provider) -> tuple[str, str]:
    """Return the variable Claude Code reads the credential of `provider` from, and what that credential is called."""
    match provider:
        case "subscription":
            return "CLAUDE_CODE_OAUTH_TOKEN", "Claude Code subscription token"
        case "api_key":
            return "ANTHROPIC_API_KEY", "Anthropic API key"
        case _:  # pragma: no cover
            assert_never(provider)


def container_environment(instance: Instance) -> dict[str, str] | str:
    """Return the variables the container of `instance` receives, by name, or why one is missing.

    They are the agent credential, `GH_TOKEN` and `TWG_TOKEN`, holding the values, read
    now, of the variables `agent.token_env`, `repository.agent_token_env` and
    `tracker.token_env` name; the worker's repository token is never among them. Then
    `TWG_SITE` and `TWG_USER`, the tracker site and account, and `IS_SANDBOX` set to `1`,
    which lets Claude Code run in `bypassPermissions` mode as root. The agent credential is
    `CLAUDE_CODE_OAUTH_TOKEN` for the `subscription` provider and `ANTHROPIC_API_KEY` for
    `api_key`. A token variable unset or empty gives the sentence naming it; no value is
    ever quoted.
    """
    agent_variable, agent_service = agent_credential(instance.agent.provider)
    environment = read_tokens(
        (
            (agent_variable, instance.agent.token_env, agent_service),
            ("GH_TOKEN", instance.repository.agent_token_env, "agent's GitHub token"),
            ("TWG_TOKEN", instance.tracker.token_env, "Jira token"),
        )
    )
    if isinstance(environment, str):
        return environment
    environment["TWG_SITE"] = instance.tracker.site
    environment["TWG_USER"] = instance.tracker.user
    environment["IS_SANDBOX"] = "1"
    return environment


def review_environment(instance: Instance) -> dict[str, str] | str:
    """Return the variables the guard review container of `instance` receives, by name, or why one is missing.

    They are the agent credential, as `container_environment` gives it, `IS_SANDBOX`
    set to `1`, and `GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_0` and `GIT_CONFIG_VALUE_0`, which
    mark `WORKDIR` as a git `safe.directory` so that git reads the mounted checkout owned
    by another user; no repository or tracker token is among them.
    """
    agent_variable, agent_service = agent_credential(instance.agent.provider)
    environment = read_tokens(
        ((agent_variable, instance.agent.token_env, agent_service),)
    )
    if isinstance(environment, str):
        return environment
    environment["IS_SANDBOX"] = "1"
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "safe.directory"
    environment["GIT_CONFIG_VALUE_0"] = WORKDIR
    return environment


def read_tokens(
    tokens: Sequence[tuple[str, str, str]],
) -> dict[str, str] | str:
    """Return each token of `tokens` by the name it is handed over under, or why one is missing.

    Each entry of `tokens` is the name a token is handed over under, the variable it is
    read from now, and what it is called. A variable unset or empty gives the sentence
    naming it; no value is ever quoted.
    """
    environment: dict[str, str] = {}
    for name, variable, service in tokens:
        value = os.environ.get(variable)
        if not value:
            return (
                f"The container was not started: the variable {variable}, which "
                f"should hold the {service}, is not set."
            )
        environment[name] = value
    return environment


def review_pull_request(
    pull_request: PullRequest,
    instance: Instance,
    worker: WorkerFile,
    directory: Path,
    began: float,
) -> tuple[str | None, str | None]:
    """Run the guard review of `pull_request`; return why it is not to be merged, None when the review found no reachable defect, and why its checkout could not be deleted, None when it was.

    `directory` is the instance directory of `instance`. The plugin the header of
    `worker` names is prepared (see `PluginCheckout.prepare`), the mirror refreshed and
    the review checkout made at the commit `pull_request.head` (see
    `Workspace.open_review`); one disposable container then runs, with the checkout
    mounted read only, the review of the change from the base branch of `pull_request`
    to that commit (see `run_review`). The review checkout is deleted afterwards,
    whatever happened (see `Workspace.close_review`); a deletion that fails is returned
    apart, never as a reason, and added to an exception raised meanwhile as a note. The
    reason lists the defects the review found, `Benign`, or says why it gave no verdict.
    """
    try:
        plugin = make_plugin(directory).prepare(worker.header.plugin)
    except PluginError as error:
        reason = (
            f"the guard review of the pull request {pull_request.url} did not run, "
            f"its plugin cannot be fetched: {error}"
        )
        return reason, None
    workspace = make_workspace(instance.repository, directory)
    try:
        reason = review_checkout(
            pull_request, instance, workspace, plugin, directory, began
        )
    except BaseException as error:
        removal = remove_review_checkout(workspace)
        if removal is not None:
            error.add_note(removal)
        raise
    return reason, remove_review_checkout(workspace)


def remove_review_checkout(workspace: Workspace) -> str | None:
    """Delete the review checkout of `workspace`; return why it failed, or None."""
    try:
        workspace.close_review()
    except WorkspaceError as error:
        return f"The review checkout cannot be deleted: {error}"
    return None


def review_checkout(
    pull_request: PullRequest,
    instance: Instance,
    workspace: Workspace,
    plugin: Path,
    directory: Path,
    began: float,
) -> str | None:
    """Make the review checkout of `pull_request` in `workspace`, then review it; return why it is not to be merged, or None.

    A mirror that cannot be refreshed or a checkout that cannot be made gives the
    reason; otherwise the review runs (see `run_review`). The checkout is left for the
    caller to delete.
    """
    try:
        workspace.refresh()
        base = workspace.open_review(pull_request.base, pull_request.head)
    except WorkspaceError as error:
        return (
            f"the guard review of the pull request {pull_request.url} did not run, "
            f"its checkout cannot be made: {error}"
        )
    return run_review(
        pull_request, base, instance, workspace.review, plugin, directory, began
    )


def run_review(
    pull_request: PullRequest,
    base: str,
    instance: Instance,
    checkout: Path,
    plugin: Path,
    directory: Path,
    began: float,
) -> str | None:
    """Run the guard review container on `checkout` for the time left in the run; return why `pull_request` is not to be merged, or None.

    The container of `instance`, kept in `directory`, mounts `checkout` read only and
    the plugin checkout `plugin` read only; it runs the review of the change from the
    commit `base` to the commit `pull_request.head` (see `review_command`) within the
    `review_max_turns` limit of `instance`, and receives only the variables
    `review_environment` gives. The time left is `run_timeout_seconds` minus the seconds
    elapsed since `began`, the `time.monotonic()` reading taken when the run started;
    none left, the container is not started. The verdict is read from what the container
    wrote (see `read_verdict`): no defect gives None; defects give a `Benign` reason
    listing each as `file:line summary`; a container that cannot be run, reaches the ceiling or gives
    no valid verdict gives the reason it gave none.
    """
    url = pull_request.url
    ceiling = instance.limits["run_timeout_seconds"]
    left = ceiling - (time.monotonic() - began)
    if left <= 0:
        return (
            f"the guard review of the pull request {url} did not run: the run "
            f"ceiling of {ceiling} seconds was reached before it."
        )
    environment = review_environment(instance)
    if isinstance(environment, str):
        return f"the guard review of the pull request {url} did not run. {environment}"
    command = review_command(
        base,
        pull_request.head,
        instance.limits["review_max_turns"],
        instance.agent.review_model,
    )
    try:
        exited = make_container(instance.container.image, directory).run(
            checkout, left, plugin, command, environment, readonly=True
        )
    except ContainerError as error:
        return f"the guard review of the pull request {url} failed: {error}."
    verdict = read_verdict(exited)
    if isinstance(verdict, str):
        return f"the guard review of the pull request {url} gave no verdict: {verdict}."
    if not verdict:
        return None
    findings = "\n".join(
        f"- {defect.file}:{defect.line} {defect.summary}" for defect in verdict
    )
    return Benign(
        f"the guard review of the pull request {url} found reachable defects; it "
        f"was not merged.\n{findings}"
    )


def run_container(
    instance: Instance,
    directory: Path,
    worktree: Path,
    plugin: Path,
    began: float,
    worker: WorkerFile,
    ticket: Ticket,
) -> Outcome | UsageLimit | str:
    """Run the container of `instance` on `worktree` for the time left in the run; return the outcome it ended on, or why there is none.

    `directory` is the instance directory of `instance`, `plugin` the plugin checkout
    the container mounts read only. The container runs Claude Code headless on `ticket`
    (see `agent_command`), with the body of `worker` appended to its system prompt, the
    `max_turns` limit of `instance` and the outcomes of the header of `worker`; it
    receives the variables `container_environment` gives, and is not started when one
    is missing.
    `began` is the `time.monotonic()` reading taken when the run started; the
    container's time is `run_timeout_seconds` minus the seconds elapsed since, minus
    the `review_timeout_seconds` kept for the guard review. No more than
    `CI_WAIT_MARGIN_SECONDS` left: the container is not started. The prompt carries the
    CI deadline, in UTC: the moment the container's time runs out, read from
    `datetime.now`, less `CI_WAIT_MARGIN_SECONDS`. The outcome
    is read from what the container wrote (see `read_outcome`); a container that cannot
    be run, or reaches the ceiling, gives none. A run that reached the usage limit of
    the agent's account gives a `UsageLimit`, which resets when the account says, or
    `usage_limit_wait_seconds` after the container ended when it does not say.
    """
    ceiling = instance.limits["run_timeout_seconds"]
    reserve = instance.limits["review_timeout_seconds"]
    left = ceiling - (time.monotonic() - began) - reserve
    if left <= 0:
        return (
            f"The container was not started: the run ceiling of {ceiling} seconds, "
            f"less the {reserve} seconds kept for the guard review, was reached "
            "before it."
        )
    if left <= CI_WAIT_MARGIN_SECONDS:
        return (
            f"The container was not started: the run ceiling of {ceiling} seconds, "
            f"less the {reserve} seconds kept for the guard review, left no more than "
            f"the {CI_WAIT_MARGIN_SECONDS} seconds kept for the agent to wait for the "
            "CI and end."
        )
    environment = container_environment(instance)
    if isinstance(environment, str):
        return environment
    outcomes = list(worker.header.outcomes.model_dump())
    ci_deadline = datetime.now(UTC) + timedelta(seconds=left - CI_WAIT_MARGIN_SECONDS)
    command = agent_command(
        ticket,
        worker.body,
        instance.limits["max_turns"],
        outcomes,
        ci_deadline,
        instance.agent.model,
    )
    try:
        exited = make_container(instance.container.image, directory).run(
            worktree, left, plugin, command, environment
        )
    except ContainerError as error:
        return f"The container failed: {error}"
    fallback_reset = datetime.now(UTC) + timedelta(
        seconds=instance.limits["usage_limit_wait_seconds"]
    )
    outcome = read_outcome(exited, outcomes, fallback_reset)
    if isinstance(outcome, str):
        return f"The run ended without a usable outcome: {outcome}."
    if isinstance(outcome, UsageLimit):
        return UsageLimit(
            outcome.resets_at,
            f"The run ended without a usable outcome: {outcome.reason}.",
        )
    return outcome


def run_in_worktree(
    handler: Handler,
    mandate: MandateInput,
    instance: Instance,
    worker: WorkerFile,
    ticket: Ticket,
    directory: Path,
    began: float,
) -> tuple[MandateOutput, Outcome | UsageLimit | str, bool]:
    """Run `handler` once the plugin is prepared, the mirror in `directory` refreshed and a fresh worktree opened.

    Returns the output with how the run ended: the outcome the container ended on when
    it is usable and the work rescued, the usage limit it reached when the work is
    rescued, else the reasons it is not, joined by a space;
    then whether no worktree was left behind: none was opened, or it was rescued and
    removed.

    The checkout in `directory` of the plugin the header of `worker` names is prepared first (see
    `PluginCheckout.prepare`); one that cannot be fetched gives a failed output naming
    the reason, without opening the worktree nor running `handler`. The worktree is on the branch the `branch` template of `instance` names for the
    ticket (see `Workspace.open`). A mirror that cannot be refreshed or a worktree that
    cannot be opened gives a failed output naming the reason, without running
    `handler`. Once opened, the worktree is rescued, then it and its local branch are
    removed, whatever `handler` returns or raises (see `end_worktree`); a rescue or a
    removal that fails keeps the output, with the reason added to `errors` and, for a
    rescue, its status set to failed, or the
    exception `handler` raised, with the reason added as a note.

    Before `handler`, one disposable container is run from the image of `instance`
    with only the worktree and, read only, the plugin checkout mounted; it runs Claude
    Code headless on `ticket`, following the body of `worker` (see `run_container`),
    and is gone once it ends or
    is removed when the run reaches its `run_timeout_seconds` limit, counted from
    `began`, the `time.monotonic()` reading taken when the run started (see
    `run_container`); a container that cannot be run, that reaches that limit, that
    the run reached it before, or that ends without a usable outcome, the usage limit
    included, gives a failed output naming the reason, without running `handler`. Otherwise `handler` gets the
    outcome and gives the output.
    """
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    branch = run_branch(instance, ticket_key)
    try:
        checkout = make_plugin(directory).prepare(worker.header.plugin)
    except PluginError as error:
        reason = f"The plugin cannot be fetched: {error}"
        return failed(mandate.task_id, reason), reason, True
    workspace = make_workspace(instance.repository, directory)
    try:
        workspace.refresh()
        workspace.open(branch)
    except WorkspaceError as error:
        reason = f"The worktree cannot be prepared: {error}"
        return failed(mandate.task_id, reason), reason, True
    try:
        ending = run_container(
            instance, directory, workspace.worktree, checkout, began, worker, ticket
        )
        if isinstance(ending, str):
            output = failed(mandate.task_id, ending)
        elif isinstance(ending, UsageLimit):
            output = failed(mandate.task_id, ending.reason)
        else:
            output = handler(mandate, instance, ending)
    except BaseException as error:
        end_error = end_worktree(workspace, branch, instance, directory)
        if end_error is not None:
            error.add_note(end_error)
        raise
    rescue_error = rescue_worktree(workspace, branch, instance, directory)
    end_error = rescue_error or close_worktree(workspace, branch)
    if rescue_error is not None:
        unusable = ending.reason if isinstance(ending, UsageLimit) else ending
        reasons = (
            [rescue_error]
            if isinstance(unusable, Outcome)
            else [unusable, rescue_error]
        )
        ending = " ".join(reasons)
        output = output.model_copy(update={"status": "failed"})
    return with_errors(output, [end_error] if end_error else []), ending, not end_error


def load_worker() -> WorkerFile:
    """Read the `WORKER.md` shipped with the package, its header validated.

    Raises `HeaderError` when its header is invalid, `OSError` when it cannot be read.
    """
    with as_file(files("dev_worker") / "WORKER.md") as path:
        return load_worker_file(path)


def run_mandate(
    mandate_name: str, raw_input: bytes, instance_path: Path
) -> MandateOutput:
    """Run the mandate named on the command line against the UTF-8 JSON input.

    The mandate gets the instance read from `instance_path` and checked against the
    header of the packaged `WORKER.md`, and runs under the lock of the instance
    directory, journaled there (see `run_journaled`). Never raises on bad input: bytes that are not
    UTF-8, JSON that cannot be parsed or written back as UTF-8, input that breaks the
    contract, an unknown mandate, and a `WORKER.md` or instance file that cannot be
    read or is invalid all become a failed output naming the reason.
    """
    try:
        text = raw_input.decode("utf-8")
    except UnicodeDecodeError as error:
        return failed(None, f"Input is not valid UTF-8: {error}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        return failed(None, f"Input is not valid JSON: {error}")
    except RecursionError:
        return failed(None, "Input JSON is nested too deeply.")
    if not isinstance(data, dict):
        return failed(None, "Input must be a JSON object.")

    raw_task_id = data.get("task_id")
    task_id = (
        raw_task_id
        if isinstance(raw_task_id, str) and is_utf8_encodable(raw_task_id)
        else None
    )
    if not is_utf8_encodable(data):
        return failed(task_id, "Input holds text that cannot be encoded as UTF-8.")

    version = data.get("contract_version")
    if version != CONTRACT_VERSION:
        return failed(
            task_id,
            f"Unsupported contract_version {version!r}, expected {CONTRACT_VERSION!r}.",
        )

    try:
        mandate = MandateInput.model_validate(data)
    except ValidationError as error:
        return failed(task_id, *describe_validation_error(error))

    if mandate.mandate != mandate_name:
        return failed(
            task_id,
            f"Mandate {mandate_name!r} on the command line differs from "
            f"{mandate.mandate!r} in the input.",
        )

    handler = MANDATES.get(mandate_name)
    if handler is None:
        return failed(task_id, f"Unknown mandate {mandate_name!r}.")

    loaded = load_checked_instance(instance_path)
    if isinstance(loaded, str):
        return failed(task_id, loaded)
    instance, worker = loaded
    return run_journaled(handler, mandate, instance, worker, instance_path).output


def wake(instance_path: Path) -> RunReport | str | None:
    """Run the `deliver_ticket` mandate on the first ticket ready to be taken; return its report, `None` when no ticket is.

    The instance is read from `instance_path` as `run` reads it. The tickets its dead
    runs left in progress are first handed back (see `recover_dead_runs`). While the
    usage limit a run of the instance reached has not reset (see `usage_limit_reset`),
    no ticket is listed nor taken, and the sentence saying so and until when, in UTC, is
    returned. Then the
    tickets in the real status of its `ready` key are listed in the tracker's rank order,
    and the first whose blockers are all in the real status of the `done` key is run
    under a fresh task_id, `wake-` followed by a UUID, as `run` runs a mandate (see
    `run_journaled`) — unless the CI of the default branch failed (see
    `CodeRepository.base_ci_failure`): then no ticket is taken, and the sentence naming
    the branch, the short SHA of the commit and its failed checks is returned. A CI
    that cannot be read does not stop the wake. An instance that cannot be loaded, a recovery that fails, or a
    tracker that cannot list the tickets gives a failed output with no task_id, as a
    failure of the worker.
    """
    loaded = load_checked_instance(instance_path)
    if isinstance(loaded, str):
        return worker_failure(None, loaded)
    instance, worker = loaded
    task_id = f"wake-{uuid.uuid4()}"
    tracker = make_tracker(instance.tracker)
    errors = recover_dead_runs(task_id, instance, tracker, instance_path)
    if errors:
        return worker_failure(None, *errors)
    resets_at = usage_limit_reset(instance_path)
    if resets_at is not None:
        return (
            "dev-worker takes no ticket: the agent's account reached its usage limit, "
            f"which resets at {resets_at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC."
        )
    ready = instance.statuses["ready"]
    try:
        tickets = tracker.tickets_in(ready)
    except TrackerError as error:
        return worker_failure(
            None, f"The tickets in {ready!r} cannot be listed from the tracker: {error}"
        )
    done = instance.statuses["done"]
    ticket = next(
        (
            ticket
            for ticket in tickets
            if all(blocker.status == done for blocker in ticket.blockers)
        ),
        None,
    )
    if ticket is None:
        return None
    try:
        failure = make_repository(instance.repository).base_ci_failure()
    except RepositoryError:
        failure = None
    if failure is not None:
        return (
            f"dev-worker takes no ticket: the CI of {failure.branch} is red at "
            f"{failure.commit[:SHORT_SHA_LENGTH]} ({', '.join(failure.checks)})."
        )
    mandate = MandateInput(
        contract_version=CONTRACT_VERSION,
        task_id=task_id,
        mandate="deliver_ticket",
        caller="wake",
        params={"ticket_key": ticket.key},
    )
    return run_journaled(deliver_ticket, mandate, instance, worker, instance_path)


def recover_dead_runs(
    task_id: str, instance: Instance, tracker: TicketTracker, instance_path: Path
) -> list[str]:
    """Hand back the tickets the dead runs of the instance left in progress; return why it failed, empty when it did not.

    The lock of the instance directory is taken for `task_id`, or taken over from a run
    that is gone and whose process no longer runs (see `take_instance_lock`), so that no
    run of the instance is alive while the tickets in the real status of the
    `in_progress` key are read; a lock held by a run that is not gone, or whose process
    still runs, skips the recovery without failing. Each ticket is handed
    back as `requeue_dead_ticket` says, until a tracker failure stops the recovery. The
    lock is released afterwards; a lock that cannot be released adds its reason. Never
    raises on a file or tracker error.
    """
    directory = made_instance_directory(instance_path)
    if isinstance(directory, str):
        return [directory]
    record = {
        "task_id": task_id,
        "start": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
    }
    lock = take_instance_lock(directory, record, instance)
    if isinstance(lock, Benign):
        return []
    if isinstance(lock, str):
        return [lock]
    try:
        problem = requeue_dead_tickets(instance, tracker)
    finally:
        try:
            release_lock(lock, record)
        except OSError as error:
            release_error = f"Lock file '{lock}' cannot be removed: {os_reason(error)}"
        else:
            release_error = None
    errors = [] if problem is None else [problem]
    if release_error is not None:
        errors.append(release_error)
    return errors


def requeue_dead_tickets(instance: Instance, tracker: TicketTracker) -> str | None:
    """Hand back each ticket in the real status of the `in_progress` key that a dead run left there; return why it failed, or None.

    Each ticket is handed back as `requeue_dead_ticket` says; the first tracker failure
    stops the recovery.
    """
    in_progress = instance.statuses["in_progress"]
    try:
        tickets = tracker.tickets_in(in_progress)
    except TrackerError as error:
        return (
            f"The tickets in {in_progress!r} cannot be listed from the tracker: {error}"
        )
    for ticket in tickets:
        problem = requeue_dead_ticket(ticket.key, instance, tracker)
        if problem is not None:
            return problem
    return None


def requeue_dead_ticket(
    key: str, instance: Instance, tracker: TicketTracker
) -> str | None:
    """Hand back the ticket `key` when a dead run of the worker left it in progress; return why it failed, or None.

    No run of the instance is to be alive. A ticket settles when it enters the real
    status of any status key of the instance but `ready` and `in_progress`. The ticket
    is the worker's when it has a claim since it last entered the real status of the
    `ready` key, the latest of which was posted since it last settled, and when it last
    entered the real status of the `in_progress` key at most `TAKE_DATING_LIMIT_SECONDS`
    after that claim was created, both times read from the tracker: the worker sends the
    move within `CLAIM_TAKE_LIMIT_SECONDS` of its claim, and the tracker may date it up
    to `TWG_TIMEOUT_SECONDS` later. Any other ticket
    was moved by a person and is left as it is. The dead run made one attempt more
    than the requeue comments since the ticket last settled.
    Below the `max_attempts` limit, a requeue comment naming the attempt is posted and
    the ticket moved to the `ready` status; at the limit, a comment asks a person to
    look at it and the ticket is moved to the `blocked` status. A ticket whose comment
    fails is not moved. After a tracker failure the sentence says whether the comment
    may have been posted, or that the status is unknown.
    """
    ready = instance.statuses["ready"]
    in_progress = instance.statuses["in_progress"]
    blocked = instance.statuses["blocked"]
    settled = frozenset(instance.statuses.values()) - {ready, in_progress}
    try:
        since_ready = tracker.claims(key, frozenset({ready}))
        since_settled = {claim.claim_id for claim in tracker.claims(key, settled)}
        if not since_ready or since_ready[-1].claim_id not in since_settled:
            return None
        claimed = since_ready[-1].created
        taken = tracker.entered(key, frozenset({in_progress}))
        take_limit = timedelta(seconds=TAKE_DATING_LIMIT_SECONDS)
        if taken is None or not claimed <= taken <= claimed + take_limit:
            return None
        attempt = tracker.requeues(key, settled) + 1
    except TrackerError as error:
        return (
            f"The comments of ticket {key!r} cannot be read from the tracker: {error}"
        )
    ceiling = instance.limits["max_attempts"]
    if attempt < ceiling:
        status = ready
        body = (
            f"{REQUEUE_MARKER}the run that took it is gone without settling it "
            f"(attempt {attempt} of {ceiling})."
        )
    else:
        status = blocked
        body = (
            f"The runs that took this ticket died {attempt} times without settling "
            f"it. A person needs to look at it, then move it back to {ready!r}."
        )
    try:
        tracker.comment(key, body)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be commented on: {error} The comment may have "
            "been posted; the ticket was not moved."
        )
    try:
        tracker.set_status(key, status)
    except TrackerError as error:
        return (
            f"Ticket {key!r} cannot be moved to {status!r}: {error} "
            "Its status is unknown and is to be read again in the tracker."
        )
    return None


def load_checked_instance(instance_path: Path) -> tuple[Instance, WorkerFile] | str:
    """Read the instance from `instance_path`, checked against the packaged `WORKER.md`.

    Returns the instance with `WORKER.md`, or the reason as a sentence
    instead when `WORKER.md` or the instance file cannot be read or is invalid. Never
    raises on a file error.
    """
    try:
        worker = load_worker()
    except HeaderError as error:
        return f"WORKER.md does not load: {error}"
    except OSError as error:
        return f"WORKER.md cannot be read: {os_reason(error)}"
    try:
        return load_instance(instance_path, worker.header), worker
    except InstanceError as error:
        return f"Instance file '{instance_path}': {error}"
    except OSError as error:
        return f"Instance file '{instance_path}' cannot be read: {os_reason(error)}"


def run_start(record: Mapping[str, object]) -> datetime | None:
    """Return the `start` of a journal entry or lock record as an aware datetime.

    `None` when `start` is missing, not an ISO 8601 string, or has no time zone.
    """
    start = record.get("start")
    if not isinstance(start, str):
        return None
    try:
        moment = datetime.fromisoformat(start)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment


def run_day(entry: Mapping[str, object]) -> date | None:
    """Return the UTC day the journaled run started, `None` when `start` is unusable."""
    moment = run_start(entry)
    return None if moment is None else moment.astimezone(UTC).date()


def run_tokens(entry: Mapping[str, object]) -> int:
    """Return the tokens the journaled run spent, 0 when `tokens` is not an integer."""
    tokens = entry.get("tokens")
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def answerable(record: Mapping[str, object]) -> bool:
    """Tell whether `record`, placed in the state, can be validated and written as JSON.

    False when a string in it holds a lone surrogate or it is nested deeper than the
    state model validates or serializes.
    """
    try:
        State.model_validate({"common": {"recent": [record]}}).model_dump_json()
    except ValueError:
        return False
    return True


def task_entries(
    entries: Sequence[Mapping[str, object]],
    locks: Sequence[Mapping[str, object] | None],
) -> list[Mapping[str, object]]:
    """Return one entry per task of the journal `entries`, its last line, oldest first.

    Tasks are ordered by their first line; a line whose `task_id` is not a string is a
    task of its own. `locks` are the lock records read before and after `entries`,
    `None` when no run held the lock. A task whose last line is the one written before
    its mandate body ran (`running` without a `duration`) is shown as `failed`,
    interrupted, unless one of `locks` is held by that task or by a run whose task is
    unknown; a last line written once the body returned keeps the status it returned.
    """
    tasks: dict[object, Mapping[str, object]] = {}
    for entry in entries:
        task_id = entry.get("task_id")
        tasks[task_id if isinstance(task_id, str) else object()] = entry
    return [
        interrupted(entry)
        if entry.get("status") == "running"
        and entry.get("duration") is None
        and not any(held_by(lock, entry) for lock in locks)
        else entry
        for entry in tasks.values()
    ]


def held_by(lock: Mapping[str, object] | None, entry: Mapping[str, object]) -> bool:
    """Tell whether the lock record `lock` may be held by the run of `entry`.

    False when no run holds the lock (`None`); true when the lock names no task.
    """
    if lock is None:
        return False
    return "task_id" not in lock or lock["task_id"] == entry.get("task_id")


def interrupted(entry: Mapping[str, object]) -> dict[str, object]:
    """Return the journal entry `entry`, written before its mandate body ran, as the
    failed run it became."""
    return {
        **entry,
        "status": "failed",
        "errors": ["The run stopped before its final journal line was written."],
    }


def read_state(instance_path: Path) -> State:
    """Build the state of the instance read from `instance_path`, writing nothing.

    `current` holds the lock record of the run in progress (`{}` when its details are
    unknown), `recent` the last `RECENT_RUNS` tasks of the journal (see
    `task_entries`), most recent first, and `cost` the number of those tasks and the
    tokens they spent on the current UTC day. The
    instance file itself is not read; an instance path without a suffix gives an empty
    state. A journal entry that `answerable` refuses is skipped, a lock record it
    refuses gives `{}`. Never raises.
    """
    try:
        directory = instance_directory(instance_path)
    except ValueError:
        directory = None
    lock = read_lock(directory) if directory else None
    if lock is not None and not answerable(lock):
        lock = {}
    entries = (
        task_entries(
            [entry for entry in read_entries(directory) if answerable(entry)],
            [lock, read_lock(directory)],
        )
        if directory
        else []
    )
    today = datetime.now(UTC).date()
    todays = [entry for entry in entries if run_day(entry) == today]
    tokens = sum(run_tokens(entry) for entry in todays)
    return State.model_validate(
        {
            "common": {
                "current": [] if lock is None else [lock],
                "recent": entries[::-1][:RECENT_RUNS],
                "pending": [],
                "cost": {
                    "day": today.isoformat(),
                    "runs": len(todays),
                    "tokens": tokens,
                },
            }
        }
    )


def unwritable_instance_directory(directory: Path) -> str | None:
    """Return why a run could not write in `directory`, or `None` when it could.

    Checks, without writing, the directory when it exists, else its nearest existing
    parent, in which a run would create it: it must be a directory the process may
    write in and enter. The journal and the guard file, when they exist, must be
    writable too.
    """
    nearest = directory
    while not nearest.exists() and nearest.parent != nearest:
        nearest = nearest.parent
    if not (nearest.is_dir() and os.access(nearest, os.W_OK | os.X_OK)):
        return (
            f"Instance directory '{directory}' cannot be written: "
            f"'{nearest}' is not a writable directory."
        )
    journal = directory / JOURNAL_FILE
    if journal.exists() and not os.access(journal, os.W_OK):
        return f"Journal '{journal}' cannot be written."
    guard = directory / GUARD_FILE
    if guard.exists() and not os.access(guard, os.R_OK | os.W_OK):
        return f"Guard file '{guard}' cannot be written."
    return None


def lock_age(directory: Path, lock: Mapping[str, object]) -> float | None:
    """Return for how many seconds the run holding the lock of `directory` has run.

    The age is counted from the `start` of the lock record, else from the time the
    lock file was last modified. `None` when the lock file is gone or cannot be read.
    """
    now = datetime.now(UTC)
    start = run_start(lock)
    if start is not None:
        return (now - start).total_seconds()
    try:
        modified = (directory / LOCK_FILE).stat().st_mtime
    except OSError:
        return None
    return now.timestamp() - modified


def gone_reason(
    directory: Path, lock: Mapping[str, object], timeout: int
) -> str | None:
    """Return why the run holding the lock of `directory` is gone, `None` if it is not.

    `lock` is the lock record. When the record names a process that can be inspected
    (see `lock_process_runs`), the run is gone once that process no longer runs or was
    started after the record's `start`, whatever the lock's age. Otherwise the run is
    gone once it has held the lock longer than `timeout` seconds (see `lock_age`).
    """
    pid = lock_pid(lock)
    runs = lock_process_runs(lock)
    if runs is True:
        return None
    if runs is False:
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"is gone: its process {pid} no longer runs."
        )
    age = lock_age(directory, lock)
    if age is not None and age > timeout:
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"started {age:.0f} seconds ago, beyond the run timeout of "
            f"{timeout} seconds."
        )
    return None


def overdue_reason(
    directory: Path, lock: Mapping[str, object], timeout: int
) -> str | None:
    """Return why the run holding the lock of `directory` is overdue, `None` if it is not.

    `lock` is the lock record. The run is overdue when its process still runs (see
    `lock_process_runs`) and it has held the lock longer than `timeout` seconds (see
    `lock_age`); such a lock is not taken over.
    """
    pid = lock_pid(lock)
    age = lock_age(directory, lock)
    if age is None or age <= timeout or lock_process_runs(lock) is not True:
        return None
    return (
        f"The run holding the lock of instance directory '{directory}' "
        f"started {age:.0f} seconds ago, beyond the run timeout of "
        f"{timeout} seconds; its process {pid} still runs, so the lock is not "
        f"taken over."
    )


def lock_pid(lock: Mapping[str, object]) -> int | None:
    """Return the process the lock record `lock` names, or `None` when it names none.

    A record names its process by a positive integer `pid` and a usable `start`.
    """
    pid = lock.get("pid")
    if (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and run_start(lock) is not None
    ):
        return pid
    return None


def lock_process_runs(lock: Mapping[str, object]) -> bool | None:
    """Tell whether the process the lock record `lock` names (see `lock_pid`) still runs, started no later than the record's `start`.

    `None` when the record names no process, or names one that cannot be inspected.
    """
    pid = lock_pid(lock)
    start = run_start(lock)
    if pid is None or start is None:
        return None
    return process_runs_since(pid, start.timestamp())


def take_over_lock(
    directory: Path,
    record: Mapping[str, object],
    timeout: int,
) -> Path | str:
    """Take the lock of `directory` from a run that is gone, writing `record` in it.

    The gone run (see `gone_reason`) is journaled first, as a failed run with the
    `task_id`, `ticket_key`, `model`, `review_model` and `start` of its lock record and the reason it is gone
    in `errors`; a torn last journal line is set aside before. Returns the lock file
    path, or why the lock cannot be taken: its run is
    not gone, another run took it first, both `Benign`, or a file cannot be read or
    written. Never raises on a file error.
    """
    in_progress = Benign(
        f"A run is already in progress in instance directory '{directory}'."
    )
    try:
        seen = (directory / LOCK_FILE).read_bytes()
        held = parse_lock(seen)
        reason = gone_reason(directory, held, timeout)
        if reason is None or not remove_lock_if_unchanged(directory, seen):
            return in_progress
    except FileNotFoundError:
        return in_progress
    except OSError as error:
        return lock_unwritable(directory, error)
    entry = {
        "task_id": held.get("task_id"),
        "ticket_key": held.get("ticket_key"),
        "model": held.get("model"),
        "review_model": held.get("review_model"),
        "status": "failed",
        "start": held.get("start"),
        "duration": None,
        "tokens": 0,
        "errors": [reason],
    }
    try:
        set_aside_torn_tail(directory)
        append_entry(directory, entry)
    except OSError as error:
        return f"Journal in '{directory}' cannot be written: {os_reason(error)}"
    try:
        return take_lock(directory, record)
    except FileExistsError:
        return in_progress
    except OSError as error:
        return lock_unwritable(directory, error)


def lock_unwritable(directory: Path, error: OSError) -> str:
    """Return the sentence saying the lock file in `directory` fails with `error`."""
    return f"Lock file in '{directory}' cannot be written: {os_reason(error)}"


def read_health(instance_path: Path) -> Health:
    """Tell whether the worker can run against the instance read from `instance_path`.

    `down` when `WORKER.md` or the instance file does not load, the instance path has
    no suffix, the instance directory or its existing journal or guard file cannot be
    written (see `unwritable_instance_directory`), or
    the journal cannot be read the way a run reads it (see `unreadable_journal`), or
    its torn last line cannot be set aside (see `unsettable_torn_line`), or the
    tracker or the repository cannot be reached with the instance credentials (see
    `unreachable_connector`), or the Docker daemon cannot be reached (see
    `unreachable_docker`); `degraded`
    when the run holding the lock is gone (see `gone_reason`), which the next run mends
    by taking the lock over, or overdue (see `overdue_reason`), or the journal holds a
    torn line (see
    `torn_journal_line`); `ok` otherwise. Writes nothing and never raises.
    """
    loaded = load_checked_instance(instance_path)
    if isinstance(loaded, str):
        return Health(status="down", reason=writable_text(loaded), active=True)
    instance, _ = loaded
    try:
        directory = instance_directory(instance_path)
    except ValueError as error:
        return Health(status="down", reason=writable_text(str(error)), active=True)
    unwritable = unwritable_instance_directory(directory)
    if unwritable is not None:
        return Health(status="down", reason=writable_text(unwritable), active=True)
    unreadable = unreadable_journal(directory)
    if unreadable is not None:
        return Health(status="down", reason=writable_text(unreadable), active=True)
    unsettable = unsettable_torn_line(directory)
    if unsettable is not None:
        return Health(status="down", reason=writable_text(unsettable), active=True)
    unreachable = unreachable_connector(instance)
    if unreachable is not None:
        return Health(status="down", reason=writable_text(unreachable), active=True)
    no_docker = unreachable_docker(instance, directory)
    if no_docker is not None:
        return Health(status="down", reason=writable_text(no_docker), active=True)
    lock = read_lock(directory)
    timeout = instance.limits["run_timeout_seconds"]
    stale = (
        None
        if lock is None
        else gone_reason(directory, lock, timeout)
        or overdue_reason(directory, lock, timeout)
    )
    if stale is not None:
        return Health(status="degraded", reason=writable_text(stale), active=True)
    torn = torn_journal_line(directory)
    if torn is not None:
        return Health(status="degraded", reason=writable_text(torn), active=True)
    return Health(status="ok", reason=None, active=True)


def unreadable_journal(directory: Path) -> str | None:
    """Return why a run could not read the journal of `directory`, `None` when it could.

    A journal that does not exist yet can be read. Its torn last line, if any, is left
    out, since the next run sets it aside (see `torn_journal_line`). Never raises.
    """
    journal = directory / JOURNAL_FILE
    if not journal.exists():
        return None
    if not journal.is_file():
        return f"Journal '{journal}' is not a regular file."
    try:
        content = journal.read_bytes()
    except OSError as error:
        return f"Journal '{journal}' cannot be read: {os_reason(error)}"
    tail = torn_tail(content)
    whole = content if tail is None else content[: len(content) - len(tail)]
    for number, line in enumerate(io.BytesIO(whole), start=1):
        if not line.strip():
            continue
        try:
            journal_entry(number, line)
        except JournalError as error:
            return f"Journal '{journal}': {error}"
    return None


def unsettable_torn_line(directory: Path) -> str | None:
    """Return why a run could not set aside the torn last journal line of `directory`.

    `None` when the journal does not end with a torn line, or when the torn file is
    absent, a link to an absent file in a directory the process may write, or a regular
    file the process can read and, unless it already ends with that line, may write.
    Never raises.
    """
    try:
        content = (directory / JOURNAL_FILE).read_bytes()
    except OSError:
        return None
    tail = torn_tail(content)
    if tail is None:
        return None
    torn = directory / TORN_FILE
    try:
        torn.stat()
    except FileNotFoundError:
        target_directory = os.path.dirname(os.path.realpath(torn))
        if torn.is_symlink() and not os.access(target_directory, os.W_OK | os.X_OK):
            return f"Torn file '{torn}' links where a run cannot create it."
        return None
    except OSError as error:
        return f"Torn file '{torn}' cannot be read: {os_reason(error)}"
    if not torn.is_file():
        return f"Torn file '{torn}' is not a regular file."
    try:
        set_aside = torn.read_bytes()
    except OSError as error:
        return f"Torn file '{torn}' cannot be read: {os_reason(error)}"
    if not holds_torn_line(set_aside, tail) and not os.access(torn, os.W_OK):
        return f"Torn file '{torn}' cannot be written."
    return None


def torn_journal_line(directory: Path) -> str | None:
    """Return why the journal of `directory` holds a torn line, `None` when it does not.

    A torn line is one in the torn file, or a torn last line of the journal that no run
    has set aside yet. A journal that cannot be read is taken as whole. Never raises.
    """
    torn = directory / TORN_FILE
    if torn.exists():
        return f"Torn journal lines are set aside in '{torn}'."
    journal = directory / JOURNAL_FILE
    try:
        content = journal.read_bytes()
    except OSError:
        return None
    if torn_tail(content) is not None:
        return f"Journal '{journal}' ends with a torn line."
    return None


def add_instance_argument(command: argparse.ArgumentParser) -> None:
    """Add the required `--instance` option naming the YAML instance file."""
    command.add_argument(
        "--instance",
        required=True,
        type=Path,
        help="path of the YAML instance file naming the tracker project and repository",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the `run`, `wake`, `state` and `health` subcommands.

    All four require `--instance`; `run` also takes the mandate name.
    """
    parser = argparse.ArgumentParser(prog="dev-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a mandate read as JSON on stdin")
    run.add_argument("mandate", help="name of the mandate to run")
    add_instance_argument(run)
    wake_command = commands.add_parser(
        "wake", help="run the first ticket ready to be taken, if any"
    )
    add_instance_argument(wake_command)
    state = commands.add_parser("state", help="print the worker state")
    add_instance_argument(state)
    health = commands.add_parser("health", help="print the worker health")
    add_instance_argument(health)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, write the JSON answer on stdout and return the exit code.

    The `run` input is read from stdin as raw bytes and the answer is written as UTF-8
    bytes, whatever the encoding of the system. `state` reads the journal and lock of the
    instance directory; `health` reads the instance and its directory (see `read_health`). `run`, `state`
    and `health` return 0 whenever a JSON document is written, even for a failed
    mandate; argparse exits with 2 on incorrect usage.

    `wake` writes the output of the run it made, or of its own failure, and nothing when
    no ticket is ready; while the usage limit has not reset, or when the CI of the
    default branch failed, it writes nothing on stdout and the reason on stderr. It returns 1 when the worker failed (see
    `RunReport`), 0 otherwise.
    """
    args = build_parser().parse_args(argv)
    if args.command == "wake":
        report = wake(args.instance)
        if report is None:
            return 0
        if isinstance(report, str):
            sys.stderr.write(f"{report}\n")
            return 0
        write_answer(report.output)
        return 1 if report.worker_failed else 0
    if args.command == "run":
        answer = run_mandate(args.mandate, sys.stdin.buffer.read(), args.instance)
    elif args.command == "state":
        answer = read_state(args.instance)
    else:
        answer = read_health(args.instance)
    write_answer(answer)
    return 0


def write_answer(answer: MandateOutput | State | Health) -> None:
    """Write `answer` on stdout as one line of UTF-8 JSON, whatever the encoding of the system."""
    sys.stdout.flush()
    sys.stdout.buffer.write(answer.model_dump_json().encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
