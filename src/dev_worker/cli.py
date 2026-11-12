"""Command line entry point: the `run`, `state` and `health` commands.

Each command names the instance file with `--instance`; `run` runs a mandate read as JSON
on stdin against that instance and the `WORKER.md` shipped with the package.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.resources import as_file, files
from pathlib import Path

from pydantic import ValidationError

from dev_worker.container import Container, ContainerError, agent_command
from dev_worker.contract import (
    CONTRACT_VERSION,
    Health,
    MandateInput,
    MandateOutput,
    State,
)
from dev_worker.github import GitHubRepository
from dev_worker.header import HeaderError, WorkerFile, load_worker_file
from dev_worker.instance import (
    Instance,
    InstanceError,
    Repository,
    Tracker,
    load_instance,
)
from dev_worker.jira import JiraTracker
from dev_worker.outcome import Outcome, read_outcome
from dev_worker.plugin import PluginCheckout, PluginError
from dev_worker.repositories import CodeRepository, HeadMovedError, RepositoryError
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
from dev_worker.tickets import Claim, Ticket, TicketTracker, TrackerError
from dev_worker.validation import describe_validation_error
from dev_worker.workspace import Workspace, WorkspaceError

TICKET_KEY = re.compile(r"(.+)-([1-9][0-9]*)")
RECENT_RUNS = 10
CLAIM_TAKE_LIMIT_SECONDS = 300
CLAIM_EXPIRY_SECONDS = 600
AGENT_CREDENTIALS = {
    "subscription": ("CLAUDE_CODE_OAUTH_TOKEN", "Claude Code subscription token"),
    "api_key": ("ANTHROPIC_API_KEY", "Anthropic API key"),
}


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


def make_repository(repository: Repository) -> CodeRepository:
    """Return the code repository that `repository` of an instance describes."""
    return GitHubRepository(repository)


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

    Each is asked to check its access with the instance credentials; nothing is changed
    on either. Never raises.
    """
    try:
        make_tracker(instance.tracker).check_access()
    except TrackerError as error:
        return f"The tracker cannot be reached: {error}"
    try:
        make_repository(instance.repository).check_access()
    except RepositoryError as error:
        return f"The repository cannot be reached: {error}"
    return None


def read_ready_ticket(
    key: str, instance: Instance, tracker: TicketTracker
) -> Ticket | str:
    """Return the ticket `key` read from `tracker`, or why it cannot be run.

    It is refused when the tracker cannot read it,
    when its status is not the real status of the `ready` key of `instance`, or when a
    ticket that blocks it is not in the real status of the `done` key.
    """
    try:
        ticket = tracker.read(key)
    except TrackerError as error:
        return f"Ticket {key!r} cannot be read from the tracker: {error}"
    ready = instance.statuses["ready"]
    if ticket.status != ready:
        return f"Ticket {key!r} is in status {ticket.status!r}, not {ready!r}."
    done = instance.statuses["done"]
    open_blockers = [
        f"{blocker.key} ({blocker.status!r})"
        for blocker in ticket.blockers
        if blocker.status != done
    ]
    if open_blockers:
        return (
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
) -> str | None:
    """Claim the ticket `key` for `task_id` through `tracker`; return why the run may not take it, or None.

    The claims posted since the ticket last entered the real status of the `ready` key
    are judged in the order of their ids: a claim wins unless an older winning claim was
    created at most `CLAIM_EXPIRY_SECONDS` before it, by the tracker's clock. An older
    claim is otherwise left by a run that is gone, and a losing claim never blocks
    another. The run may take the ticket only when its own claim wins: of runs that both
    read the ticket ready, one only goes on. A run whose claim cannot be posted, whose claims
    cannot be read, whose claims are read back too late after `start` (see
    `claim_overdue`), or whose own claim is not found among them may not take it; the
    reason says whether the claim may have been posted. A run whose claim wins reads the
    ticket again and may not take it unless it is still in the `ready` status. `start` is
    taken right before the claim is posted.
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
        claims = tracker.claims(key, ready)
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
        return (
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
        return (
            f"Ticket {key!r} is no longer in {ready!r}: it is in {ticket.status!r}. "
            f"The claim of task {task_id!r} was posted."
        )
    return None


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
) -> tuple[str, str, str | None]:
    """Merge the pull request of a run on the ticket `key` that ended on `done`; return the status key the ticket goes to, the body of its comment and why it is blocked.

    The open pull request whose head is the run branch (see `run_branch`) is found, its
    CI read once, the ticket read again through `tracker`, then the pull request merged
    as one squashed commit under its title. The status key is `done` once merged; the one
    the header of `worker` ties the `done` outcome to when the ticket carries the
    `hold_merge_label` of `instance`, the pull request left open; the one it ties
    `blocked` to when there is no pull request, its CI has not passed, or a step fails,
    nothing retried. The body is `summary` followed by what happened; after a merge that
    failed for a reason other than a moved head, it says the merge may have happened.
    The reason the ticket is blocked is `None` unless it goes to the `blocked` status key.
    """
    outcomes = worker.header.outcomes

    def blocked(reason: str) -> tuple[str, str, str]:
        """Return the landing of a ticket blocked for `reason`."""
        return (
            outcomes.blocked,
            f"{summary}\n\nThe ticket is blocked: {reason}",
            reason,
        )

    branch = run_branch(instance, key)
    repository = make_repository(instance.repository)
    try:
        pull_request = repository.find_pull_request(branch)
        if pull_request is None:
            return blocked(f"no open pull request has the branch {branch} as its head.")
        ci = repository.read_ci(pull_request)
        if ci != "passed":
            state = "failed" if ci == "failed" else "has not finished"
            return blocked(
                f"the CI of the pull request {pull_request.url} {state}; it was not "
                "merged."
            )
        held = instance.hold_merge_label in tracker.read(key).labels
    except (RepositoryError, TrackerError) as error:
        return blocked(str(error))
    if held:
        left_open = (
            f"{summary}\n\nThe pull request {pull_request.url} is left open: the "
            f"ticket carries the label {instance.hold_merge_label!r}, which holds its "
            "merge back."
        )
        return outcomes.done, left_open, None
    try:
        repository.merge(pull_request)
    except HeadMovedError as error:
        return blocked(str(error))
    except RepositoryError as error:
        return blocked(
            f"the pull request {pull_request.url} cannot be merged: {error} It may "
            "have been merged and is to be read again."
        )
    return "done", f"{summary}\n\nMerged: {pull_request.url}", None


def settle_ticket(
    key: str,
    ending: Outcome | str,
    instance: Instance,
    worker: WorkerFile,
    tracker: TicketTracker,
) -> tuple[str | None, list[str]]:
    """Comment on the ticket `key` then move it as the run's `ending` says; return why a `done` run left it blocked, and why either step failed.

    `ending` is the outcome the run ended on, usable, or why it ended without one. The
    `done` outcome first has its pull request merged, and the ticket is commented on and
    moved as `land_done` says. The comment is made writable as UTF-8 whatever it
    carries. Any other outcome has its message posted and the ticket
    moved to the real status of the status key the header of `worker` ties it to; a
    reason is posted and the ticket moved as the `blocked` outcome is. The ticket is
    moved even when the comment fails, and never to the real status of the `ready` key.
    After a tracker failure the sentence says whether the comment may have been posted,
    or that the status is unknown. The reason a `done` run left the ticket blocked is the
    one `land_done` gives, `None` for any other ending.
    """
    tied = worker.header.outcomes.model_dump()
    blocking: str | None = None
    if isinstance(ending, Outcome) and ending.outcome == "done":
        status_key, body, blocking = land_done(
            key, ending.message, instance, worker, tracker
        )
    elif isinstance(ending, Outcome):
        status_key, body = tied[ending.outcome], ending.message
    else:
        status_key, body = tied["blocked"], ending
    errors: list[str] = []
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
) -> MandateOutput:
    """Run `handler` under the lock of the instance directory and journal the run.

    `worker` is the `WORKER.md` the run follows (see `run_in_worktree`).
    The ticket key is checked first, then the ticket is read from the tracker and
    refused unless it is ready (see `read_ready_ticket`), then the lock is taken, or
    taken over from a run that is gone (see `take_over_lock`); a task_id already in the
    journal is refused, otherwise the ticket is claimed for the run (see `claim_ticket`)
    and moved to the real status of the `in_progress` key (see `take_ticket`) before the
    run is journaled. A refused run, or one whose ticket cannot be claimed or moved, is
    not journaled. A run that took its ticket settles it once it ends (see
    `run_under_lock`). The lock is
    released only if no other run has taken it over meanwhile. Every file error
    becomes a failed output naming the reason, except a lock that cannot be removed:
    the run keeps its status and the reason is added to `errors`. An exception raised
    by `handler` propagates once the lock is released, the ticket left as it is.
    """
    task_id = mandate.task_id
    refusal = check_ticket_key(mandate, instance)
    if refusal is not None:
        return failed(task_id, refusal)
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    tracker = make_tracker(instance.tracker)
    ticket = read_ready_ticket(ticket_key, instance, tracker)
    if isinstance(ticket, str):
        return failed(task_id, ticket)
    try:
        directory = instance_directory(instance_path)
    except ValueError as error:
        return failed(task_id, str(error))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return failed(
            task_id,
            f"Instance directory '{directory}' cannot be created: {os_reason(error)}",
        )
    start = datetime.now(UTC).isoformat()
    began = time.monotonic()
    record = {
        "task_id": task_id,
        "ticket_key": ticket_key,
        "start": start,
        "pid": os.getpid(),
    }
    try:
        taken: Path | str = take_lock(directory, record)
    except FileExistsError:
        taken = take_over_lock(
            directory, record, instance.limits["run_timeout_seconds"]
        )
    except OSError as error:
        taken = lock_unwritable(directory, error)
    if isinstance(taken, str):
        return failed(task_id, taken)
    lock = taken
    try:
        output = run_under_lock(
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
        return output.model_copy(update={"errors": [*output.errors, release_error]})
    return output


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
) -> MandateOutput:
    """Run `handler` unless its task_id is already journaled, journaling the run twice.

    `worker` is the `WORKER.md` the run follows and `ticket` the ticket as read before
    the lock was taken (see `run_in_worktree`).
    Before anything is journaled, a worktree an earlier run left in `directory` is
    rescued and removed (see `rescue_left_worktree`); a run that cannot do so fails
    without taking its ticket nor being journaled. Then the ticket is claimed for the run
    through `tracker` (see `claim_ticket`): the claim must be read back in time, win
    over the live claims and find the ticket still ready. The ticket is then moved to the
    real status of the `in_progress` key; a run whose ticket cannot be claimed or moved
    fails without being journaled. A `running` line is appended before `handler` runs, so that the
    task is never run again even if its final line, appended once `handler` returns,
    cannot be written; the output is then kept, with the journal error added to
    `errors`. A run whose `running` line cannot be written fails without running
    `handler`. Once the `running` line is written, `handler` runs in a fresh worktree,
    after a disposable container (see `run_in_worktree`). `start` is the ISO 8601 UTC
    start of the run, written as is in both lines, and `began` the `time.monotonic()`
    reading taken with it. A torn last journal line is first set aside in the torn file.

    Once its ticket is taken, the run settles it before the final line is written (see
    `settle_ticket`): by the outcome the run ended on when it is usable and its work
    rescued, as `blocked` with the reason otherwise, a `running` line that cannot be
    written included. A run that ended on `done` but whose ticket is left blocked fails,
    the reason as its first error ahead of the errors its output already carried, and
    is journaled `failed`. A tracker failure while
    settling adds its reason to `errors`. An exception raised by `handler` leaves the
    ticket as it is.
    """
    task_id = mandate.task_id
    try:
        set_aside_torn_tail(directory)
        repeated = journal_holds(directory, task_id)
    except OSError as error:
        return failed(
            task_id,
            f"Journal in '{directory}' cannot be read or repaired: {os_reason(error)}",
        )
    except JournalError as error:
        return failed(task_id, f"Journal in '{directory}': {error}")
    if repeated:
        return failed(task_id, f"Task {task_id!r} has already been run.")
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    refusal = rescue_left_worktree(instance, directory)
    if refusal is None:
        claim_start = start_claim()
        refusal = claim_ticket(
            ticket_key, task_id, instance, tracker, claim_start
        ) or take_ticket(ticket_key, task_id, instance, tracker, claim_start)
    if refusal is not None:
        return failed(task_id, refusal)
    entry: dict[str, object] = {
        "task_id": task_id,
        "ticket_key": mandate.params["ticket_key"],
        "status": "running",
        "start": start,
        "duration": None,
        "tokens": 0,
    }
    try:
        append_entry(directory, entry)
    except OSError as error:
        reason = f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        _, errors = settle_ticket(ticket_key, reason, instance, worker, tracker)
        return with_errors(failed(task_id, reason), errors)
    mandate_began = time.monotonic()
    output, ending = run_in_worktree(
        handler, mandate, instance, worker, ticket, directory, began
    )
    blocking, errors = settle_ticket(ticket_key, ending, instance, worker, tracker)
    if blocking is not None:
        output = failed(
            task_id,
            f"The run ended on 'done' but its ticket is blocked: {blocking}",
            *output.errors,
        )
    output = with_errors(output, errors)
    entry |= {"status": output.status, "duration": time.monotonic() - mandate_began}
    try:
        append_entry(directory, entry)
    except OSError as error:
        journal_error = (
            f"Journal in '{directory}' cannot be written: {os_reason(error)}"
        )
        return output.model_copy(
            update={"errors": [*output.errors, writable_text(journal_error)]}
        )
    return output


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


def container_environment(instance: Instance) -> dict[str, str] | str:
    """Return the variables the container of `instance` receives, by name, or why one is missing.

    They are the agent credential, `GH_TOKEN` and `TWG_TOKEN`, holding the values of the
    variables `token_env` names for the agent, the repository and the tracker, read now,
    and `TWG_SITE` and `TWG_USER`, the tracker site and account. The agent credential is
    `CLAUDE_CODE_OAUTH_TOKEN` for the `subscription` provider and `ANTHROPIC_API_KEY` for
    `api_key`. A token variable unset or empty gives the sentence naming it; no value is
    ever quoted.
    """
    agent_variable, agent_service = AGENT_CREDENTIALS[instance.agent.provider]
    tokens = (
        (agent_variable, instance.agent.token_env, agent_service),
        ("GH_TOKEN", instance.repository.token_env, "GitHub token"),
        ("TWG_TOKEN", instance.tracker.token_env, "Jira token"),
    )
    environment: dict[str, str] = {}
    for name, variable, service in tokens:
        value = os.environ.get(variable)
        if not value:
            return (
                f"The container was not started: the variable {variable}, which "
                f"should hold the {service}, is not set."
            )
        environment[name] = value
    environment["TWG_SITE"] = instance.tracker.site
    environment["TWG_USER"] = instance.tracker.user
    return environment


def run_container(
    instance: Instance,
    directory: Path,
    worktree: Path,
    plugin: Path,
    began: float,
    worker: WorkerFile,
    ticket: Ticket,
) -> Outcome | str:
    """Run the container of `instance` on `worktree` for the time left in the run; return the outcome it ended on, or why there is none.

    `directory` is the instance directory of `instance`, `plugin` the plugin checkout
    the container mounts read only. The container runs Claude Code headless on `ticket`
    (see `agent_command`), with the body of `worker` appended to its system prompt, the
    `max_turns` limit of `instance` and the outcomes of the header of `worker`; it
    receives the variables `container_environment` gives, and is not started when one
    is missing.
    `began` is the `time.monotonic()` reading taken when the run started; the time left
    is `run_timeout_seconds` minus the seconds elapsed since. None left: the container
    is not started. The outcome is read from what the container wrote (see
    `read_outcome`); a container that cannot be run, or reaches the ceiling, gives none.
    """
    ceiling = instance.limits["run_timeout_seconds"]
    left = ceiling - (time.monotonic() - began)
    if left <= 0:
        return (
            f"The container was not started: the run ceiling of {ceiling} seconds "
            "was reached before it."
        )
    environment = container_environment(instance)
    if isinstance(environment, str):
        return environment
    outcomes = list(worker.header.outcomes.model_dump())
    command = agent_command(ticket, worker.body, instance.limits["max_turns"], outcomes)
    try:
        exited = make_container(instance.container.image, directory).run(
            worktree, left, plugin, command, environment
        )
    except ContainerError as error:
        return f"The container failed: {error}"
    outcome = read_outcome(exited, outcomes)
    if isinstance(outcome, str):
        return f"The run ended without a usable outcome: {outcome}."
    return outcome


def run_in_worktree(
    handler: Handler,
    mandate: MandateInput,
    instance: Instance,
    worker: WorkerFile,
    ticket: Ticket,
    directory: Path,
    began: float,
) -> tuple[MandateOutput, Outcome | str]:
    """Run `handler` once the plugin is prepared, the mirror in `directory` refreshed and a fresh worktree opened.

    Returns the output with how the run ended: the outcome the container ended on when
    it is usable and the work rescued, else the reasons it is not, joined by a space.

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
    the run reached it before, or that ends without a usable outcome gives a failed
    output naming the reason, without running `handler`. Otherwise `handler` gets the
    outcome and gives the output.
    """
    ticket_key = mandate.params["ticket_key"]
    assert isinstance(ticket_key, str)
    branch = run_branch(instance, ticket_key)
    try:
        checkout = make_plugin(directory).prepare(worker.header.plugin)
    except PluginError as error:
        reason = f"The plugin cannot be fetched: {error}"
        return failed(mandate.task_id, reason), reason
    workspace = make_workspace(instance.repository, directory)
    try:
        workspace.refresh()
        workspace.open(branch)
    except WorkspaceError as error:
        reason = f"The worktree cannot be prepared: {error}"
        return failed(mandate.task_id, reason), reason
    try:
        ending = run_container(
            instance, directory, workspace.worktree, checkout, began, worker, ticket
        )
        if isinstance(ending, str):
            output = failed(mandate.task_id, ending)
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
        reasons = [ending, rescue_error] if isinstance(ending, str) else [rescue_error]
        ending = " ".join(reasons)
        output = output.model_copy(update={"status": "failed"})
    return with_errors(output, [end_error] if end_error else []), ending


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
    return run_journaled(handler, mandate, instance, worker, instance_path)


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

    `lock` is the lock record. The run is gone when it has held the lock longer than
    `timeout` seconds (see `lock_age`), or when the record names its process by a
    positive integer `pid` and a usable `start`, and that process no longer runs or
    was started after `start`.
    """
    age = lock_age(directory, lock)
    if age is not None and age > timeout:
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"started {age:.0f} seconds ago, beyond the run timeout of "
            f"{timeout} seconds."
        )
    pid = lock.get("pid")
    start = run_start(lock)
    if (
        isinstance(pid, int)
        and not isinstance(pid, bool)
        and pid > 0
        and start is not None
        and not process_runs_since(pid, start.timestamp())
    ):
        return (
            f"The run holding the lock of instance directory '{directory}' "
            f"is gone: its process {pid} no longer runs."
        )
    return None


def take_over_lock(
    directory: Path, record: Mapping[str, object], timeout: int
) -> Path | str:
    """Take the lock of `directory` from a run that is gone, writing `record` in it.

    The gone run (see `gone_reason`) is journaled first, as a failed run with the
    `task_id`, `ticket_key` and `start` of its lock record and the reason it is gone
    in `errors`; a torn last journal line is set aside before. Returns the lock file
    path, or why the lock cannot be taken: its run is not gone, another run took it
    first, or a file cannot be read or written. Never raises on a file error.
    """
    in_progress = f"A run is already in progress in instance directory '{directory}'."
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
    by taking the lock over, or the journal holds a torn line (see
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
    gone = None if lock is None else gone_reason(directory, lock, timeout)
    if gone is not None:
        return Health(status="degraded", reason=writable_text(gone), active=True)
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
    """Build the argument parser with the `run`, `state` and `health` subcommands.

    All three require `--instance`; `run` also takes the mandate name.
    """
    parser = argparse.ArgumentParser(prog="dev-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a mandate read as JSON on stdin")
    run.add_argument("mandate", help="name of the mandate to run")
    add_instance_argument(run)
    state = commands.add_parser("state", help="print the worker state")
    add_instance_argument(state)
    health = commands.add_parser("health", help="print the worker health")
    add_instance_argument(health)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line, write the JSON answer on stdout and return the exit code.

    The `run` input is read from stdin as raw bytes and the answer is written as UTF-8
    bytes, whatever the encoding of the system. `state` reads the journal and lock of the
    instance directory; `health` reads the instance and its directory (see `read_health`). Returns 0 whenever a
    JSON document is written, even for a failed mandate; argparse exits with 2 on
    incorrect usage.
    """
    args = build_parser().parse_args(argv)
    if args.command == "run":
        answer = run_mandate(args.mandate, sys.stdin.buffer.read(), args.instance)
    elif args.command == "state":
        answer = read_state(args.instance)
    else:
        answer = read_health(args.instance)
    sys.stdout.flush()
    sys.stdout.buffer.write(answer.model_dump_json().encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()
    return 0
