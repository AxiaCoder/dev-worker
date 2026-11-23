# dev-worker

A worker that takes a framed ticket to a pull request, and merges it when nothing holds it back. An AI sits at the centre; it is given tools, and guardrails in code that it cannot bypass.

**Status: design phase.** Nothing runs yet.

## V1 in one paragraph

One main mandate — "take this ticket to a pull request" — in which the AI chains implementation, verification and review itself, using a pinned stack of agents and skills. The worker's code owns what must not depend on the model: the disposable worktree, the turn caps, the ticket statuses, and the merge. External systems (a ticket tracker, a code host) are reached through connectors; Jira and GitHub come by default, and an instance can plug others. Secrets are never in the repository: an instance file holds the name of an environment variable, not its value.

Out of V1: a framing mandate, narrow mandates, connectors beyond Jira and GitHub.

## Adding an instance

An instance is one YAML file that points the worker at a tracker project and a repository. It is read against the header of [`WORKER.md`](src/dev_worker/WORKER.md) and refused if it loosens it, or tightens `run_timeout_seconds` so far that a run has no time left for its agent.

1. Copy [`examples/instance.yaml`](examples/instance.yaml) to a place **outside this repository**.
2. Set the tracker site address (`site`), the email of its account (`user`), the tracker project and the repository name.
3. Under `branch`, write the template of the branch a run works on; it must contain `{key}`, which the ticket key replaces (for instance `agent/{key}`).
4. Under `container.image`, name the image a run starts, pinned by a tag other than `latest` or by a digest.
5. Under `git`, set the `name` and `email` of the commits the worker makes; use the account's noreply address, so that a push is not refused for exposing a private email.
6. Map every status key of the header to the status name your tracker really uses; two keys cannot share one status.
7. Under `hold_merge_label`, name the ticket label that holds a merge back: a ticket carrying it keeps its pull request open.
8. Under `agent.provider`, choose how the agent signs in: `subscription` for a Claude Code OAuth token, `api_key` for an Anthropic API key. Under `agent.model` and `agent.review_model`, name the model the agent runs on and the one the guard review runs on; each is passed to Claude Code by `--model`, and an alias such as `opus` follows the latest version of its family.
9. Write the name of the environment variable that holds each token — the tracker's under `tracker.token_env`, the credential of the agent's provider under `agent.token_env`, and two for the repository: the agent's under `repository.agent_token_env`, the worker's under `repository.worker_token_env`, each in its own variable — and set those variables in the worker's environment. The file never holds a token. A run hands the tracker token, the agent credential, the agent's repository token, the tracker site and its account to its container, and nothing else from the worker's environment: the worker's repository token stays with the worker, which merges with it (see [Protecting the base branch](#protecting-the-base-branch)).
10. Optionally, tighten a limit under `limits`; a value above the header ceiling is refused, and so is a `run_timeout_seconds` that does not exceed `review_timeout_seconds` plus the 180 seconds kept for the agent to wait for the CI.
11. Optionally, under `tracker.required_fields`, map the id of each Jira field a ticket must fill before a run takes it — `customfield_` followed by digits, or a lowercase system field name such as `environment` — to the name the worker shows for it. A ticket that leaves one of them empty is not run: the worker comments on it, naming the empty fields, and moves it to `blocked`. The filled fields are given to the agent after the description, in the order written. A field is read as text when it holds a string, an Atlassian document, an option, or a list of strings and options; a field of any other shape makes the ticket unreadable.
12. Optionally, under `handover`, say what a human must merge: `sensitive_paths` lists glob patterns of paths, and `max_changed_lines` bounds the lines added plus removed outside the files matching `test_paths`; those last two are set together. Patterns are relative to the repository root, use forward slashes, match the whole path case-sensitively, hold no `.` or `..` segment, and `**` spans directories; a pattern ending in `/` or `/**`, or whose last segment holds none of `*`, `?` and `[`, names a directory and covers that path and everything under it. This block needs `gh` 2.48.0 or later, for `--slurp`. See [The guard review before a merge](#the-guard-review-before-a-merge).

Real instances and secrets stay outside the repository: the file under `examples/` is fictional and is the only instance committed here.

## Waking the worker

```
dev-worker wake --instance <instance file>
```

`wake` lists the tickets of the instance project in its `ready` status, in the tracker's rank order, and takes the first whose blockers are all in its `done` status. It runs the `deliver_ticket` mandate on it under a task_id of its own, `wake-` followed by a UUID, exactly as `run` would, and carries it to the end: one ticket per wake. It is meant to be started by a timer.

Before it lists the ready tickets, `wake` hands back the tickets a dead run left in the `in_progress` status — a worker killed mid-run leaves its ticket there. It does so under the instance lock, so only when no run of the instance is alive; while a run that is not gone holds the lock, or a run whose process still runs, it skips this step. A ticket settles when it enters any status of the instance other than `ready` and `in_progress` — `blocked`, `in_review`, `done`. A ticket in `in_progress` is the worker's when it carries a claim posted since it last entered `ready`, the latest such claim was posted after it last settled, and the ticket last entered `in_progress` no earlier than that claim and at most 360 seconds after it, both times as the tracker recorded them — the worker sends that move within 300 seconds of its claim, and the tracker may take up to 60 seconds more to record it; any other ticket there was moved by a person and is left alone. Each dead run counts as one attempt, counted from the comments that start with `dev-worker returns this ticket to the queue: ` posted since the ticket last settled. Below the `max_attempts` limit, the worker posts such a comment, naming the attempt, and moves the ticket back to `ready`, where this same wake may take it; at the limit, it comments that the runs died that many times, asks a person to look at the ticket and move it back to `ready`, and moves it to `blocked`. Once handed back after it settled, the ticket counts its attempts from one again.

A run that ends without an outcome because the agent's account reached its usage limit moves its ticket back to `ready`, with a comment saying when the limit resets, and is not counted as an attempt. The worker reads the reset time from the last `rate_limit_event` Claude Code wrote, when its `status` is `rejected` and the account is not using overage; when that event gives no reset time, or when there is none but the run failed with the API error 429, it waits the `usage_limit_wait_seconds` limit from the end of the run. It writes that time, in UTC, in `usage-limit.json` in the instance directory. Until then, `wake` hands back the tickets of dead runs but takes no ticket: it writes nothing on stdout, one line on stderr saying until when, and exits with 0. Once the time has passed, the file stays and is ignored.

Once it has found a ticket to take, `wake` reads the CI of the most recent of the last 10 commits of the default branch that reports a check. When one of its checks failed, it takes no ticket: it writes nothing on stdout, one line on stderr naming the branch, the commit and the failed checks, and exits with 0. A CI still running does not stop it, nor does a CI it cannot read. A failure the forge caused rather than the code — a GitHub Actions run that was cancelled or failed to start, or whose failed jobs failed only in their `Set up job` step — is rerun instead, once: a run already on its second attempt counts as failed, as does one that cannot be read or whose rerun is refused. A run cancelled on a commit older than the branch head is taken as superseded by a later push: its cancelled checks do not count, and it is not rerun — even when it was cancelled by hand and the head runs no CI of its own, as with `[skip ci]`. This check only sees a CI that runs on the default branch itself: a repository whose workflows run on `pull_request` alone reports no check on it, so `wake` is never stopped there. For the check to guard a repository, its CI also runs on `push` to the default branch.

It writes on stdout the JSON output of the run it made, and nothing when no ticket is ready. Its exit code says whether the wake failed, not whether the ticket was delivered:

- **0** when no ticket is ready; while the usage limit has not reset; when the CI of the default branch failed; when a run reached the usage limit and its ticket was moved back to `ready` with nothing else failing; when the lock is held by a run that is not gone; when another run claimed the ticket first, or the ticket is no longer ready to be taken; when the ticket left a required field empty and was commented on and moved to `blocked`; and when the run ended on an outcome of the agent — `done`, merged, held or handed over to a human, or `blocked` — with the ticket commented on and moved and nothing else failing; a `done` run left blocked counts here only when it was blocked by a verdict: no open pull request, a CI that failed, has not finished, or has reported no check yet on the head commit while the repository has an active GitHub Actions workflow file under `.github/workflows/`, a guard review that found a reachable defect, or a head that moved before the merge;
- **1** otherwise: the instance or `WORKER.md` cannot be loaded, the tracker cannot list or read the tickets or their comments, a ticket left by a dead run cannot be commented on or moved, the claim cannot be posted or read back, the lock or the journal cannot be read or written, a worktree cannot be rescued or removed, the run ended without an outcome of the agent (its plugin, worktree or container could not be prepared or run), or the ticket cannot be moved or commented on. A `done` run left blocked by the worker's own chain counts here too: the pull request, its CI or the ticket cannot be read, the repository reports no check on the pull request and has no active GitHub Actions workflow file under `.github/workflows/` — GitHub's dynamic workflows, such as Dependabot's, do not count — the merge fails, or the guard review gives no usable verdict — its plugin cannot be fetched, its checkout cannot be made, its credential is missing, its container fails, no time is left for it, it reaches its turn ceiling, its answer is not a valid verdict, or it says it did not take place. When the wake fails before it picks a ticket, its output is a failed one with no task_id.

## Protecting the base branch

Only the worker merges, and the guarantee holds only if the worker and the agent are two different GitHub accounts: a token acts with its account's role, so an agent token created on the worker's account would be let through too.

On GitHub, in the repository's **Settings → Rules → Rulesets**, add a **New branch ruleset** that targets the default branch, turn on **Restrict updates**, and add the worker's role to its bypass list — for instance Repository admin, held by the worker's account — while the agent's account is a collaborator with Write only.

On a repository owned by a user account, a collaborator cannot use a fine-grained token on it, so the agent's token is a classic token with the `repo` scope; it reaches every repository the agent's account can, so give that account access to this repository only — and it can still write to any public repository (issues, comments, pull requests), which a fine-grained token cannot. On a repository owned by an organization the agent's account is a member of, a fine-grained token limited to this repository, with **Contents** and **Pull requests** set to read and write, is enough. With this rule, a merge attempted with the agent's token is refused.

When the head commit of a pull request reports no check yet, the worker lists the repository's GitHub Actions workflows with its own token, the one under `repository.worker_token_env`; it also reads the workflow runs of the default branch and reruns those the forge failed (see `wake` above). A fine-grained worker token needs **Actions** set to read and write, which a classic token's `repo` scope already covers.

## The guard review before a merge

Once a run ends on `done`, its pull request is merged only if its CI passed, the ticket does not carry the `hold_merge_label`, and a guard review finds no reachable defect in it. The review is a second Claude Code run, in its own disposable container started from the instance image: it reads, mounted read only, a checkout of the pull request's head commit made from the mirror, and diffs it locally against the pull request's base branch. It receives the agent credential, plus the git setting that lets it read the mounted checkout; no repository or tracker token. It runs as the `reviewer` agent of the pinned plugin, within the `review_max_turns` limit and the time left in the run, of which the agent's run keeps `review_timeout_seconds` for it, and answers whether it reviewed the change, why not when it did not, and the list of the defects it found. The pull request is merged only on an answer that the change was reviewed with no defect; a defect found, a review that says it did not take place, or a review that gives no valid answer leaves the ticket blocked with the reason in its comment, and the pull request open.

An instance with a `handover` block has the files of the pull request read before that review. When one of them matches `sensitive_paths` by its path or the path it was renamed from, when the lines changed outside tests exceed `max_changed_lines` (a renamed file is a test only when both its paths match `test_paths`), or when the pull request reaches the 3000 files GitHub lists at most, the pull request stays open, unreviewed and unmerged, and the ticket goes to `in_review` with a comment naming why. When the files cannot be read, the ticket is blocked. An instance without the block merges as described above.

## Development

Requires [uv](https://docs.astral.sh/uv/). The tests create symbolic links: on Windows, turn on Developer Mode first. From the repository root:

```
uv sync
uv run poe check
```

`poe check` runs lint, format check, types, and tests with a 100 % line and branch coverage floor.

## License

[MIT](LICENSE).
