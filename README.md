# dev-worker

A worker that takes a framed ticket to a pull request, and merges it when nothing holds it back. An AI sits at the centre; it is given tools, and guardrails in code that it cannot bypass.

**Status: design phase.** Nothing runs yet.

## V1 in one paragraph

One main mandate — "take this ticket to a pull request" — in which the AI chains implementation, verification and review itself, using a pinned stack of agents and skills. The worker's code owns what must not depend on the model: the disposable worktree, the turn caps, the ticket statuses, and the merge. External systems (a ticket tracker, a code host) are reached through connectors; Jira and GitHub come by default, and an instance can plug others. Secrets are never in the repository: an instance file holds the name of an environment variable, not its value.

Out of V1: a framing mandate, narrow mandates, connectors beyond Jira and GitHub.

## Adding an instance

An instance is one YAML file that points the worker at a tracker project and a repository. It is read against the header of [`WORKER.md`](src/dev_worker/WORKER.md) and refused if it loosens it.

1. Copy [`examples/instance.yaml`](examples/instance.yaml) to a place **outside this repository**.
2. Set the tracker site address (`site`), the email of its account (`user`), the tracker project and the repository name.
3. Under `branch`, write the template of the branch a run works on; it must contain `{key}`, which the ticket key replaces (for instance `agent/{key}`).
4. Under `container.image`, name the image a run starts, pinned by a tag other than `latest` or by a digest.
5. Under `git`, set the `name` and `email` of the commits the worker makes; use the account's noreply address, so that a push is not refused for exposing a private email.
6. Map every status key of the header to the status name your tracker really uses; two keys cannot share one status.
7. Under `hold_merge_label`, name the ticket label that holds a merge back: a ticket carrying it keeps its pull request open.
8. Under `agent.provider`, choose how the agent signs in: `subscription` for a Claude Code OAuth token, `api_key` for an Anthropic API key.
9. Write the name of the environment variable that holds each token — the tracker's under `tracker.token_env`, the credential of the agent's provider under `agent.token_env`, and two for the repository: the agent's under `repository.agent_token_env`, the worker's under `repository.worker_token_env`, each in its own variable — and set those variables in the worker's environment. The file never holds a token. A run hands the tracker token, the agent credential, the agent's repository token, the tracker site and its account to its container, and nothing else from the worker's environment: the worker's repository token stays with the worker, which merges with it (see [Protecting the base branch](#protecting-the-base-branch)).
10. Optionally, tighten a limit under `limits`; a value above the header ceiling is refused.

Real instances and secrets stay outside the repository: the file under `examples/` is fictional and is the only instance committed here.

## Protecting the base branch

Only the worker merges, and the guarantee holds only if the worker and the agent are two different GitHub accounts: a token acts with its account's role, so an agent token created on the worker's account would be let through too.

On GitHub, in the repository's **Settings → Rules → Rulesets**, add a **New branch ruleset** that targets the default branch, turn on **Restrict updates**, and add the worker's role to its bypass list — for instance Repository admin, held by the worker's account — while the agent's account is a collaborator with Write only.

On a repository owned by a user account, a collaborator cannot use a fine-grained token on it, so the agent's token is a classic token with the `repo` scope; it reaches every repository the agent's account can, so give that account access to this repository only. On a repository owned by an organization the agent's account is a member of, a fine-grained token limited to this repository, with **Contents** and **Pull requests** set to read and write, is enough. With this rule, a merge attempted with the agent's token is refused.

## Development

Requires [uv](https://docs.astral.sh/uv/). The tests create symbolic links: on Windows, turn on Developer Mode first. From the repository root:

```
uv sync
uv run poe check
```

`poe check` runs lint, format check, types, and tests with a 100 % line and branch coverage floor.

## License

[MIT](LICENSE).
