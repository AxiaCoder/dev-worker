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
8. Under `token_env`, write the name of the environment variable that holds each token — the tracker's, the repository's, and under `agent` the Claude Code OAuth token — and set that variable in the worker's environment. The file never holds a token. A run hands these tokens, the tracker site and its account to its container, and nothing else from the worker's environment.
9. Optionally, tighten a limit under `limits`; a value above the header ceiling is refused.

Real instances and secrets stay outside the repository: the file under `examples/` is fictional and is the only instance committed here.

## Development

Requires [uv](https://docs.astral.sh/uv/). The tests create symbolic links: on Windows, turn on Developer Mode first. From the repository root:

```
uv sync
uv run poe check
```

`poe check` runs lint, format check, types, and tests with a 100 % line and branch coverage floor.

## License

[MIT](LICENSE).
