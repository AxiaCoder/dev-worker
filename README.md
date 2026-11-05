# dev-worker

A worker that takes a framed ticket to a pull request, and merges it when nothing holds it back. An AI sits at the centre; it is given tools, and guardrails in code that it cannot bypass.

**Status: design phase.** Nothing runs yet.

## V1 in one paragraph

One main mandate — "take this ticket to a pull request" — in which the AI chains implementation, verification and review itself, using a pinned stack of agents and skills. The worker's code owns what must not depend on the model: the disposable worktree, the turn caps, the ticket statuses, and the merge. External systems (a ticket tracker, a code host) are reached through connectors; Jira and GitHub come by default, and an instance can plug others. Secrets are never in the repository: an instance file holds the name of an environment variable, not its value.

Out of V1: a framing mandate, narrow mandates, connectors beyond Jira and GitHub.

## Adding an instance

An instance is one YAML file that points the worker at a tracker project and a repository. It is read against the header of [`WORKER.md`](src/dev_worker/WORKER.md) and refused if it loosens it.

1. Copy [`examples/instance.yaml`](examples/instance.yaml) to a place **outside this repository**.
2. Set the tracker project and the repository name.
3. Map every status key of the header to the status name your tracker really uses; two keys cannot share one status.
4. Under `token_env`, write the name of the environment variable that holds each token, and set that variable in the worker's environment. The file never holds a token.
5. Optionally, tighten a limit under `limits`; a value above the header ceiling is refused.

Real instances and secrets stay outside the repository: the file under `examples/` is fictional and is the only instance committed here.

## Development

Requires [uv](https://docs.astral.sh/uv/). From the repository root:

```
uv sync
uv run poe check
```

`poe check` runs lint, format check, types, and tests with a 100 % line and branch coverage floor.

## License

[MIT](LICENSE).
