# dev-worker

A worker that takes a framed ticket to a pull request, and merges it when nothing holds it back. An AI sits at the centre; it is given tools, and guardrails in code that it cannot bypass.

**Status: design phase.** Nothing runs yet.

## V1 in one paragraph

One main mandate — "take this ticket to a pull request" — in which the AI chains implementation, verification and review itself, using a pinned stack of agents and skills. The worker's code owns what must not depend on the model: the disposable worktree, the turn caps, the ticket statuses, and the merge. External systems (a ticket tracker, a code host) are reached through connectors; Jira and GitHub come by default, and an instance can plug others. Secrets are never in the repository: an instance file holds the name of an environment variable, not its value.

Out of V1: a framing mandate, narrow mandates, connectors beyond Jira and GitHub.

## License

[MIT](LICENSE).
