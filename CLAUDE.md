# dev-worker

## How we build this worker

One issue, one pass, with Axia steering the choices:

- an issue is framed in the issue itself, its open choices settled with Axia, then built in one go up to its pull request; Axia reviews the pull request, not the steps;
- every technical or architectural choice the issue leaves open is put to him **before** it is made, with a recommendation — during framing, or as soon as it shows up while building or reviewing;
- a choice with a single sensible answer is not a question: it is made, and said in the pull request.

## Platforms

Every tooling choice must work the same on Windows and on the Mac, where Axia develops, and on Debian, where the worker runs.

- paths handled through the language's path API only, never a hard-coded separator;
- no shell scripts for the project's own commands;
- line endings are LF everywhere (`.gitattributes`).

## Language

English for everything in this repository: code, comments, docs, issues, commits.

## History

The repository is public from its first commit. Write everything as it will be read by a stranger.

- Conventional Commits; what follows the type states a fact, not an action (`fix: long flags were let through`, not `fix: correct the guard`);
- no personal data, local paths or private references in code, docs, issues or commit messages;
- instance files committed here are examples only; real instances and secrets stay outside the repository;
- commits signed with the GitHub noreply address.
