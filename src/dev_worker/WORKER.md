---
status_keys:
  - ready
  - in_progress
  - in_review
  - blocked
  - done
outcomes:
  done: in_review
  blocked: blocked
  needs_human: blocked
  abandoned: blocked
ceilings:
  run_timeout_seconds: 1800
  max_turns: 60
plugin:
  repository: https://github.com/AxiaCoder/claude-config.git
  tag: atelier--v1.0.0
---

# Worker

This worker takes a framed ticket and carries it to a pull request.

The header above declares the status keys of the tracker, the outcome a run can end on with the
status key each outcome moves the ticket to, the absolute ceilings no instance may exceed, and the
plugin every run loads: the git repository it is cloned from and the tag it is taken at.

The rules of conduct the AI follows during a run will be written here in a later step.
