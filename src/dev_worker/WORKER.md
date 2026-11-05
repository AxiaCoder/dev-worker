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
---

# Worker

This worker takes a framed ticket and carries it to a pull request.

The header above declares the status keys of the tracker, the outcome a run can end on with the
status key each outcome moves the ticket to, and the absolute ceilings no instance may exceed.

The rules of conduct the AI follows during a run will be written here in a later step.
