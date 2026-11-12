"""A stand-in for `gh`: records how it was called and answers what the test asked for.

`FAKE_GH_RECORD` names the file it writes its arguments and `GH_*` variables to, as JSON;
each call replaces the previous one. When `FAKE_GH_CALLS` is set, each call is also appended
to that file as one JSON line, so a test can read every call in order.
`FAKE_GH_STDOUT`, `FAKE_GH_STDERR` and `FAKE_GH_EXIT` set its answer and exit code.
`FAKE_GH_ANSWERS`, a JSON list of `{"stdout", "stderr", "exit"}` objects, replaces them with
one answer per call, the nth call getting the nth answer; it needs `FAKE_GH_CALLS` to count
the calls, and a call past the end of the list exits 99.
`FAKE_GH_SLEEP`, in seconds, delays the answer after the call is recorded.
"""

import json
import os
import sys
import time
from pathlib import Path


def _count_and_append(record: str) -> int:
    """Append `record` as one line to the `FAKE_GH_CALLS` file and return how many calls preceded it."""
    calls = Path(os.environ["FAKE_GH_CALLS"])
    earlier = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    with calls.open("a", encoding="utf-8") as log:
        log.write(record + "\n")
    return len(earlier)


def main() -> int:
    """Record the call, write the canned answer, and return the canned exit code."""
    record = json.dumps(
        {
            "argv": sys.argv[1:],
            "env": {
                name: os.environ.get(name)
                for name in ("GH_TOKEN", "GH_PROMPT_DISABLED")
            },
        }
    )
    Path(os.environ["FAKE_GH_RECORD"]).write_text(record, encoding="utf-8")
    index = _count_and_append(record) if "FAKE_GH_CALLS" in os.environ else 0
    time.sleep(float(os.environ.get("FAKE_GH_SLEEP", "0")))
    if "FAKE_GH_ANSWERS" in os.environ:
        answers = json.loads(os.environ["FAKE_GH_ANSWERS"])
        if index >= len(answers):
            sys.stderr.write(f"fake gh: no answer for call {index + 1}\n")
            return 99
        answer = answers[index]
        stdout, stderr, code = answer["stdout"], answer["stderr"], answer["exit"]
    else:
        stdout = os.environ.get("FAKE_GH_STDOUT", "")
        stderr = os.environ.get("FAKE_GH_STDERR", "")
        code = int(os.environ.get("FAKE_GH_EXIT", "0"))
    sys.stdout.buffer.write(stdout.encode("utf-8"))
    sys.stderr.buffer.write(stderr.encode("utf-8"))
    return code


if __name__ == "__main__":
    sys.exit(main())
