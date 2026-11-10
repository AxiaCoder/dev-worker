"""A stand-in for `docker`: records each call and answers what the test asked for its subcommand.

`FAKE_DOCKER_RECORD` names the file it appends its arguments to, one JSON line per call.
For a subcommand such as `run`, `FAKE_DOCKER_RUN_STDOUT`, `FAKE_DOCKER_RUN_STDERR` and
`FAKE_DOCKER_RUN_EXIT` set its answer and exit code, and `FAKE_DOCKER_RUN_SLEEP`, in
seconds, delays the answer after the call is recorded.
"""

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    """Record the call, write the canned answer of its subcommand, and return its exit code."""
    arguments = sys.argv[1:]
    with Path(os.environ["FAKE_DOCKER_RECORD"]).open("a", encoding="utf-8") as record:
        record.write(json.dumps(arguments) + "\n")
    prefix = f"FAKE_DOCKER_{arguments[0].upper()}_"
    time.sleep(float(os.environ.get(f"{prefix}SLEEP", "0")))
    sys.stdout.buffer.write(os.environ.get(f"{prefix}STDOUT", "").encode("utf-8"))
    sys.stderr.buffer.write(os.environ.get(f"{prefix}STDERR", "").encode("utf-8"))
    return int(os.environ.get(f"{prefix}EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
