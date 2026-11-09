"""A stand-in for `gh`: records how it was called and answers what the test asked for.

`FAKE_GH_RECORD` names the file it writes its arguments and `GH_*` variables to, as JSON.
`FAKE_GH_STDOUT`, `FAKE_GH_STDERR` and `FAKE_GH_EXIT` set its answer and exit code.
`FAKE_GH_SLEEP`, in seconds, delays the answer after the call is recorded.
"""

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    """Record the call, write the canned answer, and return the canned exit code."""
    record = {
        "argv": sys.argv[1:],
        "env": {
            name: os.environ.get(name) for name in ("GH_TOKEN", "GH_PROMPT_DISABLED")
        },
    }
    Path(os.environ["FAKE_GH_RECORD"]).write_text(json.dumps(record), encoding="utf-8")
    time.sleep(float(os.environ.get("FAKE_GH_SLEEP", "0")))
    sys.stdout.buffer.write(os.environ.get("FAKE_GH_STDOUT", "").encode("utf-8"))
    sys.stderr.buffer.write(os.environ.get("FAKE_GH_STDERR", "").encode("utf-8"))
    return int(os.environ.get("FAKE_GH_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
