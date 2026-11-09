"""A stand-in for `twg`: records how it was called and answers what the test asked for.

`FAKE_TWG_RECORD` names the file it writes its arguments and `TWG_*` variables to, as JSON.
`FAKE_TWG_STDOUT`, `FAKE_TWG_STDERR` and `FAKE_TWG_EXIT` set its answer and exit code.
`FAKE_TWG_SLEEP`, in seconds, delays the answer after the call is recorded.
"""

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    """Record the call, wait if asked, write the canned answer, and return the exit code."""
    record = {
        "argv": sys.argv[1:],
        "env": {
            name: os.environ.get(name) for name in ("TWG_SITE", "TWG_USER", "TWG_TOKEN")
        },
    }
    Path(os.environ["FAKE_TWG_RECORD"]).write_text(json.dumps(record), encoding="utf-8")
    time.sleep(float(os.environ.get("FAKE_TWG_SLEEP", "0")))
    sys.stdout.buffer.write(os.environ.get("FAKE_TWG_STDOUT", "").encode("utf-8"))
    sys.stderr.buffer.write(os.environ.get("FAKE_TWG_STDERR", "").encode("utf-8"))
    return int(os.environ.get("FAKE_TWG_EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
