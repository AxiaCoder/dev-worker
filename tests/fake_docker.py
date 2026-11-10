"""A stand-in for `docker`: records each call and answers what the test asked for its subcommand.

`FAKE_DOCKER_RECORD` names the file it appends its arguments to, one JSON line per call.
For a subcommand such as `run`, `FAKE_DOCKER_RUN_STDOUT`, `FAKE_DOCKER_RUN_STDERR` and
`FAKE_DOCKER_RUN_EXIT` set its answer and exit code, and `FAKE_DOCKER_RUN_SLEEP`, in
seconds, delays the answer after the call is recorded. Any of them suffixed with `_<n>`,
such as `FAKE_DOCKER_PS_STDOUT_1`, overrides it for the call of that subcommand that
`n` earlier calls of it preceded.
"""

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    """Record the call, write the canned answer of its subcommand, and return its exit code."""
    arguments = sys.argv[1:]
    record = Path(os.environ["FAKE_DOCKER_RECORD"])
    earlier = (
        [json.loads(line) for line in record.read_text("utf-8").splitlines()]
        if record.exists()
        else []
    )
    with record.open("a", encoding="utf-8") as file:
        file.write(json.dumps(arguments) + "\n")
    rank = sum(call[0] == arguments[0] for call in earlier)
    prefix = f"FAKE_DOCKER_{arguments[0].upper()}_"

    def setting(name: str, default: str) -> str:
        """Return the setting `name` for this call, its unranked value, or `default`."""
        return os.environ.get(
            f"{prefix}{name}_{rank}", os.environ.get(f"{prefix}{name}", default)
        )

    time.sleep(float(setting("SLEEP", "0")))
    sys.stdout.buffer.write(setting("STDOUT", "").encode("utf-8"))
    sys.stderr.buffer.write(setting("STDERR", "").encode("utf-8"))
    return int(setting("EXIT", "0"))


if __name__ == "__main__":
    sys.exit(main())
