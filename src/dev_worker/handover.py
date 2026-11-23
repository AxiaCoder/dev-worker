"""The decision to leave a pull request open for a human, from the files it changes."""

from collections.abc import Sequence
from pathlib import PurePosixPath

from dev_worker.instance import Handover
from dev_worker.repositories import ChangedFile

MAX_LISTED_FILES = 3000
MAX_NAMED_PATHS = 20


def _matches_pattern(path: str, pattern: str, *, covers_directory_path: bool) -> bool:
    """Return whether the glob `pattern` covers `path`, matched in full and case-sensitively.

    A pattern ending in `/` or `/**` covers what is under the directory, and the
    directory path itself only when `covers_directory_path` is true, as for sensitive
    paths: a submodule is that single path. Any other pattern covers what it matches
    and, when its last segment holds none of `*`, `?` and `[`, what is under that path.
    """
    candidate = PurePosixPath(path)
    directory = pattern.rstrip("/").removesuffix("/**")
    if directory != pattern:
        return candidate.full_match(f"{directory}/**") or (
            covers_directory_path and candidate.full_match(directory)
        )
    if candidate.full_match(pattern):
        return True
    last = directory.rsplit("/", 1)[-1]
    return not set(last) & set("*?[") and candidate.full_match(f"{directory}/**")


def _matches(
    path: str, patterns: Sequence[str], *, covers_directory_path: bool
) -> bool:
    """Return whether one of `patterns` covers `path`, as `_matches_pattern` reads each with `covers_directory_path`."""
    return any(
        _matches_pattern(path, pattern, covers_directory_path=covers_directory_path)
        for pattern in patterns
    )


def _paths(file: ChangedFile) -> tuple[str, ...]:
    """Return every path `file` has: its path, then the one it was renamed from, if any."""
    if file.previous_path is None:
        return (file.path,)
    return (file.path, file.previous_path)


def _named(paths: Sequence[str]) -> str:
    """Return `paths` joined by commas, the first `MAX_NAMED_PATHS` only, then how many more there are."""
    named = ", ".join(paths[:MAX_NAMED_PATHS])
    hidden = len(paths) - MAX_NAMED_PATHS
    return f"{named} and {hidden} more" if hidden > 0 else named


def handover_reason(files: Sequence[ChangedFile], handover: Handover) -> str | None:
    """Return why a pull request changing `files` is left for a human, or `None` when nothing does.

    It is when a path or previous path of a file, tests included, matches
    `sensitive_paths`; when the lines added and removed in files that are not tests
    exceed `max_changed_lines`, a file being a test only when each of its paths matches
    `test_paths`; or when `files` reaches `MAX_LISTED_FILES`, where the list may be
    truncated. The sentence states every reason that applies, naming at most
    `MAX_NAMED_PATHS` matched paths and the lines counted against the bound.
    """
    reasons: list[str] = []
    sensitive = list(
        dict.fromkeys(
            path
            for file in files
            for path in _paths(file)
            if _matches(path, handover.sensitive_paths, covers_directory_path=True)
        )
    )
    if sensitive:
        reasons.append(f"it touches sensitive paths: {_named(sensitive)}")
    if handover.max_changed_lines is not None:
        changed = sum(
            file.additions + file.deletions
            for file in files
            if not all(
                _matches(path, handover.test_paths, covers_directory_path=False)
                for path in _paths(file)
            )
        )
        if changed > handover.max_changed_lines:
            reasons.append(
                f"it changes {changed} lines outside tests, above the bound of "
                f"{handover.max_changed_lines}"
            )
    if len(files) >= MAX_LISTED_FILES:
        reasons.append(
            f"it changes at least {MAX_LISTED_FILES} files, the most the host lists, "
            "so its list of files may be incomplete"
        )
    return f"{'; '.join(reasons)}." if reasons else None
