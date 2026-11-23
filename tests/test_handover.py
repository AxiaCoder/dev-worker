import pytest

from dev_worker.handover import MAX_LISTED_FILES, MAX_NAMED_PATHS, handover_reason
from dev_worker.instance import Handover
from dev_worker.repositories import ChangedFile

SENSITIVE = Handover(sensitive_paths=(".github/workflows/**", "src/auth/**"))
BOUNDED = Handover(max_changed_lines=10, test_paths=("tests/**",))


def changed(
    path: str,
    previous_path: str | None = None,
    additions: int = 0,
    deletions: int = 0,
) -> ChangedFile:
    """Return a `ChangedFile` of `path`, with no line changed by default."""
    return ChangedFile(path, previous_path, additions, deletions)


def test_a_pull_request_touching_nothing_declared_is_not_handed_over() -> None:
    assert handover_reason([changed("src/app.py", additions=9)], SENSITIVE) is None


def test_no_file_at_all_is_not_handed_over() -> None:
    assert handover_reason([], BOUNDED) is None


def test_a_sensitive_path_is_named_in_the_reason() -> None:
    reason = handover_reason([changed("src/auth/login.py")], SENSITIVE)

    assert reason == "it touches sensitive paths: src/auth/login.py."


@pytest.mark.parametrize(
    "path",
    [".github/workflows/ci.yml", ".github/workflows/nested/ci.yml", "src/auth/a.py"],
)
def test_a_double_star_spans_directories(path: str) -> None:
    assert handover_reason([changed(path)], SENSITIVE) is not None


@pytest.mark.parametrize(
    "path", ["SRC/AUTH/a.py", "lib/src/auth/a.py", "src/authx/a.py"]
)
def test_a_pattern_matches_the_whole_path_case_sensitively(path: str) -> None:
    assert handover_reason([changed(path)], SENSITIVE) is None


def test_a_file_moved_out_of_a_sensitive_path_is_caught_by_its_old_path() -> None:
    moved = changed("docs/ci.yml", previous_path=".github/workflows/ci.yml")

    assert (
        handover_reason([moved], SENSITIVE)
        == "it touches sensitive paths: .github/workflows/ci.yml."
    )


def test_a_file_moved_into_a_sensitive_path_is_caught_by_its_new_path() -> None:
    moved = changed("src/auth/login.py", previous_path="src/login.py")

    assert (
        handover_reason([moved], SENSITIVE)
        == "it touches sensitive paths: src/auth/login.py."
    )


def test_a_sensitive_test_file_is_handed_over() -> None:
    handover = Handover(
        sensitive_paths=("tests/conftest.py",),
        max_changed_lines=10,
        test_paths=("tests/**",),
    )

    reason = handover_reason([changed("tests/conftest.py")], handover)

    assert reason == "it touches sensitive paths: tests/conftest.py."


def test_a_path_matched_twice_is_named_once() -> None:
    files = [
        changed("src/auth/a.py"),
        changed("src/auth/b.py", previous_path="src/auth/a.py"),
    ]

    assert (
        handover_reason(files, SENSITIVE)
        == "it touches sensitive paths: src/auth/a.py, src/auth/b.py."
    )


def test_exactly_the_named_cap_lists_every_path() -> None:
    files = [changed(f"src/auth/{index}.py") for index in range(MAX_NAMED_PATHS)]

    reason = handover_reason(files, SENSITIVE)

    assert reason is not None
    assert "more" not in reason
    assert f"src/auth/{MAX_NAMED_PATHS - 1}.py" in reason


def test_paths_above_the_named_cap_are_counted_not_named() -> None:
    files = [changed(f"src/auth/{index}.py") for index in range(MAX_NAMED_PATHS + 3)]

    reason = handover_reason(files, SENSITIVE)

    assert reason is not None
    assert reason.endswith(f"src/auth/{MAX_NAMED_PATHS - 1}.py and 3 more.")
    assert f"src/auth/{MAX_NAMED_PATHS}.py" not in reason


def test_lines_exactly_at_the_bound_are_not_handed_over() -> None:
    files = [changed("src/app.py", additions=6, deletions=4)]

    assert handover_reason(files, BOUNDED) is None


def test_one_line_above_the_bound_is_handed_over() -> None:
    files = [changed("src/app.py", additions=6, deletions=5)]

    assert (
        handover_reason(files, BOUNDED)
        == "it changes 11 lines outside tests, above the bound of 10."
    )


def test_lines_are_summed_across_files() -> None:
    files = [changed("src/a.py", additions=6), changed("src/b.py", deletions=5)]

    reason = handover_reason(files, BOUNDED)

    assert reason is not None
    assert "11 lines" in reason


def test_lines_of_test_files_do_not_count() -> None:
    files = [
        changed("tests/test_app.py", additions=500, deletions=500),
        changed("src/app.py", additions=10),
    ]

    assert handover_reason(files, BOUNDED) is None


def test_a_file_renamed_inside_tests_does_not_count() -> None:
    files = [changed("tests/b.py", previous_path="tests/a.py", additions=500)]

    assert handover_reason(files, BOUNDED) is None


def test_a_production_file_renamed_into_tests_still_counts() -> None:
    files = [changed("tests/app.py", previous_path="src/app.py", deletions=11)]

    reason = handover_reason(files, BOUNDED)

    assert reason is not None
    assert "11 lines" in reason


def test_a_test_file_renamed_into_production_counts() -> None:
    files = [changed("src/app.py", previous_path="tests/app.py", additions=11)]

    reason = handover_reason(files, BOUNDED)

    assert reason is not None
    assert "11 lines" in reason


def test_no_bound_ignores_the_size() -> None:
    files = [changed("src/app.py", additions=1_000_000)]

    assert handover_reason(files, SENSITIVE) is None


def test_one_file_below_the_listing_limit_is_not_handed_over() -> None:
    files = [changed(f"src/{index}.py") for index in range(MAX_LISTED_FILES - 1)]

    assert handover_reason(files, SENSITIVE) is None


def test_a_list_at_the_listing_limit_is_handed_over_as_maybe_incomplete() -> None:
    files = [changed(f"src/{index}.py") for index in range(MAX_LISTED_FILES)]

    assert handover_reason(files, SENSITIVE) == (
        "it changes at least 3000 files, the most the host lists, so its list of "
        "files may be incomplete."
    )


def test_every_reason_that_applies_is_stated() -> None:
    handover = Handover(
        sensitive_paths=("src/auth/**",),
        max_changed_lines=10,
        test_paths=("tests/**",),
    )
    files = [changed("src/auth/a.py", additions=11)] + [
        changed(f"tests/{index}.py") for index in range(MAX_LISTED_FILES)
    ]

    assert handover_reason(files, handover) == (
        "it touches sensitive paths: src/auth/a.py; "
        "it changes 11 lines outside tests, above the bound of 10; "
        "it changes at least 3000 files, the most the host lists, so its list of "
        "files may be incomplete."
    )


@pytest.mark.parametrize("pattern", ["src/auth", "src/auth/"])
@pytest.mark.parametrize("path", ["src/auth/a.py", "src/auth/d/a.py"])
def test_a_pattern_naming_a_directory_covers_everything_under_it(
    pattern: str, path: str
) -> None:
    handover = Handover(sensitive_paths=(pattern,))

    assert (
        handover_reason([changed(path)], handover)
        == f"it touches sensitive paths: {path}."
    )


@pytest.mark.parametrize("pattern", ["src/auth", "src/auth/"])
@pytest.mark.parametrize("path", ["src/authx/a.py", "src/auth.py"])
def test_a_pattern_naming_a_directory_leaves_a_lookalike_path_alone(
    pattern: str, path: str
) -> None:
    handover = Handover(sensitive_paths=(pattern,))

    assert handover_reason([changed(path)], handover) is None


def test_a_pattern_naming_a_file_still_matches_that_file() -> None:
    handover = Handover(sensitive_paths=("README.md",))

    assert (
        handover_reason([changed("README.md")], handover)
        == "it touches sensitive paths: README.md."
    )


@pytest.mark.parametrize("pattern", ["src/auth", "src/auth/"])
def test_a_file_moved_out_of_a_named_directory_is_caught_by_its_old_path(
    pattern: str,
) -> None:
    handover = Handover(sensitive_paths=(pattern,))
    moved = changed("docs/login.py", previous_path="src/auth/login.py")

    assert (
        handover_reason([moved], handover)
        == "it touches sensitive paths: src/auth/login.py."
    )


@pytest.mark.parametrize("pattern", ["tests", "tests/"])
@pytest.mark.parametrize("path", ["tests/test_app.py", "tests/d/test_app.py"])
def test_a_test_path_naming_a_directory_leaves_its_files_out_of_the_size(
    pattern: str, path: str
) -> None:
    handover = Handover(max_changed_lines=10, test_paths=(pattern,))

    assert handover_reason([changed(path, additions=11)], handover) is None


@pytest.mark.parametrize("path", [".github/ci.yml", ".github/workflows/ci.yml"])
def test_a_double_star_glob_still_spans_directories(path: str) -> None:
    handover = Handover(sensitive_paths=(".github/**",))

    assert handover_reason([changed(path)], handover) is not None


def test_a_single_star_glob_still_matches_its_own_level() -> None:
    handover = Handover(sensitive_paths=("src/auth/*",))

    assert handover_reason([changed("src/auth/a.py")], handover) is not None


def test_a_single_star_glob_still_stops_at_a_subdirectory() -> None:
    handover = Handover(sensitive_paths=("src/auth/*",))

    assert handover_reason([changed("src/auth/d/a.py")], handover) is None


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("packages/*/auth", "packages/api/auth/login.py"),
        ("packages/*/auth", "packages/api/auth"),
        ("**/secrets", "a/secrets/k"),
        ("**/secrets", "secrets/k"),
        ("**/secrets", "secrets"),
    ],
)
def test_a_glob_whose_last_segment_is_literal_covers_the_directory_it_names(
    pattern: str, path: str
) -> None:
    handover = Handover(sensitive_paths=(pattern,))

    assert (
        handover_reason([changed(path)], handover)
        == f"it touches sensitive paths: {path}."
    )


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("packages/*/auth", "packages/api/auth/login.py"),
        ("**/secrets", "a/secrets/k"),
        ("**/secrets", "secrets/k"),
    ],
)
def test_a_test_path_whose_last_segment_is_literal_leaves_its_files_out_of_the_size(
    pattern: str, path: str
) -> None:
    handover = Handover(max_changed_lines=10, test_paths=(pattern,))

    assert handover_reason([changed(path, additions=11)], handover) is None


@pytest.mark.parametrize("pattern", ["vendor/auth/**", "vendor/auth/", "vendor/auth"])
def test_a_pattern_naming_a_directory_covers_the_directory_path_itself(
    pattern: str,
) -> None:
    handover = Handover(sensitive_paths=(pattern,))

    assert (
        handover_reason([changed("vendor/auth")], handover)
        == "it touches sensitive paths: vendor/auth."
    )


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("src/auth/*", "src/auth/d/a.py"),
        ("src/auth/*", "src/auth"),
        ("**/*.env", "a.env/k"),
        ("packages/*/auth", "packages/api/authx/a.py"),
    ],
)
def test_a_pattern_not_naming_a_directory_does_not_reach_under_its_match(
    pattern: str, path: str
) -> None:
    handover = Handover(sensitive_paths=(pattern,))

    assert handover_reason([changed(path)], handover) is None


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("**/test*/**", "src/test_utils.py"),
        ("**/test*/**", "src/testing.py"),
        ("**/test*/", "src/test_utils.py"),
        ("**/test*/", "src/testing.py"),
        ("test*/**", "testing.py"),
        ("**/*test*/**", "src/latest.py"),
        ("**/*spec*/**", "src/inspector.py"),
        ("tests/", "tests"),
        ("tests/**", "tests"),
    ],
)
def test_a_test_path_ending_in_a_slash_or_double_star_counts_a_file_named_like_it(
    pattern: str, path: str
) -> None:
    handover = Handover(max_changed_lines=10, test_paths=(pattern,))

    assert (
        handover_reason([changed(path, additions=500)], handover)
        == "it changes 500 lines outside tests, above the bound of 10."
    )


@pytest.mark.parametrize(
    ("pattern", "path"),
    [
        ("**/test*/**", "src/tests/a.py"),
        ("tests/", "tests/a.py"),
        ("tests/**", "tests/a.py"),
        ("tests/conftest.py", "tests/conftest.py"),
        ("tests", "tests/a.py"),
    ],
)
def test_a_test_path_still_leaves_out_what_it_covers(pattern: str, path: str) -> None:
    handover = Handover(max_changed_lines=10, test_paths=(pattern,))

    assert handover_reason([changed(path, additions=500)], handover) is None


@pytest.mark.parametrize("pattern", ["vendor/auth/**", "vendor/auth/"])
def test_a_sensitive_path_ending_in_a_slash_or_double_star_still_catches_the_path_itself(
    pattern: str,
) -> None:
    handover = Handover(
        sensitive_paths=(pattern,), max_changed_lines=10, test_paths=("tests/**",)
    )

    assert (
        handover_reason([changed("vendor/auth")], handover)
        == "it touches sensitive paths: vendor/auth."
    )
