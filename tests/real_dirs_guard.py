"""Guard for the `real_platform_dirs` fixture (tests/conftest.py, #456 merge check).

That fixture puts the platform's real user-directory properties back over #488's per-test redirect, for the few tests
that compare paths. On macOS and Windows they resolve to the REAL data directories, so a test that also wrote would
write there. The fixture records, for every exposed directory, whether it exists and the names of its top-level
entries, and fails the test if that changed: a test that creates one of the directories, or a file or folder in one.

Only existence and top-level NAMES are compared (GM-1): a developer's running bridge appends to its logs and rewrites
its existing state files inside these directories during a test run, and must not make the test fail. Nothing is read.
"""

from __future__ import annotations

from pathlib import Path

Fingerprint = dict[str, object]


def fingerprint(dirs: list[Path]) -> Fingerprint:
    """For each directory: None when it does not exist, else the sorted names of its top-level entries."""
    out: Fingerprint = {}
    for d in dirs:
        if d.is_dir():
            try:
                out[str(d)] = sorted(p.name for p in d.iterdir())
            except OSError:
                out[str(d)] = "unreadable"
        else:
            out[str(d)] = None if not d.exists() else "not a directory"
    return out


def assert_untouched(before: Fingerprint, after: Fingerprint) -> None:
    for d, was in before.items():
        if after.get(d) != was:
            raise AssertionError(
                f"a test using real_platform_dirs created or changed a real user directory: {d} "
                "(the fixture is for comparing paths only)"
            )
