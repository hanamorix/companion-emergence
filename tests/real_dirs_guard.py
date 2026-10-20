"""Guard for the `real_platform_dirs` fixture (tests/conftest.py, #456 merge check).

That fixture puts the platform's real user-directory properties back over #488's per-test redirect, for the few tests
that compare paths. On macOS and Windows they resolve to the REAL data directories, so a test that also wrote would
write there. The fixture takes a stat-only fingerprint of every exposed directory before the test and fails the test
if any of them was created, or anything inside was added, removed or changed. File contents are never read.
"""

from __future__ import annotations

import os
from pathlib import Path

Fingerprint = dict[str, object]


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out[os.path.relpath(p, root)] = (st.st_mtime_ns, st.st_size)
    return out


def fingerprint(dirs: list[Path]) -> Fingerprint:
    """For each directory: None when it does not exist, else every entry inside with its mtime and size."""
    return {str(d): (_tree(d) if d.is_dir() else (None if not d.exists() else "not a directory")) for d in dirs}


def assert_untouched(before: Fingerprint, after: Fingerprint) -> None:
    for d, was in before.items():
        now = after.get(d)
        if now != was:
            raise AssertionError(
                f"a test using real_platform_dirs created or changed a real user directory: {d} "
                "(the fixture is for comparing paths only)"
            )
