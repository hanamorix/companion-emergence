"""Cross-process exclusive file locks for shared JSONL data files.

Audit 2026-05-07 P2-2: ``soul_candidates.jsonl`` is appended to by the
ingest pipeline AND rewritten by the soul-review path. Without
coordination, a candidate queued between review's read and write is
silently dropped when review replaces the file. ``soul_candidate_lock``
gates both paths through the same OS-level exclusive lock so the
read-modify-rewrite is atomic against concurrent appends.

Implementation:

* POSIX (macOS / Linux): ``fcntl.flock`` on a sidecar ``<file>.lock``.
* Windows: ``msvcrt.locking`` on the same sidecar with a single-byte
  region. Tauri-on-Windows still spawns the bridge as a single
  process, so cross-process contention is rare today, but the
  pytest suite imports this module on every platform — the Windows
  branch keeps that import (and the test surface above it) working.

Sidecar lives next to the data file so file replacement during the
write phase doesn't disturb the lock. Lock blocks until acquired;
released on context exit.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    # msvcrt is a Windows-only stdlib module; importing on POSIX would
    # fail at module load. The branch here mirrors how stdlib
    # ``logging.handlers`` guards Windows-specific imports.
    import msvcrt
else:
    import fcntl


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive cross-process lock for ``path``.

    Lock lives on a sidecar ``<path>.lock`` next to the data file so
    the data file can be renamed/replaced while the lock is held.
    Blocks until acquired. Released on context exit.

    The sidecar is created as needed and never deleted — it's empty
    metadata. Cleaning it up between operations would race the lock
    acquisition itself.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh: IO[bytes] = open(lock_path, "ab")
    try:
        if _IS_WINDOWS:
            # msvcrt.locking blocks until the lock is acquired when
            # called with LK_LOCK; the byte region is one byte at the
            # current file position, which is fine for advisory use.
            # We always lock the same byte (offset 0, size 1).
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
