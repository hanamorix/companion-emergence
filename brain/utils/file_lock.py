"""Cross-process exclusive file locks for shared JSONL data files.

Audit 2026-05-07 P2-2: ``soul_candidates.jsonl`` is appended to by the
ingest pipeline AND rewritten by the soul-review path. Without
coordination, a candidate queued between review's read and write is
silently dropped when review replaces the file. ``soul_candidate_lock``
gates both paths through the same OS-level exclusive lock so the
read-modify-rewrite is atomic against concurrent appends.

Implementation: ``fcntl.flock`` on a sidecar ``<file>.lock`` (separate
from the data file so file replacement during the write phase doesn't
disturb the lock). Lock blocks until acquired; on macOS / Linux this
is the canonical advisory lock. Windows would need ``msvcrt.locking``
— not implemented because the bridge concurrency this guards against
is bridged-Python-on-Hana's-mac-shaped today; if/when Windows runs
the supervisor, this module gets the platform branch.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive POSIX advisory lock for ``path``.

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
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
