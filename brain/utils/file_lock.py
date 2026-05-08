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
            _windows_lock_acquire(fh)
            try:
                yield
            finally:
                _windows_lock_release(fh)
        else:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


# Windows: msvcrt.locking with LK_LOCK only retries every second for
# ~10s then raises. Under high contention (the concurrent-appenders
# test runs 8×5 = 40 ops fighting for the same lock) that 10s window
# can starve. Wrap with our own non-blocking attempt + sleep loop so
# we never give up — matches fcntl.flock's "block until acquired"
# semantics on POSIX.
import time as _time  # noqa: E402

_WIN_LOCK_INITIAL_SLEEP = 0.005
_WIN_LOCK_MAX_SLEEP = 0.2


def _windows_lock_acquire(fh: IO[bytes]) -> None:
    fh.seek(0)
    sleep_s = _WIN_LOCK_INITIAL_SLEEP
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            _time.sleep(sleep_s)
            if sleep_s < _WIN_LOCK_MAX_SLEEP:
                sleep_s = min(sleep_s * 2, _WIN_LOCK_MAX_SLEEP)


def _windows_lock_release(fh: IO[bytes]) -> None:
    fh.seek(0)
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        # Releasing an already-released lock isn't fatal; defensive
        # because the file handle close that follows will release any
        # remaining locks anyway.
        pass
