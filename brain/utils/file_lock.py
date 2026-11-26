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
import time
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
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Acquire an exclusive cross-process lock for ``path``.

    Lock lives on a sidecar ``<path>.lock`` next to the data file so
    the data file can be renamed/replaced while the lock is held.
    Released on context exit.

    ``blocking`` (default True): block until acquired, then yield True —
    the historical behaviour; existing callers ``with file_lock(p):``
    ignore the yielded value and keep working unchanged.

    ``blocking=False``: attempt the lock once. Yields True if acquired
    (and holds it for the ``with`` body, releasing on exit), or False if
    another holder has it (the body runs WITHOUT the lock — callers must
    check the yielded bool and skip their critical section on False).
    Used by the consolidation gate to skip a run when a concurrent gate
    is already consolidating.

    The sidecar is created as needed and never deleted — it's empty
    metadata. Cleaning it up between operations would race the lock
    acquisition itself.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh: IO[bytes] = open(lock_path, "ab")
    try:
        if _IS_WINDOWS:
            acquired = _windows_lock_acquire(fh, blocking=blocking)
            try:
                yield acquired
            finally:
                if acquired:
                    _windows_lock_release(fh)
        else:
            acquired = True
            if blocking:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            else:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError):
                    acquired = False
            try:
                yield acquired
            finally:
                if acquired:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


class LockTimeoutError(TimeoutError):
    """``locked_within`` gave up: another holder kept the lock for the whole wait. The caller's body never ran."""


@contextmanager
def locked_within(path: Path, seconds: float, *, poll_seconds: float | None = None) -> Iterator[None]:
    """Hold the ``path`` lock for the ``with`` body, waiting at most ``seconds`` to get it.

    ``file_lock(blocking=True)`` has no timeout, so a wedged holder would hang the caller for good (a bridge boot, a
    settings request). This polls the NON-blocking form (the same on POSIX and Windows) until the deadline. On the
    deadline it raises :class:`LockTimeoutError` (a ``TimeoutError``) and the body is NEVER entered: the non-blocking
    ``file_lock`` yields False and still runs its own body without the lock, so the flag is checked here and the
    caller's body runs only when it is True. An exception raised by the body propagates exactly once (the body is not
    retried) and releases the lock. The lock is not re-entrant: nesting the same ``path`` times out.
    """
    from brain import dev_constants

    step = dev_constants.SETTINGS_FILE_LOCK_POLL_S if poll_seconds is None else poll_seconds
    deadline = time.monotonic() + seconds
    while True:
        with file_lock(path, blocking=False) as acquired:
            if acquired:
                yield
                return
        if time.monotonic() >= deadline:
            raise LockTimeoutError(f"could not lock {path} within {seconds:g} s")
        time.sleep(step)


# Windows: msvcrt.locking with LK_LOCK only retries every second for
# ~10s then raises. Under high contention (the concurrent-appenders
# test runs 8×5 = 40 ops fighting for the same lock) that 10s window
# can starve. Wrap with our own non-blocking attempt + sleep loop so
# we never give up — matches fcntl.flock's "block until acquired"
# semantics on POSIX.
_WIN_LOCK_INITIAL_SLEEP = 0.005
_WIN_LOCK_MAX_SLEEP = 0.2


def _windows_lock_acquire(fh: IO[bytes], *, blocking: bool = True) -> bool:
    """Acquire the msvcrt lock. Returns True on acquire.

    blocking=True: retry until acquired (returns True). blocking=False: a single
    LK_NBLCK attempt — returns False immediately if another holder has it (the
    non-blocking mode used by the consolidation gate's skip-if-contended lock).
    """
    fh.seek(0)
    sleep_s = _WIN_LOCK_INITIAL_SLEEP
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            if not blocking:
                return False
            time.sleep(sleep_s)
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
