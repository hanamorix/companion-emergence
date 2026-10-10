"""Peak resident-memory growth of one call (name-recall fix S220, S222).

With the ONNX arena and memory patterns off (S222) the reranker returns its memory after
every call, so "RSS after minus RSS before" reads about 0 whatever the call needed (RAM grid
arm B: that difference stayed within +/-40 MB while the peak grew by up to 1.6 GB). What the
model needs is the PEAK during the call, so:

    begin()                -> VmRSS before the call, after resetting the process peak (VmHWM)
    growth_since(before)   -> max(0, VmHWM now - before)

Linux only (`/proc/self/clear_refs` = 5 resets the peak; `VmHWM` is the peak). Everywhere
else, and whenever the reset or a read fails, the reading is unavailable (`None`): the callers
then neither record a sample nor hold turns back (S206 posture: an unreadable measurement is
never read as unlimited memory, and a RAM bound without a measurement does not exist). A
failure at call time latches the reader as unavailable for the process and is warned once.
No macOS/Windows reader exists: there is no per-call resettable peak there (open question).

The reset is process-wide, so the reading includes anything else the process allocates during
the call; the callers hold the reranker's lock around the call so reranker calls do not
overlap, and a reading inflated by another thread over-states the need (never under-states).
"""

from __future__ import annotations

import logging
import os
import sys
import threading

log = logging.getLogger(__name__)

# Seams for tests (the real paths on Linux).
_PROC_STATUS = "/proc/self/status"
_CLEAR_REFS = "/proc/self/clear_refs"
_is_linux: bool = sys.platform.startswith("linux")

_lock = threading.Lock()
_latched_off = False
_warned = False


def _warn_once(reason: str) -> None:
    global _warned
    with _lock:
        first = not _warned
        _warned = True
    (log.warning if first else log.debug)("peak_rss: %s; RAM growth cannot be measured", reason)


def _latch_off(reason: str) -> None:
    global _latched_off
    with _lock:
        _latched_off = True
    _warn_once(reason)


def _read_kb(field: str) -> float | None:
    """A `kB` field of /proc/self/status in bytes, or `None`."""
    try:
        with open(_PROC_STATUS, encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return float(line.split()[1]) * 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def available() -> bool:
    """Whether a peak reading can be taken here: Linux, the status file readable, the
    reset file writable, and no earlier failure in this process. Side-effect free (no reset)."""
    if _latched_off or not _is_linux:
        return False
    return os.access(_PROC_STATUS, os.R_OK) and os.access(_CLEAR_REFS, os.W_OK)


def begin() -> float | None:
    """Reset the process's peak and return its current RSS in bytes, or `None` when
    unavailable (and latch that for the process when a reset or read failed)."""
    if not available():
        return None
    try:
        with open(_CLEAR_REFS, "w", encoding="ascii") as handle:
            handle.write("5")
    except OSError:
        _latch_off("the peak counter could not be reset")
        return None
    rss = _read_kb("VmRSS")
    if rss is None:
        _latch_off("VmRSS could not be read")
    return rss


def growth_since(before: float | None) -> float | None:
    """The peak resident growth since `begin()` returned `before`, in bytes (never
    negative), or `None` when `before` is `None` or the peak cannot be read."""
    if before is None:
        return None
    peak = _read_kb("VmHWM")
    if peak is None:
        _latch_off("VmHWM could not be read")
        return None
    return max(0.0, peak - before)


def _reset_for_tests() -> None:
    global _latched_off, _warned
    with _lock:
        _latched_off = False
        _warned = False
