"""Peak resident-memory growth of one call (name-recall fix S220, S222).

With the ONNX arena and memory patterns off (S222) the reranker returns its memory after
every call, so "RSS after minus RSS before" reads about 0 whatever the call needed (RAM grid
arm B: that difference stayed within +/-40 MB while the peak grew by up to 1.6 GB). What the
model needs is the PEAK during the call, so:

    begin()                -> VmRSS before the call, after resetting the process peak (VmHWM)
    growth_since(before)   -> max(0, VmHWM now - before)

Linux keeps this exact reading (`/proc/self/clear_refs` = 5 resets the peak; `VmHWM` is the
peak). On macOS and Windows there is no per-call resettable peak, so (S248) a background thread
SAMPLES the process RSS while the call runs and the peak is the largest sample, used RAW (no
correction factor: the fit's own average over-prediction is the cover the order names). The
sampler can miss a peak shorter than one sampling period (`_SAMPLE_INTERVAL_S`). A test can force
the sampler on Linux (`_force_sampler`). Whenever the reset or a read fails, the reading is
unavailable (`None`): the callers then neither record a sample nor hold turns back (S206
posture: an unreadable measurement is never read as unlimited memory, and a RAM bound without a
measurement does not exist). A failure at call time latches the reader as unavailable for the
process and is warned once.

Sampler containment: ONE sampler at a time; the sampler thread only reads the RSS and keeps a
local maximum and never takes `_lock`; `stop()` joins outside `_lock` with a bounded timeout;
`growth_since` returns `None` when no sampler is active (a second `begin()` stopped it:
overlapping callers of different providers get no sample, never an exception); a thread that
cannot start latches the reader off; `cancel()` (a call that raised) stops it.

The reset is process-wide, so the reading includes anything else the process allocates during
the call; the callers hold the reranker's lock around the call so reranker calls do not
overlap. Another thread's allocation during the call over-states the need; another thread FREEING
memory right after the reset (the reset restarts the peak at the current RSS, which still holds
it) can make the reading too low.
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

# S248 seams. `_disabled`: the suite's switch (fake rerankers neither touch /proc nor start a
# sampler). `_force_sampler`: use the RSS sampler on Linux too (the non-Linux path under test).
_disabled: bool = False
_force_sampler: bool = False

# S248: the sampling period. A sampler cannot see a peak shorter than this; it is a parameter of
# any sampler, not a relevance threshold. `_JOIN_TIMEOUT_S` only bounds a join that cannot hang
# in practice (the thread waits on an Event the stop sets).
_SAMPLE_INTERVAL_S = 0.005
_JOIN_TIMEOUT_S = 1.0

_lock = threading.Lock()
_latched_off = False
_warned = False
_sampler: _Sampler | None = None


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


def _uses_sampler() -> bool:
    return _force_sampler or not _is_linux


_process: object | None = None


def _rss_now() -> float | None:
    """The process's resident set in bytes (psutil, the dependency the headroom reader uses), or
    `None` when it cannot be read."""
    global _process
    try:
        if _process is None:
            import psutil

            _process = psutil.Process()
        return float(_process.memory_info().rss)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a diagnostic must never break a recall
        return None


class _Sampler:
    """A daemon thread that reads the RSS every `_SAMPLE_INTERVAL_S` until stopped and keeps the
    largest reading. It never takes `peak_rss._lock` and never raises."""

    def __init__(self, baseline: float) -> None:
        self._stop = threading.Event()
        self.largest = baseline
        self.thread = threading.Thread(target=self._run, name="peak-rss-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(_SAMPLE_INTERVAL_S):
            rss = _rss_now()
            if rss is not None and rss > self.largest:
                self.largest = rss

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> float:
        """Stop and join (bounded); the largest reading seen."""
        self._stop.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(_JOIN_TIMEOUT_S)
        return self.largest


def _take_sampler() -> _Sampler | None:
    """Detach the active sampler (under the lock); the caller stops it OUTSIDE the lock."""
    global _sampler
    with _lock:
        taken, _sampler = _sampler, None
    return taken


def cancel() -> None:
    """Stop an active sampler without a reading (the measured call raised). Idempotent; a no-op on
    the Linux path."""
    taken = _take_sampler()
    if taken is not None:
        taken.stop()


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
    reset file writable, and no earlier failure in this process. Off Linux (or when the sampler is
    forced), whether the RSS can be read. Side-effect free (no reset, no thread)."""
    if _disabled or _latched_off:
        return False
    if _uses_sampler():
        return _rss_now() is not None
    if not _is_linux:
        return False
    usable = os.access(_PROC_STATUS, os.R_OK) and os.access(_CLEAR_REFS, os.W_OK)
    if not usable:
        _warn_once("the kernel's peak counter is not accessible here")
    return usable


def begin() -> float | None:
    """Reset the process's peak and return its current RSS in bytes, or `None` when
    unavailable (and latch that for the process when a reset or read failed). With the sampler
    (S248) it returns the RSS now and starts the sampling thread, stopping an earlier one."""
    if not available():
        return None
    if _uses_sampler():
        return _begin_sampling()
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


def _begin_sampling() -> float | None:
    global _sampler
    baseline = _rss_now()
    if baseline is None:
        _latch_off("the RSS could not be read")
        return None
    sampler = _Sampler(baseline)
    try:
        sampler.start()
    except RuntimeError:  # a thread cannot be started (no memory): the reading is unavailable
        _latch_off("the RSS sampling thread could not be started")
        return None
    with _lock:
        previous, _sampler = _sampler, sampler
    if previous is not None:
        previous.stop()
    return baseline


def growth_since(before: float | None) -> float | None:
    """The peak resident growth since `begin()` returned `before`, in bytes (never
    negative), or `None` when `before` is `None` or the peak cannot be read. With the sampler
    (S248): the largest RSS sampled during the call, plus one last reading, minus `before`;
    `None` when no sampler is active."""
    if _uses_sampler():
        taken = _take_sampler()
        if taken is None:
            return None
        largest = taken.stop()
        if before is None:
            return None
        last = _rss_now()
        if last is None:
            _latch_off("the RSS could not be read")
            return None
        return max(0.0, max(largest, last) - before)
    if before is None:
        return None
    peak = _read_kb("VmHWM")
    if peak is None:
        _latch_off("VmHWM could not be read")
        return None
    return max(0.0, peak - before)


def _reset_for_tests() -> None:
    global _latched_off, _warned, _process
    cancel()
    with _lock:
        _latched_off = False
        _warned = False
    _process = None
