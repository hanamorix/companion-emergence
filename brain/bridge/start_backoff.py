"""The start back-off for service-run bridges (#458, D26).

A bridge run by an OS service manager that fails every time it starts (a corrupt database) exits non-zero at every
start, so the manager restarts it at once. This module is the bridge's own schedule for that case, the same ladder the
app's watchdog uses for the bridges it spawns (``app/src/hooks/useBridgeWatchdog.ts``): 30 s, 60 s, 300 s, then 600 s
for every later start, back to the start after success (D7). Only a service-origin ``nell supervisor run`` uses it; an
app-spawned or terminal bridge is the app watchdog's or the person's.

The record, ``bridge_start_backoff.json`` beside ``bridge.json`` (through ``json_io``): how many starts in this series
have not ended cleanly, and when the last one began. The record exists from a start until the run ends on a recorded
stop (the clean "stopped" end, any hard exit but a crash) or an explicit start deletes it (``nell supervisor start``,
the app, the service start and install paths; that also ends a wait in progress). So every way a run can end badly
leaves it, including the ones that run no code of ours (SIGKILL, out of memory, power loss), and the next service start
waits: until ``at`` plus the step for the count (a failing bridge is never started twice within a step).

After ``HEALTHY_RESET_S`` of healthy running (D7's success) the ladder starts over but the record STAYS, marked
``after_healthy`` with a count of 1 (the healthy run is the series' first start): the app waits 30 s after any crash,
even after a long healthy run, so the start that follows such a crash waits the whole first step from its own start (the
recorded time says nothing about when the crash happened) and counts as the second start of the series.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brain.bridge.shutdown import POLL_S
from brain.health.json_io import read_json, write_json

logger = logging.getLogger(__name__)

FILENAME = "bridge_start_backoff.json"

# The owner's Q4 ladder (2026-10-19) and D7's success, mirrored from the app watchdog's CRASH_RESTART_DELAYS_MS and
# HEALTHY_RESET_MS (seconds here); tests/bridge/test_start_backoff.py reads that file and fails when they differ.
DELAYS_S = (30, 60, 300, 600)
HEALTHY_RESET_S = 600


@dataclass(frozen=True)
class Record:
    """``unproven_starts`` starts since the last success, the last one at wall-clock epoch second ``at``."""

    unproven_starts: int = 0
    at: float = 0.0
    after_healthy: bool = False  # the run stayed healthy for HEALTHY_RESET_S; `at` is then no anchor for the wait


def _now() -> float:
    return time.time()


def _monotonic() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _path(persona_dir: Path) -> Path:
    return Path(persona_dir) / FILENAME


def _valid_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def read(persona_dir: Path) -> Record:
    """The record; missing, unreadable or invalid content reads as no failures (json_io logs the evidence)."""
    raw = read_json(_path(persona_dir), default=dict)
    if not isinstance(raw, dict):
        return Record()
    count, at = raw.get("unproven_starts"), raw.get("at")
    if not _valid_count(count) or isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at):
        return Record()
    return Record(count, float(at), raw.get("after_healthy") is True)


def step_s(unproven_starts: int) -> int:
    """The wait before the start that follows ``unproven_starts`` unproven ones: 0, then the ladder, then its last step."""
    if unproven_starts <= 0:
        return 0
    return DELAYS_S[min(unproven_starts, len(DELAYS_S)) - 1]


def remaining_wait_s(record: Record, now: float) -> float:
    """How much of the step is left at ``now``: 0 once it has passed, never more than the step (a wall clock that was
    set backwards cannot make a wait longer than the ladder says). After a healthy run the whole step: the recorded time
    is the end of the healthy stretch's first start, no anchor for a crash that came later."""
    step = step_s(record.unproven_starts)
    if record.after_healthy:
        return float(step)
    return min(float(step), max(0.0, record.at + step - now))


def note_start(persona_dir: Path) -> None:
    """Count this start as unproven. A failed write is logged and the start goes ahead: a start is never blocked by
    its own bookkeeping (it is then simply not backed off, as before this record existed)."""
    try:
        rec = read(persona_dir)
        write_json(_path(persona_dir), {"unproven_starts": rec.unproven_starts + 1, "at": _now()}, mode=0o600)
    except Exception:  # noqa: BLE001
        logger.warning("start back-off: could not record this start; it will not be backed off", exc_info=True)


def mark_healthy(persona_dir: Path) -> None:
    """D7's success (healthy for ``HEALTHY_RESET_S``): the ladder starts over, but the record stays: a service start
    that finds it knows the run did not end on a recorded stop (a crash, a kill) and waits the first step, as the app
    does after any crash. The healthy run counts as the series' first start (count 1)."""
    try:
        write_json(
            _path(persona_dir), {"unproven_starts": 1, "at": _now(), "after_healthy": True}, mode=0o600
        )
    except Exception:  # noqa: BLE001
        logger.warning("start back-off: could not mark the run healthy", exc_info=True)


def reset(persona_dir: Path) -> None:
    """A recorded stop or an explicit start: the next start does not wait."""
    try:
        _path(persona_dir).unlink(missing_ok=True)
    except OSError:
        logger.warning("start back-off: could not delete the record", exc_info=True)


def wait_before_start(persona_dir: Path, *, interrupted: Callable[[], bool]) -> bool:
    """Wait out the step for the recorded failures; True when the start may go ahead, False when ``interrupted()``
    (the respawn guard) says not to start.

    The wait is a countdown, not a comparison with the wall clock: when the record is first seen (or seen changed),
    the remaining time is worked out once from the wall clock (at most one step) and turned into a MONOTONIC deadline.
    A wall clock that is stepped afterwards, or a record stamped ahead of a clock that was later corrected, can neither
    extend nor cut the wait. Each poll (every ``shutdown.POLL_S``) re-reads the record: an explicit start that deleted
    it ends the wait at once, and one that rewrote it starts a new countdown. Holds no lock and writes nothing."""
    seen: Record | None = None
    deadline = 0.0
    announced = False
    while True:
        record = read(persona_dir)
        if record != seen:
            seen = record
            remaining = remaining_wait_s(record, _now())
            if remaining <= 0:
                return True
            deadline = _monotonic() + remaining
        left = deadline - _monotonic()
        if left <= 0:
            return True
        if interrupted():
            return False
        if not announced:
            announced = True
            print(
                f"waiting {int(left)} s before starting: {record.unproven_starts} earlier start(s) did not stay up",
                file=sys.stderr,
                flush=True,
            )
            logger.info("start back-off: waiting %.0f s (%d unproven starts)", left, record.unproven_starts)
        _sleep(min(POLL_S, left))


class SuccessWatch:
    """Marks the run healthy (``mark_healthy``) once the server has been started for ``HEALTHY_RESET_S`` (D7: healthy,
    then 600 s).

    ``poll_once`` is driven by its own thread (``start``), at ``shutdown.POLL_S``, until it has reset or ``stop`` is
    called at the end of the run. A server that never started never marks."""

    def __init__(self, persona_dir: Path, server: Any, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.persona_dir = Path(persona_dir)
        self.server = server
        self._clock = clock
        self._started_at: float | None = None
        self._done = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll_once(self) -> None:
        if self._done:
            return
        if self._started_at is None:
            if not getattr(self.server, "started", False):
                return
            self._started_at = self._clock()
            return
        if self._clock() - self._started_at >= HEALTHY_RESET_S:
            mark_healthy(self.persona_dir)
            self._done = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="bridge-start-backoff-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5 * POLL_S + 1.0)

    def _loop(self) -> None:
        while not self._stop.is_set() and not self._done:
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - never dies; the next poll retries
                logger.warning("start back-off watch poll failed", exc_info=True)
            self._stop.wait(POLL_S)
