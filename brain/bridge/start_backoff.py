"""The start back-off for service-run bridges (#458, D26).

A bridge run by an OS service manager that fails every time it starts (a corrupt database) exits non-zero at every
start, so the manager restarts it at once. This module is the bridge's own schedule for that case, the same ladder the
app's watchdog uses for the bridges it spawns (``app/src/hooks/useBridgeWatchdog.ts``): 30 s, 60 s, 300 s, then 600 s
for every later start, back to the start after success (D7). Only a service-origin ``nell supervisor run`` uses it; an
app-spawned or terminal bridge is the app watchdog's or the person's.

The record, ``bridge_start_backoff.json`` beside ``bridge.json`` (through ``json_io``): how many starts since the last
success have not been proven, and when the last one began. A start counts as unproven until the bridge has run healthy
for ``HEALTHY_RESET_S`` or has ended on a recorded stop, so every way a start can fail is counted, including the ones
that run no code of ours (SIGKILL, out of memory, power loss). The next start waits until ``at`` plus the step for the
count: a failing bridge is never started twice within a step. Explicit starts (``nell supervisor start``, the app,
the service start and install paths) delete the record, which also ends a wait in progress.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _now() -> float:
    return time.time()


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
    if not _valid_count(count) or isinstance(at, bool) or not isinstance(at, (int, float)):
        return Record()
    return Record(count, float(at))


def step_s(unproven_starts: int) -> int:
    """The wait before the start that follows ``unproven_starts`` unproven ones: 0, then the ladder, then its last step."""
    if unproven_starts <= 0:
        return 0
    return DELAYS_S[min(unproven_starts, len(DELAYS_S)) - 1]


def remaining_wait_s(record: Record, now: float) -> float:
    """How much of the step is left at ``now``: 0 once it has passed, never more than the step (a wall clock that was
    set backwards cannot make a wait longer than the ladder says)."""
    step = step_s(record.unproven_starts)
    return min(float(step), max(0.0, record.at + step - now))


def note_start(persona_dir: Path) -> None:
    """Count this start as unproven. A failed write is logged and the start goes ahead: a start is never blocked by
    its own bookkeeping (it is then simply not backed off, as before this record existed)."""
    try:
        rec = read(persona_dir)
        write_json(_path(persona_dir), {"unproven_starts": rec.unproven_starts + 1, "at": _now()}, mode=0o600)
    except Exception:  # noqa: BLE001
        logger.warning("start back-off: could not record this start; it will not be backed off", exc_info=True)


def reset(persona_dir: Path) -> None:
    """Success, a recorded stop or an explicit start: the next start does not wait."""
    try:
        _path(persona_dir).unlink(missing_ok=True)
    except OSError:
        logger.warning("start back-off: could not delete the record", exc_info=True)


def wait_before_start(persona_dir: Path, *, interrupted: Callable[[], bool]) -> bool:
    """Wait out the step for the recorded failures; True when the start may go ahead, False when ``interrupted()``
    (the respawn guard) says not to start. Each poll (every ``shutdown.POLL_S``) re-reads the record, so an explicit
    start that deleted it ends the wait at once. Holds no lock and writes nothing."""
    from brain.bridge.shutdown import POLL_S

    announced = False
    while True:
        record = read(persona_dir)
        remaining = remaining_wait_s(record, _now())
        if remaining <= 0:
            return True
        if interrupted():
            return False
        if not announced:
            announced = True
            import sys

            print(
                f"waiting {int(remaining)} s before starting: the bridge has not stayed up through "
                f"{record.unproven_starts} start(s) in a row",
                file=sys.stderr,
                flush=True,
            )
            logger.info("start back-off: waiting %.0f s (%d unproven starts)", remaining, record.unproven_starts)
        _sleep(min(POLL_S, remaining))


class SuccessWatch:
    """Deletes the record once the server has been started for ``HEALTHY_RESET_S`` (D7: healthy, then 600 s).

    ``poll_once`` is driven by its own thread (``start``), at ``shutdown.POLL_S``, until it has reset or ``stop`` is
    called at the end of the run. A server that never started never resets."""

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
            reset(self.persona_dir)
            self._done = True

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="bridge-start-backoff-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        from brain.bridge.shutdown import POLL_S

        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5 * POLL_S + 1.0)

    def _loop(self) -> None:
        from brain.bridge.shutdown import POLL_S

        while not self._stop.is_set() and not self._done:
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - never dies; the next poll retries
                logger.warning("start back-off watch poll failed", exc_info=True)
            self._stop.wait(POLL_S)
