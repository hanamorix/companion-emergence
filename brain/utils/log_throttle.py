"""Warn about a fail-soft failure without flooding the log (#367).

Readers that substitute a safe default on error (a recovery flag, a presence
signal, a pending-write record) run on every poll or tick. Logging each failure
at WARNING floods the runtime log; logging at DEBUG hides it, since the runtime
handler is INFO. ``warn_throttled`` warns with the active traceback at most
once per key per ``INTERVAL_S`` and logs at DEBUG in between, so a persistent
failure is reported a few times a day and a later re-break warns again.
"""

from __future__ import annotations

import logging
import threading
import time

INTERVAL_S = 6 * 3600.0

_last: dict[str, float] = {}
_counts: dict[str, int] = {}
_lock = threading.Lock()


def _due(key: str, interval_s: float) -> bool:
    """True (and restart the interval) if ``key`` has not logged loudly within ``interval_s``.
    Caller holds ``_lock``."""
    now = time.monotonic()
    last = _last.get(key)
    if last is None or now - last >= interval_s:
        _last[key] = now
        return True
    return False


def warn_throttled(
    logger: logging.Logger, key: str, msg: str, *args: object, exc_info: bool = True
) -> None:
    """Log ``msg % args`` (with the active exception unless ``exc_info=False``):
    WARNING if ``key`` has not warned in the last ``INTERVAL_S``, DEBUG otherwise."""
    with _lock:
        warn = _due(key, INTERVAL_S)
    (logger.warning if warn else logger.debug)(msg, *args, exc_info=exc_info)


def count_throttled(
    logger: logging.Logger,
    key: str,
    level: int,
    msg: str,
    *args: object,
    interval_s: float = INTERVAL_S,
) -> int:
    """Count one firing of a cap or gate under ``key`` (#368) and log ``msg % args``
    with the running count: at ``level`` if ``key`` has not logged at that level in
    the last ``interval_s``, DEBUG otherwise. Returns the count since process start.

    For events that are expected, not failures, so there is no traceback. Counts are
    in-process only and reset on restart; firings after the last loud line show
    only at DEBUG until the next interval. Surfacing the counts in /health is a
    follow-up (#368)."""
    with _lock:
        n = _counts[key] = _counts.get(key, 0) + 1
        loud = _due(key, interval_s)
    logger.log(level if loud else logging.DEBUG, msg + " (%d since start)", *args, n)
    return n
