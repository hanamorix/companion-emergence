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
_lock = threading.Lock()


def warn_throttled(
    logger: logging.Logger, key: str, msg: str, *args: object, exc_info: bool = True
) -> None:
    """Log ``msg % args`` (with the active exception unless ``exc_info=False``):
    WARNING if ``key`` has not warned in the last ``INTERVAL_S``, DEBUG otherwise."""
    now = time.monotonic()
    with _lock:
        last = _last.get(key)
        warn = last is None or now - last >= INTERVAL_S
        if warn:
            _last[key] = now
    (logger.warning if warn else logger.debug)(msg, *args, exc_info=exc_info)
