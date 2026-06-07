"""Process-global CLI throttle. Interactive chat has absolute priority and
never waits; background CLI consumers (dreams, reflex, research, soul review,
initiate, voice reflection, the reflection passes, backfills) yield while a
chat turn is in-flight / recently active, and respect a small concurrency cap.

Interactive code is never throttled — it simply does not call acquire_background.

Fails open: on any internal error, background is allowed and the error is logged
at WARNING every time (background calls are cadence-driven and infrequent, so
repeated warnings signal a genuine lock malfunction rather than spam). Closes
deferred item 26."""
from __future__ import annotations

import contextlib
import logging
import threading
import time

log = logging.getLogger(__name__)

_IDLE_SECONDS = 300.0          # match emotion_backfill _ACTIVE_CHAT_IDLE_MINUTES (5 min)
_MAX_CONCURRENT_BACKGROUND = 1

_lock = threading.Lock()
_last_interactive_monotonic: float = -1e9
_inflight_background = 0


def reset() -> None:  # test helper
    global _last_interactive_monotonic, _inflight_background
    with _lock:
        _last_interactive_monotonic = -1e9
        _inflight_background = 0


def mark_interactive_active(at: float | None = None) -> None:
    global _last_interactive_monotonic
    with _lock:
        _last_interactive_monotonic = time.monotonic() if at is None else at


def should_yield(*, now: float | None = None) -> bool:
    """Read-only peek: True if a chat turn is recently active (same idle-window
    check as acquire_background).  Does NOT touch the semaphore — safe to call
    inside a held background_slot to decide whether to break mid-batch."""
    with _lock:
        t = time.monotonic() if now is None else now
        return (t - _last_interactive_monotonic) < _IDLE_SECONDS


def acquire_background(*, now: float | None = None) -> bool:
    """True → caller may make its background CLI call (and MUST call
    release_background() after). False → defer to next tick."""
    global _inflight_background
    try:
        t = time.monotonic() if now is None else now
        with _lock:
            if (t - _last_interactive_monotonic) < _IDLE_SECONDS:
                return False
            if _inflight_background >= _MAX_CONCURRENT_BACKGROUND:
                return False
            _inflight_background += 1
            return True
    except Exception:  # noqa: BLE001 — fail open
        log.warning("cli_throttle.acquire_background failed; allowing (fail-open)", exc_info=True)
        return True


def release_background() -> None:
    global _inflight_background
    with _lock:
        _inflight_background = max(0, _inflight_background - 1)


@contextlib.contextmanager
def background_slot(*, now: float | None = None):
    """Context manager wrapping acquire/release. Yields True if the slot was
    acquired (caller should do its CLI work), False if it should defer.
    Always releases on exit when acquired."""
    acquired = acquire_background(now=now)
    if not acquired:
        yield False
        return
    try:
        yield True
    finally:
        release_background()
