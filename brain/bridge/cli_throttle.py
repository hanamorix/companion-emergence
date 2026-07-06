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

from brain import tunables

log = logging.getLogger(__name__)

_IDLE_SECONDS = tunables.register("throttle.background_min_idle_seconds", 300.0)          # match emotion_backfill _ACTIVE_CHAT_IDLE_MINUTES (5 min)
_MAX_CONCURRENT_BACKGROUND = tunables.register("throttle.max_concurrent_background", 1)


def _idle_seconds() -> float:
    return tunables.get_tunable("throttle.background_min_idle_seconds", _IDLE_SECONDS)


def _max_concurrent_background() -> int:
    return tunables.get_tunable("throttle.max_concurrent_background", _MAX_CONCURRENT_BACKGROUND)


_lock = threading.Lock()
_last_interactive_monotonic: float = -1e9
_inflight_background = 0


class ThrottleDeferred(RuntimeError):  # noqa: N818 — a control signal, not an error
    """A background runner raises this when the throttle denied its slot.

    The tick treats it as a quiet no-op (retry next tick) — NOT a failure: no
    error log, no traceback, no cooldown penalty, no budget spend. Distinct from
    a generic Exception so the tick can tell a transient yield apart from a real
    making/compose error.
    """


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
        return (t - _last_interactive_monotonic) < _idle_seconds()


def slot_available(*, now: float | None = None, min_idle: float | None = None) -> bool:
    """Read-only peek: True if a background slot could be acquired right now —
    chat idle long enough AND the concurrency cap not full. Does NOT touch the
    semaphore, so it is safe as a pre-flight gate before a tick commits budget /
    cooldown. Fails OPEN (True) on internal error, matching acquire_background —
    a defer must never be reported when the throttle itself malfunctions."""
    try:
        with _lock:
            t = time.monotonic() if now is None else now
            idle = _idle_seconds() if min_idle is None else min_idle
            if (t - _last_interactive_monotonic) < idle:
                return False
            return _inflight_background < _max_concurrent_background()
    except Exception:  # noqa: BLE001 — fail open
        log.warning("cli_throttle.slot_available failed; reporting available (fail-open)", exc_info=True)
        return True


def acquire_background(*, now: float | None = None, min_idle: float | None = None) -> bool:
    """True → caller may make its background CLI call (and MUST call
    release_background() after). False → defer to next tick.

    ``min_idle`` overrides the chat-idle window (default ``_IDLE_SECONDS``, tuned
    for the cadence engines). Turn-coupled callers (the pass-2 queue worker) pass
    a shorter value so they drain soon after a turn finishes rather than waiting
    the full cadence window — while still respecting the concurrency cap.
    """
    global _inflight_background
    try:
        t = time.monotonic() if now is None else now
        idle = _idle_seconds() if min_idle is None else min_idle
        with _lock:
            if (t - _last_interactive_monotonic) < idle:
                return False
            if _inflight_background >= _max_concurrent_background():
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
