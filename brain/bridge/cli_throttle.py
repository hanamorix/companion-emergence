"""Process-global CLI throttle. Interactive chat has absolute priority and
never waits; background CLI consumers (dreams, reflex, research, soul review,
initiate, voice reflection, the reflection passes, backfills) yield while a
chat turn is in-flight / recently active, and respect a small concurrency cap.

Interactive code is never throttled — it simply does not call acquire_background.

The idle check (``is_chat_idle``) is the SOLE reader of the one registered
``chat.idle_lull_seconds`` tunable (C4(b), S40/S72) — every other background/
cadence caller in this module routes through it, so a single lull value
governs all of them. It fails CLOSED (reports "not idle") on any internal
error and logs the failure at a rate-limited pace (S63) — background work is
paused, never silently unblocked, while the check itself is malfunctioning.

The slot-cap half of this module (``_max_concurrent_background`` /
``_inflight_background``) keeps its own FAIL-OPEN posture on internal error —
it is not the idle check, and a malfunction there must not itself become a
second reason for background work to starve.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time

from brain import tunables

log = logging.getLogger(__name__)

_MAX_CONCURRENT_BACKGROUND = tunables.register("throttle.max_concurrent_background", 1)


def _max_concurrent_background() -> int:
    return tunables.get_tunable("throttle.max_concurrent_background", _MAX_CONCURRENT_BACKGROUND)


def _lull_seconds() -> float:
    """The lull duration — the ONLY place the ``chat.idle_lull_seconds``
    tunable key is read (C4(b)). Called only from ``is_chat_idle``."""
    return tunables.get_tunable("chat.idle_lull_seconds", tunables.CHAT_IDLE_LULL_SECONDS)


_lock = threading.Lock()
# S35: a fresh process has never seen a message, so it counts as idle from
# its very first instant — -inf makes `now - _last_message_mono` always
# exceed any finite lull.
_last_message_mono: float = float("-inf")
_inflight_replies: int = 0
_inflight_background = 0

# Rate-limited error logging state for is_chat_idle (S63; r2 m1 fix: bounded
# volume under a persistent fault). Guarded by _lock alongside the state above.
_err_last_key: str | None = None
_err_last_logged_mono: float = float("-inf")
_err_suppressed_count: int = 0
_ERROR_LOG_INTERVAL_S = 600.0  # 10 min


class ThrottleDeferred(RuntimeError):  # noqa: N818 — a control signal, not an error
    """A background runner raises this when the throttle denied its slot.

    The tick treats it as a quiet no-op (retry next tick) — NOT a failure: no
    error log, no traceback, no cooldown penalty, no budget spend. Distinct from
    a generic Exception so the tick can tell a transient yield apart from a real
    making/compose error.
    """


def reset() -> None:  # test helper
    global _last_message_mono, _inflight_background, _inflight_replies
    global _err_last_key, _err_last_logged_mono, _err_suppressed_count
    with _lock:
        _last_message_mono = float("-inf")
        _inflight_background = 0
        _inflight_replies = 0
        _err_last_key = None
        _err_last_logged_mono = float("-inf")
        _err_suppressed_count = 0


def mark_interactive_active(at: float | None = None) -> None:
    """Test/back-compat helper: stamp the last-message anchor as "just now",
    without modeling a full turn's in-flight window.

    Production code should use ``note_user_message``/``note_reply_end``
    (``chat/engine.py``'s ``respond`` wrapper) instead — this is kept as the
    single-call convenience many existing tests already use to simulate
    "chat was recently active" (same underlying anchor ``is_chat_idle`` reads).
    """
    global _last_message_mono
    with _lock:
        _last_message_mono = time.monotonic() if at is None else at


def note_user_message(at: float | None = None) -> None:
    """Call at the start of a chat turn: stamps activity and marks a reply
    in flight (is_chat_idle is False until the matching note_reply_end)."""
    global _last_message_mono, _inflight_replies
    with _lock:
        _last_message_mono = time.monotonic() if at is None else at
        _inflight_replies += 1


def note_reply_end(at: float | None = None) -> None:
    """Call when a chat turn's reply finishes (success OR error): re-stamps
    activity — a long call's idle window can't expire mid-flight (3376b2c1) —
    and clears this reply's in-flight marker."""
    global _last_message_mono, _inflight_replies
    with _lock:
        _last_message_mono = time.monotonic() if at is None else at
        _inflight_replies = max(0, _inflight_replies - 1)


def time_since_last_message(*, now: float | None = None) -> float:
    """Seconds since the last user message / reply-end, whichever is later.

    Read-only accessor of the SAME monotonic anchor ``is_chat_idle`` reads
    internally — it does NOT read the lull tunable/key (S72/2-plan §3.3a):
    for a caller (the empty-session prune) that needs "how long has the
    current idle window been open", not "is it open long enough to act"."""
    with _lock:
        t = time.monotonic() if now is None else now
        return t - _last_message_mono


def _log_is_chat_idle_error(exc: Exception) -> None:
    global _err_last_key, _err_last_logged_mono, _err_suppressed_count
    key = f"{type(exc).__name__}:{exc}"
    now = time.monotonic()
    with _lock:
        changed = key != _err_last_key
        if changed:
            _err_last_key = key
            _err_last_logged_mono = now
            _err_suppressed_count = 0
            should_log, first_or_changed, suppressed = True, True, 0
        elif (now - _err_last_logged_mono) >= _ERROR_LOG_INTERVAL_S:
            _err_last_logged_mono = now
            suppressed = _err_suppressed_count
            _err_suppressed_count = 0
            should_log, first_or_changed = True, False
        else:
            _err_suppressed_count += 1
            should_log, first_or_changed, suppressed = False, False, 0
    if not should_log:
        return
    if first_or_changed:
        log.error(
            "cli_throttle.is_chat_idle failed; reporting NOT IDLE (fail-closed)",
            exc_info=exc,
        )
    else:
        log.error(
            "cli_throttle.is_chat_idle still failing (%s); reporting NOT IDLE "
            "(fail-closed) — background work is paused while this persists "
            "(%d suppressed repeats)",
            key, suppressed,
        )


def _note_is_chat_idle_recovery() -> None:
    global _err_last_key, _err_suppressed_count
    with _lock:
        was_failing = _err_last_key is not None
        _err_last_key = None
        _err_suppressed_count = 0
    if was_failing:
        log.info("cli_throttle.is_chat_idle recovered — background work may resume")


def is_chat_idle(*, now: float | None = None) -> bool:
    """True iff no reply is in flight and the lull has elapsed since the
    last user message / reply end (S35/S42). The sole reader of the
    ``chat.idle_lull_seconds`` tunable key (via ``_lull_seconds``, C4(b)).

    Fails CLOSED on any internal error (S63): returns False ("not idle") and
    logs at a rate-limited pace (see ``_log_is_chat_idle_error``) so a
    persistent fault stays visible without flooding."""
    try:
        with _lock:
            t = time.monotonic() if now is None else now
            inflight = _inflight_replies
            elapsed = t - _last_message_mono
        idle = inflight == 0 and elapsed >= _lull_seconds()
        _note_is_chat_idle_recovery()
        return idle
    except Exception as exc:  # noqa: BLE001 — fail CLOSED (S63)
        # The rate-limited logger itself touches time.monotonic()/_lock; a
        # secondary failure there (e.g. the same fault that broke the clock)
        # must never escape and override the fail-closed return below.
        try:
            _log_is_chat_idle_error(exc)
        except Exception:  # noqa: BLE001 — logging must never defeat fail-closed
            pass
        return False


def should_yield(*, now: float | None = None) -> bool:
    """Read-only peek: True if chat is NOT idle right now (a turn is
    in-flight or the lull hasn't elapsed). Does NOT touch the semaphore —
    safe to call inside a held background_slot to decide whether to break
    mid-batch."""
    return not is_chat_idle(now=now)


def slot_available(*, now: float | None = None) -> bool:
    """Read-only peek: True if a background slot could be acquired right
    now — chat idle AND the concurrency cap not full. Does NOT touch the
    semaphore, so it is safe as a pre-flight gate before a tick commits
    budget/cooldown. The concurrency-cap read fails OPEN (True) on internal
    error, matching acquire_background — is_chat_idle itself already fails
    CLOSED internally, so a malfunction there already reports not-idle."""
    try:
        if not is_chat_idle(now=now):
            return False
        with _lock:
            return _inflight_background < _max_concurrent_background()
    except Exception:  # noqa: BLE001 — fail open (slot-cap half only)
        log.warning("cli_throttle.slot_available failed; reporting available (fail-open)", exc_info=True)
        return True


def acquire_background(*, now: float | None = None) -> bool:
    """True → caller may make its background CLI call (and MUST call
    release_background() after). False → defer to next tick.

    Idle half: is_chat_idle() (the one registered lull, fail-closed).
    Concurrency half: the max_concurrent_background cap (fail-open on
    internal error, unrelated to the idle check)."""
    global _inflight_background
    try:
        if not is_chat_idle(now=now):
            return False
        with _lock:
            if _inflight_background >= _max_concurrent_background():
                return False
            _inflight_background += 1
            return True
    except Exception:  # noqa: BLE001 — fail open (slot-cap half only)
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
