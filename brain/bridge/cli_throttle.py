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

At bridge start, ``seed_last_message_from_active_conversations`` seeds the
idle anchor from the newest message timestamp already SAVED on disk (S82) —
narrowing "a fresh process is idle" to "idle once the lull has passed since
the last saved message"; see that function's docstring.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from brain import tunables
from brain.health.json_io import Recovery, read_jsonl_tail

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
# S218: replies STARTED in this process (monotone; `reset()` leaves it alone, so a snapshot
# compared by equality can never be confused by a reset). A reply that starts and ends
# between two looks is invisible to `_inflight_replies` and visible here.
_replies_started: int = 0
_inflight_background = 0
# S218: signalled by `note_reply_end` (built on `_lock`, so waiting and the counters share it).
_reply_ended = threading.Condition(_lock)
# S223: callbacks registered through `ReplyWatch.on_reply_start`, run by `note_user_message` (after
# its counters changed and OUTSIDE `_lock`). Guarded by `_lock`.
_start_hooks: list[_StartHook] = []

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
        _start_hooks.clear()
        _reply_ended.notify_all()


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
    global _last_message_mono, _inflight_replies, _replies_started
    with _lock:
        _last_message_mono = time.monotonic() if at is None else at
        _inflight_replies += 1
        _replies_started += 1
        hooks = list(_start_hooks)
    # S223: outside the lock (a callback may call back into this module) and never raising: the
    # reply path must not break because a measurement's stop callback failed.
    for hook in hooks:
        hook.fire()


def note_reply_end(at: float | None = None) -> None:
    """Call when a chat turn's reply finishes (success OR error): re-stamps
    activity — a long call's idle window can't expire mid-flight (3376b2c1) —
    and clears this reply's in-flight marker."""
    global _last_message_mono, _inflight_replies
    with _lock:
        _last_message_mono = time.monotonic() if at is None else at
        _inflight_replies = max(0, _inflight_replies - 1)
        _reply_ended.notify_all()


class _StartHook:
    """One registration of `ReplyWatch.on_reply_start`: runs its callback at most once."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._fired = False
        self._guard = threading.Lock()

    def fire(self) -> None:
        with self._guard:
            if self._fired:
                return
            self._fired = True
        try:
            self._callback()
        except Exception:  # noqa: BLE001 - a stop callback must never break a reply
            log.warning("cli_throttle: a reply-start callback raised", exc_info=True)


class ReplyWatch:
    """S218: a snapshot of the reply activity. Calling it returns `still_clear()` (see
    `reply_watch`). S223: `on_reply_start` lets a measurement be told the moment a reply starts."""

    def __init__(self) -> None:
        with _lock:
            self._started = _replies_started
            self._idle = _inflight_replies == 0

    def __call__(self) -> bool:
        if not self._idle:
            return False
        with _lock:
            return _replies_started == self._started

    def on_reply_start(self, callback: Callable[[], None]) -> Callable[[], None]:
        """S223: run `callback()` once, on the thread that calls `note_user_message`, when a reply
        starts (after the counters changed, outside the module lock; an exception it raises is
        logged and swallowed). If this watch is already not clear (a reply was in flight at the
        snapshot, or one started since) the callback runs at once, on the calling thread. Returns
        an idempotent `unregister()`; a callback unregistered before a reply starts never runs.
        Used by the RAM ladder to stop its running ONNX call (the reply must not wait for it).
        The callback runs on the REPLY thread, inside `note_user_message`: it must be quick and
        must not block."""
        hook = _StartHook(callback)
        with _lock:
            _start_hooks.append(hook)
        if not self():
            hook.fire()

        def unregister() -> None:
            with _lock:
                with contextlib.suppress(ValueError):
                    _start_hooks.remove(hook)

        return unregister


def reply_watch() -> ReplyWatch:
    """S218: a snapshot of the reply activity, as a callable `still_clear()`.

    `still_clear()` is True while no reply was in flight when the snapshot was taken AND no
    reply has started since (a reply that started after the snapshot counts even if it has
    already ended: its allocations may be inside whatever was measured meanwhile). If a reply
    was in flight at the snapshot, it is False from the start, even after that reply ends.
    The RAM ladder of `nell chat --no-bridge` abandons the rung in progress when this
    turns False (the process-wide RSS reading would be inflated by the reply), and (S223)
    registers `on_reply_start` to stop the rung's running ONNX call at that moment."""
    return ReplyWatch()


def wait_for_no_reply_in_flight() -> None:
    """S218: return at once when no reply is in flight, else block until the last one ends
    (`note_reply_end`, which `respond()` calls in a `finally`). No timeout and no polling
    interval. For a background thread; never call it on the reply path."""
    with _reply_ended:
        _reply_ended.wait_for(lambda: _inflight_replies == 0)


def reply_in_flight() -> bool:
    """True iff a chat reply is currently being generated in this process
    (``_inflight_replies > 0``, the same counter ``is_chat_idle`` reads).

    Distinct from ``is_chat_idle`` (which ALSO requires the lull to have
    elapsed): the heartbeat's own start check (S21) cares only about "is a
    reply in flight right now", not the lull — once started, a heartbeat
    pass runs to completion even if a reply starts mid-pass, so this is
    checked once, at the very start of a pass, never again during it."""
    with _lock:
        return _inflight_replies > 0


def _parse_saved_ts(raw: object) -> datetime:
    """Parse a saved active-conversation turn's ``ts`` field (ISO-8601 UTC,
    written by ``brain.ingest.buffer._now_iso``) into an aware datetime.

    Same tolerant shape already used elsewhere for this exact on-disk field
    (``brain/chat/session.py``'s hydration, ``brain/ingest/buffer.py``'s
    ``session_silence_minutes``): accepts a trailing ``Z`` as well as an
    explicit offset, and treats a naive result as UTC. Raises
    ValueError/TypeError on anything unparseable or missing — the caller
    (``seed_last_message_from_active_conversations``) treats that as an
    is_chat_idle fault (S63) rather than silently guessing.
    """
    if raw is None:
        raise ValueError("turn has no ts field")
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def seed_last_message_from_active_conversations(
    persona_dir: Path,
    *,
    now_wall: datetime | None = None,
    now_mono: float | None = None,
) -> None:
    """S82: seed the is_chat_idle anchor from the newest message timestamp
    already SAVED on disk in ``<persona_dir>/active_conversations/*.jsonl``
    (wall clock) — narrows S23/S35's "a fresh bridge counts as idle" to "idle
    once the lull has passed since the last SAVED message", not merely "the
    process just started". No new persisted file: the timestamps already on
    each turn are the only source of truth.

    Call once, synchronously, at bridge start — BEFORE anything can call
    ``is_chat_idle``/``time_since_last_message`` (same ordering rationale as
    ``tunables_migration.migrate_idle_keys``): both functions read the SAME
    ``_last_message_mono`` anchor this seeds, so seeding it once here keeps
    them consistent with each other for free — there is no separate anchor
    for the S72 prune accessor to fall out of step with.

    - **No saved messages** (no ``active_conversations`` dir, no ``.jsonl``
      files, or every buffer has zero turns): the anchor is left at its
      module default (``-inf``) — a fresh install still counts as idle
      immediately, matching S35's untouched semantics.
    - **A malformed/unparseable saved ``ts``** (missing field, corrupt
      value) — INCLUDING a syntactically corrupt/truncated trailing JSONL
      line (e.g. a crash/OOM mid-append; deliberately NOT silently skipped
      the way ``read_jsonl_skipping_corrupt`` skips a corrupt line elsewhere
      — see the strict tail read below, because silently falling back to an
      OLDER good line here would understate how recent the chat really was,
      the wrong direction for a fail-closed idle gate): treated as an
      is_chat_idle FAULT per S63's fail-closed rule — logged once at ERROR
      and the anchor is seeded to "right now" (not idle), rather than
      crashing bridge startup or silently guessing which timestamp to trust.
    - **A saved ``ts`` in the future** (clock skew): clamped to elapsed=0
      (seeded to "right now") rather than pinning the anchor ahead of the
      real monotonic clock, which would otherwise hold the bridge non-idle
      indefinitely (until real time caught up to the bad timestamp) instead
      of for at most one lull.
    """
    wall_now = now_wall if now_wall is not None else datetime.now(UTC)
    mono_now = now_mono if now_mono is not None else time.monotonic()

    global _last_message_mono
    try:
        from brain.ingest.buffer import list_active_sessions

        newest: datetime | None = None
        for session_id in list_active_sessions(persona_dir):
            path = persona_dir / "active_conversations" / f"{session_id}.jsonl"
            # strict + RAISE: a corrupt, truncated or non-object trailing line (a crash/OOM mid-append) must
            # raise into the fail-closed handler below, never fall back to an OLDER good turn (S63).
            tail = read_jsonl_tail(path, 1, strict=True, recover=Recovery.RAISE)
            if not tail:
                continue
            parsed = _parse_saved_ts(tail[0].get("ts"))
            if newest is None or parsed > newest:
                newest = parsed
    except Exception as exc:  # noqa: BLE001 — malformed ts or any read fault: fail CLOSED (S63)
        log.error(
            "cli_throttle.seed_last_message_from_active_conversations failed; "
            "seeding as NOT IDLE (fail-closed)",
            exc_info=exc,
        )
        with _lock:
            _last_message_mono = mono_now
        return

    if newest is None:
        return  # no saved messages anywhere -> leave the -inf default (idle)

    elapsed = max(0.0, (wall_now - newest).total_seconds())
    with _lock:
        _last_message_mono = mono_now - elapsed


def time_since_last_message(*, now: float | None = None) -> float:
    """Seconds since the last user message / reply-end, whichever is later.

    Read-only accessor of the SAME monotonic anchor ``is_chat_idle`` reads
    internally — it does NOT read the lull tunable/key (S72/2-plan §3.3a):
    for a caller (the empty-session prune) that needs "how long has the
    current idle window been open", not "is it open long enough to act"."""
    with _lock:
        t = time.monotonic() if now is None else now
        return t - _last_message_mono


def chat_activity_marker() -> float:
    """An opaque token that changes whenever the user chats: the monotonic
    anchor ``is_chat_idle`` reads (last user message / reply end). Callers
    compare it for equality only, e.g. the floor-bootstrap retry rule ("retry
    at the next lull" = once chat has happened since the failed attempt,
    name-recall fix S85)."""
    with _lock:
        return _last_message_mono


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
