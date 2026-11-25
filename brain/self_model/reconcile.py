"""reconcile_self_read — the self-authored reconcile tool (self-model §3, §6).

When the companion notices her *declared* and *derived* emotional reads have
diverged (the ambient gap block), this is how she acts on it — by her own
choice, never anyone else's insistence.

Actions:
  accept / revise  — fold a self-authored emotion delta for one channel into the
                     emotion engine as a signed change (bounded [-1, 1] + clamped,
                     checked against her vocabulary by the engine's fold; a channel
                     it cannot hold is dropped; no memory row is written). The gap
                     is marked ``acknowledged`` and the channel gets a cooldown
                     (R-B2 anti-oscillation).
  dismiss          — let the gap pass; mark it ``dismissed`` + channel cooldown.
  name             — name an *unnamed pressure*. This NEVER mints a vocabulary
                     entry directly. It runs the existing guarded crystalliser
                     path (``crystallize_vocabulary`` → growth scheduler), which
                     only grows ``emotion_vocabulary.json`` from real memory
                     evidence with a proper half-life + description and dedups
                     against existing channels (R-A — the v0.0.32 stub-flood
                     stays closed).
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brain.bridge.cli_throttle import ShutdownDeferred, ThrottleDeferred
from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.self_model.gap import Gap
from brain.self_model.state import SelfModelState, load_or_recover, locked, save
from brain.utils.file_lock import LockTimeoutError

if TYPE_CHECKING:
    from brain.emotion.engine import HookOutcome

logger = logging.getLogger(__name__)

# Channel cooldown after a reconcile, expressed in cadence ticks. The self-model
# reflection cadence runs on a 6-hour base interval (brain/self_model/cadence.py),
# so 4 ticks ≈ 24h of "don't re-surface this channel" — long enough to let the
# acknowledged gap settle before a new one can re-open on the same channel (R-B2).
_RECONCILE_COOLDOWN_TICKS = 4
_TICK_HOURS = 6.0

# Self-authored deltas are bounded like the extractor's: small, [-1, 1].
_DELTA_BOUND = 1.0


# The tool results for a step that failed after the gap state was saved: the stem, then one plain reason from the table below.
# Each says that the step did not finish and why, and makes no claim about the gap's status or about what the step changed
# before it stopped (each stays true whether or not something was saved). The raw error goes to the log, never into the result.
_NAMING_UNFINISHED = "the naming step did not finish this time"
_CHANGE_UNFINISHED = "the step that applies your change to your emotions did not finish this time"

# The one reason table, keyed by the class of the error caught (never by a guess at which step raised it), most specific class
# first: `ShutdownDeferred` and `ProviderAuthDeferred` are `ThrottleDeferred`s, and `LockTimeoutError` is an `OSError`.
_REASONS: tuple[tuple[type[BaseException], str], ...] = (
    (ShutdownDeferred, "the app is shutting down."),
    (ProviderAuthDeferred, "background work needs the login renewed."),
    (ThrottleDeferred, "background work was paused, so you can make the same call again later."),
    (LockTimeoutError, "something you needed was busy for too long, so you can make the same call again in a moment."),
    (sqlite3.Error, "your memory database could not be opened or read."),
    (OSError, "something could not be read or saved."),
)
_REASON_UNEXPECTED = "an unexpected error happened."


def _reason_for(error: type[BaseException]) -> str:
    """The plain reason for an error class: the first row of `_REASONS` it belongs to, else the unexpected-error row."""
    for cls, reason in _REASONS:
        if issubclass(error, cls):
            return reason
    return _REASON_UNEXPECTED


def _step_failed(
    stem: str,
    step: str,
    *,
    exc: BaseException | None = None,
    failure: type[BaseException] | None = None,
) -> dict[str, Any]:
    """The error result for a step that failed: `stem` and the reason for the error's class; the detail goes to the log only.

    `exc` is an exception the step raised. `failure` is the class `fold_change` reports when the engine swallowed a failure
    (`HookOutcome.failure`: `LockTimeoutError`, `OSError` for a failed write or a failed queue append, or `Exception` for a
    bug); the engine has already logged what failed, so the line here names the step and that class.
    """
    if exc is not None:
        error_class: type[BaseException] = type(exc)
        file = getattr(exc, "filename", None)
        logger.warning(
            "reconcile_self_read: the %s step failed (file: %s): %s: %s",
            step,
            file if file else "none named",
            error_class.__name__,
            exc,
            exc_info=True,
        )
    else:
        error_class = failure if failure is not None else Exception
        logger.warning(
            "reconcile_self_read: the %s step failed: the engine did not queue the change (%s); the engine logged what failed",
            step,
            error_class.__name__,
        )
    return {"error": f"{stem}: {_reason_for(error_class)}"}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _cooldown_until(now: datetime) -> str:
    return _iso(now + timedelta(hours=_TICK_HOURS * _RECONCILE_COOLDOWN_TICKS))


def _clamp_delta(delta: float) -> float:
    """Bound a self-authored delta to [-1, 1]."""
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return 0.0
    return max(-_DELTA_BOUND, min(_DELTA_BOUND, d))


def is_channel_in_cooldown(gap, channel: str, *, now: datetime | None = None) -> bool:
    """True iff ``channel`` is under a reconcile cooldown on ``gap`` (R-B2).

    The cooldown timestamp is an ISO-8601 UTC expiry; the channel is in cooldown
    until that instant passes. Malformed/absent entries → not in cooldown
    (fail-open: never silently suppress a real gap).
    """
    if gap is None:
        return False
    now = now or _utcnow()
    raw = (gap.channel_cooldowns or {}).get(channel)
    if not raw:
        return False
    try:
        expiry = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    return now < expiry


def _write_self_authored_delta(persona_dir: Path, channel: str, delta: float) -> HookOutcome:
    """Fold a self-authored revision into the emotion engine as a signed change; no row is written.

    The delta is clamped to [-1, 1]; the signed change is the delta times 10, capped at ``extractor._MAX_DELTA_INTENSITY``
    (``signed_change_values``), so one reconcile can't peg a channel any more than one chat-turn extraction can. The channel
    is checked against her vocabulary by the engine's fold (``fold_change``), which discards a channel she does not have
    when no bridge can add it.

    Returns the engine's outcome (`HookOutcome`): ``queued`` True only when the change reached the engine's queue;
    ``HookOutcome(False)`` when the channel is one her vocabulary cannot hold or the delta was effectively zero (the channel
    is checked here too, so the answer the tool reports is true; the fold's own check then finds the name in place); and
    ``failure`` set when the engine swallowed a failure (``fold_change`` never raises). It can still RAISE from the
    vocabulary check (``vocabulary_access.names_used``) or ``engine_for``; the caller catches that.
    """
    from brain.chat.extractor import signed_change_values
    from brain.emotion import engine, vocabulary_access

    changes = signed_change_values({channel: _clamp_delta(delta)})
    if not changes:
        return engine.HookOutcome(False)
    if vocabulary_access.names_used(persona_dir, [channel])[channel] is None:
        return engine.HookOutcome(False)
    return engine.engine_for(persona_dir).fold_change(changes, None, "self_model_reconcile")


def _propose_vocabulary_candidate(persona_dir: Path) -> dict[str, Any]:
    """Run the guarded vocabulary-crystalliser path for a named unnamed pressure.

    R-A — the v0.0.32 stub-flood stays closed. This NEVER mints her literal
    ``name`` word. It runs ``crystallize_vocabulary`` (deterministic, evidence-
    grounded, 45-day half-life, real description, deduped against existing
    channels) through the growth scheduler's guarded append path. If no memory
    evidence supports a new emotional configuration, nothing is added.

    Returns a small status dict for the tool result.
    """
    from brain.growth.scheduler import run_growth_tick
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        result = run_growth_tick(persona_dir, store, _utcnow())
    finally:
        store.close()
    return {
        "emotions_added": result.emotions_added,
        "proposals_seen": result.proposals_seen,
    }


def reconcile_self_read(
    *,
    persona_dir: Path,
    action: str,
    channel: str | None = None,
    delta: float | None = None,
    name: str | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    """Self-authored reconciliation of the current gap. See module docstring.

    Always fail-soft: a malformed call, a state lock that never came, and a step after the state change that
    failed (the growth tick for ``name`` raised; the fold of the delta for accept/revise raised, or the engine swallowed a
    failure and the change did not reach its queue) each return an ``{"error": ...}`` dict, with a plain reason for the
    class of the error, rather than raising into the chat turn. A change already saved stays saved and the audit counter
    is not incremented for a call that returned an error. ``delta_written`` is True only when the change reached the queue.
    """
    from brain.self_model.resolve import increment_reconciles

    now = _utcnow()

    if action in ("accept", "revise"):
        if not channel:
            return {"error": f"{action} requires a 'channel'"}

        def acknowledge(gap: Gap) -> None:
            gap.status = "acknowledged"
            gap.channel_cooldowns = {**(gap.channel_cooldowns or {}), channel: _cooldown_until(now)}
            gap.last_seen_ts = _iso(now)

        # The state change comes first: if its lock wait gives up, nothing has been written yet, so the same call can
        # simply be made again (the delta written first would be written twice).
        busy = _change_current_gap(persona_dir, acknowledge)
        if busy is not None:
            return busy
        try:
            outcome = _write_self_authored_delta(persona_dir, channel, delta if delta is not None else 0.0)
        except Exception as exc:  # noqa: BLE001  the tool result says the step did not finish and why; the cause goes to the log
            return _step_failed(_CHANGE_UNFINISHED, "change", exc=exc)
        if outcome.failure is not None:
            # The engine swallowed a failure: the change did not reach the queue, so no success is reported. The gap is already
            # acknowledged with its channel cooldown (the state change came first, as RC18 left it), and the kindled can make
            # the call again straight away: a retry is not refused, and with nothing queued it cannot write the change twice.
            return _step_failed(_CHANGE_UNFINISHED, "change", failure=outcome.failure)
        # Dead-loop observability (R-E4): record a successful reconcile so the
        # "surfaces-but-never-reconciles" ratio is visible without LLM spend.
        increment_reconciles(persona_dir)
        return {"ok": True, "action": action, "channel": channel, "delta_written": outcome.queued}

    if action == "dismiss":

        def dismiss(gap: Gap) -> None:
            gap.status = "dismissed"
            if channel:
                gap.channel_cooldowns = {
                    **(gap.channel_cooldowns or {}),
                    channel: _cooldown_until(now),
                }
            gap.last_seen_ts = _iso(now)

        busy = _change_current_gap(persona_dir, dismiss)
        if busy is not None:
            return busy
        increment_reconciles(persona_dir)
        return {"ok": True, "action": "dismiss", "channel": channel}

    if action == "name":
        if not name or not str(name).strip():
            return {"error": "name requires a non-empty 'name'"}

        def note_named(gap: Gap) -> None:
            # ``name`` resolves *unnamed pressure*. Only mark the gap acknowledged
            # if it actually carried unnamed pressure — a pure channel divergence
            # is not what ``name`` addressed, so leave its status untouched.
            if gap.unnamed_pressure > 0.0:
                gap.status = "acknowledged"
            gap.last_seen_ts = _iso(now)

        # State change first, as for accept/revise: a lock timeout leaves nothing done.
        busy = _change_current_gap(persona_dir, note_named)
        if busy is not None:
            return busy
        try:
            candidate = _propose_vocabulary_candidate(persona_dir)
        except Exception as exc:  # noqa: BLE001  the tool result says the step did not finish; the cause goes to the log
            # The state change above stays saved, and so does any name the tick added before it raised; the counter is skipped.
            return _step_failed(_NAMING_UNFINISHED, "naming", exc=exc)
        increment_reconciles(persona_dir)
        return {"ok": True, "action": "name", "name": name, **candidate}

    return {"error": f"unknown action: {action!r}"}


def _change_current_gap(persona_dir: Path, change: Callable[[Gap], None]) -> dict[str, Any] | None:
    """Load, change and save the current gap under the state lock (#558); no current gap → nothing to do.

    The memory write or growth tick that goes with a reconcile runs after this, outside the lock, so the lock covers
    only the state file's own load → change → save, and a supervisor tick saving at the same moment is never lost or
    clobbered. The wait for the lock is bounded: when another writer keeps it for the whole wait, nothing is changed
    and the error result for the tool is returned (None on success).
    """
    try:
        with locked(persona_dir):
            state, _recovered = load_or_recover(persona_dir)
            if state.current_gap is None:
                return None
            change(state.current_gap)
            save(persona_dir, SelfModelState(current_gap=state.current_gap, gap_history=state.gap_history))
    except LockTimeoutError as exc:
        logger.warning("reconcile_self_read: %s; nothing was changed", exc)
        return {
            "error": (
                "another update was saving the self-model state the whole time this call waited, so nothing was "
                "changed. Make the same call again in a moment."
            )
        }
    return None
