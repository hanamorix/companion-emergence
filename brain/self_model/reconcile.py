"""reconcile_self_read — the self-authored reconcile tool (self-model §3, §6).

When the companion notices her *declared* and *derived* emotional reads have
diverged (the ambient gap block), this is how she acts on it — by her own
choice, never anyone else's insistence.

Actions:
  accept / revise  — write a self-authored emotion delta for one channel,
                     bounded [-1, 1] + clamped, routed through the registered
                     vocabulary filter (off-vocab channels are dropped). The gap
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

The emotion write mirrors ``brain/chat/extractor.py::_apply_emotion_delta`` — a
tiny emotion-carrying Memory committed through MemoryStore, importance from
magnitude. There is no separate persisted emotion file; felt state is derived
from memory aggregation, so this is the correct (non-bypassing) way to influence
it.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain.self_model.state import SelfModelState, load_or_recover, save

logger = logging.getLogger(__name__)

# Channel cooldown after a reconcile, expressed in cadence ticks. The self-model
# reflection cadence runs on a 6-hour base interval (brain/self_model/cadence.py),
# so 4 ticks ≈ 24h of "don't re-surface this channel" — long enough to let the
# self-authored delta settle into the derived read before a new gap can re-open
# on the same channel (R-B2).
_RECONCILE_COOLDOWN_TICKS = 4
_TICK_HOURS = 6.0

# Self-authored deltas are bounded like the extractor's: small, [-1, 1].
_DELTA_BOUND = 1.0


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


def _write_self_authored_delta(persona_dir: Path, channel: str, delta: float) -> bool:
    """Commit a tiny emotion-carrying memory for a self-authored revision.

    Mirrors brain/chat/extractor.py::_apply_emotion_delta: the delta is routed
    through ``_filter_to_registered`` (off-vocab channel → dropped) and clamped
    to [-1, 1]; importance is abs(delta) * 10 on the 0..10 MemoryStore scale.

    Returns True if a memory was written, False if the channel was off-vocab or
    the delta was effectively zero (nothing to write).
    """
    from brain.chat.extractor import _filter_to_registered
    from brain.memory.store import Memory, MemoryStore

    clamped = _clamp_delta(delta)
    registered = _filter_to_registered({channel: clamped})
    emotions = {ch: abs(v) * 10.0 for ch, v in registered.items() if abs(v) > 1e-9}
    if not emotions:
        return False

    store = MemoryStore(persona_dir / "memories.db")
    try:
        channel_str = ", ".join(f"{ch}:{v:.2f}" for ch, v in emotions.items())
        mem = Memory.create_new(
            content=f"[self-model reconcile: {channel_str}]",
            memory_type="self_model_reconcile",
            domain="self",
            emotions=emotions,
            importance=max(emotions.values()),
        )
        store.create(mem)
    finally:
        store.close()
    return True


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

    Always fail-soft: a malformed call returns an ``{"error": ...}`` dict rather
    than raising into the chat turn.
    """
    now = _utcnow()
    state, _recovered = load_or_recover(persona_dir)
    gap = state.current_gap

    if action in ("accept", "revise"):
        if not channel:
            return {"error": f"{action} requires a 'channel'"}
        wrote = _write_self_authored_delta(persona_dir, channel, delta if delta is not None else 0.0)
        if gap is not None:
            gap.status = "acknowledged"
            gap.channel_cooldowns = {**(gap.channel_cooldowns or {}), channel: _cooldown_until(now)}
            gap.last_seen_ts = _iso(now)
            save(persona_dir, SelfModelState(current_gap=gap, gap_history=state.gap_history))
        return {"ok": True, "action": action, "channel": channel, "delta_written": wrote}

    if action == "dismiss":
        if gap is not None:
            gap.status = "dismissed"
            if channel:
                gap.channel_cooldowns = {
                    **(gap.channel_cooldowns or {}),
                    channel: _cooldown_until(now),
                }
            gap.last_seen_ts = _iso(now)
            save(persona_dir, SelfModelState(current_gap=gap, gap_history=state.gap_history))
        return {"ok": True, "action": "dismiss", "channel": channel}

    if action == "name":
        if not name or not str(name).strip():
            return {"error": "name requires a non-empty 'name'"}
        candidate = _propose_vocabulary_candidate(persona_dir)
        if gap is not None:
            gap.status = "acknowledged"
            gap.last_seen_ts = _iso(now)
            save(persona_dir, SelfModelState(current_gap=gap, gap_history=state.gap_history))
        return {"ok": True, "action": "name", "name": name, **candidate}

    return {"error": f"unknown action: {action!r}"}
