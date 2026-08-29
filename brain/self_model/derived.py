"""Derived emotional read — a normalized recency-weighted MEAN cross.

Two reads of the SAME memory list, differing only in time horizon:
  - baseline (long)  = compute_baseline(memories): a normalized time-weighted
                       mean per channel over that channel's own bearing events,
                       at a LONG half-life (30d). "Where this channel usually
                       sits."
  - derived  (short) = compute_derived(memories, ...): the same normalized mean
                       at a SHORT half-life (3d), plus the body's unnamed_pressure
                       residual. "Where this channel has been running lately."

The gap (short - long, in gap.py) is therefore BIDIRECTIONAL: >0 running above
baseline lately (golden cross), <0 below baseline lately (death cross), ~0 at
baseline.

Each channel is normalized by ITS OWN bearing-event weights only (never by the
weight of all memories). Two consequences the earlier designs got wrong:
  - Padding the store with memories that DON'T bear a channel cannot move that
    channel (constraint 1) — this is what the total-mass-normalized MEAN got
    wrong (the "magnitude-354" dilution artifact), and what the windowed MAX-pool
    over-corrected into a one-sided-negative gap.
  - A global time-shift (an idle/away persona) multiplies every event weight by
    the same factor, which cancels in numerator and denominator, so short and
    long are each shift-invariant and the gap is ~0 rather than a spurious
    universal negative (constraint 2).

There is NO body nudge added to any channel value: under a bidirectional signal a
fixed additive nudge manufactures fake positives. The body still contributes only
via unnamed_pressure (a separate scalar, never added to a channel).

This module is pure compute. No I/O, no LLM, no side effects.
Fail-open: any error in compute_derived / compute_baseline returns an empty read.

──────────────────────────────────────────────────────────────────
unnamed_pressure (body residual with no channel home)
──────────────────────────────────────────────────────────────────
When the body is in an extreme low-arousal state (exhaustion >= 8 OR energy <= 1)
and NONE of the low-arousal channels have any bearing event in history (so the
body signal has nowhere to map), the residual is reported as unnamed_pressure.
Otherwise unnamed_pressure == 0.0 (ordinary states are exactly 0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from brain.emotion.vocabulary import get as _get_emotion

logger = logging.getLogger(__name__)

# ── decay half-lives (hours) ────────────────────────────────────────────────
#
# short = "this week" (a 3-day-old memory carries half weight); long = "this
# month" (30-day half-life). The 10x separation gives a clean recent-vs-baseline
# contrast; the discrimination is robust across a wide Hs x Hl range (vetted), so
# these are not knife-edge values.
_SHORT_HALF_LIFE_HOURS: float = 72.0
_LONG_HALF_LIFE_HOURS: float = 720.0

# ── unnamed_pressure / body constants ───────────────────────────────────────

# Scale of the residual body signal reported as unnamed_pressure. (Retained here;
# also the unit in which the residual is expressed.)
_BODY_NUDGE_AMOUNT: float = 0.4

# Extreme body-state floor for unnamed_pressure to fire (must be extreme —
# conservative, so ordinary states report exactly 0).
_UNNAMED_EXHAUSTION_FLOOR: int = 8   # exhaustion >= this
_UNNAMED_ENERGY_CEIL: int = 1        # energy <= this

# Low-arousal channels the extreme body signal would map onto (registered-only
# filtering applied at runtime).
_LOW_AROUSAL_CHANNELS: frozenset[str] = frozenset(
    {"grief", "loneliness", "rest_need", "comfort_seeking"}
)


# ── dataclass ──────────────────────────────────────────────────────────────


@dataclass
class DerivedRead:
    """Output of compute_derived / compute_baseline.

    Attributes:
        channels: {emotion_name: normalized recency-weighted mean} — registered
            channels with at least one bearing event only.
        unnamed_pressure: magnitude of an extreme low-arousal body signal that
            maps to no channel present in history. 0.0 for ordinary states and
            for the baseline read.
        sources: {source_label: contribution} — informational; empty here (no
            per-channel body adjustment is applied).
    """

    channels: dict[str, float] = field(default_factory=dict)
    unnamed_pressure: float = 0.0
    sources: dict[str, float] = field(default_factory=dict)


# ── core: per-channel normalized recency-weighted mean ──────────────────────


def _channel_ema(memories: list, *, now: datetime, half_life_hours: float) -> dict[str, float]:
    """Normalized time-weighted mean per REGISTERED channel over its OWN bearing
    events: ema(c) = Σ x_i·d_i / Σ d_i, d_i = 0.5 ** ((now - t_i)/half_life).

    Normalizing by the channel's own event weights (not all-memory weight) is what
    makes non-bearing memories invisible (constraint 1) and a global time-shift
    cancel (constraint 2). Robust to bad timestamps / non-numeric / non-positive
    values (all skipped). A channel with no bearing event is absent from the dict.
    """
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for mem in memories:
        try:
            emotions = mem.emotions
            if not emotions or not isinstance(emotions, dict):
                continue
            created = mem.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
        except Exception:
            continue
        age_h = (now - created).total_seconds() / 3600.0
        weight = 0.5 ** (age_h / half_life_hours)
        for name, raw in emotions.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            if _get_emotion(name) is None:
                continue
            num[name] = num.get(name, 0.0) + value * weight
            den[name] = den.get(name, 0.0) + weight

    out: dict[str, float] = {}
    for name, d in den.items():
        if d > 0.0:
            out[name] = num[name] / d
    return out


# ── entry points ────────────────────────────────────────────────────────────


def compute_derived(
    memories: list,
    *,
    body_energy: int,
    body_exhaustion: int,
    now: datetime | None = None,
) -> DerivedRead:
    """The SHORT (felt-lately) read: normalized recency-weighted mean per channel
    at the short half-life, plus the body's unnamed_pressure residual.

    Args:
        memories: list of Memory-like objects (.emotions dict, .created_at datetime).
        body_energy: 1-10 (from BodyState.energy).
        body_exhaustion: 0-9 (from BodyState.exhaustion).
        now: reference instant (defaults to datetime.now(UTC)).

    Returns:
        DerivedRead (registered channels only). Fail-open → empty read.
    """
    try:
        now = now or datetime.now(UTC)
        channels = _channel_ema(memories, now=now, half_life_hours=_SHORT_HALF_LIFE_HOURS)
        unnamed_pressure = _compute_unnamed_pressure(
            channels, body_energy=body_energy, body_exhaustion=body_exhaustion
        )
        return DerivedRead(channels=channels, unnamed_pressure=unnamed_pressure, sources={})
    except Exception:
        logger.exception("compute_derived: unexpected error — returning empty read (fail-open)")
        return DerivedRead({}, 0.0, {})


def compute_baseline(memories: list, *, now: datetime | None = None) -> DerivedRead:
    """The LONG (baseline) read: normalized recency-weighted mean per channel at
    the long half-life. No body term (unnamed_pressure is a felt-read concern).

    Returns:
        DerivedRead (registered channels only), unnamed_pressure 0.0. Fail-open →
        empty read.
    """
    try:
        now = now or datetime.now(UTC)
        channels = _channel_ema(memories, now=now, half_life_hours=_LONG_HALF_LIFE_HOURS)
        return DerivedRead(channels=channels, unnamed_pressure=0.0, sources={})
    except Exception:
        logger.exception("compute_baseline: unexpected error — returning empty read (fail-open)")
        return DerivedRead({}, 0.0, {})


# ── internal: unnamed_pressure ───────────────────────────────────────────────


def _compute_unnamed_pressure(
    channels: dict[str, float],
    *,
    body_energy: int,
    body_exhaustion: int,
) -> float:
    """Residual body signal with no channel home (conservative floor).

    Fires only when the body is in an extreme low-arousal state AND no low-arousal
    channel has any bearing event (so the body signal maps nowhere). Otherwise 0.0.
    """
    extreme_body = (body_exhaustion >= _UNNAMED_EXHAUSTION_FLOOR) or (body_energy <= _UNNAMED_ENERGY_CEIL)
    if not extreme_body:
        return 0.0
    # "present" now means: has at least one bearing event in history (in channels).
    low_arousal_present = any(ch in channels for ch in _LOW_AROUSAL_CHANNELS)
    if low_arousal_present:
        return 0.0
    exhaustion_excess = max(0, body_exhaustion - _UNNAMED_EXHAUSTION_FLOOR + 1)
    energy_deficit = max(0, _UNNAMED_ENERGY_CEIL - body_energy + 1)
    return float(_BODY_NUDGE_AMOUNT * max(exhaustion_excess, energy_deficit))
