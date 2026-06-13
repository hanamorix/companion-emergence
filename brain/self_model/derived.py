"""Derived emotional read — recency-weighted mean + body physiology.

Orthogonal to the declared read (aggregate_state / max-pool):
  - declared = peak signal across memories (max-pool)
  - derived  = trend signal (recency-weighted mean) + body nudge

This module is pure compute. No I/O, no LLM, no side effects.
Fail-open: any error in compute_derived returns an empty DerivedRead.

──────────────────────────────────────────────────────────────────
Body → channel mapping (module constants, conservative)
──────────────────────────────────────────────────────────────────
The body adjustment is a *small nudge*, not a dominant term.
The recency-weighted mean is the primary signal.

Low-arousal nudge (low energy OR high exhaustion):
  Applied when body_energy <= 3 OR body_exhaustion >= 7.
  Channels nudged UP (by _BODY_NUDGE_AMOUNT):
    "calm", "grief", "loneliness", "rest_need", "comfort_seeking"

High-arousal nudge (high energy AND low exhaustion):
  Applied when body_energy >= 8 AND body_exhaustion <= 3.
  Channels nudged UP (by _BODY_NUDGE_AMOUNT):
    "joy", "desire", "curiosity", "arousal"

The nudge is added only for channels already present in the recency
mean (i.e. channels with non-zero recency weight). This prevents the
body from "inventing" channels that have no memory support.

──────────────────────────────────────────────────────────────────
unnamed_pressure (Task 1b)
──────────────────────────────────────────────────────────────────
When the body is in an extreme low-arousal state (exhaustion >= 8 OR
energy <= 1), the body nudge maps onto low-arousal channels. If NONE
of those target channels appear in the recency mean, the body signal
has "nowhere to go" — this residual is reported as unnamed_pressure.

Floor condition (module const _UNNAMED_THRESHOLD = 0.0):
  unnamed_pressure > 0 ONLY when:
    - (exhaustion >= _UNNAMED_EXHAUSTION_FLOOR OR
       energy <= _UNNAMED_ENERGY_CEIL)                  ← extreme body
    AND
    - no low-arousal channel has any recency-mean support ← no home
  Otherwise named_pressure == 0.0 (R-E5: ordinary states are exactly 0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from brain.emotion.vocabulary import get as _get_emotion

logger = logging.getLogger(__name__)

# ── body nudge constants ────────────────────────────────────────────────────

# Maximum amount (intensity units) added by the body adjustment per channel.
# Kept SMALL relative to typical recency-mean values (0–10 scale).
_BODY_NUDGE_AMOUNT: float = 0.4

# Low-arousal threshold: energy <= this OR exhaustion >= this triggers nudge
_LOW_ENERGY_THRESH: int = 3
_HIGH_EXHAUSTION_THRESH: int = 7

# High-arousal threshold: energy >= this AND exhaustion <= this triggers nudge
_HIGH_ENERGY_THRESH: int = 8
_LOW_EXHAUSTION_THRESH: int = 3

# Channel sets for body nudges (registered-only filtering applied at runtime)
_LOW_AROUSAL_CHANNELS: frozenset[str] = frozenset(
    {"calm", "grief", "loneliness", "rest_need", "comfort_seeking"}
)
_HIGH_AROUSAL_CHANNELS: frozenset[str] = frozenset(
    {"joy", "desire", "curiosity", "arousal"}
)

# ── unnamed_pressure constants ──────────────────────────────────────────────

# Extreme body state floor for unnamed_pressure to fire.
# Must be MORE extreme than the nudge thresholds — R-E5 conservatism.
_UNNAMED_EXHAUSTION_FLOOR: int = 8   # exhaustion >= this
_UNNAMED_ENERGY_CEIL: int = 1        # energy <= this


# ── dataclass ──────────────────────────────────────────────────────────────


@dataclass
class DerivedRead:
    """Output of compute_derived.

    Attributes:
        channels: {emotion_name: derived_intensity} — registered channels only.
        unnamed_pressure: magnitude of body signal that maps to no channel
            present in the recency mean (above the conservative floor).
            0.0 for ordinary states.
        sources: {source_label: contribution} — informational; maps the
            origin of each channel value for debugging.
    """

    channels: dict[str, float] = field(default_factory=dict)
    unnamed_pressure: float = 0.0
    sources: dict[str, float] = field(default_factory=dict)


# ── main entry point ────────────────────────────────────────────────────────


def compute_derived(
    memories: list,
    *,
    body_energy: int,
    body_exhaustion: int,
) -> DerivedRead:
    """Compute the derived emotional read.

    Strategy:
      1. Recency-weighted mean over memories' emotion vectors.
         weight(m) = 1 / (1 + age_days)  →  newer memories dominate.
         This captures TREND vs declared's max-pool which captures PEAK.
      2. Small body-physiology adjustment (see module-level docstring).
      3. unnamed_pressure: body residual with no channel home (conservative).
      4. Fail-open: any exception → empty DerivedRead, never raises.

    Args:
        memories: list of Memory objects (or anything with .emotions dict
            and .created_at datetime).
        body_energy: 1-10 (from BodyState.energy).
        body_exhaustion: 0-9 (from BodyState.exhaustion).

    Returns:
        DerivedRead with registered channels only.
    """
    try:
        return _compute(memories, body_energy=body_energy, body_exhaustion=body_exhaustion)
    except Exception:
        logger.exception("compute_derived: unexpected error — returning empty read (fail-open)")
        return DerivedRead({}, 0.0, {})


# ── internal implementation ─────────────────────────────────────────────────


def _compute(
    memories: list,
    *,
    body_energy: int,
    body_exhaustion: int,
) -> DerivedRead:
    now = datetime.now(UTC)

    # ── step 1: recency-weighted mean ────────────────────────────────────
    #
    # Orthogonality vs max-pool (declared):
    #   max-pool picks the PEAK per channel across all memories.
    #   We compute a TREND: each memory contributes proportionally to its
    #   recency weight, and we normalise by the SUM OF ALL MEMORY WEIGHTS
    #   (not per-channel). This means a recent memory at 3.0 can outweigh
    #   an ancient memory at 9.0 — the old peak "fades" in the trend view.
    #
    # weight(m) = 1 / (1 + age_days)
    # channel_value = Σ(weight(m) * intensity(m, ch)) / Σ(weight(m))
    #   where the denominator sums ALL memories (not just those with ch)
    #
    # This is the key orthogonality: a single channel appearing only in
    # an old memory will have its value scaled down by the recency
    # denominator that includes all more-recent memories.

    # First pass: compute weight and weighted contributions per channel
    weighted_sum: dict[str, float] = {}
    total_weight: float = 0.0

    valid_memories: list[tuple[float, dict[str, float]]] = []

    for mem in memories:
        try:
            emotions = mem.emotions
            if not emotions or not isinstance(emotions, dict):
                continue
            created = mem.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            weight = 1.0 / (1.0 + age_days)
        except Exception:
            continue

        filtered: dict[str, float] = {}
        for name, raw in emotions.items():
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            # Filter to registered vocabulary only
            if _get_emotion(name) is None:
                continue
            filtered[name] = value

        if filtered:
            valid_memories.append((weight, filtered))
            total_weight += weight
            for name, value in filtered.items():
                weighted_sum[name] = weighted_sum.get(name, 0.0) + value * weight

    # Normalise by total weight (ALL memories, not per-channel)
    # This is the orthogonality vs max-pool: an old peak is diluted by
    # the weight mass of all more-recent memories.
    recency_mean: dict[str, float] = {}
    if total_weight > 0.0:
        for name, wsum in weighted_sum.items():
            recency_mean[name] = wsum / total_weight

    if not recency_mean:
        # Empty input or all filtered out → empty read, no pressure
        return DerivedRead({}, 0.0, {})

    # ── step 2: body-physiology adjustment ───────────────────────────────
    channels = dict(recency_mean)
    sources: dict[str, float] = {}

    low_arousal_active = (body_energy <= _LOW_ENERGY_THRESH) or (body_exhaustion >= _HIGH_EXHAUSTION_THRESH)
    high_arousal_active = (body_energy >= _HIGH_ENERGY_THRESH) and (body_exhaustion <= _LOW_EXHAUSTION_THRESH)

    if low_arousal_active:
        for ch in _LOW_AROUSAL_CHANNELS:
            if ch in channels and _get_emotion(ch) is not None:
                old = channels[ch]
                channels[ch] = min(old + _BODY_NUDGE_AMOUNT, _get_emotion(ch).intensity_clamp)  # type: ignore[union-attr]
                if channels[ch] != old:
                    sources[f"body_low_arousal:{ch}"] = channels[ch] - old

    if high_arousal_active:
        for ch in _HIGH_AROUSAL_CHANNELS:
            if ch in channels and _get_emotion(ch) is not None:
                old = channels[ch]
                channels[ch] = min(old + _BODY_NUDGE_AMOUNT, _get_emotion(ch).intensity_clamp)  # type: ignore[union-attr]
                if channels[ch] != old:
                    sources[f"body_high_arousal:{ch}"] = channels[ch] - old

    # ── step 3: unnamed_pressure ─────────────────────────────────────────
    unnamed_pressure = 0.0
    extreme_body = (body_exhaustion >= _UNNAMED_EXHAUSTION_FLOOR) or (body_energy <= _UNNAMED_ENERGY_CEIL)
    if extreme_body:
        # Check whether any low-arousal channel has recency-mean support
        low_arousal_present = any(ch in recency_mean for ch in _LOW_AROUSAL_CHANNELS)
        if not low_arousal_present:
            # Body is screaming low-arousal but no channel exists to carry it
            # Scale pressure by how extreme the body state is
            exhaustion_excess = max(0, body_exhaustion - _UNNAMED_EXHAUSTION_FLOOR + 1)
            energy_deficit = max(0, _UNNAMED_ENERGY_CEIL - body_energy + 1)
            unnamed_pressure = float(_BODY_NUDGE_AMOUNT * max(exhaustion_excess, energy_deficit))

    return DerivedRead(channels=channels, unnamed_pressure=unnamed_pressure, sources=sources)
