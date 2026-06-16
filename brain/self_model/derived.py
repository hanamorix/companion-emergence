"""Derived emotional read — recency-WINDOWED peak + body physiology.

Orthogonal to the declared read (aggregate_state / max-pool over all memories):
  - declared = peak intensity per channel over the persona's whole recent
               history (the 200-memory lifetime window the caller passes).
  - derived  = peak intensity per channel over only the most-recent
               _RECENT_WINDOW_COUNT memories — "what I've actually felt lately"
               vs "the strongest I've ever claimed to feel it".

Both reads are PEAKS on the same 0–10 intensity scale, so the gap (derived −
declared) is meaningful: a channel felt at peak recently → derived ≈ declared →
~0 gap; a channel whose peak is older than the window and hasn't recurred →
derived 0, declared > 0 → an honest negative gap ("I claim this but haven't felt
it lately").

This replaces the original total-mass-normalised recency MEAN, which divided
each channel's recency-weighted sum by the weight of ALL memories (not just the
channel-bearing ones). That structurally diluted every channel to a small
fraction of its peak and produced a large uniform-negative gap for any populated
persona — the live magnitude-354 self_model_state.json artifact ("almost
everything I claim to feel is arriving at a fraction of the strength"). The
windowed peak is commensurable with the declared peak, so the gap reflects
genuine recent-vs-lifetime divergence, not a scale offset.

This module is pure compute. No I/O, no LLM, no side effects.
Fail-open: any error in compute_derived returns an empty DerivedRead.

──────────────────────────────────────────────────────────────────
Body → channel mapping (module constants, conservative)
──────────────────────────────────────────────────────────────────
The body adjustment is a *small nudge*, not a dominant term.
The windowed peak is the primary signal.

Low-arousal nudge (low energy OR high exhaustion):
  Applied when body_energy <= 3 OR body_exhaustion >= 7.
  Channels nudged UP (by _BODY_NUDGE_AMOUNT):
    "grief", "loneliness", "rest_need", "comfort_seeking"

High-arousal nudge (high energy AND low exhaustion):
  Applied when body_energy >= 8 AND body_exhaustion <= 3.
  Channels nudged UP (by _BODY_NUDGE_AMOUNT):
    "joy", "desire", "curiosity", "arousal"

The nudge is added only for channels already present in the windowed peak
(i.e. channels actually felt recently). This prevents the body from
"inventing" channels that have no recent memory support.

──────────────────────────────────────────────────────────────────
unnamed_pressure (Task 1b)
──────────────────────────────────────────────────────────────────
When the body is in an extreme low-arousal state (exhaustion >= 8 OR
energy <= 1), the body nudge maps onto low-arousal channels. If NONE
of those target channels appear in the windowed peak, the body signal
has "nowhere to go" — this residual is reported as unnamed_pressure.

Floor condition (module const _UNNAMED_THRESHOLD = 0.0):
  unnamed_pressure > 0 ONLY when:
    - (exhaustion >= _UNNAMED_EXHAUSTION_FLOOR OR
       energy <= _UNNAMED_ENERGY_CEIL)                  ← extreme body
    AND
    - no low-arousal channel has any windowed-peak support ← no home
  Otherwise unnamed_pressure == 0.0 (R-E5: ordinary states are exactly 0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from brain.emotion.vocabulary import get as _get_emotion

logger = logging.getLogger(__name__)

# ── recency window ──────────────────────────────────────────────────────────

# The derived read is the peak over the most-recent N emotion-bearing memories.
# Smaller than the caller's ~200-memory "lifetime" set, so derived captures
# "lately" against declared's "ever". Count-based (not days-based) so it is
# never empty for an away/sparse persona — the failure mode that would re-create
# the dilution artifact. If the persona has <= N memories total, derived ==
# declared and the gap is zero (a tiny history has no recent-vs-lifetime split).
_RECENT_WINDOW_COUNT: int = 30

# ── body nudge constants ────────────────────────────────────────────────────

# Maximum amount (intensity units) added by the body adjustment per channel.
# Kept SMALL relative to typical peak values (0–10 scale).
_BODY_NUDGE_AMOUNT: float = 0.4

# Low-arousal threshold: energy <= this OR exhaustion >= this triggers nudge
_LOW_ENERGY_THRESH: int = 3
_HIGH_EXHAUSTION_THRESH: int = 7

# High-arousal threshold: energy >= this AND exhaustion <= this triggers nudge
_HIGH_ENERGY_THRESH: int = 8
_LOW_EXHAUSTION_THRESH: int = 3

# Channel sets for body nudges (registered-only filtering applied at runtime)
_LOW_AROUSAL_CHANNELS: frozenset[str] = frozenset(
    {"grief", "loneliness", "rest_need", "comfort_seeking"}
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
            present in the windowed peak (above the conservative floor).
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
      1. Peak (max-pool) over the most-recent _RECENT_WINDOW_COUNT memories'
         emotion vectors. This captures RECENT FELT INTENSITY on the same scale
         as declared's lifetime peak — the two diverge only when a channel's
         peak is older than the window.
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
    # ── step 1: recency-windowed peak ────────────────────────────────────
    #
    # Select the most-recent _RECENT_WINDOW_COUNT emotion-bearing memories,
    # then max-pool per channel over that window. Robust to bad timestamps
    # (skipped) and sparsity (count-based window is never empty for >=1 memory).
    dated: list[tuple[datetime, object]] = []
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
        dated.append((created, mem))

    dated.sort(key=lambda t: t[0], reverse=True)
    window = dated[:_RECENT_WINDOW_COUNT]

    peak: dict[str, float] = {}
    for _created, mem in window:
        for name, raw in mem.emotions.items():  # type: ignore[attr-defined]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            # Filter to registered vocabulary only
            if _get_emotion(name) is None:
                continue
            if value > peak.get(name, 0.0):
                peak[name] = value

    if not peak:
        # Empty input or all filtered out → empty read, no pressure
        return DerivedRead({}, 0.0, {})

    # ── step 2: body-physiology adjustment ───────────────────────────────
    channels = dict(peak)
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
        # Check whether any low-arousal channel has windowed-peak support
        low_arousal_present = any(ch in peak for ch in _LOW_AROUSAL_CHANNELS)
        if not low_arousal_present:
            # Body is screaming low-arousal but no channel exists to carry it
            # Scale pressure by how extreme the body state is
            exhaustion_excess = max(0, body_exhaustion - _UNNAMED_EXHAUSTION_FLOOR + 1)
            energy_deficit = max(0, _UNNAMED_ENERGY_CEIL - body_energy + 1)
            unnamed_pressure = float(_BODY_NUDGE_AMOUNT * max(exhaustion_excess, energy_deficit))

    return DerivedRead(channels=channels, unnamed_pressure=unnamed_pressure, sources=sources)
