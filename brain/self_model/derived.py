"""Unnamed pressure — the body's residual signal with no emotion channel to land on.

The emotion engine's short and long averages and its shift rule decide which channels diverge (`gap.gap_from_read`).
This module holds the one remaining derived number of the self-model: `compute_unnamed_pressure`, the body's
extreme low-arousal signal when none of the low-arousal channels is present in the engine's read.

When the body is in an extreme low-arousal state (exhaustion >= 8 OR energy <= 1) and NONE of the low-arousal
channels is present in the engine's read (so the body signal has nowhere to map), the residual is reported as
unnamed_pressure. Otherwise unnamed_pressure == 0.0 (ordinary states are exactly 0).

This module is pure compute. No I/O, no LLM, no side effects.
"""

from __future__ import annotations

# ── unnamed_pressure / body constants ───────────────────────────────────────

# Scale of the residual body signal reported as unnamed_pressure.
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


def compute_unnamed_pressure(
    channels: dict[str, float],
    *,
    body_energy: int,
    body_exhaustion: int,
) -> float:
    """Residual body signal with no channel home (conservative floor).

    Fires only when the body is in an extreme low-arousal state AND no low-arousal
    channel is present in `channels` (the engine's short averages, so the body signal maps nowhere).
    Otherwise 0.0.
    """
    extreme_body = (body_exhaustion >= _UNNAMED_EXHAUSTION_FLOOR) or (body_energy <= _UNNAMED_ENERGY_CEIL)
    if not extreme_body:
        return 0.0
    # "present" means: the channel is in `channels`, i.e. present in the engine's read.
    low_arousal_present = any(ch in channels for ch in _LOW_AROUSAL_CHANNELS)
    if low_arousal_present:
        return 0.0
    exhaustion_excess = max(0, body_exhaustion - _UNNAMED_EXHAUSTION_FLOOR + 1)
    energy_deficit = max(0, _UNNAMED_ENERGY_CEIL - body_energy + 1)
    return float(_BODY_NUDGE_AMOUNT * max(exhaustion_excess, energy_deficit))
