"""Intensity-weighted lived-age scalar.

advance() integrates lived_age over wall-clock dt with three named drivers
(emotional_intensity, body_strain, chat_activity), each normalized to [0, 1]
and weighted by intuition-driven default coefficients α=0.5 β=0.4 γ=0.3.
Quiet baseline (all drivers ≈ 0) collapses to lived-hours == wall-hours;
max drivers age 2.2× as fast.

Clock-skew safety per spec §5: negative dt returns prev unchanged (handles
clock rollback), forward jumps >6h treated as "system was asleep" — pause,
don't accumulate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Coefficients:
    """Intensity driver weights (α, β, γ, δ)."""

    alpha: float  # emotional_intensity weight
    beta: float  # body_strain weight
    gamma: float  # chat_activity weight
    delta: float  # narrative_weight weight


DEFAULTS = Coefficients(alpha=0.5, beta=0.4, gamma=0.3, delta=0.5)


@dataclass(frozen=True)
class IntensityDrivers:
    """Named intensity drivers, all normalized to [0, 1]."""

    emotional_intensity: float = 0.0
    body_strain: float = 0.0
    chat_activity: float = 0.0
    narrative_weight: float = 0.0


MAX_FORWARD_DT_S = 6 * 3600.0  # 6 hours

# Lived-hours over which an open arc reaches full age-weight in narrative_weight().
NARRATIVE_WEIGHT_HORIZON_HOURS = 168.0


def narrative_weight(arc_inputs: list[tuple[float, float]], *, horizon: float) -> float:
    """Felt weight of the heaviest open arc, in [0, 1].

    arc_inputs: list of (open_lived_hours, emotion_normalised) per open arc.
    Each arc's weight = age_factor * emotion, where
    age_factor = clamp(open_lived_hours / horizon). Returns the max across
    arcs (the single most-weighing arc sets the felt heaviness); 0.0 when empty.
    """
    best = 0.0
    for open_lived_hours, emotion in arc_inputs:
        age_factor = _clamp(open_lived_hours / horizon)
        w = age_factor * _clamp(emotion)
        if w > best:
            best = w
    return _clamp(best)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def rate_per_hour(drivers: IntensityDrivers, *, coef: Coefficients = DEFAULTS) -> float:
    """Rate at which lived_age advances per wall-hour.

    lived_rate = 1.0 + α·emotional_intensity + β·body_strain + γ·chat_activity

    Each driver clamped to [0, 1]. Quiet baseline (all ≈ 0) => rate = 1.0
    (lived-hours == wall-hours). Max drivers => rate = 1 + α + β + γ.

    Args:
        drivers: IntensityDrivers with emotional_intensity, body_strain, chat_activity.
        coef: Coefficients to use; defaults to DEFAULTS (α=0.5, β=0.4, γ=0.3).

    Returns:
        Rate multiplier (>= 1.0).
    """
    intensity = _clamp(drivers.emotional_intensity)
    strain = _clamp(drivers.body_strain)
    activity = _clamp(drivers.chat_activity)
    narrative = _clamp(drivers.narrative_weight)

    return (
        1.0
        + coef.alpha * intensity
        + coef.beta * strain
        + coef.gamma * activity
        + coef.delta * narrative
    )


def advance(
    *,
    prev_lived_hours: float,
    dt_seconds: float,
    drivers: IntensityDrivers,
    coef: Coefficients = DEFAULTS,
) -> float:
    """Advance lived_age by dt_seconds with intensity modulation.

    Clock-skew safety:
    - dt_seconds <= 0 => return prev_lived_hours unchanged (handles clock rollback).
    - dt_seconds > MAX_FORWARD_DT_S (6h) => return prev_lived_hours unchanged
      (system was asleep; don't accumulate fake lived-age).

    Otherwise:
    - Convert dt_seconds to wall-hours.
    - Compute rate multiplier via rate_per_hour(drivers, coef=coef).
    - Return prev_lived_hours + wall_hours * rate_multiplier.

    Args:
        prev_lived_hours: Previous lived-age in hours.
        dt_seconds: Wall-clock delta in seconds.
        drivers: IntensityDrivers.
        coef: Coefficients; defaults to DEFAULTS.

    Returns:
        Updated lived_age in hours.
    """
    if dt_seconds <= 0:
        return prev_lived_hours

    if dt_seconds > MAX_FORWARD_DT_S:
        return prev_lived_hours

    wall_hours = dt_seconds / 3600.0
    rate = rate_per_hour(drivers, coef=coef)

    return prev_lived_hours + wall_hours * rate
