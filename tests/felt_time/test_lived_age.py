"""Tests for intensity-weighted lived-age scalar."""

import pytest

from brain.felt_time.lived_age import (
    DEFAULTS,
    NARRATIVE_WEIGHT_HORIZON_HOURS,
    IntensityDrivers,
    advance,
    narrative_weight,
    rate_per_hour,
)


def test_rate_per_hour_quiet_baseline_is_one():
    """Quiet baseline (all drivers ≈ 0) => lived-hours == wall-hours."""
    drivers = IntensityDrivers(emotional_intensity=0.0, body_strain=0.0, chat_activity=0.0)
    rate = rate_per_hour(drivers)
    assert rate == pytest.approx(1.0, abs=1e-6)


def test_rate_per_hour_max_drivers_is_one_plus_sum_of_coefficients():
    """Max drivers (all 1.0) => rate = 1 + sum(coefficients)."""
    drivers = IntensityDrivers(emotional_intensity=1.0, body_strain=1.0, chat_activity=1.0)
    rate = rate_per_hour(drivers)
    expected = 1.0 + DEFAULTS.alpha + DEFAULTS.beta + DEFAULTS.gamma
    assert rate == pytest.approx(expected, abs=1e-6)


def test_rate_per_hour_emotional_dominates_per_alpha_default():
    """Emotional intensity applies its coefficient multiplicatively."""
    drivers_emotion = IntensityDrivers(emotional_intensity=1.0, body_strain=0.0, chat_activity=0.0)
    rate_emotion = rate_per_hour(drivers_emotion)
    expected_emotion = 1.0 + DEFAULTS.alpha
    assert rate_emotion == pytest.approx(expected_emotion, abs=1e-6)


def test_advance_is_monotonic_with_positive_dt():
    """Positive dt always increases lived_age."""
    prev_lived = 10.0
    dt_seconds = 3600.0  # 1 hour
    drivers = IntensityDrivers(emotional_intensity=0.5, body_strain=0.3, chat_activity=0.1)

    new_lived = advance(prev_lived_hours=prev_lived, dt_seconds=dt_seconds, drivers=drivers)

    assert new_lived > prev_lived


def test_advance_clamps_negative_dt_to_zero():
    """Negative dt (clock rollback) returns prev unchanged."""
    prev_lived = 10.0
    dt_seconds = -100.0
    drivers = IntensityDrivers(emotional_intensity=0.9, body_strain=0.8, chat_activity=0.7)

    new_lived = advance(prev_lived_hours=prev_lived, dt_seconds=dt_seconds, drivers=drivers)

    assert new_lived == prev_lived


def test_advance_clamps_large_forward_jump_to_pause():
    """Forward jump > 6h treated as 'system was asleep' — pause, don't accumulate."""
    prev_lived = 10.0
    dt_seconds = 7 * 3600.0  # 7 hours
    drivers = IntensityDrivers(emotional_intensity=1.0, body_strain=1.0, chat_activity=1.0)

    new_lived = advance(prev_lived_hours=prev_lived, dt_seconds=dt_seconds, drivers=drivers)

    # Should return prev unchanged (paused during sleep)
    assert new_lived == prev_lived


def test_narrative_weight_empty_is_zero():
    assert narrative_weight([], horizon=NARRATIVE_WEIGHT_HORIZON_HOURS) == 0.0


def test_narrative_weight_heavy_long_open_is_high():
    w = narrative_weight([(NARRATIVE_WEIGHT_HORIZON_HOURS, 1.0)], horizon=NARRATIVE_WEIGHT_HORIZON_HOURS)
    assert w == 1.0


def test_narrative_weight_trivial_long_open_is_low():
    w = narrative_weight([(NARRATIVE_WEIGHT_HORIZON_HOURS, 0.1)], horizon=NARRATIVE_WEIGHT_HORIZON_HOURS)
    assert w == pytest.approx(0.1, abs=1e-9)


def test_narrative_weight_just_opened_is_low():
    w = narrative_weight([(1.0, 1.0)], horizon=NARRATIVE_WEIGHT_HORIZON_HOURS)
    assert w == pytest.approx(1.0 / NARRATIVE_WEIGHT_HORIZON_HOURS, abs=1e-9)


def test_narrative_weight_takes_max_across_arcs():
    w = narrative_weight(
        [(10.0, 0.2), (NARRATIVE_WEIGHT_HORIZON_HOURS, 0.9)],
        horizon=NARRATIVE_WEIGHT_HORIZON_HOURS,
    )
    assert w == pytest.approx(0.9, abs=1e-9)


def test_narrative_weight_clamps_overlong():
    w = narrative_weight(
        [(NARRATIVE_WEIGHT_HORIZON_HOURS * 5, 0.5)], horizon=NARRATIVE_WEIGHT_HORIZON_HOURS
    )
    assert w == pytest.approx(0.5, abs=1e-9)


def test_rate_increases_with_narrative_weight():
    base = rate_per_hour(IntensityDrivers())
    weighted = rate_per_hour(IntensityDrivers(narrative_weight=1.0))
    assert weighted == pytest.approx(base + DEFAULTS.delta, abs=1e-9)


def test_rate_unchanged_when_narrative_weight_zero():
    assert rate_per_hour(IntensityDrivers()) == pytest.approx(1.0, abs=1e-9)
