from brain.felt_time.lived_age import (
    _MIGRATION_SEED_RATE,
    MAX_LIVED_RATE,
    IntensityDrivers,
    rate_per_hour,
)


def test_max_lived_rate_is_derived_from_all_ones():
    # Derived, not hardcoded: the real max includes δ·narrative_weight (2.7),
    # which the old rate_per_hour docstring omitted.
    assert MAX_LIVED_RATE == rate_per_hour(IntensityDrivers(1.0, 1.0, 1.0, 1.0))
    assert abs(MAX_LIVED_RATE - 2.7) < 1e-9


def test_migration_seed_rate_is_midpoint():
    assert _MIGRATION_SEED_RATE == (1.0 + MAX_LIVED_RATE) / 2
    assert abs(_MIGRATION_SEED_RATE - 1.85) < 1e-9
