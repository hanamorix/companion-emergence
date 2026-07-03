from datetime import datetime, timedelta

from brain.felt_time import _compute_first_tick_ts
from brain.felt_time.lived_age import _MIGRATION_SEED_RATE
from brain.felt_time.state import FeltTimeState

_NOW = "2026-07-03T12:00:00+00:00"


def test_fresh_persona_seeds_now():
    # last_tick None + first_tick None => the true first tick.
    prev = FeltTimeState(last_tick_ts=None, first_tick_ts=None)
    assert _compute_first_tick_ts(prev, _NOW) == _NOW


def test_carries_forward_existing_anchor():
    prev = FeltTimeState(first_tick_ts="2026-01-01T00:00:00+00:00", last_tick_ts=_NOW)
    assert _compute_first_tick_ts(prev, _NOW) == "2026-01-01T00:00:00+00:00"


def test_existing_persona_back_dates_at_midpoint():
    # last_tick set + first_tick None => existing persona; back-date by
    # lived_age / _MIGRATION_SEED_RATE hours.
    prev = FeltTimeState(lived_age_hours=500.0, last_tick_ts=_NOW, first_tick_ts=None)
    got = datetime.fromisoformat(_compute_first_tick_ts(prev, _NOW))
    expected = datetime.fromisoformat(_NOW) - timedelta(hours=500.0 / _MIGRATION_SEED_RATE)
    assert abs((got - expected).total_seconds()) < 1.0
