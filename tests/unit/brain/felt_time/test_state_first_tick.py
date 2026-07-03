import tempfile
from pathlib import Path

from brain.felt_time.state import FeltTimeState, load_or_recover, persist


def test_first_tick_ts_defaults_none():
    assert FeltTimeState().first_tick_ts is None


def test_first_tick_ts_survives_persist_load():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        persist(FeltTimeState(lived_age_hours=5.0, first_tick_ts="2026-06-01T00:00:00+00:00"), pd)
        state, _ = load_or_recover(pd)
        assert state.first_tick_ts == "2026-06-01T00:00:00+00:00"


def test_missing_first_tick_ts_loads_as_none():
    # Old JSON without the key → None (back-compat).
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        persist(FeltTimeState(lived_age_hours=5.0), pd)  # first_tick_ts default None
        state, _ = load_or_recover(pd)
        assert state.first_tick_ts is None
