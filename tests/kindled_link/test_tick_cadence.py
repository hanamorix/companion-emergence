"""T5 — persisted tick cadence (kindled_link/cadence.py).

One test at a time (tdd-guard). Mirror brain/kindled_link/relationship.py cadence idiom.
"""
from datetime import UTC, datetime, timedelta

from brain.kindled_link.cadence import (
    save_tick_cadence,
    tick_is_due,
)

_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)


def test_not_due_within_interval(tmp_path):
    """After saving, tick is NOT due until the interval has elapsed."""
    save_tick_cadence(tmp_path, _NOW)
    assert tick_is_due(tmp_path, _NOW + timedelta(minutes=4)) is False


def test_due_after_interval(tmp_path):
    """After the interval has elapsed the tick is due."""
    save_tick_cadence(tmp_path, _NOW)
    assert tick_is_due(tmp_path, _NOW + timedelta(minutes=6)) is True
