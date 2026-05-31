"""Tests for the daily budget tracker — autonomous-behaviour recipe item 2."""
from datetime import UTC, datetime
from pathlib import Path

from brain.attunement.budget import consume_call, get_remaining


def test_initial_remaining_equals_default(tmp_path: Path):
    assert get_remaining(tmp_path, now=datetime(2026, 5, 31, 12, 0, tzinfo=UTC)) == 150


def test_consume_call_decrements_remaining(tmp_path: Path):
    now = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    assert consume_call(tmp_path, now=now) is True
    assert get_remaining(tmp_path, now=now) == 149


def test_consume_call_returns_false_when_cap_reached(tmp_path: Path):
    now = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    for _ in range(150):
        consume_call(tmp_path, now=now)
    assert consume_call(tmp_path, now=now) is False
    assert get_remaining(tmp_path, now=now) == 0


def test_counter_resets_at_local_midnight(tmp_path: Path):
    # Use noon vs. the following noon to guarantee different local dates
    # regardless of timezone offset (avoids false failures near day boundaries).
    day1 = datetime(2026, 5, 31, 12, 0, tzinfo=UTC)
    day2 = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    for _ in range(150):
        consume_call(tmp_path, now=day1)
    assert get_remaining(tmp_path, now=day1) == 0
    assert get_remaining(tmp_path, now=day2) == 150


def test_corrupt_budget_file_resets_to_default(tmp_path: Path):
    (tmp_path / "attunement").mkdir()
    (tmp_path / "attunement" / "daily_budget.json").write_text("not json")
    assert get_remaining(tmp_path, now=datetime(2026, 5, 31, 12, 0, tzinfo=UTC)) == 150
