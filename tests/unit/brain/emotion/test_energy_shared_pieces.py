"""The emotion engine's shared pieces that body energy reuses (#650 step 2, spec 1.8 and 1.9)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.emotion import engine_utils
from brain.emotion.engine_utils import (
    Average,
    SnapshotDay,
    decay,
    fold_average,
    persona_past_long_window,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def test_the_gate_is_closed_for_a_persona_with_no_start() -> None:
    assert persona_past_long_window(None, START + timedelta(days=400)) is False


def test_the_gate_opens_at_exactly_thirty_days() -> None:
    thirty = timedelta(days=30)
    assert persona_past_long_window(START, START + thirty - timedelta(seconds=1)) is False
    assert persona_past_long_window(START, START + thirty) is True
    assert persona_past_long_window(START, START + thirty + timedelta(seconds=1)) is True
    assert persona_past_long_window(START, START) is False


def test_the_gate_reads_the_long_half_life_constant_at_call_time(monkeypatch) -> None:
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "EMOTION_LONG_HALF_LIFE_D", 10.0)
    assert persona_past_long_window(START, START + timedelta(days=10)) is True
    assert persona_past_long_window(START, START + timedelta(days=9)) is False


def test_shift_of_asks_the_public_gate_with_the_end_of_its_last_snapshot_day(monkeypatch) -> None:
    """One function, reused and not copied: the engine's shift rule follows whatever the public gate answers."""
    asked: list[tuple[datetime | None, datetime]] = []

    def closed(persona_start_at, end):
        asked.append((persona_start_at, end))
        return False

    monkeypatch.setattr(engine_utils, "persona_past_long_window", closed)
    days = [SnapshotDay("2026-02-10", 5.0, 5.0, True, None)]
    assert engine_utils._shift_of(days, START) == 0
    assert asked == [(START, engine_utils._day_start("2026-02-11"))]


def test_shift_of_with_no_snapshot_days_is_zero_without_asking_the_gate(monkeypatch) -> None:
    def boom(*_a):
        raise AssertionError("no gate question without a snapshot day")

    monkeypatch.setattr(engine_utils, "persona_past_long_window", boom)
    assert engine_utils._shift_of([], START) == 0


def test_decay_halves_the_weight_once_per_half_life() -> None:
    assert decay(0.0, 24.0) == 1.0
    assert decay(24.0, 24.0) == 0.5
    assert decay(720.0, 720.0) == 0.5
    assert decay(1440.0, 720.0) == 0.25


def test_fold_average_sets_the_first_value_and_then_moves_by_weight() -> None:
    average = Average()
    fold_average(average, 8.0, 1.0)
    assert (average.avg, average.weight) == (8.0, 1.0)
    fold_average(average, 4.0, 1.0)
    assert (average.avg, average.weight) == (6.0, 2.0)
    fold_average(average, 100.0, 0.0)  # a weight of 0 folds nothing
    assert (average.avg, average.weight) == (6.0, 2.0)
    fold_average(average, 100.0, 1.0)  # held to the emotion cap
    assert average.avg == 10.0


def test_the_private_names_are_gone() -> None:
    assert not hasattr(engine_utils, "_fold_average")
    assert not hasattr(engine_utils, "_decay")
