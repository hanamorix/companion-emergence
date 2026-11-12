"""Engine 6, ONE4 item 2: the one per-memory read of a memory's feelings, peak and score.

Covers `engine_utils.memory_feelings` / `MemoryFeelings` / `faded_emotion` (the pure read, which needs no engine running) and
the engine's `memory_feelings(memory)` wrapper, and that `faded_value` and `_resolved_faded` are the same fade and the same hold.
Every expected number is hand-worked in the comment beside it. Synthetic memories only.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brain.emotion import engine_utils
from brain.emotion.engine import EmotionEngine
from brain.emotion.engine_utils import MemoryFeelings, feelings_of, memory_feelings
from tests.unit.brain.emotion._engine_support import local_day, local_noon, make_live, row

T = [local_noon(local_day(n)) for n in range(60)]  # T[n] = local noon of day n

_HALF_LIVES = {"joy": 3.0, "grief": 10.0, "love": 730.0}


def _half_life(name: str) -> float:
    return _HALF_LIVES.get(name, 5.0)


def _memory(created: datetime, emotions: dict, *, importance: float = 0.0, peak: float | None = None, score: float | None = None):
    base = row(created, emotions=emotions, mid="m")
    changes: dict = {"importance": importance}
    if peak is not None:
        changes["_stored_peak"] = peak
    if score is not None:
        changes["_stored_score"] = score
    return dataclasses.replace(base, **changes)


# --- the pure read -----------------------------------------------------------------------------------------------


def test_before_the_switch_nothing_fades_and_the_stored_values_are_held_to_0_10() -> None:
    memory = _memory(T[0], {"joy": 8.0, "grief": 14.0, "pride": -3.0}, peak=14.0, score=19.0)
    # `now` is 50 days on, which would fade everything if it were used: with no switch it is not
    read = memory_feelings(memory, _half_life, None, T[50])
    assert read.feelings == {"joy": 8.0, "grief": 10.0, "pride": 0.0}  # 14 held to 10, -3 held to 0
    assert read.peak == 10.0  # the stored peak 14 held to 10
    assert read.score == 19.0  # the stored score as it is, not held


def test_after_the_switch_the_feelings_fade_from_the_later_of_creation_and_switch_and_peak_and_score_do_not() -> None:
    memory = _memory(T[0], {"joy": 8.0, "grief": 4.0}, peak=8.0, score=12.0)
    # created day 0, switched day 44, now day 50: the age is 6 days. joy half-life 3: 8 x 0.5^2 = 2.0. grief half-life 10:
    # 4 x 0.5^0.6. Importance 0 (stretch 1).
    read = memory_feelings(memory, _half_life, T[44], T[50])
    assert read.feelings["joy"] == pytest.approx(2.0)
    assert read.feelings["grief"] == pytest.approx(4.0 * 0.5**0.6)
    assert read.peak == 8.0  # the stored highest value, not faded
    assert read.score == 12.0  # the stored score, not faded
    # a switch EARLIER than the creation counts from the creation: created day 40, switched day 10, now day 46 = 6 days
    later = _memory(T[40], {"joy": 8.0})
    assert memory_feelings(later, _half_life, T[10], T[46]).feelings["joy"] == pytest.approx(2.0)


def test_the_fade_uses_the_importance_and_the_floor() -> None:
    # importance 10: stretch 1 + 52 = 53 (HORIZON_IMPORTANCE_GAIN 52): 8 x 0.5^(3 / (3 x 53))
    important = _memory(T[0], {"joy": 8.0}, importance=10.0)
    assert memory_feelings(important, _half_life, T[0], T[3]).feelings["joy"] == pytest.approx(8.0 * 0.5 ** (1 / 53))
    # love: half-life 730 days; after 1460 days the value would be 2.0, but the floor is a third of the stored value
    love = _memory(T[0], {"love": 8.0})
    assert memory_feelings(love, _half_life, T[0], T[0] + timedelta(days=1460)).feelings["love"] == pytest.approx(8.0 / 3)


def test_the_feelings_are_not_computed_when_only_the_peak_and_score_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def spy(memory: object, emotion: str, half_lives: object, switch_over_at: object, now: object) -> float:
        calls.append(emotion)
        return 1.0

    monkeypatch.setattr(engine_utils, "faded_emotion", spy)
    read = memory_feelings(_memory(T[0], {"joy": 8.0, "grief": 4.0}, peak=8.0, score=12.0), _half_life, T[0], T[3])
    assert (read.peak, read.score) == (8.0, 12.0)
    assert calls == []  # no fade yet
    assert read.feelings == {"joy": 1.0, "grief": 1.0}
    assert sorted(calls) == ["grief", "joy"]  # computed on first access
    assert read.feelings is read.feelings  # and kept: a second access computes nothing
    assert len(calls) == 2


def test_the_result_is_frozen() -> None:
    read = memory_feelings(_memory(T[0], {"joy": 8.0}), _half_life, None, T[1])
    assert isinstance(read, MemoryFeelings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        read.peak = 1.0  # type: ignore[misc]


def test_a_negative_or_unusable_stored_value_reads_as_0_in_the_feelings_and_the_peak() -> None:
    memory = _memory(T[0], {"joy": -3.0, "grief": 5.0}, peak=-2.0, score=2.0)
    for switch in (None, T[0]):
        read = memory_feelings(memory, _half_life, switch, T[3])
        assert read.feelings["joy"] == 0.0  # -3 (faded or not) is held to 0
        assert read.peak == 0.0
        assert read.score == 2.0  # the score is as stored
    assert memory_feelings(_memory(T[0], {"joy": 8.0}, peak=math.nan), _half_life, None, T[3]).peak == 0.0


def test_two_spellings_of_one_name_are_averaged_and_a_bad_value_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    memory = _memory(T[0], {"Joy": 4.0, "joy": 6.0, "grief": math.nan})
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine_utils"):
        read = memory_feelings(memory, _half_life, None, T[1])
    assert read.feelings == {"joy": 5.0}  # the same step the engine folds with (`resolve_emotions`)
    assert any("grief" in r.getMessage() for r in caplog.records)


def test_a_name_whose_fade_raises_is_left_out_with_a_line_and_the_others_stay(caplog: pytest.LogCaptureFixture) -> None:
    def half_life(name: str) -> float:
        if name == "grief":
            raise RuntimeError("no half-life")
        return 3.0

    memory = _memory(T[0], {"joy": 8.0, "grief": 4.0})
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine_utils"):
        read = memory_feelings(memory, half_life, T[0], T[3])
    assert read.feelings == {"joy": pytest.approx(4.0)}
    assert any("grief" in r.getMessage() for r in caplog.records)


def test_faded_emotion_names_the_memory_and_the_emotion_when_it_is_absent() -> None:
    with pytest.raises(KeyError, match=r"m.*'grief'"):
        engine_utils.faded_emotion(_memory(T[0], {"joy": 8.0}), "grief", _half_life, None, T[1])


# --- the engine's reads: one fade, one hold ------------------------------------------------------------------------


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_faded_value_and_the_pure_read_agree_for_the_same_inputs(tmp_path: Path) -> None:
    clock = _Clock(T[50])
    make_live(tmp_path, T[0])
    instance = EmotionEngine(tmp_path, now=clock)
    memories = [
        _memory(T[0], {"joy": 8.0, "grief": 4.0, "love": 6.0}, importance=0.0),
        _memory(T[2], {"joy": 3.0}, importance=7.0),
        _memory(T[55], {"joy": 8.0}),  # dated in the future: held at age 0
    ]
    for memory in memories:
        read = memory_feelings(memory, instance.half_life, T[0], T[50])
        for name in memory._stored_emotions:
            assert instance.faded_value(memory, name) == pytest.approx(read.feelings[name])
        assert instance.memory_feelings(memory).feelings == read.feelings


def test_the_engine_and_the_pure_read_use_the_one_per_name_fade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_live(tmp_path, T[0])
    instance = EmotionEngine(tmp_path, now=lambda: T[3])
    monkeypatch.setattr(engine_utils, "faded_emotion", lambda memory, emotion, half_lives, switch_over_at, now: 1.25)
    monkeypatch.setattr("brain.emotion.engine.faded_emotion", engine_utils.faded_emotion)  # the engine imports the name
    memory = _memory(T[0], {"joy": 8.0})
    assert instance.faded_value(memory, "joy") == 1.25
    assert instance.memory_feelings(memory).feelings == {"joy": 1.25}
    assert memory_feelings(memory, _half_life, T[0], T[3]).feelings == {"joy": 1.25}


def test_a_stored_negative_reads_as_0_through_faded_value_live_and_before_the_switch(tmp_path: Path) -> None:
    clock = _Clock(T[3])
    instance = EmotionEngine(tmp_path, now=clock)
    memory = _memory(T[0], {"joy": -3.0, "grief": 14.0})
    assert instance.faded_value(memory, "joy") == 0.0  # not switched: the stored -3 is held to 0
    assert instance.faded_value(memory, "grief") == 10.0  # and 14 to 10
    make_live(tmp_path, T[0])
    assert instance.faded_value(memory, "joy") == 0.0  # live: -3 faded is still negative, held to 0
    # grief live: 14 x 0.5^(3/10) = 11.37, still over 10, held to 10
    assert instance.faded_value(memory, "grief") == 10.0


def test_the_engine_wrapper_uses_the_engines_switch_time_and_clock(tmp_path: Path) -> None:
    clock = _Clock(T[50])
    instance = EmotionEngine(tmp_path, now=clock)
    memory = _memory(T[0], {"joy": 8.0})
    # not switched: the stored value, whatever the age
    assert instance.memory_feelings(memory).feelings == {"joy": 8.0}
    make_live(tmp_path, T[44])
    # live now; the age counts from the switch, day 44, to the clock, day 50: 6 days, joy half-life 3: 8 x 0.25
    read = instance.memory_feelings(memory)
    assert read.feelings["joy"] == pytest.approx(2.0)
    assert (read.peak, read.score) == (memory._stored_peak, memory._stored_score)
    # the clock is read when the read is made, not when the feelings are first asked for
    lazy = instance.memory_feelings(memory)
    clock.now = T[56]
    assert lazy.feelings["joy"] == pytest.approx(2.0)


def test_resolved_faded_is_the_feelings_map_of_the_engine_read(tmp_path: Path) -> None:
    make_live(tmp_path, T[0])
    instance = EmotionEngine(tmp_path, now=lambda: T[6])
    memory = _memory(T[0], {"Joy": 4.0, "joy": 8.0, "grief": -2.0, "love": math.nan})
    resolved = instance._resolved_faded(memory)
    assert resolved == instance.memory_feelings(memory).feelings
    # joy: both spellings fade two half-lives (6 days, half-life 3): 4 -> 1.0 and 8 -> 2.0, averaged after the fade = 1.5
    assert resolved["joy"] == pytest.approx((4.0 * 0.25 + 8.0 * 0.25) / 2)
    assert resolved["grief"] == 0.0
    assert "love" not in resolved


# --- feelings_of: the one choice between an engine's read and the pure read -------------------------------------------


def test_feelings_of_with_an_engine_is_the_engines_read(tmp_path: Path) -> None:
    make_live(tmp_path, T[44])
    instance = EmotionEngine(tmp_path, now=lambda: T[50])
    memory = _memory(T[0], {"joy": 8.0}, peak=8.0, score=12.0)
    read = feelings_of(memory, instance)
    # live, 6 days from the switch to now, joy half-life 3: 8 x 0.25 = 2.0 (the pure read would give 8.0)
    assert read.feelings["joy"] == pytest.approx(2.0)
    assert read.feelings == instance.memory_feelings(memory).feelings
    assert (read.peak, read.score) == (8.0, 12.0)


def test_feelings_of_without_an_engine_is_the_pure_read_of_the_stored_values_held_to_0_10() -> None:
    memory = _memory(T[0], {"joy": 8.0, "grief": 14.0, "pride": -3.0}, peak=14.0, score=19.0)
    read = feelings_of(memory, None)
    assert read.feelings == {"joy": 8.0, "grief": 10.0, "pride": 0.0}
    assert (read.peak, read.score) == (10.0, 19.0)
    assert read.feelings == memory_feelings(memory, None, None, None).feelings


def test_feelings_of_an_engine_not_switched_over_is_the_pure_read(tmp_path: Path) -> None:
    instance = EmotionEngine(tmp_path, now=lambda: T[50])  # no state file: not live, so nothing fades
    memory = _memory(T[0], {"joy": 8.0})
    assert feelings_of(memory, instance).feelings == feelings_of(memory, None).feelings == {"joy": 8.0}
