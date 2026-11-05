"""The emotion engine's pure core (Engine 1, #576): the fold, the quiet-day close, the fade and the climax reset.

Pure-function tests, no files. Every time is built from a LOCAL civil date (local noon converted to UTC) because the
engine's day is the machine's local calendar day; nothing here sets the process time zone (`time.tzset` does not exist on
Windows). The worked numbers are the ones printed in the engine design (A3 "Worked numbers"): entries at local noon,
quiet-day zeros dated at the first instant after each quiet day.

Not tested here, by rule (one end-to-end simulation instead of unit tests beside it): `run_tracker`, `shift_for` and the
snapshot carry in `close_days`. Engine 4's simulation (Check 25 and Check 21) is their test.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import logging
import math
from datetime import UTC, date, datetime, time, timedelta

import pytest

from brain import dev_constants
from brain.emotion import engine_utils as eu
from brain.memory import memory_utils
from brain.memory.store import Memory

BASE = date(2026, 5, 1)  # the 120-day cases end by Aug 29: no daylight-saving start in common zones
LOGGER = "brain.emotion.engine_utils"


def _at(day: int, hour: int = 12, minute: int = 0) -> datetime:
    """Local `hour:minute` on day `day` after BASE, as an aware UTC time."""
    return datetime.combine(BASE + timedelta(days=day), time(hour, minute)).astimezone(UTC)


def _day_start(day: int) -> datetime:
    return _at(day, 0, 0)


def _event(kind: str, at: datetime, emotions: dict[str, float], memory_id: str = "m", **extra: float) -> eu.Event:
    return eu.Event(kind=kind, at=at, memory_id=memory_id, emotions=emotions, **extra)


def _fold(state: eu.EngineState, event: eu.Event, now: datetime | None = None) -> None:
    eu.fold_event(state, event, event.at if now is None else now)


def _close(state: eu.EngineState, day: int) -> None:
    eu.close_days(state, _at(day))


def _short(state: eu.EngineState, name: str = "joy") -> float:
    return state.emotions[name].short.avg


def _long(state: eu.EngineState, name: str = "joy") -> float:
    return state.emotions[name].long.avg


def _deep_copy(state: eu.EngineState) -> eu.EngineState:
    return copy.deepcopy(state)


def _lone_entry_then_quiet_days(value: float, quiet_days: int) -> eu.EngineState:
    """One entry at noon on day 0, then the app runs on each of the next `quiet_days` days and the last of them closes."""
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": value}, "m0"))
    for day in range(1, quiet_days + 2):
        _close(state, day)
    return state


# --- E2.1 worked numbers -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quiet_days", "short", "long"),
    [(1, 2.09, 3.93), (3, 0.38, 1.91), (7, 0.02, 0.91), (30, 0.00, 0.18)],
)
def test_joy_8_once_then_quiet_days(quiet_days: int, short: float, long: float) -> None:
    state = _lone_entry_then_quiet_days(8.0, quiet_days)
    assert _short(state) == pytest.approx(short, abs=0.005)
    assert _long(state) == pytest.approx(long, abs=0.005)


def test_fully_quiet_emotion_settles_at_known_weights() -> None:
    state = _lone_entry_then_quiet_days(1.0, 400)
    assert state.emotions["joy"].short.weight == pytest.approx(2.0, abs=1e-6)
    assert state.emotions["joy"].long.weight == pytest.approx(43.8, abs=0.1)


def test_whole_off_days_do_not_age_the_weights() -> None:
    # Joy 8 on day 0, the app off on days 1-3, running quietly on day 4: the same as one quiet day, not 0.34 / 3.79.
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": 8.0}))
    _close(state, 4)
    _close(state, 5)
    assert _short(state) == pytest.approx(2.09, abs=0.005)
    assert _long(state) == pytest.approx(3.93, abs=0.005)


def test_daily_entries_hold_the_value_then_one_quiet_day() -> None:
    state = eu.EngineState()
    for day in range(120):
        _close(state, day)
        _fold(state, _event("commit", _at(day), {"joy": 6.0}, f"m{day}"))
    assert _short(state) == pytest.approx(6.00, abs=0.005)
    assert _long(state) == pytest.approx(6.00, abs=0.005)
    _close(state, 120)
    _close(state, 121)
    assert _short(state) == pytest.approx(2.49, abs=0.005)
    assert _long(state) == pytest.approx(5.85, abs=0.005)


def test_sixty_days_then_fourteen_quiet_then_a_change_and_a_zero_entry() -> None:
    state = eu.EngineState()
    for day in range(60):
        _close(state, day)
        _fold(state, _event("commit", _at(day), {"joy": 6.0}, f"m{day}"))
    for day in range(60, 75):
        _close(state, day)
    assert _short(state) == pytest.approx(0.00, abs=0.005)
    assert _long(state) == pytest.approx(3.96, abs=0.005)
    changed = _deep_copy(state)
    _fold(changed, _event("change", _at(74), {"joy": -2.0}, "x"))
    assert _long(changed) == pytest.approx(3.90, abs=0.005)
    zeroed = _deep_copy(state)
    _fold(zeroed, _event("commit", _at(74), {"joy": 0.0}, "y"))
    assert _long(zeroed) == pytest.approx(3.85, abs=0.005)


def test_one_entry_day_in_three() -> None:
    state = eu.EngineState()
    for day in range(120):
        _close(state, day)
        if day % 3 == 0:
            _fold(state, _event("commit", _at(day), {"joy": 6.0}, f"m{day}"))
    _close(state, 120)
    assert _short(state) == pytest.approx(0.63, abs=0.005)
    assert _long(state) == pytest.approx(1.94, abs=0.005)


def test_anger_ten_quiet_days_then_a_new_entry() -> None:
    state = _lone_entry_then_quiet_days(8.0, 10)
    state.emotions["anger"] = state.emotions.pop("joy")
    assert _short(state, "anger") == pytest.approx(0.00, abs=0.005)
    assert _long(state, "anger") == pytest.approx(0.64, abs=0.005)
    _fold(state, _event("commit", _at(11), {"anger": 3.0}, "a"))
    assert _short(state, "anger") == pytest.approx(1.24, abs=0.005)
    assert _long(state, "anger") == pytest.approx(0.86, abs=0.005)


def test_entry_just_after_midnight_then_quiet() -> None:
    # An entry of 6 at 00:30 on day D: nothing folds on D or D+1, the zero for D+1 is dated 47.5 h on.
    state = eu.EngineState()
    eu.close_days(state, _at(0, 0, 30))
    _fold(state, _event("commit", _at(0, 0, 30), {"joy": 6.0}))
    _close(state, 1)
    _close(state, 2)
    assert _short(state) == pytest.approx(1.2, abs=0.02)


# --- E2.11: the result does not depend on the order the entries arrive in -------------------------------------------


def _state_signature(state: eu.EngineState) -> tuple:
    rows = {
        name: (
            pytest.approx(e.short.avg, rel=1e-9, abs=1e-12),
            pytest.approx(e.short.weight, rel=1e-9, abs=1e-12),
            pytest.approx(e.long.avg, rel=1e-9, abs=1e-12),
            pytest.approx(e.long.weight, rel=1e-9, abs=1e-12),
            e.last_entry_day,
        )
        for name, e in state.emotions.items()
    }
    return rows, state.last_update


def test_the_same_entries_in_any_order_give_the_same_state() -> None:
    # Three entries over five days; the second is dated inside the stretch where the app was off (days 1-3).
    entries = [
        ("joy", 8.0, 1.0, _at(0)),
        ("joy", 3.0, 0.5, _at(2, 18)),
        ("grief", 5.0, 1.0, _at(4)),
    ]
    results = []
    for order in itertools.permutations(entries):
        state = eu.EngineState()
        for name, value, w, at in order:
            eu.fold_entry(state, name, value, w, at)
        results.append(state)
    first = results[0]
    expected_rows, expected_last = _state_signature(first)
    assert first.last_update == _at(4)
    for other in results[1:]:
        rows, last = _state_signature(other)
        assert last == expected_last
        assert set(rows) == set(expected_rows)
        for name, row in rows.items():
            assert row == expected_rows[name]
    assert _short(first) != 0.0


def test_an_entry_at_or_below_the_average_never_raises_it() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(0))
    previous = _short(state)
    for offset in (0.0, 1.0, 5.0, 0.0):
        value = max(previous - offset, 0.0)  # at or below the average as it stands now
        eu.fold_entry(state, "joy", value, 1.0, _at(0, 13))
        assert _short(state) <= previous
        previous = _short(state)


# --- null versus 0, and the change base ---------------------------------------------------------------------------


def test_an_emotion_missing_from_an_event_is_untouched_and_a_present_zero_counts() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 6.0, "grief": 4.0}, "a"))
    joy_before = _short(state)
    _fold(state, _event("commit", _at(0, 13), {"grief": 2.0}, "b"))
    assert _short(state) == joy_before  # joy absent from the event: not folded as 0
    assert "anger" not in state.emotions  # never seen: stays absent
    _fold(state, _event("commit", _at(0, 14), {"joy": 0.0}, "c"))
    assert _short(state) < joy_before  # joy present at 0 counts as a 0


def test_change_on_an_emotion_with_no_history_takes_base_zero() -> None:
    state = eu.EngineState()
    _fold(state, _event("change", _at(0), {"joy": 3.0}, "s"))
    assert _short(state) == 3.0 and _long(state) == 3.0
    _fold(state, _event("change", _at(0), {"grief": -2.0}, "s"))
    assert _short(state, "grief") == 0.0 and _long(state, "grief") == 0.0
    # that 0 entered at full weight as a present value
    assert state.emotions["grief"].short.weight == pytest.approx(1.0)
    assert state.emotions["grief"].long.weight == pytest.approx(1.0)


def test_change_base_is_each_averages_own_stored_value() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 8.0}, "a"))
    _close(state, 0)
    _close(state, 1)
    _close(state, 2)  # day 1 had no entry: a zero; short and long now differ
    short_before, long_before = _short(state), _long(state)
    assert short_before != pytest.approx(long_before)
    _fold(state, _event("change", _at(2), {"joy": 1.0}, "s"))
    # the new average sits between the old average and (old average + 1) for each, so each moved relative to itself
    assert short_before < _short(state) < short_before + 1.0
    assert long_before < _long(state) < long_before + 1.0
    _fold(state, _event("change", _at(2), {"joy": 50.0}, "s"))  # held to 0-10 before it is folded
    assert _short(state) <= dev_constants.EMOTION_VALUE_CAP
    assert _long(state) <= dev_constants.EMOTION_VALUE_CAP


# --- E2.12 future dates -------------------------------------------------------------------------------------------


def test_an_entry_dated_in_the_future_is_dated_now() -> None:
    state = eu.EngineState()
    now = _at(3)
    eu.fold_event(state, _event("commit", now + timedelta(days=2), {"joy": 5.0}), now)
    assert state.last_update == now
    assert state.emotions["joy"].last_entry_day == eu.day_of(now)


def test_a_last_update_ahead_of_now_is_pulled_back_to_now() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(10))
    now = _at(3)
    eu.fold_event(state, _event("showing", _at(3, 11), {"joy": 5.0}), now)
    assert state.last_update == now


# --- E3.1 / E3.2: the fade -------------------------------------------------------------------------------------


def test_fade_halves_after_one_half_life_at_importance_zero() -> None:
    assert eu.fade(8.0, 14.0, 14.0, 0.0) == pytest.approx(4.0)
    assert eu.fade(8.0, 0.0, 14.0, 0.0) == 8.0


def test_importance_stretches_the_half_life() -> None:
    stretched = 14.0 * memory_utils.importance_stretch(5.0)
    assert eu.fade(8.0, stretched, 14.0, 5.0) == pytest.approx(4.0)
    assert eu.fade(8.0, 14.0, 14.0, 5.0) > eu.fade(8.0, 14.0, 14.0, 0.0)


def test_fade_holds_the_importance_to_zero_to_ten_like_the_forgetting_pass() -> None:
    assert eu.fade(8.0, 30.0, 14.0, 15.0) == eu.fade(8.0, 30.0, 14.0, 10.0)
    assert eu.fade(8.0, 30.0, 14.0, -4.0) == eu.fade(8.0, 30.0, 14.0, 0.0)


def test_fade_takes_no_importance_gain_argument() -> None:
    assert list(inspect.signature(eu.fade).parameters) == [
        "value",
        "age_days",
        "half_life_days",
        "importance",
        "floor_fraction",
    ]


def test_fade_floor_holds_for_love_and_belonging_only() -> None:
    for name in ("love", "belonging"):
        floor = dev_constants.EMOTION_FADE_FLOOR_FRACTION[name]
        assert eu.fade(9.0, 10_000.0, 365.0, 0.0, floor) == pytest.approx(9.0 * floor)
    assert eu.fade(9.0, 10_000.0, 3.0, 0.0) < 1e-6
    # a value above its floor is untouched by the floor
    assert eu.fade(9.0, 365.0, 365.0, 0.0, 1.0 / 3.0) == pytest.approx(4.5)


def test_fade_age_is_the_callers_age_so_the_legacy_anchor_is_the_callers() -> None:
    # A memory created 100 days ago, switch-over 10 days ago: the caller passes 10, and that is all fade uses.
    assert eu.fade(8.0, 10.0, 10.0, 0.0) == pytest.approx(4.0)
    assert eu.fade(8.0, 100.0, 10.0, 0.0) < 0.01


def test_fade_never_grows_a_value_for_a_row_dated_in_the_future() -> None:
    assert eu.fade(8.0, -5.0, 14.0, 0.0) == 8.0


# --- E4: the climax reset ------------------------------------------------------------------------------------------


def _climax_state() -> eu.EngineState:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"arousal": 6.0, "desire": 5.0, "comfort_seeking": 9.0, "rest_need": 3.0}, "p"))
    return state


def test_climax_reset_folds_the_memorys_own_emotions_first_then_resets_short_only() -> None:
    event_emotions = {"arousal": 4.0, "desire": 7.0, "comfort_seeking": 2.0, "rest_need": 3.0, "climax": 8.0}
    plain = _climax_state()
    _fold(plain, _event("commit", _at(1), event_emotions, "c", climax=0.0))
    reset = _climax_state()
    _fold(reset, _event("commit", _at(1), event_emotions, "c", climax=8.0))
    for name, expected in (
        ("arousal", max(0.5, plain.emotions["arousal"].short.avg * 0.2)),
        ("desire", plain.emotions["desire"].short.avg * 0.6),
        ("comfort_seeking", min(10.0, plain.emotions["comfort_seeking"].short.avg + 2.0)),
        ("rest_need", min(10.0, plain.emotions["rest_need"].short.avg + 2.0)),
        ("climax", 0.0),
    ):
        assert reset.emotions[name].short.avg == pytest.approx(expected)
    assert plain.emotions["climax"].short.avg == 8.0  # the memory's own climax entry was folded, then cleared by the reset
    for name in event_emotions:
        assert reset.emotions[name].long.avg == plain.emotions[name].long.avg
        assert reset.emotions[name].short.weight == plain.emotions[name].short.weight
        assert reset.emotions[name].long.weight == plain.emotions[name].long.weight
        assert reset.emotions[name].last_entry_day == plain.emotions[name].last_entry_day


def test_climax_below_the_threshold_does_not_reset() -> None:
    state = _climax_state()
    before = state.emotions["arousal"].short.avg
    _fold(state, _event("commit", _at(1), {"arousal": 6.0}, "c", climax=6.9))
    assert state.emotions["arousal"].short.avg == pytest.approx(before, rel=0.5)  # folded, not damped by 0.2
    assert state.emotions["arousal"].short.avg > 1.0


def test_climax_reset_leaves_an_absent_emotion_absent_and_skips_the_bump() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "arousal", 0.4, 1.0, _at(0))
    eu.apply_climax_reset(state)
    assert set(state.emotions) == {"arousal"}  # no comfort_seeking, rest_need, desire or climax conjured
    assert state.emotions["arousal"].short.avg == dev_constants.EMOTION_CLIMAX_AROUSAL_FLOOR


def test_climax_bump_is_held_to_ten() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "comfort_seeking", 9.5, 1.0, _at(0))
    eu.apply_climax_reset(state)
    assert state.emotions["comfort_seeking"].short.avg == dev_constants.EMOTION_VALUE_CAP


def test_climax_reset_does_not_touch_last_entry_day_or_add_weight() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "arousal", 8.0, 1.0, _at(0))
    before = (state.emotions["arousal"].short.weight, state.emotions["arousal"].last_entry_day, state.last_update)
    eu.apply_climax_reset(state)
    assert (state.emotions["arousal"].short.weight, state.emotions["arousal"].last_entry_day, state.last_update) == before


# --- repeats, persona start, the event kinds ------------------------------------------------------------------------


def _showing_weights(state: eu.EngineState, hours: tuple[int, ...]) -> list[float]:
    """The weight each showing of memory "m" at the given hours of day 0 adds to joy, beyond the ageing of what was there."""
    weights = []
    for hour in hours:
        before = state.emotions["joy"].short.weight
        last = state.last_update
        _fold(state, _event("showing", _at(0, hour), {"joy": 5.0}, "m"))
        assert last is not None
        aged = before * 0.5 ** ((_at(0, hour) - last).total_seconds() / 3600.0 / dev_constants.EMOTION_SHORT_HALF_LIFE_H)
        weights.append(state.emotions["joy"].short.weight - aged)
    return weights


def test_repeat_factor_steps_down_with_each_showing_within_the_window() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m"))
    entry = state.repeats["m"]
    assert (entry.first_at, entry.count) == (_at(0), 0)
    factor = dev_constants.EMOTION_REPEAT_FACTOR
    assert _showing_weights(state, (13, 14)) == pytest.approx([factor, factor**2])
    assert state.repeats["m"].count == 2


def test_the_repeat_factor_is_read_from_the_constant_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_REPEAT_FACTOR", 0.25)
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m"))
    assert _showing_weights(state, (13, 14)) == pytest.approx([0.25, 0.0625])


def test_a_showing_outside_the_window_is_a_new_first_showing() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m"))
    _fold(state, _event("showing", _at(0, 13), {"joy": 5.0}, "m"))
    later = _at(0) + timedelta(hours=dev_constants.EMOTION_REPEAT_WINDOW_H)
    before = state.emotions["joy"].short.weight
    _fold(state, _event("showing", later, {"joy": 5.0}, "m"))
    aged = before * 0.5 ** ((later - _at(0, 13)).total_seconds() / 3600.0 / dev_constants.EMOTION_SHORT_HALF_LIFE_H)
    assert state.emotions["joy"].short.weight - aged == pytest.approx(1.0)
    assert (state.repeats["m"].first_at, state.repeats["m"].count) == (later, 0)


def test_a_commit_counts_as_the_first_showing_and_resets_an_existing_entry() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m"))
    _fold(state, _event("showing", _at(0, 13), {"joy": 5.0}, "m"))
    _fold(state, _event("commit", _at(0, 14), {"joy": 5.0}, "m"))
    assert (state.repeats["m"].first_at, state.repeats["m"].count) == (_at(0, 14), 0)


def test_the_modifier_scales_the_weight() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m", modifier=0.1))
    assert state.emotions["joy"].short.weight == pytest.approx(0.1)


def test_a_change_folds_at_weight_one_with_no_modifier_and_no_repeat_factor() -> None:
    state = eu.EngineState()
    _fold(state, _event("change", _at(0), {"joy": 3.0}, "s"))
    _fold(state, _event("change", _at(0), {"joy": 3.0}, "s"))  # the same id again: a change is not a memory showing
    assert state.emotions["joy"].short.weight == pytest.approx(2.0)
    assert state.emotions["joy"].long.weight == pytest.approx(2.0)
    assert state.repeats == {}


def test_a_change_has_no_modifier_so_the_event_field_is_not_read() -> None:
    state = eu.EngineState()
    _fold(state, _event("change", _at(0), {"joy": 3.0}, "s", modifier=0.1))
    assert state.emotions["joy"].short.weight == pytest.approx(1.0)
    _fold(state, _event("change", _at(0), {"grief": 3.0}, "s", modifier=math.nan))  # not read, so not refused
    assert state.emotions["grief"].short.weight == pytest.approx(1.0)
    eu.fold_event(state, eu.Event(kind="start", at=_at(0), modifier=math.inf), _at(1))  # a start carries no modifier either
    assert state.persona_start_at == _at(0)


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_modifier_is_refused_for_the_kinds_that_carry_one(kind: str, bad: float) -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "a"))
    before = _deep_copy(state)
    with pytest.raises(ValueError, match="non-finite"):
        _fold(state, _event(kind, _at(1), {"joy": 5.0}, "b", modifier=bad))
    assert state == before


@pytest.mark.parametrize("kind", ["commit", "assigned", "showing", "change"])
@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_a_non_finite_amount_is_refused_for_every_kind_with_emotions(kind: str, bad: float) -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "a"))
    before = _deep_copy(state)
    with pytest.raises(ValueError, match="non-finite"):
        _fold(state, _event(kind, _at(1), {"joy": 5.0, "grief": bad}, "b"))
    assert state == before


# --- the repeat record of an assigned event (anchored at the memory's created_at) --------------------------------


def _window() -> timedelta:
    return timedelta(hours=dev_constants.EMOTION_REPEAT_WINDOW_H)


def _grief_showing_weight(state: eu.EngineState, at: datetime, now: datetime) -> float:
    """The weight a showing of memory "m" gives grief, an emotion no earlier event of these tests touched."""
    _fold(state, _event("showing", at, {"grief": 5.0}, "m"), now)
    return state.emotions["grief"].short.weight


def _grief_assigned_weight(state: eu.EngineState, at: datetime) -> float:
    _fold(state, _event("assigned", at, {"grief": 5.0}, "m"))
    return state.emotions["grief"].short.weight


def test_an_assigned_event_for_an_old_memory_keeps_no_repeat_record() -> None:
    created = _at(0)
    now = created + _window() + timedelta(hours=6)
    state = eu.EngineState()
    _fold(state, _event("assigned", created, {"joy": 5.0}, "m"), now)
    assert state.emotions["joy"].short.weight == pytest.approx(1.0)  # the assignment itself still folds at full weight
    assert "m" not in state.repeats
    assert _grief_showing_weight(state, now, now) == pytest.approx(1.0)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=now, count=0)  # the showing is the memory's first, with its own record


def test_a_showing_dated_inside_the_old_assignments_window_is_still_a_first_showing() -> None:
    created = _at(0)
    now = created + _window() + timedelta(hours=6)
    state = eu.EngineState()
    _fold(state, _event("assigned", created, {"joy": 5.0}, "m"), now)
    assert _grief_showing_weight(state, created + timedelta(hours=2), now) == pytest.approx(1.0)


def test_an_assigned_event_for_a_recent_memory_keeps_the_remaining_window() -> None:
    created = _at(0)
    now = created + timedelta(hours=1)
    state = eu.EngineState()
    _fold(state, _event("assigned", created, {"joy": 5.0}, "m"), now)
    assert state.emotions["joy"].short.weight == pytest.approx(1.0)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=created, count=0)  # anchored at created_at, not at the fold time
    later = created + timedelta(hours=2)
    assert _grief_showing_weight(state, later, later) == pytest.approx(dev_constants.EMOTION_REPEAT_FACTOR)
    assert state.repeats["m"].count == 1
    # the window ends 24 h after created_at, not 24 h after the assignment was folded
    after = created + _window() + timedelta(minutes=1)
    _grief_showing_weight(state, after, after)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=after, count=0)


def test_the_assigned_record_is_kept_one_second_before_the_window_ends_and_not_at_it() -> None:
    created = _at(0)
    kept = eu.EngineState()
    _fold(kept, _event("assigned", created, {"joy": 5.0}, "m"), created + _window() - timedelta(seconds=1))
    assert kept.repeats["m"] == eu.RepeatEntry(first_at=created, count=0)
    dropped = eu.EngineState()
    _fold(dropped, _event("assigned", created, {"joy": 5.0}, "m"), created + _window())
    assert "m" not in dropped.repeats


def test_a_future_dated_assignment_is_held_to_now_and_kept() -> None:
    now = _at(0)
    state = eu.EngineState()
    _fold(state, _event("assigned", now + timedelta(hours=5), {"joy": 5.0}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=now, count=0)


def test_an_assigned_event_for_an_old_memory_removes_a_stale_record_and_a_recent_one_replaces_it() -> None:
    old = eu.EngineState()
    _fold(old, _event("commit", _at(0), {"joy": 5.0}, "m"))
    _fold(old, _event("assigned", _at(3), {"joy": 5.0}, "m"), _at(6))
    assert "m" not in old.repeats  # the day-0 record was out of its window; the day-3 one would be out of its own
    recent = eu.EngineState()
    _fold(recent, _event("commit", _at(0), {"joy": 5.0}, "m"))
    _fold(recent, _event("assigned", _at(3), {"joy": 5.0}, "m"), _at(3, 13))
    assert recent.repeats["m"] == eu.RepeatEntry(first_at=_at(3), count=0)


def test_an_old_assignment_leaves_a_record_dated_after_it_alone() -> None:
    state = eu.EngineState()
    now = _at(6)
    _fold(state, _event("showing", _at(5, 20), {"joy": 5.0}, "m"), now)  # a live record, newer than the assignment below
    _fold(state, _event("assigned", _at(3), {"grief": 5.0}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=_at(5, 20), count=0)


def test_a_commit_of_an_old_memory_keeps_no_repeat_record() -> None:
    created = _at(0)
    now = created + _window() + timedelta(hours=6)
    state = eu.EngineState()
    _fold(state, _event("commit", created, {"joy": 5.0}, "m"), now)
    assert state.emotions["joy"].short.weight == pytest.approx(1.0)  # the commit itself still folds at full weight
    assert state.persona_start_at == created  # and still keeps the persona's start date
    assert "m" not in state.repeats
    assert _grief_showing_weight(state, now, now) == pytest.approx(1.0)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=now, count=0)  # the showing is the memory's first, with its own record


def test_the_climax_reset_still_applies_to_a_commit_whose_record_is_not_kept() -> None:
    created = _at(0)
    now = created + _window() + timedelta(hours=6)
    state = eu.EngineState()
    _fold(state, _event("commit", created, {"arousal": 9.0}, "m", climax=8.0), now)
    assert "m" not in state.repeats
    expected = max(dev_constants.EMOTION_CLIMAX_AROUSAL_FLOOR, 9.0 * dev_constants.EMOTION_CLIMAX_AROUSAL_FACTOR)
    assert _short(state, "arousal") == pytest.approx(expected)


def test_a_showing_dated_inside_the_old_commits_window_is_still_a_first_showing() -> None:
    created = _at(0)
    now = created + _window() + timedelta(hours=6)
    state = eu.EngineState()
    _fold(state, _event("commit", created, {"joy": 5.0}, "m"), now)
    assert _grief_showing_weight(state, created + timedelta(hours=2), now) == pytest.approx(1.0)


def test_a_commit_of_a_recent_memory_keeps_the_remaining_window() -> None:
    created = _at(0)
    now = created + timedelta(hours=1)
    state = eu.EngineState()
    _fold(state, _event("commit", created, {"joy": 5.0}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=created, count=0)  # anchored at created_at, not at the fold time
    later = created + timedelta(hours=2)
    assert _grief_showing_weight(state, later, later) == pytest.approx(dev_constants.EMOTION_REPEAT_FACTOR)
    assert state.repeats["m"].count == 1
    after = created + _window() + timedelta(minutes=1)  # the window ends 24 h after created_at, not 24 h after the fold
    _grief_showing_weight(state, after, after)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=after, count=0)


def test_the_commit_record_is_kept_one_second_before_the_window_ends_and_not_at_it() -> None:
    created = _at(0)
    kept = eu.EngineState()
    _fold(kept, _event("commit", created, {"joy": 5.0}, "m"), created + _window() - timedelta(seconds=1))
    assert kept.repeats["m"] == eu.RepeatEntry(first_at=created, count=0)
    dropped = eu.EngineState()
    _fold(dropped, _event("commit", created, {"joy": 5.0}, "m"), created + _window())
    assert "m" not in dropped.repeats


def test_a_future_dated_commit_is_held_to_now_and_kept() -> None:
    now = _at(0)
    state = eu.EngineState()
    _fold(state, _event("commit", now + timedelta(hours=5), {"joy": 5.0}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=now, count=0)


def test_a_commit_of_an_old_memory_removes_a_stale_record_even_one_still_live_at_its_own_time() -> None:
    state = eu.EngineState()
    _fold(state, _event("showing", _at(3), {"joy": 5.0}, "m"))  # a record that is still live an hour later
    _fold(state, _event("commit", _at(3) + timedelta(hours=1), {"joy": 5.0}, "m"), _at(6))
    assert "m" not in state.repeats  # dated before the commit's created_at, and past its own window at the fold time
    recent = eu.EngineState()
    _fold(recent, _event("showing", _at(3), {"joy": 5.0}, "m"))
    _fold(recent, _event("commit", _at(3) + timedelta(hours=1), {"joy": 5.0}, "m"), _at(3) + timedelta(hours=2))
    assert recent.repeats["m"] == eu.RepeatEntry(first_at=_at(3) + timedelta(hours=1), count=0)  # a recent commit resets


def test_an_old_commit_leaves_a_record_dated_after_it_alone() -> None:
    state = eu.EngineState()
    now = _at(6)
    _fold(state, _event("showing", _at(5, 20), {"joy": 5.0}, "m"), now)  # a live record, newer than the commit below
    _fold(state, _event("commit", _at(3), {"grief": 5.0}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=_at(5, 20), count=0)


def test_an_old_commit_with_no_emotions_touches_no_repeat_record() -> None:
    state = eu.EngineState()
    now = _at(6)
    _fold(state, _event("showing", _at(5, 20), {"joy": 5.0}, "m"), now)
    _fold(state, _event("commit", _at(3), {}, "m"), now)
    assert state.repeats["m"] == eu.RepeatEntry(first_at=_at(5, 20), count=0)
    assert state.persona_start_at == _at(3)


@pytest.mark.parametrize("age_h", [1, 23, 24, 30])
def test_a_commit_and_an_assigned_event_leave_the_same_repeat_record(age_h: int) -> None:
    created = _at(0)
    now = created + timedelta(hours=age_h)
    committed, assigned = eu.EngineState(), eu.EngineState()
    _fold(committed, _event("commit", created, {"joy": 5.0}, "m"), now)
    _fold(assigned, _event("assigned", created, {"joy": 5.0}, "m"), now)
    assert committed.repeats == assigned.repeats


def test_an_assigned_event_over_a_live_record_takes_the_next_repeat_factor() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "m"))
    assert _grief_assigned_weight(state, _at(0, 13)) == pytest.approx(dev_constants.EMOTION_REPEAT_FACTOR)
    assert state.repeats["m"].count == 1


def test_a_showing_with_no_emotions_writes_no_repeat_record_and_advances_none() -> None:
    state = eu.EngineState()
    _fold(state, _event("showing", _at(0), {}, "m"))
    assert state.repeats == {} and state.emotions == {} and state.last_update is None
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "n"))
    _fold(state, _event("showing", _at(0, 13), {}, "n"))  # nothing folds in, so the count does not move
    assert state.repeats["n"] == eu.RepeatEntry(first_at=_at(0), count=0)
    _fold(state, _event("showing", _at(0, 14), {"grief": 5.0}, "n"))
    assert state.emotions["grief"].short.weight == pytest.approx(dev_constants.EMOTION_REPEAT_FACTOR)


def test_persona_start_is_the_earliest_commit_and_a_start_event_folds_nothing_else() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(3), {"joy": 5.0}, "a"))
    assert state.persona_start_at == _at(3)
    _fold(state, _event("commit", _at(5), {"joy": 5.0}, "b"))
    assert state.persona_start_at == _at(3)
    _fold(state, _event("commit", _at(1), {"joy": 5.0}, "c"))
    assert state.persona_start_at == _at(1)
    only = eu.EngineState()
    eu.fold_event(only, eu.Event(kind="start", at=_at(0)), _at(9))
    assert only.persona_start_at == _at(0)
    assert only.emotions == {} and only.last_update is None and only.repeats == {}
    eu.fold_event(only, eu.Event(kind="start", at=_at(4)), _at(9))
    assert only.persona_start_at == _at(0)


def test_last_entry_day_follows_the_entries_never_a_zero() -> None:
    state = eu.EngineState()
    _fold(state, _event("showing", _at(2), {"joy": 5.0}, "a"))
    _fold(state, _event("change", _at(1), {"joy": 1.0}, "s"))
    assert state.emotions["joy"].last_entry_day == eu.day_of(_at(2))  # the later of the two days
    _fold(state, _event("assigned", _at(0), {"grief": 3.0}, "b"))
    assert state.emotions["grief"].last_entry_day == eu.day_of(_at(0))
    state2 = eu.EngineState()
    eu.fold_entry(state2, "joy", 5.0, 1.0, _at(5))
    eu.fold_entry(state2, "joy", 0.0, 1.0, _at(1), zero=True)
    assert state2.emotions["joy"].last_entry_day == eu.day_of(_at(5))


def test_a_first_fold_never_leaves_an_average_above_the_value_cap() -> None:
    """A weight of 0.22876222127045265 at the value 10 folds to 10.000000000000002 by the delta form without the hold."""
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 10.0, 0.22876222127045265, _at(0))
    assert state.emotions["joy"].short.avg <= dev_constants.EMOTION_VALUE_CAP
    assert state.emotions["joy"].long.avg <= dev_constants.EMOTION_VALUE_CAP
    assert state.emotions["joy"].short.avg == pytest.approx(10.0)
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 10.0, 0.22876222127045265, _at(0))
    eu.fold_entry(state, "joy", 10.0, 0.22876222127045265, _at(0, 13))
    assert 0.0 <= state.emotions["joy"].short.avg <= dev_constants.EMOTION_VALUE_CAP


@pytest.mark.parametrize("kind", ["commit", "start"])
def test_a_first_row_dated_in_the_future_starts_the_persona_at_now_and_an_earlier_row_still_lowers_it(kind: str) -> None:
    """The start gate counts from `persona_start_at`; a future date there would keep it shut until an earlier row arrived."""
    now = _at(3)
    state = eu.EngineState()
    eu.fold_event(state, _event(kind, now + timedelta(days=5), {"joy": 5.0} if kind == "commit" else {}, "a"), now)
    assert state.persona_start_at == now
    eu.fold_event(state, _event(kind, _at(1), {"joy": 5.0} if kind == "commit" else {}, "b"), now)
    assert state.persona_start_at == _at(1)


def test_an_unknown_event_kind_raises() -> None:
    with pytest.raises(ValueError, match="kind"):
        eu.fold_event(eu.EngineState(), eu.Event(kind="merge", at=_at(0)), _at(0))


def test_an_entry_so_old_that_its_weight_underflows_adds_nothing_and_does_not_crash() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(100_000))
    eu.fold_entry(state, "grief", 5.0, 1.0, _at(0))  # about 275 years earlier: the weight rounds to 0
    assert "grief" not in state.emotions
    assert state.emotions["joy"].short.avg == 5.0


# --- close_days: when a day opens, closes, or does nothing -----------------------------------------------------------


def test_close_days_on_an_unset_day_only_opens_it() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(0))
    eu.close_days(state, _at(3))
    assert state.active_day == eu.day_of(_at(3))
    assert state.emotions["joy"].snapshot == []
    assert state.emotions["joy"].short.weight == 1.0  # no zero entry folded


def test_close_days_on_the_same_day_or_an_earlier_day_does_nothing() -> None:
    state = eu.EngineState()
    _close(state, 5)
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(5))
    snapshot = _deep_copy(state)
    _close(state, 5)
    _close(state, 3)  # the clock was set back
    assert state == snapshot


def test_close_days_gives_no_zero_to_an_absent_emotion_or_to_one_with_an_entry_that_day() -> None:
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": 5.0}))
    _close(state, 1)  # closes day 0: joy had an entry that day
    assert state.emotions["joy"].short.weight == 1.0
    assert set(state.emotions) == {"joy"}  # nothing else appears
    _close(state, 2)  # closes day 1: joy had none, gets one zero, dated the first instant after day 1
    assert state.emotions["joy"].short.weight == pytest.approx(0.5 ** (1.5) + 1.0)
    assert state.emotions["joy"].last_entry_day == eu.day_of(_at(0))  # the zero did not set it


def test_close_days_with_no_entry_yet_does_not_crash_and_appends_nothing() -> None:
    state = eu.EngineState()
    _close(state, 0)
    _close(state, 4)
    assert state.active_day == eu.day_of(_at(4))
    assert state.emotions == {}
    assert state.last_update is None


def test_close_days_appends_one_snapshot_day_per_present_emotion() -> None:
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": 5.0, "grief": 2.0}))
    _close(state, 1)
    for emotion in state.emotions.values():
        assert [d.day for d in emotion.snapshot] == [eu.day_of(_at(0))]
        assert emotion.snapshot[0].felt is True
    _close(state, 2)
    assert [d.felt for d in state.emotions["joy"].snapshot] == [True, False]


def test_snapshot_keeps_at_most_thirty_days() -> None:
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": 5.0}))
    for day in range(1, 60):
        _close(state, day)
    days = [d.day for d in state.emotions["joy"].snapshot]
    assert len(days) == 30
    assert days[-1] == eu.day_of(_at(58))


# --- which memories count, and what they weigh -------------------------------------------------------------------


def _memory(memory_type: str, **fields: object) -> Memory:
    metadata = fields.pop("metadata", {})
    tags = fields.pop("tags", [])
    return Memory.create_new(
        content="c", memory_type=memory_type, domain="us", emotions={"joy": 5.0}, tags=tags, metadata=metadata, **fields
    )


@pytest.mark.parametrize(
    ("memory", "kind"),
    [
        (_memory("conversation"), "user"),
        (_memory("fact", tags=["conversation"]), "user"),
        (_memory("observation", metadata={"source_summary": "conversation:abc"}), "user"),
        (_memory("observation", metadata={"tags": [], "importance": 6}), "user"),
        (_memory("making"), "own"),
        (_memory("initiate_outbound", metadata={"initiate_state": "delivered"}), "own"),
        (_memory("initiate_outbound", metadata={"initiate_state": "dismissed"}), "own"),
        (_memory("initiate_outbound", metadata={"initiate_state": "note"}), "own"),
        (_memory("kindled_peer", metadata={"peer_id": "p"}), "peer"),
        (_memory("research"), "research"),
        (_memory("initiate_outbound", metadata={"initiate_state": "pending"}), None),
        (_memory("initiate_outbound", metadata={"initiate_state": "bogus"}), None),
        (_memory("initiate_outbound"), None),
        (_memory("observation", metadata={"tags": [], "importance": 6, "extra": 1}), None),
        (_memory("monologue_emotion"), None),
        (_memory("journal_entry"), None),
        (_memory("dream"), None),
        (_memory("heartbeat"), None),
        (_memory("self_model_resolved"), None),
    ],
)
def test_counted_kind(memory: Memory, kind: str | None) -> None:
    assert eu.counted_kind(memory) == kind


def test_every_initiate_state_after_pending_counts() -> None:
    from typing import get_args

    from brain.initiate.schemas import StateName

    for state_name in get_args(StateName):
        counted = eu.counted_kind(_memory("initiate_outbound", metadata={"initiate_state": state_name}))
        assert (counted == "own") is (state_name != "pending")


def test_source_modifier_by_kind() -> None:
    assert eu.source_modifier(_memory("conversation"), 0.0) == 1.0
    assert eu.source_modifier(_memory("making"), 0.0) == 1.0
    peer = _memory("kindled_peer", metadata={"peer_id": "p"})
    assert eu.source_modifier(peer, 0.0) == pytest.approx(0.1)
    assert eu.source_modifier(peer, 0.5) == pytest.approx(0.55)
    assert eu.source_modifier(peer, 1.0) == pytest.approx(1.0)
    assert eu.source_modifier(peer, 7.0) == pytest.approx(1.0)  # trust is held to 0-1
    assert eu.source_modifier(peer, math.nan) == pytest.approx(0.1)  # a NaN weight would poison the state for good


@pytest.mark.parametrize(
    ("rating", "modifier"),
    [(8, 0.82), ("8", 0.82), (None, 0.1), ("abc", 0.1), (15, 1.0), (-3, 0.1), (math.nan, 0.1), (math.inf, 0.1)],
)
def test_research_modifier_reads_the_rating(rating: object, modifier: float) -> None:
    metadata = {} if rating is None else {"engagement_rating": rating}
    assert eu.source_modifier(_memory("research", metadata=metadata), 0.0) == pytest.approx(modifier)


def test_source_modifier_of_an_uncounted_memory_is_a_caller_error() -> None:
    with pytest.raises(ValueError, match="count"):
        eu.source_modifier(_memory("dream"), 0.0)


# --- the one name-resolution step ---------------------------------------------------------------------------------


def test_resolve_emotions_canonicalises_averages_and_holds() -> None:
    resolved = eu.resolve_emotions({"Joy": 4.0, "joy": 6.0, " Touch-Hunger ": 15, "grief": -1})
    assert resolved == {"joy": 5.0, "touch_hunger": 10.0, "grief": 0.0}


def test_resolve_emotions_averages_before_it_holds() -> None:
    assert eu.resolve_emotions({"joy": 20.0, "Joy": 0.0}) == {"joy": 10.0}


def test_resolve_emotions_drops_a_value_that_is_not_a_finite_number_with_a_log_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        resolved = eu.resolve_emotions({"joy": math.nan, "grief": math.inf, "anger": "x", "love": 2.0, "awe": True})
    assert resolved == {"love": 2.0}
    assert len(caplog.records) == 4
    assert all(r.name == LOGGER for r in caplog.records)


def test_a_dropped_spelling_does_not_spoil_the_other() -> None:
    assert eu.resolve_emotions({"joy": math.nan, "Joy": 4.0}) == {"joy": 4.0}


def test_resolve_emotions_signed_keeps_the_sign() -> None:
    assert eu.resolve_emotions({"Joy": -2.0, "grief": 30.0}, signed=True) == {"joy": -2.0, "grief": 30.0}


# --- review-driven additions: the edges the first draft of these tests did not pin ---------------------------------------


def test_a_climax_of_exactly_the_threshold_resets() -> None:
    state = _climax_state()
    _fold(state, _event("commit", _at(1), {"arousal": 6.0, "climax": 7.0}, "c", climax=dev_constants.EMOTION_CLIMAX_THRESHOLD))
    assert state.emotions["climax"].short.avg == 0.0


@pytest.mark.parametrize("kind", ["showing", "assigned"])
def test_only_a_commit_triggers_the_climax_reset(kind: str) -> None:
    state = _climax_state()
    _fold(state, _event(kind, _at(1), {"climax": 9.0}, "c", climax=9.0))
    assert state.emotions["climax"].short.avg == 9.0


def test_fold_entry_holds_the_value_to_the_emotion_scale() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 25.0, 1.0, _at(0))
    eu.fold_entry(state, "grief", -4.0, 1.0, _at(0))
    eu.fold_entry(state, "anger", 99.0, 1.0, _at(0))
    assert state.emotions["joy"].short.avg == dev_constants.EMOTION_VALUE_CAP
    assert state.emotions["grief"].short.avg == 0.0
    assert state.emotions["anger"].long.avg == dev_constants.EMOTION_VALUE_CAP


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_fold_entry_refuses_a_non_finite_value(bad: float) -> None:
    state = eu.EngineState()
    with pytest.raises(ValueError, match="non-finite"):
        eu.fold_entry(state, "joy", bad, 1.0, _at(0))
    assert state == eu.EngineState()


def test_a_bad_event_leaves_the_state_untouched() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(0), {"joy": 5.0}, "a"))
    before = _deep_copy(state)
    with pytest.raises(ValueError, match="non-finite"):
        _fold(state, _event("commit", _at(1), {"joy": 5.0, "grief": math.nan}, "b"))
    with pytest.raises(ValueError, match="non-finite"):
        _fold(state, _event("showing", _at(1), {"joy": 5.0}, "a", modifier=math.nan))
    assert state == before


def test_a_zero_entry_for_an_absent_emotion_changes_nothing() -> None:
    state = eu.EngineState()
    eu.fold_entry(state, "joy", 5.0, 1.0, _at(0))
    before = _deep_copy(state)
    eu.fold_entry(state, "grief", 0.0, 1.0, _at(1), zero=True)
    assert state == before


def test_the_snapshot_stores_each_average_in_its_own_place() -> None:
    state = eu.EngineState()
    _close(state, 0)
    _fold(state, _event("commit", _at(0), {"joy": 8.0}))
    _close(state, 1)
    _close(state, 2)  # closes day 1: the zero pulls the short average far below the long one
    first, second = state.emotions["joy"].snapshot
    assert (first.short, first.long) == (8.0, 8.0)
    assert second.short == _short(state) and second.long == _long(state)
    assert second.short < second.long


def test_a_commit_without_emotions_still_keeps_the_persona_start() -> None:
    state = eu.EngineState()
    _fold(state, _event("commit", _at(2), {}, "e"))
    assert state.persona_start_at == _at(2)
    assert state.emotions == {} and state.repeats == {}


def test_resolve_emotions_drops_an_integer_too_large_for_a_float(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert eu.resolve_emotions({"joy": 10**400, "grief": 3}) == {"grief": 3.0}
    assert len(caplog.records) == 1
