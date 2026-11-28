"""The pure body energy step (#650 spec section 2): the funnel, the used-to multipliers, mood, the message bracket, the drift,
the single hold, the long-average fold and the caught-up heartbeats. The worked numbers are the spec's section 8 values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain import dev_constants
from brain.body import energy
from brain.body.energy import EnergyCarry, EnergyInputs, EnergyMessage, UsedTo
from brain.emotion import engine_utils

HER = EnergyMessage(weight=1.0, length_factor=1.0)
USER = EnergyMessage(weight=0.5, length_factor=1.0)
T0 = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


def _step(level, *, short=None, high=(), messages=(), x=0.0, multipliers=None):
    return energy.step_heartbeat(
        level,
        short=short or {},
        high=frozenset(high),
        messages=tuple(messages),
        x=x,
        multipliers=multipliers or UsedTo(),
    )


def _no_drift(monkeypatch):
    monkeypatch.setattr(dev_constants, "BODY_ENERGY_DRIFT_STEP", 0.0)


# ── A4 the funnel ───────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("level", "want_drop", "want_rise"),
    [(8.0, 0.625, 1.6), (2.0, 2.5, 0.4), (5.0, 1.0, 1.0)],
)
def test_funnel(level, want_drop, want_rise):
    assert energy.drop(level) == pytest.approx(want_drop, abs=1e-12)
    assert energy.rise(level) == pytest.approx(want_rise, abs=1e-12)


# ── A5, A6, A19 the drift ───────────────────────────────────────────────────────────────────────────────────────────


def _drift_steps(start, n, short=None):
    level = start
    for _ in range(n):
        level = _step(level, short=short)
    return level


@pytest.mark.parametrize(
    ("start", "steps", "want"),
    [
        (1.0, 95, 6.3809),
        (1.0, 96, 6.5),
        (9.0, 39, 6.5377),
        (9.0, 40, 6.5),
        (4.0, 24, 6.3885),
        (4.0, 25, 6.5),
    ],
)
def test_drift_alone_reaches_the_baseline_and_stops(start, steps, want):
    assert _drift_steps(start, steps) == pytest.approx(want, abs=5e-5)


def test_drift_never_passes_the_baseline_and_stays():
    assert _drift_steps(1.0, 200) == 6.5
    assert _drift_steps(9.0, 200) == 6.5


def test_drift_with_peace_8_is_paced_by_the_strongest_calm_feeling():
    assert _step(5.0, short={"peace": 8.0}) == pytest.approx(5.0 + 0.0985 * 1.8, abs=1e-12)
    assert _step(5.0, short={"peace": 8.0, "calm": 3.0}) == pytest.approx(5.0 + 0.0985 * 1.8, abs=1e-12)
    assert _drift_steps(1.0, 53, {"peace": 8.0}) == pytest.approx(6.3397, abs=5e-5)
    assert _drift_steps(1.0, 54, {"peace": 8.0}) == 6.5


def test_drift_acts_in_a_heartbeat_with_messages_and_in_one_without():
    drift_only = _step(4.0)
    assert drift_only == pytest.approx(4.0 + 0.0985 * 0.8, abs=1e-12)
    with_message = _step(4.0, messages=[HER])
    assert with_message == pytest.approx(4.0 + 0.0985 * 0.8 - 0.375 * 1.25, abs=1e-12)


# ── A7 mood ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_mood_lift_and_drain_at_level_4(monkeypatch):
    _no_drift(monkeypatch)
    short = {"joy": 8.0, "grief": 5.0}
    assert _step(4.0, short={"joy": 8.0}, high={"joy"}) == pytest.approx(4.24, abs=1e-12)
    assert _step(4.0, short={"grief": 5.0}, high={"grief"}) == pytest.approx(4.0 - 0.234375, abs=1e-12)
    assert _step(4.0, short=short, high={"joy", "grief"}) == pytest.approx(4.005625, abs=1e-12)


def test_mood_at_the_baseline():
    assert _step(6.5, short={"joy": 8.0}, high={"joy"}) == pytest.approx(6.5 + 0.39, abs=1e-12)
    assert _step(6.5, short={"grief": 6.0}, high={"grief"}) == pytest.approx(6.5 - 0.1731, abs=5e-5)


def test_a_feeling_that_is_not_high_contributes_nothing():
    assert _step(6.5, short={"joy": 9.0, "grief": 9.0}, high=()) == 6.5


def test_feelings_on_neither_list_contribute_nothing():
    short = {"anger": 9.0, "nostalgia": 9.0, "fear": 9.0, "contentment": 0.0}
    assert _step(6.5, short=short, high=set(short)) == 6.5


def test_the_lists_are_exactly_the_owner_set():
    assert dev_constants.BODY_ENERGY_LIFT_FEELINGS == ("joy", "love", "gratitude", "belonging", "defiance")
    for name in ("anger", "nostalgia", "fear", "contentment"):
        assert name not in dev_constants.BODY_ENERGY_LIFT_FEELINGS + dev_constants.BODY_ENERGY_DRAIN_FEELINGS


# ── A8, A21 length factor and the base cost ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("ratio", "want"), [(0.25, 0.25), (0.5, 0.5), (1.0, 1.0), (2.0, 1.5), (4.0, 2.5)])
def test_length_factor(ratio, want):
    assert energy.length_factor(int(ratio * 40), 40.0) == pytest.approx(want, abs=1e-12)


def test_a_speaker_with_no_earlier_message_or_an_average_of_zero_gets_one():
    assert energy.length_factor(120, 0.0) == 1.0
    assert energy.length_factor(0, 0.0) == 1.0


def test_base_cost_at_the_baseline_with_no_feeling_effect():
    assert _step(6.5, messages=[HER]) == pytest.approx(6.5 - 0.28846, abs=5e-6)
    assert _step(6.5, messages=[USER]) == pytest.approx(6.5 - 0.14423, abs=5e-6)
    long_one = EnergyMessage(weight=1.0, length_factor=1.5)
    assert _step(6.5, messages=[long_one]) == pytest.approx(6.5 - 0.43269, abs=5e-6)
    assert _step(6.5, messages=[HER, USER]) == pytest.approx(6.5 - 0.43269, abs=5e-6)


def test_a_busy_heartbeat_follows_the_worked_numbers():
    assert _step(6.5, messages=[HER, USER] * 3) == pytest.approx(5.2019, abs=5e-5)
    assert _step(6.5, messages=[HER, USER] * 10) == pytest.approx(2.1731, abs=5e-5)


def test_the_hold_is_applied_once_to_the_sum_not_to_each_term():
    # The messages alone would take her far below 1, the mood adds a little back: one hold on the sum gives exactly 1.0,
    # a hold after each term would give more.
    result = _step(1.5, short={"joy": 8.0}, high={"joy"}, messages=[HER] * 20)
    assert result == energy.hold(1.0)
    assert result == 1.0


def test_a_sum_that_would_dip_below_one_in_the_middle_is_not_held_there():
    # Messages alone (-4.6875) would take 1.2 below 1, the mood (+9.0 at V_lift 100) brings it back well above: one hold on
    # the sum leaves the exact sum; a hold after the messages would give 1.0 first and end at the 10 cap.
    drift = 0.0985 * energy.rise(1.2)
    result = _step(
        1.2, short={"joy": 10.0}, high={"joy"}, messages=[HER] * 3, multipliers=UsedTo(v_lift=100.0)
    )
    assert result == pytest.approx(1.2 + 9.0 - 3 * 0.375 * energy.drop(1.2) + drift, abs=1e-9)


def test_the_level_stays_on_the_scale_and_a_stored_level_off_the_scale_is_held_first():
    assert _step(50.0, short={"joy": 9.0}, high={"joy"}) <= 10.0
    assert _step(0.0, messages=[HER]) == 1.0  # no division by zero in the funnel
    assert _step(-3.0) >= 1.0


# ── A9, A23 the feeling movement X ─────────────────────────────────────────────────────────────────────────────────


def test_x_is_lift_moves_minus_drain_moves_over_ten():
    saved = {"joy": 4.0, "love": 1.0, "grief": 2.0}
    short = {"joy": 6.0, "love": 1.0, "grief": 5.0, "calm": 9.0}
    assert energy.feeling_movement(short, saved) == pytest.approx((2.0 - 3.0) / 10.0, abs=1e-12)


def test_x_is_zero_without_saved_values_or_a_reading():
    assert energy.feeling_movement({"joy": 9.0}, None) == 0.0
    assert energy.feeling_movement({}, {"joy": 9.0}) == 0.0


def test_x_counts_a_feeling_absent_on_one_side_as_zero():
    assert energy.feeling_movement({"joy": 3.0}, {}) == pytest.approx(0.3, abs=1e-12)
    assert energy.feeling_movement({}, {"joy": 3.0}) == 0.0  # nothing readable: no movement
    assert energy.feeling_movement({"love": 1.0}, {"joy": 3.0}) == pytest.approx((1.0 - 3.0) / 10.0, abs=1e-12)


def test_no_listed_feeling_moved_gives_the_base_cost_only():
    saved = {"joy": 4.0}
    assert energy.feeling_movement({"joy": 4.0}, saved) == 0.0
    assert _step(6.5, messages=[HER]) == pytest.approx(6.5 - 0.28846, abs=5e-6)


# ── A24 the bracket ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_bracket_funnelled_by_its_sign_at_the_baseline():
    assert _step(6.5, x=0.3, messages=[HER]) == pytest.approx(6.5 - 0.05769, abs=5e-6)
    assert _step(6.5, x=0.3, messages=[USER]) == pytest.approx(6.5 - 0.02885, abs=5e-6)
    assert _step(6.5, x=0.3, messages=[HER, USER]) == pytest.approx(6.5 - 0.08654, abs=5e-6)
    assert _step(6.5, x=0.5, messages=[HER]) == pytest.approx(6.5 + 0.1625, abs=5e-6)
    assert _step(6.5, x=0.5, messages=[HER, USER]) == pytest.approx(6.5 + 0.24375, abs=5e-6)


def test_bracket_at_level_8():
    # The drift (a drop toward 6.5) is part of the step, so take it out.
    drift = 0.0985 * energy.drop(8.0)
    assert _step(8.0, x=0.3, messages=[HER]) == pytest.approx(8.0 - 0.046875 - drift, abs=1e-9)
    assert _step(8.0, x=0.3, messages=[USER]) == pytest.approx(8.0 - 0.02344 - drift, abs=1e-5)


def test_bracket_with_the_feeling_part_scaled_by_v_only():
    mult = UsedTo(v_lift=2.5, v_drain=0.4)
    assert _step(6.5, x=0.6, messages=[HER], multipliers=mult) == pytest.approx(6.5 + 1.4625, abs=5e-6)


def test_the_funnel_acts_on_the_feeling_total_by_its_sign(monkeypatch):
    monkeypatch.setattr(dev_constants, "BODY_ENERGY_STEP", 0.0)
    _no_drift(monkeypatch)
    assert _step(3.0, x=0.1, messages=[HER]) == pytest.approx(3.0 + 0.06, abs=1e-12)
    assert _step(8.0, x=0.1, messages=[HER]) == pytest.approx(8.0 + 0.16, abs=1e-12)
    # X below 0 uses the drop funnel and V_drain.
    assert _step(8.0, x=-0.1, messages=[HER], multipliers=UsedTo(v_drain=0.4)) == pytest.approx(
        8.0 - 0.1 * 0.4 * 0.625, abs=1e-12
    )


# ── A25 which multiplier scales what ───────────────────────────────────────────────────────────────────────────────


def test_u_scales_the_drift_and_the_base_cost_but_v_does_not():
    mult = UsedTo(u_rise=2.0, u_drop=3.0, v_lift=5.0, v_drain=7.0)
    assert _step(4.0, multipliers=mult) == pytest.approx(4.0 + 0.0985 * 0.8 * 2.0, abs=1e-12)  # drift up, U_rise only
    assert _step(9.0, multipliers=mult) == pytest.approx(9.0 - 0.0985 * energy.drop(9.0) * 3.0, abs=1e-12)
    cost = 0.375 * energy.drop(6.5) * 3.0
    assert _step(6.5, messages=[HER], multipliers=mult) == pytest.approx(6.5 - cost, abs=1e-12)  # base cost: U_drop only


def test_mood_takes_both_u_and_v():
    mult = UsedTo(u_rise=1.2, v_lift=2.5, u_drop=0.9, v_drain=0.4)
    assert _step(6.5, short={"joy": 8.0}, high={"joy"}, multipliers=mult) == pytest.approx(6.5 + 0.39 * 1.2 * 2.5, abs=1e-12)
    assert _step(6.5, short={"grief": 6.0}, high={"grief"}, multipliers=mult) == pytest.approx(
        6.5 - 0.6 * 0.375 * energy.drop(6.5) * 0.9 * 0.4, abs=1e-12
    )


# ── A10, A27 the used-to multipliers ───────────────────────────────────────────────────────────────────────────────


def test_gate_closed_all_multipliers_are_one():
    assert energy.used_to(8.0, 1.0, {"grief": 9.0}, gate_open=False) == UsedTo()


def test_u_follows_the_long_average_once_the_gate_is_open():
    first = energy.used_to(8.0, 1.0, {}, gate_open=True)
    assert first.u_drop == pytest.approx(0.8125, abs=1e-12)
    assert first.u_rise == pytest.approx(1.2308, abs=5e-5)
    second = energy.used_to(5.0, 1.0, {}, gate_open=True)
    assert second.u_drop == pytest.approx(1.3, abs=1e-12)
    assert second.u_rise == pytest.approx(0.7692, abs=5e-5)


def test_gate_open_with_no_long_average_yet_u_is_one_and_the_unset_average_is_not_read():
    result = energy.used_to(float("nan"), 0.0, {}, gate_open=True)
    assert (result.u_rise, result.u_drop) == (1.0, 1.0)


def test_v_compares_the_drain_mean_with_the_lift_mean_and_floors_each_at_one():
    long = dict.fromkeys(dev_constants.BODY_ENERGY_DRAIN_FEELINGS, 5.0)
    long.update(dict.fromkeys(dev_constants.BODY_ENERGY_LIFT_FEELINGS, 2.0))
    result = energy.used_to(6.5, 1.0, long, gate_open=True)
    assert result.v_lift == pytest.approx(2.5, abs=1e-12)
    assert result.v_drain == pytest.approx(0.4, abs=1e-12)
    low = dict.fromkeys(dev_constants.BODY_ENERGY_DRAIN_FEELINGS, 0.5)
    low.update(dict.fromkeys(dev_constants.BODY_ENERGY_LIFT_FEELINGS, 2.0))
    assert energy.used_to(6.5, 1.0, low, gate_open=True).v_lift == pytest.approx(0.5, abs=1e-12)  # 1 / 2


def test_a_feeling_the_kindled_lacks_counts_zero_in_the_means():
    # Only grief present with long 8: the drain mean is 8 / 8 = 1 (the floor), the lift mean 0 floors to 1.
    assert energy.used_to(6.5, 1.0, {"grief": 8.0}, gate_open=True).v_lift == pytest.approx(1.0, abs=1e-12)
    # All eight drain feelings at 10 and nothing lifts: mean 10 / floor 1.
    full = dict.fromkeys(dev_constants.BODY_ENERGY_DRAIN_FEELINGS, 10.0)
    assert energy.used_to(6.5, 1.0, full, gate_open=True).v_lift == pytest.approx(10.0, abs=1e-12)


# ── the long-average fold and the caught-up heartbeats (A17) ────────────────────────────────────────────────────────


def _inputs(*, start=T0, short=None, long=None, high=(), messages=(), persona_start_at=None, heartbeat_s=900.0):
    return EnergyInputs(
        interval_start=start,
        heartbeat_s=heartbeat_s,
        short=short or {},
        long=long or {},
        high=frozenset(high),
        messages=tuple(messages),
        persona_start_at=persona_start_at,
    )


def _spy_beats(monkeypatch):
    calls = []
    real = energy._beat

    def spy(carry, *, inputs, at, hours, messages, x):
        calls.append({"at": at, "hours": hours, "messages": messages, "x": x})
        return real(carry, inputs=inputs, at=at, hours=hours, messages=messages, x=x)

    monkeypatch.setattr(energy, "_beat", spy)
    return calls


@pytest.mark.parametrize(
    ("elapsed_s", "caught_up"),
    [(0, 0), (-5000, 0), (600, 0), (899, 0), (905, 0), (1799, 0), (1800, 1), (2700, 2), (8 * 3600, 31)],
)
def test_the_caught_up_count_is_whole_intervals_less_the_current_heartbeat(monkeypatch, elapsed_s, caught_up):
    calls = _spy_beats(monkeypatch)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(),
        now=T0 + timedelta(seconds=elapsed_s),
    )
    assert len(calls) == caught_up + 1


@pytest.mark.parametrize(
    ("heartbeat_s", "elapsed_s", "caught_up"),
    [(450.0, 905, 1), (450.0, 1800, 3), (450.0, 2700, 5), (450.0, 449, 0), (1800.0, 2700, 0), (1800.0, 3600, 1)],
)
def test_the_caught_up_count_counts_closed_time_in_the_supervisors_own_heartbeat_length(
    monkeypatch, heartbeat_s, elapsed_s, caught_up
):
    """One value: the length handed in with the inputs (the supervisor's `heartbeat_interval_s`), not a copy of 900."""
    calls = _spy_beats(monkeypatch)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(heartbeat_s=heartbeat_s),
        now=T0 + timedelta(seconds=elapsed_s),
    )
    assert len(calls) == caught_up + 1
    for k, call in enumerate(calls[:-1], start=1):
        assert call["at"] == T0 + timedelta(seconds=k * heartbeat_s)
        assert call["hours"] == pytest.approx(heartbeat_s / 3600.0)


@pytest.mark.parametrize("heartbeat_s", [0.0, -900.0])
def test_a_heartbeat_length_of_zero_or_less_has_no_caught_up_heartbeats_and_no_division(monkeypatch, heartbeat_s):
    calls = _spy_beats(monkeypatch)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(heartbeat_s=heartbeat_s),
        now=T0 + timedelta(hours=8),
    )
    assert len(calls) == 1


def test_caught_up_heartbeats_have_no_messages_and_no_x_and_the_current_one_has_both(monkeypatch):
    calls = _spy_beats(monkeypatch)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings={"joy": 0.0},
        inputs=_inputs(short={"joy": 5.0}, messages=[HER]),
        now=T0 + timedelta(seconds=2700),
    )
    assert [c["messages"] for c in calls] == [(), (), (HER,)]
    assert [c["x"] for c in calls] == [0.0, 0.0, 0.5]


def test_each_caught_up_heartbeat_has_its_own_time_and_folds_at_one_heartbeat_of_ageing(monkeypatch):
    calls = _spy_beats(monkeypatch)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(),
        now=T0 + timedelta(seconds=2700 + 100),
    )
    assert [c["at"] for c in calls[:2]] == [T0 + timedelta(seconds=900), T0 + timedelta(seconds=1800)]
    assert [c["hours"] for c in calls[:2]] == [0.25, 0.25]
    assert calls[2]["hours"] == pytest.approx((2800 - 1800) / 3600.0, abs=1e-12)


def test_the_gate_uses_each_heartbeats_own_time(monkeypatch):
    seen = []
    real = energy.used_to

    def spy(long_avg, long_weight, long, gate_open):
        seen.append(gate_open)
        return real(long_avg, long_weight, long, gate_open)

    monkeypatch.setattr(energy, "used_to", spy)
    # 30 days old at +1800 s: closed at +900 s, open at +1800 s (the caught-up heartbeat) and at now.
    start = T0 - timedelta(days=30) + timedelta(seconds=1000)
    energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(persona_start_at=start),
        now=T0 + timedelta(seconds=2700),
    )
    assert seen == [False, True, True]


def test_a_gap_over_six_hours_is_not_skipped_and_there_is_no_cap(monkeypatch):
    # Drift only from 1.0: 8 hours is 31 caught-up heartbeats plus the current one: 32 drift steps.
    result = energy.advance_energy(
        EnergyCarry(1.0, 0.0, 0.0),
        prev_feelings=None,
        inputs=_inputs(),
        now=T0 + timedelta(hours=8),
    )
    assert result.carry.level == pytest.approx(_drift_steps(1.0, 32), abs=1e-12)


def test_closed_time_mood_acts_in_every_caught_up_heartbeat_and_can_lower_her():
    # Grief high all the time she was away: each heartbeat drains, the drift only returns her to the baseline.
    inputs = _inputs(short={"grief": 9.0}, high={"grief"})
    result = energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0), prev_feelings=None, inputs=inputs, now=T0 + timedelta(hours=6)
    )
    assert result.carry.level < 6.5


def test_the_first_fold_sets_the_average_to_the_level_and_the_weight_to_one():
    result = energy.advance_energy(EnergyCarry(6.5, 0.0, 0.0), prev_feelings=None, inputs=_inputs(), now=T0 + timedelta(seconds=901))
    assert result.carry.long_avg == pytest.approx(result.carry.level, abs=1e-12)
    assert result.carry.long_weight == 1.0


def test_a_later_fold_ages_the_weight_by_the_engines_decay_in_hours():
    half_life_hours = dev_constants.EMOTION_LONG_HALF_LIFE_D * 24.0
    result = energy.advance_energy(
        EnergyCarry(6.5, 6.5, 2.0), prev_feelings=None, inputs=_inputs(), now=T0 + timedelta(seconds=900)
    )
    # elapsed 900 s: one whole interval less the current heartbeat = 0 caught up; the average is aged 0.25 h and folded.
    assert result.carry.long_weight == pytest.approx(2.0 * engine_utils.decay(0.25, half_life_hours) + 1.0, abs=1e-12)


def test_the_ageing_never_grows_the_weight_when_the_clock_stepped_back():
    result = energy.advance_energy(
        EnergyCarry(6.5, 6.5, 2.0), prev_feelings=None, inputs=_inputs(), now=T0 - timedelta(hours=5)
    )
    assert result.carry.long_weight == pytest.approx(3.0, abs=1e-12)


def test_the_saved_feelings_are_the_lift_and_drain_short_averages_only():
    short = {"joy": 3.5, "grief": 2.25, "peace": 8.0, "anger": 1.0}
    result = energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0), prev_feelings=None, inputs=_inputs(short=short), now=T0 + timedelta(seconds=901)
    )
    assert result.feelings == {"joy": 3.5, "grief": 2.25}


def test_nothing_readable_leaves_the_saved_feelings_alone_and_x_is_zero():
    result = energy.advance_energy(
        EnergyCarry(6.5, 0.0, 0.0), prev_feelings={"joy": 9.0}, inputs=_inputs(messages=[HER]), now=T0 + timedelta(seconds=901)
    )
    assert result.feelings is None
    assert result.carry.level == pytest.approx(6.5 - 0.28846, abs=5e-6)


def test_no_energy_code_reads_the_change_log_or_the_shift_and_difference():
    import inspect

    from brain.body import energy_inputs

    for module in (energy, energy_inputs):
        source = inspect.getsource(module)
        for banned in ("read_changes", ".shift", ".difference", "BODY_ENERGY_HIGH_COUNT"):
            assert banned not in source, (module.__name__, banned)
