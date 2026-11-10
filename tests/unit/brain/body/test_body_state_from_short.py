"""`body_state_from_short`: the body state built from the engine's short averages (B2 phase B, step 01)."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.body.state import BODY_EMOTION_NAMES, body_state_from_short, compute_body_state

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
INPUTS = {"session_hours": 1.5, "words_written": 3000, "days_since_contact": 2.0, "now": NOW}
FRESH = {"session_hours": 0.0, "words_written": 0, "days_since_contact": 0.0, "now": NOW}


def _none_high(name: str) -> bool:
    return False


def _old_levels(short: dict[str, float]):
    """A predicate that says "high" exactly where `compute_body_state` fires on its fixed levels."""
    levels = {"belonging": 8.0, "love": 8.0, "climax": 5.0}
    return lambda name: short.get(name, 0.0) >= levels.get(name, 7.0)


def test_projections_match_compute_body_state_when_high_is_where_the_old_levels_fire():
    short = {"arousal": 6.0, "love": 8.0, "rest_need": 7.5, "peace": 2.0, "desire": 7.0}
    from_short = body_state_from_short(short, feels_high=_old_levels(short), **INPUTS)
    reference = compute_body_state(emotions=short, **INPUTS)
    assert (from_short.energy, from_short.temperature, from_short.exhaustion) == (
        reference.energy,
        reference.temperature,
        reference.exhaustion,
    )
    assert from_short.computed_at == NOW


def test_body_emotions_absent_from_short_are_absent_from_the_state():
    short = {"arousal": 6.0, "love": 8.0}
    body = body_state_from_short(short, feels_high=_none_high, **INPUTS)
    assert set(body.body_emotions) == {"arousal"}
    # The projection that fills six names with 0.0 is the one the helper replaces.
    assert set(compute_body_state(emotions=short, **INPUTS).body_emotions) == BODY_EMOTION_NAMES


def test_a_present_zero_is_kept():
    body = body_state_from_short(
        {"touch_hunger": 0.0, "desire": 3.0}, feels_high=_none_high, **INPUTS
    )
    assert body.body_emotions == {"touch_hunger": 0.0, "desire": 3.0}


def test_an_empty_short_gives_baseline_projections_and_no_body_emotions():
    body = body_state_from_short({}, feels_high=_none_high, **INPUTS)
    assert body.body_emotions == {}
    assert "body_emotions" in body.to_dict()
    assert body.to_dict()["body_emotions"] == {}
    reference = compute_body_state(emotions={}, **INPUTS)
    assert (body.energy, body.temperature, body.exhaustion) == (
        reference.energy,
        reference.temperature,
        reference.exhaustion,
    )


# --- the predicate decides, not a fixed level (S53) -----------------------------------------------------------------

BASE_ENERGY = body_state_from_short({}, feels_high=_none_high, **FRESH).energy  # 8
BASE_TEMPERATURE = body_state_from_short({}, feels_high=_none_high, **FRESH).temperature  # 4


def _high(*names: str):
    return lambda name: name in names


def _body(short: dict[str, float], *high: str):
    return body_state_from_short(short, feels_high=_high(*high), **FRESH)


def test_more_than_six_high_feelings_lower_energy_by_one_and_six_do_not():
    names = [f"feeling_{i}" for i in range(7)]
    short = dict.fromkeys(names, 0.5)
    assert _body(short, *names[:6]).energy == BASE_ENERGY
    assert _body(short, *names).energy == BASE_ENERGY - 1


def test_the_high_feeling_count_is_one_dev_constant_read_at_call_time_on_both_paths(monkeypatch):
    """The count that lowers energy ("more than six") is `dev_constants.BODY_ENERGY_HIGH_COUNT`, used by the engine path
    (`body_state_from_short`) and the old path (`compute_body_state`) alike: lowering it to 2 makes three high feelings enough."""
    from brain import dev_constants

    names = [f"feeling_{i}" for i in range(3)]
    short = dict.fromkeys(names, 0.5)
    old_short = dict.fromkeys(names, 7.5)  # at the old path's fixed level of 7.0
    assert dev_constants.BODY_ENERGY_HIGH_COUNT == 6
    assert _body(short, *names).energy == BASE_ENERGY
    assert compute_body_state(emotions=old_short, **FRESH).energy == BASE_ENERGY
    monkeypatch.setattr(dev_constants, "BODY_ENERGY_HIGH_COUNT", 2)
    assert _body(short, *names).energy == BASE_ENERGY - 1
    assert compute_body_state(emotions=old_short, **FRESH).energy == BASE_ENERGY - 1


def test_all_values_at_the_top_but_nothing_high_leaves_energy_and_temperature_at_baseline():
    names = [f"feeling_{i}" for i in range(8)] + [
        "rest_need",
        "arousal",
        "desire",
        "belonging",
        "love",
        "climax",
        "body_grief",
        "touch_hunger",
    ]
    body = _body(dict.fromkeys(names, 10.0))
    assert (body.energy, body.temperature) == (BASE_ENERGY, BASE_TEMPERATURE)


def test_rest_need_high_lowers_energy_by_one():
    assert _body({"rest_need": 0.5}, "rest_need").energy == BASE_ENERGY - 1


def test_peace_high_in_a_fresh_session_raises_energy_by_one_and_not_in_a_long_one():
    short = {"peace": 0.5}
    assert _body(short, "peace").energy == min(10, BASE_ENERGY + 1)
    long_session = {**FRESH, "session_hours": 1.0}
    assert (
        body_state_from_short(short, feels_high=_high("peace"), **long_session).energy
        == body_state_from_short(short, feels_high=_none_high, **long_session).energy
    )


def test_each_warming_feeling_that_is_high_raises_temperature_by_one():
    for name in ("arousal", "desire", "belonging", "love", "climax"):
        assert _body({name: 0.5}, name).temperature == BASE_TEMPERATURE + 1, name


def test_each_cooling_feeling_that_is_high_lowers_temperature_by_one():
    for name in ("body_grief", "touch_hunger"):
        assert _body({name: 0.5}, name).temperature == BASE_TEMPERATURE - 1, name


def test_a_feeling_absent_from_short_is_not_asked_to_be_high_for_its_body_value():
    body = _body(
        {"desire": 0.5}, "desire", "love"
    )  # love is "high" but not in short: only desire is a body emotion read
    assert set(body.body_emotions) == {"desire"}
