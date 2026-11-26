"""`body_state_from_short`: the body state built from the engine's short averages (B2 phase B, step 01)."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.body.state import body_state_from_short

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
INPUTS = {"session_hours": 1.5, "energy": 4.0, "days_since_contact": 2.0, "now": NOW}
FRESH = {"session_hours": 0.0, "energy": 6.5, "days_since_contact": 0.0, "now": NOW}


def _none_high(name: str) -> bool:
    return False


def test_the_projections_combine_the_given_energy_the_predicate_and_the_contact_rules():
    """Pinned values: the saved level 4.0 comes out as given; love and desire high (two warming feelings) and 2 days
    since contact give temperature 6, and exhaustion follows the level (7 - 4 = 3)."""
    short = {"arousal": 6.0, "love": 8.0, "rest_need": 7.5, "peace": 2.0, "desire": 7.0}
    body = body_state_from_short(short, feels_high=lambda name: name in ("love", "desire", "rest_need"), **INPUTS)
    assert (body.energy, body.temperature, body.exhaustion) == (4.0, 6, 3)
    assert body.computed_at == NOW


def test_body_emotions_absent_from_short_are_absent_from_the_state():
    short = {"arousal": 6.0, "love": 8.0}
    body = body_state_from_short(short, feels_high=_none_high, **INPUTS)
    assert set(body.body_emotions) == {"arousal"}
    # The body emotions are exactly these six names; a feeling outside the set is never copied in.
    six = {"arousal": 1.0, "desire": 2.0, "climax": 3.0, "touch_hunger": 4.0, "comfort_seeking": 5.0, "rest_need": 6.0}
    assert body_state_from_short({**six, "love": 9.0}, feels_high=_none_high, **INPUTS).body_emotions == six


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
    assert (body.energy, body.temperature, body.exhaustion) == (4.0, 4, 3)  # pinned, as above


# --- the predicate decides the temperature, the saved level is the energy (S53, #650) --------------------------------

BASE_ENERGY = body_state_from_short({}, feels_high=_none_high, **FRESH).energy  # 6.5, the level handed in
BASE_TEMPERATURE = body_state_from_short({}, feels_high=_none_high, **FRESH).temperature  # 4


def _high(*names: str):
    return lambda name: name in names


def _body(short: dict[str, float], *high: str):
    return body_state_from_short(short, feels_high=_high(*high), **FRESH)


def test_no_feeling_being_high_changes_the_energy_it_is_given():
    """No count of high feelings, no rest need, no peace and no session length move the energy: it is the saved level
    (A11: nothing here counts how many feelings are high for energy)."""
    names = [f"feeling_{i}" for i in range(9)] + ["rest_need", "peace"]
    short = dict.fromkeys(names, 9.0)
    assert _body(short, *names).energy == BASE_ENERGY
    assert _body({"rest_need": 0.5}, "rest_need").energy == BASE_ENERGY
    assert _body({"peace": 0.5}, "peace").energy == BASE_ENERGY
    long_session = {**FRESH, "session_hours": 9.0}
    assert body_state_from_short({"peace": 0.5}, feels_high=_high("peace"), **long_session).energy == BASE_ENERGY
    assert body_state_from_short({}, feels_high=_none_high, **{**FRESH, "energy": 3.25}).energy == 3.25


def test_all_values_at_the_top_but_nothing_high_leaves_energy_and_temperature_at_baseline():
    names = [f"feeling_{i}" for i in range(8)] + [
        "rest_need",
        "arousal",
        "desire",
        "belonging",
        "love",
        "climax",
        "touch_hunger",
    ]
    body = _body(dict.fromkeys(names, 10.0))
    assert (body.energy, body.temperature) == (BASE_ENERGY, BASE_TEMPERATURE)


def test_each_warming_feeling_that_is_high_raises_temperature_by_one():
    for name in ("arousal", "desire", "belonging", "love", "climax"):
        assert _body({name: 0.5}, name).temperature == BASE_TEMPERATURE + 1, name


def test_each_cooling_feeling_that_is_high_lowers_temperature_by_one():
    assert _body({"touch_hunger": 0.5}, "touch_hunger").temperature == BASE_TEMPERATURE - 1


def test_body_grief_high_does_not_change_temperature():
    """`body_grief` is not a predefined feeling (#652), so it has no temperature rule: high or not, it moves nothing."""
    assert _body({"body_grief": 0.5}, "body_grief").temperature == BASE_TEMPERATURE
    assert _body({"body_grief": 0.5}, "body_grief", "touch_hunger").temperature == BASE_TEMPERATURE - 1


def test_a_feeling_absent_from_short_is_not_asked_to_be_high_for_its_body_value():
    body = _body(
        {"desire": 0.5}, "desire", "love"
    )  # love is "high" but not in short: only desire is a body emotion read
    assert set(body.body_emotions) == {"desire"}
