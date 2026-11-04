"""`body_state_from_short`: the body state built from the engine's short averages (B2 phase B, step 01)."""

from __future__ import annotations

from datetime import UTC, datetime

from brain.body.state import BODY_EMOTION_NAMES, body_state_from_short, compute_body_state

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
INPUTS = {"session_hours": 1.5, "words_written": 3000, "days_since_contact": 2.0, "now": NOW}


def test_projections_match_compute_body_state_for_the_same_map():
    short = {"arousal": 6.0, "love": 8.0, "rest_need": 7.5, "peace": 2.0, "desire": 7.0}
    from_short = body_state_from_short(short, **INPUTS)
    reference = compute_body_state(emotions=short, **INPUTS)
    assert (from_short.energy, from_short.temperature, from_short.exhaustion) == (
        reference.energy,
        reference.temperature,
        reference.exhaustion,
    )
    assert from_short.computed_at == NOW


def test_body_emotions_absent_from_short_are_absent_from_the_state():
    short = {"arousal": 6.0, "love": 8.0}
    body = body_state_from_short(short, **INPUTS)
    assert set(body.body_emotions) == {"arousal"}
    # The projection that fills six names with 0.0 is the one the helper replaces.
    assert set(compute_body_state(emotions=short, **INPUTS).body_emotions) == BODY_EMOTION_NAMES


def test_a_present_zero_is_kept():
    body = body_state_from_short({"touch_hunger": 0.0, "desire": 3.0}, **INPUTS)
    assert body.body_emotions == {"touch_hunger": 0.0, "desire": 3.0}


def test_an_empty_short_gives_baseline_projections_and_no_body_emotions():
    body = body_state_from_short({}, **INPUTS)
    assert body.body_emotions == {}
    assert "body_emotions" in body.to_dict()
    assert body.to_dict()["body_emotions"] == {}
    reference = compute_body_state(emotions={}, **INPUTS)
    assert (body.energy, body.temperature, body.exhaustion) == (
        reference.energy,
        reference.temperature,
        reference.exhaustion,
    )
