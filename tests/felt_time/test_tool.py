import pytest

from brain.felt_time import FeltTime, TickContext
from brain.felt_time.lived_age import IntensityDrivers
from brain.felt_time.tool import felt_time_now, pressure_since


@pytest.fixture
def primed_persona(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(
        TickContext(
            now_iso="2026-05-17T22:00:00+00:00",
            heartbeats_in_tick=5,
            chat_turns_in_tick=2,
            reflex_firings_in_tick=1,
            wall_clock_s_in_tick=900.0,
            drivers=IntensityDrivers(),
        )
    )
    return tmp_path


def test_felt_time_now_returns_full_state(primed_persona):
    result = felt_time_now(persona_dir=primed_persona)
    assert "lived_age_hours" in result
    assert "anchors" in result
    assert "pressure_since_last_anchor" in result
    assert result["lived_age_hours"] > 0.0


def test_pressure_since_returns_per_anchor_vector(primed_persona):
    result = pressure_since(arguments={"anchor_type": "dream"}, persona_dir=primed_persona)
    assert "heartbeats" in result
    assert "chat_turns" in result
    assert "reflex_firings" in result
    assert "wall_clock_s" in result


def test_pressure_since_rejects_bad_anchor_type(primed_persona):
    with pytest.raises(ValueError):
        pressure_since(arguments={"anchor_type": "garbage"}, persona_dir=primed_persona)


def test_felt_time_now_cold_start_returns_null_anchors(tmp_path):
    result = felt_time_now(persona_dir=tmp_path)
    assert result["lived_age_hours"] == 0.0
    assert result["anchors"] == {}
