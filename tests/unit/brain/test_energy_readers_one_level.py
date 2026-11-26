"""Every reader of the body energy reads the one saved level (#650 spec 1.5, 4; acceptance A1 and A3).

One test writes the level into a synthetic persona's `felt_time_state.json` and asks all six readers (her prompt line, the
`get_body_state` tool, the panel data, felt-time strain, the self-model's unnamed pressure and the rest gate) what they
saw. The level is unrounded (6.326923...), so a reader that rounded early or recomputed from the feelings is caught. The
comparison cases run through the real readers: the rest gate on at 3.0 and off at 3.4, unnamed pressure on at 1.0 and off
at 1.4. No real persona is touched.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge import persona_state, supervisor
from brain.bridge.provider import FakeProvider
from brain.chat import prompt
from brain.emotion.engine import Reading
from brain.felt_time.state import FeltTimeState, persist
from brain.tools.impls.get_body_state import get_body_state

LEVEL = 6.326923076923077


class _Bus:
    def publish(self, event: object) -> None:
        pass


def _persona(tmp_path: Path, level: float) -> Path:
    persona = tmp_path / "persona"
    persona.mkdir(parents=True)
    (persona / "active_conversations").mkdir()
    (persona / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    persist(FeltTimeState(energy_level=level), persona)
    return persona


def _quiet_engine(stub) -> None:
    # Feelings that would have moved the old computed energy: loud, long, all high. None of them may reach the level.
    names = ("rest_need", "peace", "love", "joy", "grief", "desire")
    stub.readings = {n: Reading(short=9.0, long=1.0, difference=8.0, shift=0) for n in names}
    stub.high = set(names)


def test_all_six_readers_see_the_same_saved_level(tmp_path, monkeypatch, emotion_engine_stub):
    from brain.bridge import cli_throttle

    persona = _persona(tmp_path, LEVEL)
    _quiet_engine(emotion_engine_stub)

    seen: dict[str, float] = {}
    real_pressure = supervisor.compute_unnamed_pressure
    real_rest = supervisor._rest_state_from_energy

    def pressure(channels, *, body_energy):
        seen["self_model"] = body_energy
        return real_pressure(channels, body_energy=body_energy)

    def rest(energy):
        seen["rest_gate"] = energy
        return real_rest(energy)

    monkeypatch.setattr(supervisor, "compute_unnamed_pressure", pressure)
    monkeypatch.setattr(supervisor, "_rest_state_from_energy", rest)

    block = prompt._build_body_block(persona)
    match = re.search(r"energy: (\d+\.\d)/10", block)
    assert match is not None, block
    seen["prompt"] = float(match.group(1))

    seen["tool"] = get_body_state(store=MagicMock(), hebbian=MagicMock(), persona_dir=persona)["energy"]
    panel = persona_state._build_body(persona, now=datetime(2026, 11, 26, 12, 0, tzinfo=UTC))
    assert panel is not None
    seen["panel"] = panel["energy"]
    strain = supervisor._derive_intensity_drivers(persona, 0, 900.0).body_strain

    cli_throttle.reset()  # a fresh persona's tick is due; the throttle must not hold a chat spell from another test
    supervisor._run_self_model_tick(persona, provider=FakeProvider(), event_bus=_Bus())
    supervisor._run_initiate_review_tick(persona, FakeProvider(), _Bus())

    assert seen["prompt"] == seen["tool"] == seen["panel"] == 6.3  # to one decimal, for display only
    assert strain == pytest.approx(1.0 - LEVEL / 10.0, abs=1e-12)
    assert seen["self_model"] == LEVEL  # unrounded
    assert seen["rest_gate"] == LEVEL  # unrounded


def test_the_rest_gate_through_the_real_reader_is_on_at_3_and_off_at_3_4(tmp_path, monkeypatch, emotion_engine_stub):
    captured: dict[str, bool] = {}

    def fake_review_tick(persona_dir, *, provider, voice_template, cap_per_tick, user_presence, is_rest_state=False, should_pause=None, paused_out=None):
        captured["rest"] = is_rest_state

    monkeypatch.setattr(supervisor, "run_initiate_review_tick", fake_review_tick)
    monkeypatch.setattr(supervisor, "compute_user_presence", lambda persona_dir: None)
    answers = {}
    for index, level in enumerate((3.0, 3.4, 2.0, 6.5)):
        persona = _persona(tmp_path / f"case{index}", level)
        supervisor._run_initiate_review_tick(persona, FakeProvider(), _Bus())
        answers[level] = captured["rest"]
    assert answers == {3.0: True, 3.4: False, 2.0: True, 6.5: False}


def test_unnamed_pressure_through_the_real_tick_is_on_only_at_the_floor(tmp_path, monkeypatch, emotion_engine_stub):
    from brain.bridge import cli_throttle

    emotion_engine_stub.readings = {"joy": Reading(short=4.0, long=0.0, difference=0.0, shift=0)}  # no low-arousal channel
    got: list[float] = []
    real_pressure = supervisor.compute_unnamed_pressure

    def pressure(channels, *, body_energy):
        value = real_pressure(channels, body_energy=body_energy)
        got.append(value)
        return value

    monkeypatch.setattr(supervisor, "compute_unnamed_pressure", pressure)
    answers = {}
    for index, level in enumerate((1.0, 1.4, 1.5, 6.5)):
        persona = _persona(tmp_path / f"case{index}", level)
        cli_throttle.reset()
        got.clear()
        supervisor._run_self_model_tick(persona, provider=FakeProvider(), event_bus=_Bus())
        assert len(got) == 1
        answers[level] = got[0] > 0.0
    assert answers == {1.0: True, 1.4: False, 1.5: False, 6.5: False}


def test_no_reader_recomputes_the_energy_from_the_feelings_or_the_session(tmp_path, emotion_engine_stub):
    """The saved level is the energy whatever the feelings say: a read full of loud high feelings and a long session
    leave the prompt line and the tool at the saved 6.5."""
    persona = _persona(tmp_path, 6.5)
    _quiet_engine(emotion_engine_stub)
    assert "energy: 6.5/10" in prompt._build_body_block(persona)
    out = get_body_state(store=MagicMock(), hebbian=MagicMock(), persona_dir=persona, session_hours=12.0)
    assert out["energy"] == 6.5
