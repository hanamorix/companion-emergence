"""The body energy step inside `FeltTime.tick` (#650 spec 1.2, 1.10, 2.4): saved in felt time's single write."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import brain.felt_time as felt_time_module
from brain.body import energy
from brain.body.energy import EnergyInputs, EnergyMessage
from brain.felt_time import FeltTime, TickContext
from brain.felt_time.lived_age import IntensityDrivers
from brain.felt_time.state import FeltTimeState, load_or_recover, persist

T0 = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
HER = EnergyMessage(weight=1.0, length_factor=1.0)


def _inputs(start, *, short=None, messages=(), high=()):
    return EnergyInputs(
        interval_start=start,
        heartbeat_s=900.0,
        short=short or {},
        long={},
        high=frozenset(high),
        messages=tuple(messages),
        persona_start_at=None,
    )


def _ctx(now: datetime, energy: EnergyInputs | None, wall_s: float = 900.0) -> TickContext:
    return TickContext(
        now_iso=now.isoformat(),
        heartbeats_in_tick=1,
        chat_turns_in_tick=0,
        reflex_firings_in_tick=0,
        wall_clock_s_in_tick=wall_s,
        drivers=IntensityDrivers(),
        energy=energy,
    )


def _saved(persona_dir: Path) -> dict:
    return json.loads((persona_dir / "felt_time_state.json").read_text())


def test_a_fresh_kindled_and_a_file_from_before_the_fields_start_at_the_baseline(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    assert ft.get_state().energy_level == 6.5
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0))))
    assert ft.get_state().energy_level == 6.5  # no messages, at the baseline: the drift is 0
    # A file written before #650 has none of the four fields.
    path = tmp_path / "felt_time_state.json"
    data = json.loads(path.read_text())
    for key in ("energy_level", "energy_long_avg", "energy_long_weight", "energy_feelings"):
        data.pop(key, None)
    path.write_text(json.dumps(data))
    old = FeltTime(persona_dir=tmp_path).get_state()
    assert (old.energy_level, old.energy_long_weight, old.energy_feelings) == (6.5, 0.0, None)


def test_the_step_is_saved_with_the_tick_and_survives_a_reload(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    start = ft.interval_start(T0, 900.0)
    ft.tick(_ctx(T0, _inputs(start, short={"joy": 3.5, "peace": 8.0}, messages=[HER])))
    expected = 6.5 - 0.375 * (5.0 / 6.5)
    state = ft.get_state()
    assert state.energy_level == pytest.approx(expected, abs=1e-12)
    assert state.energy_long_weight == 1.0
    assert state.energy_long_avg == pytest.approx(expected, abs=1e-12)
    on_disk = _saved(tmp_path)
    assert on_disk["energy_level"] == pytest.approx(expected, abs=1e-12)  # unrounded
    assert on_disk["energy_feelings"] == {"joy": 3.5}  # lift and drain feelings only, not peace
    reloaded, _ = load_or_recover(tmp_path)
    assert reloaded.energy_level == state.energy_level
    assert reloaded.energy_feelings == {"joy": 3.5}


def test_the_whole_tick_is_one_write(tmp_path, monkeypatch):
    writes = []
    real = felt_time_module.persist
    monkeypatch.setattr(felt_time_module, "persist", lambda state, persona_dir: (writes.append(state), real(state, persona_dir))[1])
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0), short={"joy": 1.0}, messages=[HER])))
    assert len(writes) == 1
    assert writes[0].energy_level < 6.5 and writes[0].energy_feelings == {"joy": 1.0}


def test_the_saved_feelings_are_a_new_dict_and_the_old_one_is_not_mutated(tmp_path):
    old = {"joy": 1.0}
    persist(replace(FeltTimeState.cold_start(), energy_feelings=old), tmp_path)
    ft = FeltTime(persona_dir=tmp_path)
    before = ft.get_state().energy_feelings
    assert before == {"joy": 1.0}
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0), short={"joy": 4.0})))
    after = ft.get_state().energy_feelings
    assert after == {"joy": 4.0}
    assert after is not before
    assert before == {"joy": 1.0}


def test_the_next_heartbeats_x_comes_from_the_values_the_last_one_saved(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0), short={"joy": 2.0})))
    assert ft.get_state().energy_feelings == {"joy": 2.0}
    level = ft.get_state().energy_level
    later = T0 + timedelta(seconds=900)
    ft.tick(_ctx(later, _inputs(ft.interval_start(later, 900.0), short={"joy": 5.0}, messages=[HER])))
    # X = (5 - 2) / 10 = 0.3: the bracket 0.3 - 0.375 is a drop, funnelled at the start level.
    assert ft.get_state().energy_level == pytest.approx(level - 0.075 * (5.0 / level), abs=1e-9)


def test_a_heartbeat_with_no_messages_still_overwrites_the_saved_feelings(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0), short={"joy": 2.0, "grief": 1.0})))
    later = T0 + timedelta(seconds=900)
    ft.tick(_ctx(later, _inputs(ft.interval_start(later, 900.0), short={"joy": 7.0})))
    assert ft.get_state().energy_feelings == {"joy": 7.0}


def test_recovery_pending_keeps_the_saved_feelings_and_steps_without_x(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, _inputs(ft.interval_start(T0, 900.0), short={"joy": 2.0})))
    saved = ft.get_state().energy_feelings
    later = T0 + timedelta(seconds=900)
    ft.tick(_ctx(later, _inputs(ft.interval_start(later, 900.0), short={}, messages=[HER])))
    assert ft.get_state().energy_feelings == {"joy": 2.0}
    assert ft.get_state().energy_feelings is not saved  # a copy
    assert ft.get_state().energy_level < 6.5  # the base cost still acts


def test_closed_time_since_the_saved_tick_is_caught_up_in_the_same_tick(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, None))  # saves last_tick_ts
    persist(replace(ft.get_state(), energy_level=1.0), tmp_path)
    ft = FeltTime(persona_dir=tmp_path)
    later = T0 + timedelta(hours=8)
    start = ft.interval_start(later, 8 * 3600.0)
    assert start == T0
    ft.tick(_ctx(later, _inputs(start), wall_s=8 * 3600.0))
    expected = 1.0
    for _ in range(32):  # eight hours: 31 caught-up heartbeats and the current one, drift only
        expected = energy.step_heartbeat(
            expected, short={}, high=frozenset(), messages=(), x=0.0, multipliers=energy.UsedTo()
        )
    assert ft.get_state().energy_level == pytest.approx(expected, abs=1e-12)
    assert expected > 1.5


def test_the_interval_starts_at_the_saved_last_tick_or_wall_seconds_before_now(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    assert ft.interval_start(T0, 600.0) == T0 - timedelta(seconds=600)
    ft.tick(_ctx(T0, None))
    assert FeltTime(persona_dir=tmp_path).interval_start(T0 + timedelta(seconds=77), 5.0) == T0


def test_without_inputs_no_step_is_taken_and_the_fields_carry(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    persist(replace(ft.get_state(), energy_level=3.0, energy_feelings={"joy": 1.0}), tmp_path)
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_ctx(T0, None))
    assert ft.get_state().energy_level == 3.0
    assert ft.get_state().energy_feelings == {"joy": 1.0}


def test_the_replay_branch_is_not_changed_by_the_step():
    # A22: `_replay_from_logs` still builds its state from the dataclass defaults (the replay wipe is #385's, spec 1.7).
    import inspect

    assert "energy" not in inspect.getsource(felt_time_module._replay_from_logs)
