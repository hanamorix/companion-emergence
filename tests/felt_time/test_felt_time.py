"""Integration tests for FeltTime orchestrator + log replay.

Task 6.1, spec §2 __init__.py + §3 recovery model.
"""

import json
from pathlib import Path

from brain.felt_time import FeltTime, TickContext
from brain.felt_time.lived_age import IntensityDrivers
from brain.felt_time.state import FeltTimeState, load_or_recover


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def test_felt_time_first_tick_cold_start_advances_lived_age(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(
        TickContext(
            now_iso="2026-05-17T22:00:00+00:00",
            heartbeats_in_tick=1,
            chat_turns_in_tick=0,
            reflex_firings_in_tick=0,
            wall_clock_s_in_tick=900.0,
            drivers=IntensityDrivers(),
        )
    )
    state = ft.get_state()
    assert state.lived_age_hours > 0.0
    assert state.last_tick_ts == "2026-05-17T22:00:00+00:00"
    # State should be persisted to disk.
    assert (tmp_path / "felt_time_state.json").exists()


def test_felt_time_tick_picks_up_new_anchor(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [{"ts": "2026-05-17T21:00:00+00:00", "summary": "the boat one"}],
    )
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(
        TickContext(
            now_iso="2026-05-17T22:00:00+00:00",
            heartbeats_in_tick=1,
            chat_turns_in_tick=0,
            reflex_firings_in_tick=0,
            wall_clock_s_in_tick=3600.0,
            drivers=IntensityDrivers(),
        )
    )
    state = ft.get_state()
    assert "dream" in state.anchors
    assert state.anchors["dream"].label == "the boat one"
    # Anchor in tick → pressure counters reset to zero.
    assert state.pressure.heartbeats == 0


def test_felt_time_replay_rebuilds_state_from_logs(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [
            {"ts": "2026-05-17T10:00:00+00:00", "summary": "first dream"},
            {"ts": "2026-05-17T18:00:00+00:00", "summary": "second dream"},
        ],
    )
    _write_jsonl(
        tmp_path / "growth.log.jsonl",
        [{"ts": "2026-05-17T15:00:00+00:00", "title": "first growth"}],
    )

    ft = FeltTime.from_logs(persona_dir=tmp_path)
    state = ft.get_state()
    assert state.anchors["dream"].label == "second dream"  # latest wins
    assert state.anchors["growth"].label == "first growth"
    # No tick has run yet, lived_age stays at 0.
    assert state.lived_age_hours == 0.0


def test_load_or_recover_replays_when_logs_newer_than_state(tmp_path):
    from brain.felt_time.state import persist

    # Persist an old state.
    persist(FeltTimeState(last_tick_ts="2026-05-17T18:00:00+00:00"), tmp_path)
    # Now write a newer dream.
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [{"ts": "2026-05-17T20:00:00+00:00", "summary": "later"}],
    )

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.anchors["dream"].label == "later"


def test_replay_marks_state_as_replayed(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [{"ts": "2026-05-17T20:00:00+00:00", "summary": "test dream"}],
    )
    ft = FeltTime.from_logs(persona_dir=tmp_path)
    assert ft.get_state().replayed is True


def test_tick_clears_replayed_flag(tmp_path):
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [{"ts": "2026-05-17T20:00:00+00:00", "summary": "test dream"}],
    )
    ft = FeltTime.from_logs(persona_dir=tmp_path)
    assert ft.get_state().replayed is True  # precondition

    ft.tick(
        TickContext(
            now_iso="2026-05-17T22:00:00+00:00",
            heartbeats_in_tick=1,
            chat_turns_in_tick=0,
            reflex_firings_in_tick=0,
            wall_clock_s_in_tick=900.0,
            drivers=IntensityDrivers(),
        )
    )
    assert ft.get_state().replayed is False
    # Persisted state also has replayed=False so the bridge flag clears.
    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    assert data["replayed"] is False
