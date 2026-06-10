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


# ---------------------------------------------------------------------------
# Horizon bucket tests (Task 3)
# ---------------------------------------------------------------------------



def _neutral_ctx(now_iso: str, chat_turns: int = 0) -> TickContext:
    return TickContext(
        now_iso=now_iso,
        heartbeats_in_tick=1,
        chat_turns_in_tick=chat_turns,
        reflex_firings_in_tick=0,
        wall_clock_s_in_tick=900.0,
        drivers=IntensityDrivers(),
    )


def test_tick_populates_horizon_pressure(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00", chat_turns=3))
    state = ft.get_state()
    assert "week" in state.horizon_pressure
    assert "month" in state.horizon_pressure
    assert state.horizon_pressure["week"].counters.chat_turns == 3
    assert state.horizon_pressure["month"].counters.chat_turns == 3


def test_tick_horizon_persists_across_reload(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00", chat_turns=5))
    ft2 = FeltTime(persona_dir=tmp_path)
    assert ft2.get_state().horizon_pressure["week"].counters.chat_turns == 5


def test_tick_horizon_rollover_after_7_days(tmp_path):
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-01T10:00:00+00:00", chat_turns=10))
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00", chat_turns=2))
    state = ft.get_state()
    assert state.horizon_pressure["week"].prev_counters.chat_turns == 10
    assert state.horizon_pressure["week"].counters.chat_turns == 2


# ---------------------------------------------------------------------------
# Arc anchor accumulation tests (Task 4)
# ---------------------------------------------------------------------------


def _write_arc_event(persona_dir: Path, event: str, title: str, ts_iso: str) -> None:
    arc_log = persona_dir / "arcs.log.jsonl"
    with arc_log.open("a") as f:
        f.write(json.dumps({"event": event, "ts_iso": ts_iso, "title": title}) + "\n")


def test_tick_appends_arc_anchor(tmp_path):
    _write_arc_event(tmp_path, "arc_opened", "The Long Work", "2026-06-07T10:00:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00"))
    state = ft.get_state()
    assert len(state.arc_anchors) == 1
    assert state.arc_anchors[0].label == "The Long Work"
    assert state.arc_anchors[0].event_type == "arc_opened"


def test_tick_arc_anchor_syncs_anchors_dict(tmp_path):
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-06-07T10:00:00+00:00")
    _write_arc_event(tmp_path, "arc_opened", "Thread B", "2026-06-07T11:00:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00"))
    state = ft.get_state()
    assert len(state.arc_anchors) == 2
    assert state.anchors["arc"].ts == state.arc_anchors[-1].ts


def test_tick_arc_anchor_cap_at_20(tmp_path):
    for i in range(21):
        _write_arc_event(
            tmp_path, "arc_opened", f"Thread {i}",
            f"2026-06-0{(i % 7) + 1}T{i:02d}:00:00+00:00"
        )
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00"))
    state = ft.get_state()
    assert len(state.arc_anchors) == 20


def test_tick_arc_anchors_not_duplicated_on_second_tick(tmp_path):
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-06-07T10:00:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-06-08T10:00:00+00:00"))
    ft.tick(_neutral_ctx("2026-06-08T10:15:00+00:00"))  # no new arc events
    state = ft.get_state()
    assert len(state.arc_anchors) == 1  # not duplicated


def test_replay_seeds_arc_anchors_from_logs(tmp_path):
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-06-07T10:00:00+00:00")
    _write_arc_event(tmp_path, "arc_closed", "Thread A", "2026-06-08T10:00:00+00:00")
    ft = FeltTime.from_logs(persona_dir=tmp_path)
    state = ft.get_state()
    assert len(state.arc_anchors) == 2
    assert state.arc_anchors[0].event_type == "arc_opened"
    assert state.arc_anchors[1].event_type == "arc_closed"
