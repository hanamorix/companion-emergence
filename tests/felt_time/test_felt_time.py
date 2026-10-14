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


def test_load_or_recover_keeps_saved_state_when_logs_newer_than_state(tmp_path):
    from brain.felt_time.state import persist

    # Persist a valid saved state with real lived age.
    persist(
        FeltTimeState(lived_age_hours=1.25, last_tick_ts="2026-05-17T18:00:00+00:00"),
        tmp_path,
    )
    # A dream is logged after the last tick (the normal case between ticks).
    _write_jsonl(
        tmp_path / "dreams.log.jsonl",
        [{"ts": "2026-05-17T20:00:00+00:00", "summary": "later"}],
    )

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False  # saved state is authoritative; no replay
    assert loaded.lived_age_hours == 1.25
    # The new anchor is folded in by the next tick, not at load time (OQ1).
    assert "dream" not in loaded.anchors

    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(
        TickContext(
            now_iso="2026-05-17T21:00:00+00:00",
            heartbeats_in_tick=1,
            chat_turns_in_tick=0,
            reflex_firings_in_tick=0,
            wall_clock_s_in_tick=3600.0,
            drivers=IntensityDrivers(),
        )
    )
    assert ft.get_state().anchors["dream"].label == "later"


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


# ---------------------------------------------------------------------------
# Lived age survives anchor events logged between ticks (#384)
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from brain.felt_time.state import (  # noqa: E402
    Anchor,
    HorizonBucket,
    PressureCounters,
    persist,
)

# source name -> (log file, label key)
_EVENT_SOURCES = {
    "dream": ("dreams.log.jsonl", "summary"),
    "growth": ("growth.log.jsonl", "title"),
    "soul": ("soul.log.jsonl", "moment_label"),
    "weather_shift": ("weather_shifts.log.jsonl", "label"),
}
_T0 = "2026-10-01T10:00:00+00:00"  # last tick of the saved state


def _append_event(persona_dir: Path, kind: str, label: str, ts: str) -> None:
    filename, label_key = _EVENT_SOURCES[kind]
    with (persona_dir / filename).open("a") as f:
        f.write(json.dumps({"ts": ts, label_key: label}) + "\n")


def _rich_saved_state(persona_dir: Path, *, saved_only: str) -> FeltTimeState:
    """A valid persisted state holding values the log replay cannot reproduce."""
    state = FeltTimeState(
        lived_age_hours=3.5,
        anchors={
            saved_only: Anchor(
                type=saved_only,
                ts="2026-09-30T08:00:00+00:00",
                label="saved only",
                source_ref=f"{saved_only}.log.jsonl:99",
            )
        },
        pressure=PressureCounters(heartbeats=5, chat_turns=2, reflex_firings=1, wall_clock_s=4500.0),
        last_tick_ts=_T0,
        first_tick_ts="2026-09-25T10:00:00+00:00",
        weather_baselines={"joy": {"mean": 0.4}},
        replayed=False,
        horizon_pressure={
            "week": HorizonBucket(
                counters=PressureCounters(heartbeats=7),
                period_start_ts="2026-09-28T00:00:00+00:00",
            )
        },
    )
    persist(state, persona_dir)
    return state


def _saved_only_for(kind: str) -> str:
    return "growth" if kind == "soul" else "soul"


def test_lived_age_survives_dream_logged_between_ticks(tmp_path):
    """The #384 repro: 5 ticks -> 1.25 h; a dream is logged; the next FeltTime()
    must still read 1.25 h (was 0.0 with recovered_from_logs=True)."""
    from datetime import datetime, timedelta

    start = datetime.fromisoformat("2026-10-01T10:00:00+00:00")
    for i in range(1, 6):  # five 15-minute ticks, a fresh FeltTime each time (as the supervisor does)
        ft = FeltTime(persona_dir=tmp_path)
        ft.tick(_neutral_ctx((start + timedelta(minutes=15 * i)).isoformat()))
    assert ft.get_state().lived_age_hours == pytest.approx(1.25)
    last_tick = ft.get_state().last_tick_ts

    assert last_tick == "2026-10-01T11:15:00+00:00"
    _append_event(tmp_path, "dream", "x", "2026-10-01T11:20:00+00:00")

    ft2 = FeltTime(persona_dir=tmp_path)
    assert ft2.get_state().lived_age_hours == pytest.approx(1.25)
    assert ft2.recovered_from_logs is False

    ft2.tick(_neutral_ctx("2026-10-01T11:30:00+00:00"))
    assert ft2.get_state().lived_age_hours > 1.25


@pytest.mark.parametrize("kind", ["dream", "growth", "soul", "weather_shift"])
def test_saved_state_preserved_across_anchor_event(tmp_path, kind):
    saved = _rich_saved_state(tmp_path, saved_only=_saved_only_for(kind))
    _append_event(tmp_path, kind, "new event", "2026-10-01T10:05:00+00:00")

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.replayed is False
    assert loaded.lived_age_hours == saved.lived_age_hours
    assert loaded.first_tick_ts == saved.first_tick_ts
    assert loaded.last_tick_ts == saved.last_tick_ts
    assert loaded.pressure == saved.pressure
    assert loaded.weather_baselines == saved.weather_baselines
    assert loaded.horizon_pressure == saved.horizon_pressure
    assert loaded.anchors == saved.anchors
    assert loaded.arc_anchors == saved.arc_anchors


def test_saved_state_preserved_across_arc_event_control(tmp_path):
    """arcs.log.jsonl is not in LOG_SOURCES, so this never reset; control for C2."""
    saved = _rich_saved_state(tmp_path, saved_only="soul")
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-10-01T10:05:00+00:00")
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.lived_age_hours == saved.lived_age_hours
    assert loaded.pressure == saved.pressure
    assert loaded.anchors == saved.anchors


@pytest.mark.parametrize("kind", ["dream", "growth", "soul", "weather_shift"])
def test_tick_folds_newer_anchor_into_preserved_state(tmp_path, kind):
    saved_only = _saved_only_for(kind)
    saved = _rich_saved_state(tmp_path, saved_only=saved_only)
    _append_event(tmp_path, kind, "new event", "2026-10-01T10:05:00+00:00")

    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T11:00:00+00:00"))  # dt = 1 h, quiet drivers
    state = ft.get_state()

    assert state.lived_age_hours == pytest.approx(saved.lived_age_hours + 1.0)
    assert state.anchors[kind].label == "new event"
    assert state.anchors[kind].ts == "2026-10-01T10:05:00+00:00"
    assert state.anchors[saved_only] == saved.anchors[saved_only]  # not discarded
    assert state.pressure == PressureCounters()  # new anchor zeroes pressure
    assert state.weather_baselines == saved.weather_baselines
    assert state.first_tick_ts == saved.first_tick_ts
    assert state.last_tick_ts == "2026-10-01T11:00:00+00:00"
    assert state.replayed is False

    persisted = json.loads((tmp_path / "felt_time_state.json").read_text())
    assert persisted["lived_age_hours"] == pytest.approx(saved.lived_age_hours + 1.0)
    assert persisted["anchors"][kind]["label"] == "new event"


def test_tick_folds_newer_arc_event_exactly_once(tmp_path):
    _rich_saved_state(tmp_path, saved_only="soul")
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-10-01T10:30:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T11:00:00+00:00"))
    state = ft.get_state()
    assert [a.label for a in state.arc_anchors] == ["Thread A"]
    assert state.anchors["arc"] == state.arc_anchors[-1]
    ft.tick(_neutral_ctx("2026-10-01T11:15:00+00:00"))
    assert len(ft.get_state().arc_anchors) == 1


def test_repeated_loads_between_ticks_are_stable_control(tmp_path):
    """C8 control: constructing FeltTime twice before the tick does not double-fold."""

    def run(root: Path, loads: int):
        root.mkdir()
        _rich_saved_state(root, saved_only="soul")
        _append_event(root, "dream", "new event", "2026-10-01T10:05:00+00:00")
        _write_arc_event(root, "arc_opened", "Thread A", "2026-10-01T10:30:00+00:00")
        for _ in range(loads):
            ft = FeltTime(persona_dir=root)
        ft.tick(_neutral_ctx("2026-10-01T11:00:00+00:00"))
        ft.tick(_neutral_ctx("2026-10-01T11:15:00+00:00"))
        return ft.get_state()

    one = run(tmp_path / "one", 1)
    two = run(tmp_path / "two", 2)
    assert two.lived_age_hours == one.lived_age_hours
    assert two.anchors == one.anchors
    assert two.arc_anchors == one.arc_anchors
    assert len(two.arc_anchors) == 1


def test_arc_opened_with_seed_time_stamp_is_folded_once(tmp_path):
    """C11: arc_opened is stamped with the SEED anchor's time (<= last tick), so a
    timestamp scan never sees it. It must still reach arc_anchors, exactly once,
    without resetting pressure."""
    _append_event(tmp_path, "dream", "the seed", "2026-10-01T10:00:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:05:00+00:00"))
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:15:00+00:00"))
    assert ft.get_state().pressure.heartbeats == 1  # accumulating, no anchor this tick

    # Narrative pass opens the arc later, stamped with the seed's own time.
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-10-01T10:00:00+00:00")

    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:20:00+00:00"))
    state = ft.get_state()
    assert [a.label for a in state.arc_anchors] == ["Thread A"]
    assert state.anchors["arc"] == state.arc_anchors[-1]
    assert state.pressure.heartbeats == 2  # NOT reset by the late arc fold-in

    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:35:00+00:00"))
    assert len(ft.get_state().arc_anchors) == 1
    assert ft.get_state().pressure.heartbeats == 3


def test_arc_fold_in_from_previously_persisted_state_adds_only_unfolded(tmp_path):
    """C11b: state saved by earlier code already holds arc anchors 1 and 2; a third
    stale-stamped arc_opened is added, the first two are not re-added."""
    _write_arc_event(tmp_path, "arc_opened", "Thread A", "2026-09-20T10:00:00+00:00")
    _write_arc_event(tmp_path, "arc_closed", "Thread A", "2026-09-21T10:00:00+00:00")
    folded = [
        Anchor("arc", "2026-09-20T10:00:00+00:00", "Thread A", "arcs.log.jsonl:1", "arc_opened"),
        Anchor("arc", "2026-09-21T10:00:00+00:00", "Thread A", "arcs.log.jsonl:2", "arc_closed"),
    ]
    persist(
        FeltTimeState(
            lived_age_hours=2.0, last_tick_ts=_T0, arc_anchors=folded, anchors={"arc": folded[-1]}
        ),
        tmp_path,
    )
    _write_arc_event(tmp_path, "arc_opened", "Thread B", "2026-09-20T09:00:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:15:00+00:00"))
    labels = [(a.label, a.event_type) for a in ft.get_state().arc_anchors]
    assert labels == [
        ("Thread A", "arc_opened"),
        ("Thread A", "arc_closed"),
        ("Thread B", "arc_opened"),
    ]


def test_arc_cap_is_stable_and_idempotent_across_ticks(tmp_path):
    """C12: 25 arc events; first tick keeps the 20 most recent; later ticks do not
    re-add the 5 evicted ones."""
    for i in range(25):
        _write_arc_event(
            tmp_path, "arc_opened", f"Thread {i:02d}", f"2026-09-{i + 1:02d}T10:00:00+00:00"
        )
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:00:00+00:00"))
    first = [a.label for a in ft.get_state().arc_anchors]
    assert first == [f"Thread {i:02d}" for i in range(5, 25)]
    for minute in ("10:15", "10:30"):
        ft = FeltTime(persona_dir=tmp_path)
        ft.tick(_neutral_ctx(f"2026-10-01T{minute}:00+00:00"))
        assert [a.label for a in ft.get_state().arc_anchors] == first


def test_event_stamped_exactly_at_last_tick_is_not_folded_again(tmp_path):
    """The anchor scan is strictly newer than the last tick: a row stamped at the
    last tick was already seen by that tick and must not reset pressure again."""
    _append_event(tmp_path, "dream", "x", "2026-10-01T10:05:00+00:00")
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:05:00+00:00"))  # folds the dream, zeroes pressure
    assert ft.get_state().pressure.heartbeats == 0
    ft = FeltTime(persona_dir=tmp_path)
    ft.tick(_neutral_ctx("2026-10-01T10:20:00+00:00"))
    assert ft.get_state().pressure.heartbeats == 1
