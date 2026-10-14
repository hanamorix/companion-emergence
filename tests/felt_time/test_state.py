import json

import pytest

from brain.felt_time.state import (
    Anchor,
    FeltTimeState,
    PressureCounters,
    load_or_recover,
    persist,
)


def test_felt_time_state_cold_start_defaults():
    s = FeltTimeState.cold_start()
    assert s.lived_age_hours == 0.0
    assert s.anchors == {}
    assert s.pressure == PressureCounters()
    assert s.last_tick_ts is None
    assert s.weather_baselines == {}
    assert s.replayed is False


def test_persist_writes_atomic_json(tmp_path):
    s = FeltTimeState(
        lived_age_hours=42.7,
        anchors={
            "dream": Anchor(
                type="dream",
                ts="2026-05-17T20:00:00+00:00",
                label="the boat one",
                source_ref="dreams.log.jsonl:42",
            )
        },
        pressure=PressureCounters(
            heartbeats=12, chat_turns=3, reflex_firings=1, wall_clock_s=5400.0
        ),
        last_tick_ts="2026-05-17T22:00:00+00:00",
    )

    persist(s, tmp_path)

    state_file = tmp_path / "felt_time_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["lived_age_hours"] == 42.7
    assert data["anchors"]["dream"]["label"] == "the boat one"
    assert data["pressure"]["heartbeats"] == 12
    assert data["last_tick_ts"] == "2026-05-17T22:00:00+00:00"
    # No leftover tmpfile
    assert not list(tmp_path.glob("felt_time_state.json.tmp*"))


def test_load_or_recover_returns_persisted_state_when_present(tmp_path):
    s = FeltTimeState(lived_age_hours=10.0, last_tick_ts="2026-05-17T22:00:00+00:00")
    persist(s, tmp_path)

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.lived_age_hours == 10.0
    assert loaded.last_tick_ts == "2026-05-17T22:00:00+00:00"


def test_load_or_recover_returns_cold_start_when_no_state_file(tmp_path):
    loaded, recovered = load_or_recover(tmp_path)
    # No state file AND no logs to recover from → cold start, recovered flag still True
    # (anything that wasn't a clean load counts as recovery for the banner).
    assert loaded.lived_age_hours == 0.0
    assert recovered is True


def test_load_or_recover_keeps_state_when_log_entry_is_newer_than_last_tick(tmp_path):
    # Persist state with last_tick_ts BEFORE a (fake) dream log entry. Log rows newer
    # than the last tick are the normal case between ticks; they are not staleness.
    persist(
        FeltTimeState(lived_age_hours=2.0, last_tick_ts="2026-05-17T20:00:00+00:00"),
        tmp_path,
    )
    dreams_log = tmp_path / "dreams.log.jsonl"
    dreams_log.write_text('{"ts": "2026-05-17T21:00:00+00:00", "summary": "the boat one"}\n')

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.lived_age_hours == 2.0
    assert loaded.replayed is False


def test_load_is_read_only_and_keeps_age_when_log_is_newer(tmp_path):
    persist(FeltTimeState(lived_age_hours=2.0, last_tick_ts="2026-05-17T20:00:00+00:00"), tmp_path)
    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T21:00:00+00:00", "summary": "x"}\n'
    )
    state_file = tmp_path / "felt_time_state.json"
    before = state_file.read_bytes()
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 2.0
    assert state_file.read_bytes() == before


def test_saved_state_with_no_last_tick_is_loaded_not_replayed(tmp_path):
    persist(FeltTimeState(lived_age_hours=2.0, last_tick_ts=None), tmp_path)
    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T21:00:00+00:00", "summary": "x"}\n'
    )
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.lived_age_hours == 2.0
    assert loaded.replayed is False


def test_missing_state_file_with_logs_replays(tmp_path):
    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T21:00:00+00:00", "summary": "from the log"}\n'
    )
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.replayed is True
    assert loaded.lived_age_hours == 0.0
    assert loaded.anchors["dream"].label == "from the log"


def test_read_paths_never_write_after_a_replay(tmp_path):
    """C13: a reader that lands on a missing state file (e.g. the save-time rotation
    window) gets a replayed 0.0 in memory only. Nothing is written, so it can never
    overwrite the real state when the file reappears. Only FeltTime.tick persists."""
    from brain.felt_time import FeltTime
    from brain.felt_time.tool import felt_time_now

    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T21:00:00+00:00", "summary": "x"}\n'
    )
    before = sorted(p.name for p in tmp_path.iterdir())

    load_or_recover(tmp_path)
    ft = FeltTime(persona_dir=tmp_path)
    ft.get_state()
    felt_time_now(persona_dir=tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / "felt_time_state.json").exists()


def test_load_or_recover_treats_corrupt_state_file_as_recovery_trigger(tmp_path):
    (tmp_path / "felt_time_state.json").write_text("{not valid json")
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.lived_age_hours == 0.0


def test_persist_round_trip_preserves_replayed_flag(tmp_path):
    s = FeltTimeState(lived_age_hours=1.0, replayed=True)
    persist(s, tmp_path)

    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    assert data["replayed"] is True

    loaded, _recovered = load_or_recover(tmp_path)
    assert loaded.replayed is True


def test_persist_round_trip_replayed_false_round_trips_correctly(tmp_path):
    s = FeltTimeState(lived_age_hours=5.0, last_tick_ts="2026-05-17T22:00:00+00:00", replayed=False)
    persist(s, tmp_path)

    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    assert data["replayed"] is False

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.replayed is False


def test_felt_time_state_new_field_defaults():
    s = FeltTimeState()
    assert s.horizon_pressure == {}
    assert s.arc_anchors == []


def test_anchor_event_type_defaults_to_empty():
    a = Anchor(type="arc", ts="2026-06-01T00:00:00+00:00", label="X", source_ref="arcs.log.jsonl:1")
    assert a.event_type == ""


def test_anchor_event_type_explicit():
    a = Anchor(type="arc", ts="2026-06-01T00:00:00+00:00", label="X",
               source_ref="arcs.log.jsonl:1", event_type="arc_opened")
    assert a.event_type == "arc_opened"


def test_horizon_bucket_defaults():
    from brain.felt_time.state import HorizonBucket
    b = HorizonBucket()
    assert b.counters == PressureCounters()
    assert b.prev_counters == PressureCounters()
    assert b.period_start_ts is None


def test_persist_roundtrips_new_fields(tmp_path):
    from brain.felt_time.state import HorizonBucket
    s = FeltTimeState(
        lived_age_hours=10.0,
        last_tick_ts="2026-06-08T09:00:00+00:00",
        horizon_pressure={
            "week": HorizonBucket(
                counters=PressureCounters(chat_turns=5, heartbeats=10),
                prev_counters=PressureCounters(chat_turns=3),
                period_start_ts="2026-06-01T00:00:00+00:00",
            )
        },
        arc_anchors=[
            Anchor(type="arc", ts="2026-06-01T00:00:00+00:00", label="The Long Work",
                   source_ref="arcs.log.jsonl:1", event_type="arc_opened")
        ],
    )
    persist(s, tmp_path)
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.horizon_pressure["week"].counters.chat_turns == 5
    assert loaded.horizon_pressure["week"].prev_counters.chat_turns == 3
    assert loaded.horizon_pressure["week"].period_start_ts == "2026-06-01T00:00:00+00:00"
    assert len(loaded.arc_anchors) == 1
    assert loaded.arc_anchors[0].label == "The Long Work"
    assert loaded.arc_anchors[0].event_type == "arc_opened"


def test_load_old_state_missing_new_fields_defaults_gracefully(tmp_path):
    old_state = {
        "lived_age_hours": 42.0,
        "anchors": {},
        "pressure": {"heartbeats": 0, "chat_turns": 0, "reflex_firings": 0, "wall_clock_s": 0.0},
        "last_tick_ts": "2026-06-08T09:00:00+00:00",
        "weather_baselines": {},
        "replayed": False,
    }
    (tmp_path / "felt_time_state.json").write_text(json.dumps(old_state))
    state, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert state.horizon_pressure == {}
    assert state.arc_anchors == []


def test_load_seeds_arc_anchors_from_existing_arc_anchor(tmp_path):
    anchor_dict = {
        "type": "arc", "ts": "2026-06-01T00:00:00+00:00",
        "label": "The Long Work", "source_ref": "arcs.log.jsonl:1",
    }
    old_state = {
        "lived_age_hours": 10.0,
        "anchors": {"arc": anchor_dict},
        "pressure": {"heartbeats": 0, "chat_turns": 0, "reflex_firings": 0, "wall_clock_s": 0.0},
        "last_tick_ts": "2026-06-08T09:00:00+00:00",
        "weather_baselines": {},
        "replayed": False,
    }
    (tmp_path / "felt_time_state.json").write_text(json.dumps(old_state))
    state, _ = load_or_recover(tmp_path)
    assert len(state.arc_anchors) == 1
    assert state.arc_anchors[0].label == "The Long Work"


# ---------------------------------------------------------------------------
# Corrupt state file: read-only .bak walk, then log replay (#384, Planning ruling)
# ---------------------------------------------------------------------------


def _persist_generations(tmp_path, ages):
    """Persist one state per age, oldest first, so .bak1 is the previous one."""
    for age in ages:
        persist(FeltTimeState(lived_age_hours=age, last_tick_ts="2026-05-17T20:00:00+00:00"), tmp_path)


def _corrupt(path):
    path.write_text("{not valid json")


def _dream_log(tmp_path):
    (tmp_path / "dreams.log.jsonl").write_text(
        '{"ts": "2026-05-17T21:00:00+00:00", "summary": "from the log"}\n'
    )


def test_corrupt_state_uses_good_bak1(tmp_path):
    _persist_generations(tmp_path, [5.0, 7.0])  # main=7.0, .bak1=5.0
    _corrupt(tmp_path / "felt_time_state.json")
    loaded, recovered = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 5.0
    assert recovered is True  # not a clean load of the live file
    assert loaded.replayed is False  # restored from a backup, not rebuilt from logs


def test_corrupt_state_and_bad_bak1_uses_bak2(tmp_path):
    _persist_generations(tmp_path, [3.0, 5.0, 7.0])  # main=7, .bak1=5, .bak2=3
    _corrupt(tmp_path / "felt_time_state.json")
    _corrupt(tmp_path / "felt_time_state.json.bak1")
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 3.0


def test_corrupt_state_walks_to_bak3(tmp_path):
    _persist_generations(tmp_path, [1.0, 3.0, 5.0, 7.0])  # .bak3=1
    for name in ("", ".bak1", ".bak2"):
        _corrupt(tmp_path / f"felt_time_state.json{name}")
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 1.0


def test_corrupt_state_with_all_baks_bad_replays_from_logs(tmp_path):
    _persist_generations(tmp_path, [3.0, 5.0, 7.0])
    for name in ("", ".bak1", ".bak2"):
        _corrupt(tmp_path / f"felt_time_state.json{name}")
    _dream_log(tmp_path)
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.replayed is True
    assert loaded.lived_age_hours == 0.0
    assert loaded.anchors["dream"].label == "from the log"


def test_corrupt_state_with_no_baks_and_no_logs_cold_starts(tmp_path):
    _corrupt(tmp_path / "felt_time_state.json")
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.lived_age_hours == 0.0
    assert loaded.anchors == {}


def test_corrupt_state_with_no_baks_replays_from_logs(tmp_path):
    _corrupt(tmp_path / "felt_time_state.json")
    _dream_log(tmp_path)
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True
    assert loaded.replayed is True
    assert loaded.anchors["dream"].label == "from the log"


_WRONG_TYPED = [
    {"lived_age_hours": None},
    {"lived_age_hours": "a lot"},
    {"anchors": {"dream": "not a dict"}},
    {"anchors": [1, 2]},
    {"pressure": "x"},
    {"arc_anchors": [3]},
    {"arc_anchors": "x"},
    {"horizon_pressure": {"week": 5}},
    {"horizon_pressure": [1]},
    {"last_tick_ts": 12345},
    {"weather_baselines": [1]},
]


@pytest.mark.parametrize("bad", _WRONG_TYPED, ids=lambda b: next(iter(b)) + "=" + repr(next(iter(b.values())))[:12])
def test_wrong_typed_field_is_corrupt_not_a_raise(tmp_path, bad):
    """A parseable file with a wrong-typed field takes the corrupt path: good .bak wins."""
    _persist_generations(tmp_path, [5.0, 7.0])
    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    data.update(bad)
    (tmp_path / "felt_time_state.json").write_text(json.dumps(data))
    loaded, _ = load_or_recover(tmp_path)  # must not raise
    assert loaded.lived_age_hours == 5.0


def test_wrong_typed_field_without_baks_falls_back_to_logs_or_cold_start(tmp_path):
    persist(FeltTimeState(lived_age_hours=7.0), tmp_path)
    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    data["lived_age_hours"] = None
    (tmp_path / "felt_time_state.json").write_text(json.dumps(data))
    loaded, recovered = load_or_recover(tmp_path)  # no raise, no logs -> cold start
    assert (loaded.lived_age_hours, recovered) == (0.0, True)
    _dream_log(tmp_path)
    loaded, recovered = load_or_recover(tmp_path)  # with logs -> replay
    assert loaded.replayed is True and recovered is True


def test_wrong_typed_field_with_newer_log_row_does_not_raise(tmp_path):
    """The regression the #384 fix would otherwise introduce: before it, a newer log
    row sent a parseable-but-wrong-typed file to replay; the fix removed that bypass,
    so this must not raise on load."""
    persist(FeltTimeState(lived_age_hours=7.0, last_tick_ts="2026-05-17T20:00:00+00:00"), tmp_path)
    data = json.loads((tmp_path / "felt_time_state.json").read_text())
    data["lived_age_hours"] = None
    (tmp_path / "felt_time_state.json").write_text(json.dumps(data))
    _dream_log(tmp_path)
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True and loaded.lived_age_hours == 0.0


def test_corrupt_paths_write_nothing(tmp_path):
    """C13 extended: the .bak walk and the replay fallback are read-only. Every file,
    including the corrupt ones and the backups, is byte-identical afterwards."""
    from brain.felt_time import FeltTime
    from brain.felt_time.tool import felt_time_now

    _persist_generations(tmp_path, [3.0, 5.0, 7.0])
    _corrupt(tmp_path / "felt_time_state.json")
    _corrupt(tmp_path / "felt_time_state.json.bak1")
    _dream_log(tmp_path)

    def snapshot():
        return {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir())}

    before = snapshot()
    load_or_recover(tmp_path)
    FeltTime(persona_dir=tmp_path).get_state()
    felt_time_now(persona_dir=tmp_path)
    assert snapshot() == before


def test_newest_good_backup_wins_over_older_ones(tmp_path):
    _persist_generations(tmp_path, [3.0, 5.0, 7.0])  # main=7, .bak1=5, .bak2=3 (both baks good)
    _corrupt(tmp_path / "felt_time_state.json")
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 5.0  # .bak1 before .bak2


def test_non_object_json_state_is_corrupt(tmp_path):
    _persist_generations(tmp_path, [5.0, 7.0])
    (tmp_path / "felt_time_state.json").write_text("[1, 2, 3]")
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.lived_age_hours == 5.0
