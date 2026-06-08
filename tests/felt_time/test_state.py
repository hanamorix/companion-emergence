import json

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


def test_load_or_recover_triggers_recovery_when_state_older_than_newest_log(tmp_path):
    # Persist state with last_tick_ts BEFORE a (fake) dream log entry.
    persist(
        FeltTimeState(last_tick_ts="2026-05-17T20:00:00+00:00"),
        tmp_path,
    )
    dreams_log = tmp_path / "dreams.log.jsonl"
    dreams_log.write_text('{"ts": "2026-05-17T21:00:00+00:00", "summary": "the boat one"}\n')

    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is True  # newer log entry forced re-derivation


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
