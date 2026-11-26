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


# --- body energy fields (#650 step 2, spec 1.1, 1.3, 1.6) ---------------------------------------------------------------------


def _write_state_file(persona_dir, data):
    (persona_dir / "felt_time_state.json").write_text(json.dumps(data))


def test_energy_fields_start_at_the_baseline_with_no_long_average_and_no_saved_feelings():
    from brain import dev_constants

    s = FeltTimeState.cold_start()
    assert s.energy_level == dev_constants.BODY_ENERGY_BASELINE == 6.5
    assert s.energy_long_avg == 0.0
    assert s.energy_long_weight == 0.0
    assert s.energy_feelings is None


def test_energy_fields_round_trip_unrounded(tmp_path):
    s = FeltTimeState(
        last_tick_ts="2026-05-17T22:00:00+00:00",
        energy_level=6.326923076923077,
        energy_long_avg=5.25,
        energy_long_weight=12.5,
        energy_feelings={"joy": 3.1415926, "grief": 0.0},
    )
    persist(s, tmp_path)
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.energy_level == 6.326923076923077
    assert loaded.energy_long_avg == 5.25
    assert loaded.energy_long_weight == 12.5
    assert loaded.energy_feelings == {"joy": 3.1415926, "grief": 0.0}


def test_missing_saved_feelings_are_absent_from_the_file_and_load_as_none(tmp_path):
    persist(FeltTimeState(last_tick_ts="2026-05-17T22:00:00+00:00"), tmp_path)
    assert "energy_feelings" not in json.loads((tmp_path / "felt_time_state.json").read_text())
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.energy_feelings is None


def test_an_empty_saved_feelings_map_is_not_missing(tmp_path):
    persist(FeltTimeState(last_tick_ts="2026-05-17T22:00:00+00:00", energy_feelings={}), tmp_path)
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.energy_feelings == {}


def test_an_old_file_without_the_energy_fields_loads_the_defaults(tmp_path):
    _write_state_file(tmp_path, {"lived_age_hours": 77.0, "last_tick_ts": "2026-05-17T22:00:00+00:00"})
    loaded, recovered = load_or_recover(tmp_path)
    assert recovered is False
    assert loaded.lived_age_hours == 77.0
    assert (loaded.energy_level, loaded.energy_long_avg, loaded.energy_long_weight, loaded.energy_feelings) == (6.5, 0.0, 0.0, None)


@pytest.mark.parametrize("bad", ["7.5", None, True, float("nan"), float("inf"), [7.5], {"a": 1}])
def test_an_unsound_energy_field_loads_as_its_default_and_does_not_raise(tmp_path, bad):
    _write_state_file(
        tmp_path,
        {
            "last_tick_ts": "2026-05-17T22:00:00+00:00",
            "energy_level": bad,
            "energy_long_avg": bad,
            "energy_long_weight": bad,
        },
    )
    loaded, _ = load_or_recover(tmp_path)
    assert (loaded.energy_level, loaded.energy_long_avg, loaded.energy_long_weight) == (6.5, 0.0, 0.0)


@pytest.mark.parametrize(
    "bad",
    ["joy", [1.0], {"joy": "3"}, {"joy": None}, {"joy": float("nan")}, {"joy": True}, {"joy": 2.0, "grief": "x"}],
)
def test_an_unsound_saved_feelings_map_loads_as_absent(tmp_path, bad):
    _write_state_file(tmp_path, {"last_tick_ts": "2026-05-17T22:00:00+00:00", "energy_feelings": bad})
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.energy_feelings is None


def test_read_energy_level_returns_the_saved_level_unrounded(tmp_path):
    from brain.felt_time.state import read_energy_level

    persist(FeltTimeState(last_tick_ts="2026-05-17T22:00:00+00:00", energy_level=6.326923076923077), tmp_path)
    assert read_energy_level(tmp_path) == 6.326923076923077


def test_read_energy_level_does_not_round_or_hold_a_stored_number(tmp_path):
    from brain.felt_time.state import read_energy_level

    _write_state_file(tmp_path, {"energy_level": 50.0})
    assert read_energy_level(tmp_path) == 50.0
    _write_state_file(tmp_path, {"energy_level": 0.25})
    assert read_energy_level(tmp_path) == 0.25


def test_read_energy_level_is_the_baseline_when_there_is_no_file(tmp_path):
    from brain.felt_time.state import read_energy_level

    assert read_energy_level(tmp_path) == 6.5


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]", "null", "{}", '{"lived_age_hours": 4}'])
def test_read_energy_level_is_the_baseline_for_an_unreadable_file_or_an_absent_field(tmp_path, text):
    from brain.felt_time.state import read_energy_level

    (tmp_path / "felt_time_state.json").write_text(text)
    assert read_energy_level(tmp_path) == 6.5


@pytest.mark.parametrize("bad", ["7.5", None, True, float("nan"), float("inf"), [7.5]])
def test_read_energy_level_is_the_baseline_for_an_unsound_field(tmp_path, bad):
    from brain.felt_time.state import read_energy_level

    _write_state_file(tmp_path, {"energy_level": bad})
    assert read_energy_level(tmp_path) == 6.5


def test_read_energy_level_is_a_light_read_that_never_rewrites_the_file(tmp_path, monkeypatch):
    from brain.felt_time import state as state_module
    from brain.felt_time.state import read_energy_level

    def boom(*_a, **_k):
        raise AssertionError("the accessor must not use the full loader or scan the logs")

    monkeypatch.setattr(state_module, "load_or_recover", boom)
    monkeypatch.setattr(state_module, "_newest_log_ts", boom)
    path = tmp_path / "felt_time_state.json"
    path.write_text("not json")
    before = (path.read_bytes(), sorted(p.name for p in tmp_path.iterdir()))
    assert read_energy_level(tmp_path) == 6.5
    assert (path.read_bytes(), sorted(p.name for p in tmp_path.iterdir())) == before


def test_the_baseline_default_is_read_at_call_time(monkeypatch):
    from brain import dev_constants

    monkeypatch.setattr(dev_constants, "BODY_ENERGY_BASELINE", 4.0)
    assert FeltTimeState.cold_start().energy_level == 4.0


def test_weather_shift_is_not_an_anchor_type():
    """#785: the weather-shift anchor never fired, so it is gone everywhere an anchor type is named."""
    from brain.felt_time.anchors import _SOURCES
    from brain.felt_time.state import ANCHOR_TYPES, LOG_SOURCES
    from brain.tools.schemas import SCHEMAS

    assert "weather_shift" not in ANCHOR_TYPES
    assert "weather_shifts.log.jsonl" not in LOG_SOURCES
    assert "weather_shift" not in _SOURCES
    enum = SCHEMAS["pressure_since"]["parameters"]["properties"]["anchor_type"]["enum"]
    assert "weather_shift" not in enum
    assert "weather_shift" not in SCHEMAS["felt_time_now"]["description"]
