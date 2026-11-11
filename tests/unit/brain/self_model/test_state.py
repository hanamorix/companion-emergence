"""Tests for brain/self_model/state.py — persisted SelfModelState."""
from __future__ import annotations

from brain.self_model.state import (
    SelfModelState,
    load_or_recover,
    save,
)


def test_missing_file_returns_fresh_not_recovered(tmp_path):
    state, recovered = load_or_recover(tmp_path)
    assert not recovered
    assert state.current_gap is None
    assert state.gap_history == []


def test_round_trip_current_gap_and_history(tmp_path):
    from brain.self_model.gap import Gap
    gap = Gap(per_channel={"joy": -1.5}, magnitude=1.5, unnamed_pressure=0.0,
              status="open", first_seen_ts="2026-01-01T00:00:00+00:00",
              last_seen_ts="2026-01-01T00:00:00+00:00")
    old = Gap(per_channel={"calm": 2.0}, magnitude=2.0, unnamed_pressure=0.0,
              status="resolved")
    state = SelfModelState(current_gap=gap, gap_history=[old])
    save(tmp_path, state)
    loaded, recovered = load_or_recover(tmp_path)
    assert not recovered
    assert loaded.current_gap is not None
    assert loaded.current_gap.magnitude == 1.5
    assert loaded.current_gap.status == "open"
    assert len(loaded.gap_history) == 1
    assert loaded.gap_history[0].status == "resolved"


def test_corrupt_file_returns_fresh_recovered_true(tmp_path):
    (tmp_path / "self_model_state.json").write_text("{invalid json{{{{", encoding="utf-8")
    state, recovered = load_or_recover(tmp_path)
    assert recovered
    assert state.current_gap is None
    assert state.gap_history == []


def test_history_cap_at_20(tmp_path):
    from brain.self_model.gap import Gap
    from brain.self_model.state import _GAP_HISTORY_CAP, push_gap
    assert _GAP_HISTORY_CAP == 20
    # Build a state with 20 resolved gaps already in history
    history = [
        Gap(per_channel={"calm": float(i)}, magnitude=float(i),
            unnamed_pressure=0.0, status="resolved")
        for i in range(20)
    ]
    state = SelfModelState(current_gap=None, gap_history=history)
    # Push a current gap
    g1 = Gap(per_channel={"joy": 1.0}, magnitude=1.0, unnamed_pressure=0.0, status="open")
    s1 = push_gap(state, g1)
    assert len(s1.gap_history) == 20  # unchanged (no displacement yet)
    # Push another — displaces g1 into history making 21, capped to 20
    g2 = Gap(per_channel={"joy": 2.0}, magnitude=2.0, unnamed_pressure=0.0, status="open")
    s2 = push_gap(s1, g2)
    assert len(s2.gap_history) == 20, f"Expected 20, got {len(s2.gap_history)}"


def test_displaced_unresolved_gap_survives_in_history_and_warns(caplog):
    import logging

    from brain.self_model.gap import Gap
    from brain.self_model.state import push_gap
    existing = Gap(per_channel={"joy": 1.0}, magnitude=1.0, unnamed_pressure=0.0, status="open")
    state = SelfModelState(current_gap=existing, gap_history=[])
    new_gap = Gap(per_channel={"calm": 0.5}, magnitude=0.5, unnamed_pressure=0.0, status="open")

    with caplog.at_level(logging.WARNING, logger="brain.self_model.state"):
        result = push_gap(state, new_gap)

    # §7 tripwire: displaced open gap survives in history
    assert len(result.gap_history) == 1
    assert result.gap_history[0].status == "open"
    assert result.gap_history[0].magnitude == 1.0
    # WARN must be logged
    assert any("unresolved" in r.message.lower() for r in caplog.records)


def test_displaced_resolved_gap_no_warning(caplog):
    import logging

    from brain.self_model.gap import Gap
    from brain.self_model.state import push_gap
    existing = Gap(per_channel={"joy": 1.0}, magnitude=1.0, unnamed_pressure=0.0, status="resolved")
    state = SelfModelState(current_gap=existing, gap_history=[])
    new_gap = Gap(per_channel={"calm": 0.5}, magnitude=0.5, unnamed_pressure=0.0, status="open")

    with caplog.at_level(logging.WARNING, logger="brain.self_model.state"):
        result = push_gap(state, new_gap)

    # resolved displacement is silent — no unresolved WARN
    assert not any("unresolved" in r.message.lower() for r in caplog.records)
    # but the displaced gap still lands in history
    assert len(result.gap_history) == 1


def test_recovering_from_a_bad_state_file_is_logged(tmp_path, monkeypatch, caplog):
    """#367: all five callers discard the `recovered` flag, so a corrupt file
    silently became an empty gap history. The loader now says so itself."""
    import logging

    from brain.self_model.state import _state_path, load_or_recover
    from brain.utils import log_throttle

    for bad in ("{ not json", "[1, 2]", '{"gap_history": 7}'):
        monkeypatch.setattr(log_throttle, "_last", {})
        _state_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        _state_path(tmp_path).write_text(bad, encoding="utf-8")
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _state, recovered = load_or_recover(tmp_path)
        assert recovered is True
        # An unreadable or non-object file is reported by brain.health.json_io; a malformed gap by this module.
        assert any(
            "self-model state" in r.getMessage() or "unusable" in r.getMessage() for r in caplog.records
        ), bad


def test_a_missing_state_file_is_not_logged(tmp_path, caplog):
    import logging

    from brain.self_model.state import load_or_recover

    with caplog.at_level(logging.WARNING, logger="brain.self_model.state"):
        assert load_or_recover(tmp_path)[1] is False
    assert caplog.records == []



def test_a_corrupt_state_file_warns_once_not_every_turn(tmp_path, monkeypatch, caplog):
    """#367 review: chat turns load this; a corrupt file must not warn per turn."""
    import logging

    from brain.self_model.state import _state_path, load_or_recover
    from brain.utils import log_throttle

    monkeypatch.setattr(log_throttle, "_last", {})
    _state_path(tmp_path).write_text("{ not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="brain.self_model.state"):
        for _ in range(3):
            load_or_recover(tmp_path)
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


# ── B2 phase B: Gap.directions and Gap.reset_seen survive the file ────────────


def test_directions_and_reset_seen_survive_save_and_load(tmp_path):
    from brain.self_model.gap import Gap

    gap = Gap(
        per_channel={"joy": 1.5, "calm": -0.5},
        magnitude=2.0,
        unnamed_pressure=0.0,
        directions={"joy": 1, "calm": -1},
        reset_seen="2026-11-04T10:00:00+00:00",
    )
    save(tmp_path, SelfModelState(current_gap=gap, gap_history=[gap]))
    loaded, recovered = load_or_recover(tmp_path)
    assert not recovered
    assert loaded.current_gap.directions == {"joy": 1, "calm": -1}
    assert loaded.current_gap.reset_seen == "2026-11-04T10:00:00+00:00"
    assert loaded.gap_history[0].directions == {"joy": 1, "calm": -1}


def test_a_stored_none_reset_seen_reads_back_as_none_not_unset(tmp_path):
    from brain.self_model.gap import UNSET, Gap

    gap = Gap(per_channel={}, magnitude=0.0, unnamed_pressure=0.0, reset_seen=None)
    save(tmp_path, SelfModelState(current_gap=gap))
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.current_gap.reset_seen is None
    assert loaded.current_gap.reset_seen is not UNSET


def test_a_gap_that_never_stored_reset_seen_reads_back_as_unset_and_stays_without_the_key(tmp_path):
    import json

    from brain.self_model.gap import UNSET, Gap

    gap = Gap(per_channel={"joy": 1.0}, magnitude=1.0, unnamed_pressure=0.0)
    assert gap.reset_seen is UNSET
    save(tmp_path, SelfModelState(current_gap=gap))
    raw = json.loads((tmp_path / "self_model_state.json").read_text(encoding="utf-8"))
    assert "reset_seen" not in raw["current_gap"]
    assert raw["current_gap"]["directions"] == {}
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.current_gap.reset_seen is UNSET


def test_a_pre_field_gap_history_item_still_loads(tmp_path):
    import json

    from brain.self_model.gap import UNSET

    old_item = {
        "per_channel": {"calm": 2.0},
        "magnitude": 2.0,
        "unnamed_pressure": 0.0,
        "status": "resolved",
    }
    (tmp_path / "self_model_state.json").write_text(
        json.dumps({"current_gap": None, "gap_history": [old_item]}), encoding="utf-8"
    )
    loaded, recovered = load_or_recover(tmp_path)
    assert not recovered
    assert loaded.gap_history[0].directions == {}
    assert loaded.gap_history[0].reset_seen is UNSET


def test_a_malformed_directions_value_loads_as_empty(tmp_path):
    import json

    item = {"per_channel": {"joy": 1.0}, "magnitude": 1.0, "unnamed_pressure": 0.0, "directions": "up"}
    (tmp_path / "self_model_state.json").write_text(
        json.dumps({"current_gap": item, "gap_history": []}), encoding="utf-8"
    )
    loaded, _ = load_or_recover(tmp_path)
    assert loaded.current_gap.directions == {}
