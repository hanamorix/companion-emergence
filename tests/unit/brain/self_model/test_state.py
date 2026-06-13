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
