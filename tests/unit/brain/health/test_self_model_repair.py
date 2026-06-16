"""One-time self-model gap reset migration (windowed-peak fix)."""

from __future__ import annotations

from brain.health.self_model_repair import (
    run_self_model_repair,
    should_run_self_model_repair,
)
from brain.self_model import state as sm_state
from brain.self_model.gap import Gap


def _bogus_gap() -> Gap:
    # The live artifact shape: every channel large-negative, magnitude in the
    # hundreds, status open.
    return Gap(
        per_channel={"love": -8.3, "joy": -5.7, "trust": -7.3},
        magnitude=354.0,
        unnamed_pressure=0.0,
        status="open",
        first_seen_ts="2026-06-15T23:20:48+00:00",
        last_seen_ts="2026-06-15T23:20:48+00:00",
        sustained_ticks=1,
    )


def test_repair_clears_open_gap_once(tmp_path) -> None:
    sm_state.save(tmp_path, sm_state.SelfModelState(current_gap=_bogus_gap(), gap_history=[]))
    assert should_run_self_model_repair(tmp_path) is True

    run_self_model_repair(tmp_path)

    st, _ = sm_state.load_or_recover(tmp_path)
    assert st.current_gap is None
    # marker written → never runs again
    assert should_run_self_model_repair(tmp_path) is False


def test_repair_noop_when_no_state(tmp_path) -> None:
    # No self_model_state.json at all → repair is a no-op but still marks done.
    assert should_run_self_model_repair(tmp_path) is True
    run_self_model_repair(tmp_path)
    assert should_run_self_model_repair(tmp_path) is False
    st, _ = sm_state.load_or_recover(tmp_path)
    assert st.current_gap is None
