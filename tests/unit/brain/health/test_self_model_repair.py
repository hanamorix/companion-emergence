"""One-time self-model gap reset migration (windowed-peak fix)."""

from __future__ import annotations

import pytest

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


def test_a_failed_repair_writes_no_marker_and_the_next_start_runs_it(tmp_path, monkeypatch) -> None:
    # #737: a save that raises (not a lock timeout) left the marker written, so the bogus gap stayed for good.
    sm_state.save(tmp_path, sm_state.SelfModelState(current_gap=_bogus_gap(), gap_history=[]))
    real_save = sm_state.save

    def failing_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sm_state, "save", failing_save)
    run_self_model_repair(tmp_path)
    assert should_run_self_model_repair(tmp_path) is True, "a failed repair must run again at the next start"
    assert sm_state.load_or_recover(tmp_path)[0].current_gap is not None

    monkeypatch.setattr(sm_state, "save", real_save)
    run_self_model_repair(tmp_path)
    assert sm_state.load_or_recover(tmp_path)[0].current_gap is None
    assert should_run_self_model_repair(tmp_path) is False


def test_a_repair_that_keeps_failing_gives_up_after_three_starts(tmp_path, monkeypatch, caplog) -> None:
    # #737: the retry is bounded, so a fault that never clears does not log a traceback on every start.
    import logging

    sm_state.save(tmp_path, sm_state.SelfModelState(current_gap=_bogus_gap(), gap_history=[]))

    def failing_save(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(sm_state, "save", failing_save)
    starts = 0
    with caplog.at_level(logging.WARNING, logger="brain.health.self_model_repair"):
        while should_run_self_model_repair(tmp_path) and starts < 10:
            run_self_model_repair(tmp_path)
            starts += 1
    assert starts == 3
    tracebacks = [r for r in caplog.records if r.exc_info]
    assert len(tracebacks) == 3
    assert any("gave up" in r.getMessage() for r in caplog.records), caplog.records


@pytest.mark.no_base_failure(
    reason="the base read any marker file as done; this pins that an unreadable one still does once the marker carries a count"
)
def test_an_unreadable_marker_reads_as_done_so_a_real_gap_is_kept(tmp_path) -> None:
    # Running again would clear a gap the reflection tick computed honestly since the repair.
    (tmp_path / "self_model_repair_state.json").write_text("{not json", encoding="utf-8")
    assert should_run_self_model_repair(tmp_path) is False


def test_repair_noop_when_no_state(tmp_path) -> None:
    # No self_model_state.json at all → repair is a no-op but still marks done.
    assert should_run_self_model_repair(tmp_path) is True
    run_self_model_repair(tmp_path)
    assert should_run_self_model_repair(tmp_path) is False
    st, _ = sm_state.load_or_recover(tmp_path)
    assert st.current_gap is None
