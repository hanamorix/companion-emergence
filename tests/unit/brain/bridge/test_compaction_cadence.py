"""Tests for the compaction_cadence.json persisted-cadence wiring.

Verifies that:
  - compaction_cadence.json is created and advances after a tick fires.
  - is_due fires when the cadence is past its next_at (86400s interval).
  - _run_compaction_tick is importable and has the expected signature.
  - run_folded accepts compaction_interval_s as a keyword argument.
"""
import inspect
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge import persisted_cadence as pc


def test_run_compaction_tick_importable_and_callable():
    from brain.bridge.supervisor import _run_compaction_tick

    sig = inspect.signature(_run_compaction_tick)
    params = list(sig.parameters)
    assert params == ["persona_dir", "provider"], (
        f"expected (persona_dir, provider), got {params}"
    )


def test_compaction_cadence_due_now_when_missing():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.load_cadence(pd, "compaction_cadence.json")
        assert pc.is_due(state, now=now) is True


def test_compaction_cadence_not_due_after_advance_and_fires_at_86400():
    now = datetime(2026, 6, 29, 12, tzinfo=UTC)
    state = pc.advance(now=now, interval_s=86400.0)
    assert pc.is_due(state, now=now) is False
    assert pc.is_due(state, now=now + timedelta(seconds=86400)) is True


def test_compaction_cadence_save_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.advance(now=now, interval_s=86400.0)
        pc.save_cadence(pd, "compaction_cadence.json", state)
        loaded = pc.load_cadence(pd, "compaction_cadence.json")
        assert loaded.next_at == state.next_at
        assert not list(pd.glob("*.tmp")), "atomic write must leave no .tmp"


def test_run_folded_accepts_compaction_interval_s():
    from brain.bridge.supervisor import run_folded

    sig = inspect.signature(run_folded)
    assert "compaction_interval_s" in sig.parameters
    param = sig.parameters["compaction_interval_s"]
    assert param.default == 86400.0, f"expected default 86400.0, got {param.default!r}"
