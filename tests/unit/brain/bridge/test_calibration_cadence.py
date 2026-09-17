"""Tests for the calibration_cadence.json persisted-cadence wiring (F2a #250
inc5, spec Section 5).

Mirrors test_compaction_cadence.py's coverage for the 4th sibling cadence:
  - calibration_cadence.json is created and advances after a tick fires.
  - is_due fires when the cadence is past its next_at (86400s interval).
  - _run_calibration_tick is importable and has the expected signature.
  - run_folded accepts calibration_interval_s as a keyword argument.

Plus the acceptance-5b retention-pruning test: after simulating logs spanning
more than the retention window, the tick prunes old rows while rows inside
the window (needed for the current floor sample) survive.
"""
import inspect
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge import persisted_cadence as pc


def test_run_calibration_tick_importable_and_callable():
    from brain.bridge.supervisor import _run_calibration_tick

    sig = inspect.signature(_run_calibration_tick)
    params = list(sig.parameters)
    # This increment (inc5) is a retention-pruning-only scaffold — no
    # provider is needed yet (the judge/floor-derivation work that will need
    # one lands in inc6/inc7). The idle-gate ``is_session_busy`` keyword
    # mirrors compaction's, defaulted so the startup catch-up can call it
    # with no argument.
    assert params == ["persona_dir", "is_session_busy"], (
        f"expected (persona_dir, *, is_session_busy), got {params}"
    )
    busy = sig.parameters["is_session_busy"]
    assert busy.kind is inspect.Parameter.KEYWORD_ONLY
    assert busy.default is None


def test_calibration_cadence_due_now_when_missing():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.load_cadence(pd, "calibration_cadence.json")
        assert pc.is_due(state, now=now) is True


def test_calibration_cadence_not_due_after_advance_and_fires_at_86400():
    now = datetime(2026, 6, 29, 12, tzinfo=UTC)
    state = pc.advance(now=now, interval_s=86400.0)
    assert pc.is_due(state, now=now) is False
    assert pc.is_due(state, now=now + timedelta(seconds=86400)) is True


def test_calibration_cadence_save_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        now = datetime(2026, 6, 29, 12, tzinfo=UTC)
        state = pc.advance(now=now, interval_s=86400.0)
        pc.save_cadence(pd, "calibration_cadence.json", state)
        loaded = pc.load_cadence(pd, "calibration_cadence.json")
        assert loaded.next_at == state.next_at
        assert not list(pd.glob("*.tmp")), "atomic write must leave no .tmp"


def test_run_folded_accepts_calibration_interval_s():
    from brain.bridge.supervisor import run_folded

    sig = inspect.signature(run_folded)
    assert "calibration_interval_s" in sig.parameters
    param = sig.parameters["calibration_interval_s"]
    assert param.default == 86400.0, f"expected default 86400.0, got {param.default!r}"


def test_calibration_tick_skips_entirely_when_a_session_is_busy():
    """Unlike compaction's per-session skip, this tick's work (a table-wide
    prune) is corpus-global, so a busy session defers the WHOLE tick rather
    than partially pruning. Seed one genuinely active session (a real
    buffer file under active_conversations/, via ingest_turn — the same
    write path list_active_sessions reads) so is_session_busy is actually
    consulted and reports busy, then assert the tick defers before ever
    opening the store: no memories.db is created, so the prune cannot have
    run. Without a seeded active session `list_active_sessions` returns []
    and `any(...)` over an empty list is vacuously False regardless of what
    is_session_busy would say, which is exactly the theater this rewrite
    closes."""
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.ingest.buffer import ingest_turn, list_active_sessions

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        sid = ingest_turn(pd, {"session_id": "sess_busy", "speaker": "user", "text": "hi"})
        assert list_active_sessions(pd) == [sid], "fixture must seed a real active session"

        _run_calibration_tick(pd, is_session_busy=lambda s: True)

        assert not (pd / "memories.db").exists(), (
            "tick must defer before opening the store when a session is busy"
        )


def test_calibration_tick_prunes_old_rows_keeps_recent_rows_within_window():
    """Acceptance 5b: after simulating calibration_log rows spanning more than
    the retention window (many synthetic day_buckets), the daily tick prunes
    rows outside the rolling window so the table stays bounded, while rows
    INSIDE the window (needed for the current floor sample) are retained."""
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory.store import CALIBRATION_LOG_RETENTION_WINDOW_DAYS, MemoryStore

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        now = datetime.now(UTC)
        window = CALIBRATION_LOG_RETENTION_WINDOW_DAYS

        # Old rows spanning several buckets, all well outside the window.
        old_buckets = [
            (now - timedelta(days=window + 5)).strftime("%Y-%m-%d"),
            (now - timedelta(days=window + 40)).strftime("%Y-%m-%d"),
            (now - timedelta(days=window + 400)).strftime("%Y-%m-%d"),
        ]
        # Recent rows spanning several buckets, all safely inside the window.
        recent_buckets = [
            now.strftime("%Y-%m-%d"),
            (now - timedelta(days=1)).strftime("%Y-%m-%d"),
            (now - timedelta(days=max(0, window - 1))).strftime("%Y-%m-%d"),
        ]
        for bucket in old_buckets + recent_buckets:
            store._conn.execute(
                "INSERT INTO calibration_log "
                "(day_bucket, query, candidate_ids, reranker_scores, reranker_model_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (bucket, f"query for {bucket}", json.dumps(["a"]), json.dumps([1.0]), "test-model"),
            )
        store._conn.commit()
        store.close()

        _run_calibration_tick(pd)

        store2 = MemoryStore(pd / "memories.db", integrity_check=False)
        rows = store2._conn.execute("SELECT day_bucket FROM calibration_log").fetchall()
        store2.close()
        remaining_buckets = {row["day_bucket"] for row in rows}

        for bucket in old_buckets:
            assert bucket not in remaining_buckets, f"old bucket {bucket} should have been pruned"
        for bucket in recent_buckets:
            assert bucket in remaining_buckets, f"recent bucket {bucket} should have survived the prune"
        # Bounded: no rows older than the window remain at all.
        cutoff = (now - timedelta(days=window)).strftime("%Y-%m-%d")
        assert all(b >= cutoff for b in remaining_buckets)
