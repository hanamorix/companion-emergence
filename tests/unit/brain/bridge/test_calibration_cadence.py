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
    # inc6 (#250 §6) adds the judge-labeling pass: `provider` (the Haiku
    # tie-break's generation provider — None means "build one via
    # build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER) inside
    # this function", mirroring consolidation.run_consolidation's per-call
    # construction) and `judge` (test-injection point for a
    # RelevanceJudgeProvider — None means "build the real torch judge
    # lazily"). Both keyword-only, both defaulted, so every existing
    # zero-arg call site (run_folded's startup catch-up + periodic fire)
    # keeps working unmodified.
    assert params == ["persona_dir", "is_session_busy", "provider", "judge"], (
        f"expected (persona_dir, *, is_session_busy, provider, judge), got {params}"
    )
    busy = sig.parameters["is_session_busy"]
    assert busy.kind is inspect.Parameter.KEYWORD_ONLY
    assert busy.default is None
    provider_param = sig.parameters["provider"]
    assert provider_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert provider_param.default is None
    judge_param = sig.parameters["judge"]
    assert judge_param.kind is inspect.Parameter.KEYWORD_ONLY
    assert judge_param.default is None


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
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
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

        # This test is pruning-only (acceptance 5b); inc6's judge-labeling
        # pass is exercised separately in test_relevance_judge.py, so a
        # FakeRelevanceJudgeProvider + no Haiku provider keeps this test
        # offline/hermetic and focused on the prune assertion below (every
        # sampled row's candidate id "a" isn't a real memory, so every row
        # labels "unknown" regardless of the injected judge).
        _run_calibration_tick(pd, judge=FakeRelevanceJudgeProvider())

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


def test_calibration_tick_judge_failure_does_not_crash_and_prune_already_committed(
    monkeypatch,
):
    """The tick-level fault-isolation wrapper (spec Section 5/6: 'must not
    crash the tick or the bridge') around `_run_calibration_tick`'s
    judge-labeling pass has no dedicated test — `label_calibration_sample`'s
    OWN internal fault isolation is covered in test_relevance_judge.py, but
    the try/except `_run_calibration_tick` wraps around *calling* it is not.

    Monkeypatches `relevance_judge.label_calibration_sample` (the name
    `_run_calibration_tick` imports fresh, inside its own try block, on
    every call) to raise, and asserts:
      (a) `_run_calibration_tick` completes without the exception
          propagating (the tick/bridge does not crash), and
      (b) the prune step that ran BEFORE the failing judge pass stays
          committed — an old bucket is gone and a recent bucket survives —
          proving the failure did not undo the already-committed prune.
    """
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory import relevance_judge as rj_mod
    from brain.memory.store import CALIBRATION_LOG_RETENTION_WINDOW_DAYS, MemoryStore

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)
        store = MemoryStore(pd / "memories.db", integrity_check=False)
        now = datetime.now(UTC)
        window = CALIBRATION_LOG_RETENTION_WINDOW_DAYS

        old_bucket = (now - timedelta(days=window + 5)).strftime("%Y-%m-%d")
        recent_bucket = now.strftime("%Y-%m-%d")
        for bucket in (old_bucket, recent_bucket):
            store._conn.execute(
                "INSERT INTO calibration_log "
                "(day_bucket, query, candidate_ids, reranker_scores, reranker_model_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (bucket, f"query for {bucket}", json.dumps(["a"]), json.dumps([1.0]), "test-model"),
            )
        store._conn.commit()
        store.close()

        def _raising_label_calibration_sample(*args, **kwargs):
            raise RuntimeError("judge blew up")

        monkeypatch.setattr(rj_mod, "label_calibration_sample", _raising_label_calibration_sample)

        # (a) must not raise — a propagating exception here would fail this
        # test just as surely as an explicit assertion would.
        _run_calibration_tick(pd)

        # (b) the prune that ran before the judge pass failed must stand.
        store2 = MemoryStore(pd / "memories.db", integrity_check=False)
        rows = store2._conn.execute("SELECT day_bucket FROM calibration_log").fetchall()
        store2.close()
        remaining_buckets = {row["day_bucket"] for row in rows}

        assert old_bucket not in remaining_buckets, (
            "prune must already have run/committed before the judge-labeling failure"
        )
        assert recent_bucket in remaining_buckets


# ---------------------------------------------------------------------------
# F2a #250 inc7 (spec Section 7): floor derivation + the acceptance-2b
# precision-cache invalidation, as wired into the tick itself. The fit/EMA/
# stability-gate/cold-start MECHANISM has its own dedicated coverage in
# test_floor_calibration.py; these tests are scoped to the TICK's
# orchestration — does it call derive_and_persist_floor for the right
# model_id, and does it invalidate the reranker precision cache ONLY on an
# ACCEPTED write, never on a held cycle or a derivation failure.
# ---------------------------------------------------------------------------


def test_calibration_tick_resets_precision_cache_on_an_accepted_floor_write(
    monkeypatch,
):
    """Acceptance 2b's integration half: when the tick's floor derivation
    is ACCEPTED (a floor was actually written this cycle), the tick must
    call reranker.reset_precision_decision_for_floor_change() exactly
    once, so the next build_reranker_provider() call re-runs the
    fp16/fp32 self-check under the freshly written floor."""
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
    from brain.memory.reranker import FakeRerankerProvider

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda: FakeRerankerProvider())
        scripted = FloorDerivationOutcome(
            accepted=True, floor=1.0, raw_fit_floor=1.0, sample_pairs=200,
            is_cold_start=False, held_for_stability=False,
        )
        calls: dict[str, list] = {"derive": [], "reset": []}
        monkeypatch.setattr(
            fc_mod,
            "derive_and_persist_floor",
            lambda store, model_id, **kw: (calls["derive"].append(model_id), scripted)[1],
        )
        monkeypatch.setattr(
            reranker_mod, "reset_precision_decision_for_floor_change",
            lambda: calls["reset"].append(True),
        )

        _run_calibration_tick(pd, judge=FakeRelevanceJudgeProvider())

        assert calls["derive"] == ["fake-reranker"], (
            "the tick must derive a floor for the CURRENT production reranker's model_id"
        )
        assert len(calls["reset"]) == 1, (
            "an ACCEPTED floor write must reset the precision cache exactly once"
        )


def test_calibration_tick_does_not_reset_precision_cache_on_a_held_cycle(monkeypatch):
    """The other half of acceptance 2b's integration: a HELD cycle (the
    stability gate tripped — nothing was written) must NOT reset the
    precision cache. Nothing changed, so there is nothing to re-evaluate
    the fp16/fp32 decision against — resetting anyway would defeat §2's
    one-time-cost design on every held cycle too."""
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.floor_calibration import FloorDerivationOutcome
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
    from brain.memory.reranker import FakeRerankerProvider

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda: FakeRerankerProvider())
        held = FloorDerivationOutcome(
            accepted=False, floor=1.0, raw_fit_floor=2.0, sample_pairs=200,
            is_cold_start=False, held_for_stability=True,
        )
        reset_calls: list = []
        monkeypatch.setattr(fc_mod, "derive_and_persist_floor", lambda store, model_id, **kw: held)
        monkeypatch.setattr(
            reranker_mod, "reset_precision_decision_for_floor_change",
            lambda: reset_calls.append(True),
        )

        _run_calibration_tick(pd, judge=FakeRelevanceJudgeProvider())

        assert reset_calls == [], "a HELD (unaccepted) cycle must never reset the precision cache"


def test_calibration_tick_floor_derivation_failure_does_not_crash_the_tick(monkeypatch):
    """Spec Section 5/6's 'must not crash the tick or the bridge' fault-
    isolation posture extends to inc7's floor-derivation step too — wrapped
    in its own try/except mirroring the judge-labeling step immediately
    above it (see test_calibration_tick_judge_failure_does_not_crash_and_
    prune_already_committed above for that step's own coverage)."""
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory import floor_calibration as fc_mod
    from brain.memory import reranker as reranker_mod
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
    from brain.memory.reranker import FakeRerankerProvider

    with tempfile.TemporaryDirectory() as d:
        pd = Path(d)

        monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda: FakeRerankerProvider())

        def _raising_derive(store, model_id, **kw):
            raise RuntimeError("simulated floor-derivation failure")

        monkeypatch.setattr(fc_mod, "derive_and_persist_floor", _raising_derive)

        # Must not raise.
        _run_calibration_tick(pd, judge=FakeRelevanceJudgeProvider())
