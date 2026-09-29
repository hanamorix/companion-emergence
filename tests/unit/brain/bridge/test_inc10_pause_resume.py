"""ram-spike-fix INC-10 — per-item pause/resume, crash resume, judge release
on pause (spec §4; S14/S31/S32/S36/S41/S65).

Criteria covered here: C8 (per-item pause for every pausable job in the S32
table), C31 (the new job_progress shared-state accessor's atomicity), and a
regression guard for the emotion_backfill "wrongly marks complete on pause"
bug found and fixed while building this increment. C16 (real-subprocess
crash resume) and C2's calibration pause-arm (real-model RSS) live in their
own files (test_c16_crash_resume.py, test_judge_release_rss.py) — the
former needs a real subprocess kill, the latter needs the real torch judge.

Deterministic throughout: injected `should_pause`/`between_items` callables
(a stateful counter that flips True after N calls), no sleeps, no real
LLM/model — synthetic tmp personas only.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from brain.bridge import job_progress


def _persona(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


def _pause_on_call(k: int):
    """A should_pause callable that returns False for calls 1..k-1 and True
    from call k onward (1-indexed): with a between-items check made once per
    completed item, `_pause_on_call(1)` pauses right after the FIRST item."""
    count = {"n": 0}

    def _f() -> bool:
        count["n"] += 1
        return count["n"] >= k

    return _f


# ---------------------------------------------------------------------------
# C31 — job_progress atomicity (the new shared-state accessor)
# ---------------------------------------------------------------------------


def test_c31_save_progress_is_atomic_temp_rename(tmp_path: Path) -> None:
    """A concurrent reader never observes a torn/partial write: save_progress
    always leaves either the old complete JSON or the new complete JSON."""
    persona_dir = _persona(tmp_path)
    job_progress.save_progress(persona_dir, "forgetting", {"last_id": "a"})
    path = persona_dir / "cadence" / "forgetting_progress.json"
    assert path.exists()
    assert json.loads(path.read_text()) == {"last_id": "a"}
    # No .tmp sidecar left behind after a successful save.
    assert not path.with_suffix(".json.tmp").exists()

    # A mid-write reader (simulated: read while a second save is committed
    # via rename, which is atomic on POSIX) sees one full state or the other,
    # never a half-written file.
    seen: list[dict] = []
    barrier = threading.Barrier(2)

    def _writer():
        barrier.wait()
        for i in range(50):
            job_progress.save_progress(persona_dir, "forgetting", {"last_id": f"item-{i}"})

    def _reader():
        barrier.wait()
        for _ in range(50):
            try:
                raw = path.read_text()
            except OSError:
                continue
            if raw:
                seen.append(json.loads(raw))

    t1, t2 = threading.Thread(target=_writer), threading.Thread(target=_reader)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert all("last_id" in d for d in seen), "every read must be a complete, valid JSON object"


def test_c31_load_progress_missing_or_corrupt_starts_from_scratch(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    assert job_progress.load_progress(persona_dir, "forgetting") == {}
    path = persona_dir / "cadence" / "forgetting_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json{{{")
    assert job_progress.load_progress(persona_dir, "forgetting") == {}


def test_c31_clear_progress_removes_the_file(tmp_path: Path) -> None:
    persona_dir = _persona(tmp_path)
    job_progress.save_progress(persona_dir, "forgetting", {"last_id": "a"})
    job_progress.clear_progress(persona_dir, "forgetting")
    assert job_progress.load_progress(persona_dir, "forgetting") == {}
    assert not (persona_dir / "cadence" / "forgetting_progress.json").exists()


# ---------------------------------------------------------------------------
# C8 — pass 2 (S32 row 1): should_pause already existed pre-INC-10 at the
# pass2_queue level; this proves the JOB-LEVEL JobOutcome mapping INC-10
# added (_pass2_run in supervisor.py) correctly distinguishes a real pause
# (items remain) from a full drain.
# ---------------------------------------------------------------------------


def test_c8_pass2_job_reports_paused_with_items_remaining_then_resumes(tmp_path: Path) -> None:
    from brain.bridge.central_cadence import JobOutcome
    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import _build_gated_jobs
    from brain.chat import pass2_queue

    persona_dir = _persona(tmp_path)
    ran: list[str] = []
    for _i in range(3):
        rid = pass2_queue.new_record_id()

        def _fx(_rid=rid):
            ran.append(_rid)

        pass2_queue.register_test_side_effect(rid, _fx)
        pass2_queue.enqueue({"id": rid, "kind": "test_probe"}, persona_dir=persona_dir)

    pause_flag = {"on": True}
    pauser = lambda: pause_flag["on"]  # noqa: E731 — toggled between calls below
    jobs = _build_gated_jobs(
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=EventBus(),
        is_session_busy=None,
        finalize_after_hours=24.0,
        finalize_interval_s=None,
        initiate_review_interval_s=None,
        maintenance_interval_s=None,
        self_model_interval_s=None,
        compaction_interval_s=None,
        calibration_interval_s=None,
        interest_sweep_interval_s=None,
        judge_selftune_interval_s=None,
        clustering_interval_s=None,
        intensity_drivers=lambda: None,
        tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
        between_items=pauser,
    )
    pass2_job = next(j for j in jobs if j.name == "pass2")
    outcome = pass2_job.run()
    assert outcome is JobOutcome.PAUSED, "items remain queued -> must report PAUSED, not COMPLETED"
    assert len(ran) == 1, "exactly one item drained before the pause"
    assert pass2_queue.queue_length(persona_dir) == 2

    # Resume: idle now (no more pauses) -> drains the rest, item 1 never re-runs.
    pause_flag["on"] = False
    outcome2 = pass2_job.run()
    assert outcome2 is JobOutcome.COMPLETED
    assert len(ran) == 3
    assert pass2_queue.queue_length(persona_dir) == 0
    assert len(set(ran)) == 3, "no item drained twice"


# ---------------------------------------------------------------------------
# C8 — emotion backfill (S32 row 3): also the regression-guard for the real
# bug found while building this (a paused run wrongly marked status=complete).
# ---------------------------------------------------------------------------


def test_c8_emotion_backfill_pauses_between_memories_and_resumes(tmp_path: Path) -> None:
    from brain.ingest.emotion_backfill import run_emotion_backfill
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    ids = []
    for i in range(3):
        m = Memory.create_new(content=f"memory number {i} with enough content", memory_type="conversation", domain="us")
        store.create(m)
        ids.append(m.id)
    ids.sort()  # run_emotion_backfill processes in id order

    tagged_order: list[str] = []

    def _tagger(memory):
        tagged_order.append(memory.id)
        return {"joy": 5.0}

    pause = _pause_on_call(2)  # checked BEFORE each item; pause before item 1
    state = run_emotion_backfill(
        persona_dir, tagger_fn=_tagger, store=store, delay_s=0, should_pause=pause
    )
    assert state.status == "running", "a genuine between-items pause must NOT mark complete"
    assert tagged_order == [ids[0]]
    assert state.last_cursor == ids[0]

    # Regression guard: has_emotion_backfill_work must still see work (the
    # bug this run fixed would have wrongly reported "complete" here).
    from brain.ingest.emotion_backfill import has_emotion_backfill_work

    assert has_emotion_backfill_work(persona_dir, store=store) is True

    # Resume: no more pausing -> finishes the rest, item 0 never re-tagged.
    state2 = run_emotion_backfill(persona_dir, tagger_fn=_tagger, store=store, delay_s=0)
    assert state2.status == "complete"
    assert tagged_order == [ids[0], ids[1], ids[2]], "item 0 must not be re-tagged on resume"
    store.close()


def test_c8_emotion_backfill_zero_tagged_guard_is_not_confused_with_pause(tmp_path: Path) -> None:
    """A systematic tagger failure (processed candidates, tagged none) sets
    status='running' too, but for an UNRELATED reason -- supervisor.py's
    _emotion_backfill_job must only report JobOutcome.PAUSED for a REAL
    between-items pause, never for this case (else it would wrongly stop
    the whole S55 sequence for a tagger bug, not a chat-idle event) -- see
    test_supervisor.py::test_supervisor_tick_embeds_backlogged_memory for
    the end-to-end proof this exact confusion broke."""
    from brain.bridge.central_cadence import JobOutcome
    from brain.bridge.events import EventBus
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import _build_gated_jobs
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    m = Memory.create_new(content="a memory long enough to tag", memory_type="conversation", domain="us")
    store.create(m)
    store.close()

    jobs = _build_gated_jobs(
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=EventBus(),
        is_session_busy=None,
        finalize_after_hours=24.0,
        finalize_interval_s=None,
        initiate_review_interval_s=None,
        maintenance_interval_s=None,
        self_model_interval_s=None,
        compaction_interval_s=None,
        calibration_interval_s=None,
        interest_sweep_interval_s=None,
        judge_selftune_interval_s=None,
        clustering_interval_s=None,
        intensity_drivers=lambda: None,
        tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
        between_items=lambda: False,  # never idle-paused
    )
    job = next(j for j in jobs if j.name == "emotion_backfill")
    outcome = job.run()
    # FakeProvider's .complete() output isn't valid tagger JSON -> the
    # default tagger raises for every candidate -> zero tagged, status stays
    # "running" -- but this is NOT a chat-idle pause.
    assert outcome is JobOutcome.COMPLETED


# ---------------------------------------------------------------------------
# C8 — daily calibration (S32 row 11): offline judge-release pause proof
# (the real-model RSS measurement lives in test_judge_release_rss.py::
# test_c2c_calibration_tick_pause_releases_judge_rss).
# ---------------------------------------------------------------------------


def test_c8_calibration_pauses_between_rows_releases_judge_resumes(tmp_path: Path) -> None:
    from brain.bridge.provider import FakeProvider
    from brain.bridge.supervisor import _run_calibration_tick
    from brain.memory.relevance_judge import FakeRelevanceJudgeProvider
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona(tmp_path)
    store = MemoryStore(persona_dir / "memories.db")
    ids = []
    for i in range(2):
        m = Memory.create_new(content=f"calibration content {i}", memory_type="conversation", domain="us")
        store.create(m)
        store.log_calibration_sample(
            query=f"q{i}", candidate_ids=[m.id], reranker_scores=[5.0], reranker_model_id="m"
        )
        ids.append(m.id)
    store.close()

    judge = FakeRelevanceJudgeProvider(scores={(f"q{i}", f"calibration content {i}"): 10.0 for i in range(2)})
    pause = _pause_on_call(1)
    ran = _run_calibration_tick(
        persona_dir, provider=FakeProvider(), judge=judge, should_pause=pause
    )
    assert ran is None, "a mid-labeling pause must report None (paused)"

    store2 = MemoryStore(persona_dir / "memories.db")
    rows = store2._conn.execute(
        "SELECT COUNT(*) FROM calibration_log WHERE local_judge_label IS NOT NULL"
    ).fetchone()[0]
    unlabeled = store2._conn.execute(
        "SELECT COUNT(*) FROM calibration_log WHERE local_judge_label IS NULL"
    ).fetchone()[0]
    assert rows == 1, "exactly one row labeled before the pause"
    assert unlabeled == 1
    store2.close()

    # Resume: no more pausing -> the remaining row gets labeled, the first
    # (already local_judge_label IS NOT NULL) is never re-sampled/re-scored.
    ran2 = _run_calibration_tick(persona_dir, provider=FakeProvider(), judge=judge)
    assert ran2 is True
    store3 = MemoryStore(persona_dir / "memories.db")
    rows3 = store3._conn.execute(
        "SELECT COUNT(*) FROM calibration_log WHERE local_judge_label IS NOT NULL"
    ).fetchone()[0]
    assert rows3 == 2
    store3.close()


# ---------------------------------------------------------------------------
# C8 — maintenance/forgetting (S32 row 5): the NEW cursor mechanism.
# ---------------------------------------------------------------------------


def test_c8_forgetting_pauses_between_memories_new_cursor_resumes(tmp_path: Path, monkeypatch) -> None:
    from brain.bridge.events import EventBus
    from brain.forgetting import run_pass
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    ids = []
    for i in range(3):
        m = Memory.create_new(content=f"forgetting content {i}", memory_type="conversation", domain="us")
        store.create(m)
        ids.append(m.id)
    ids.sort()
    store.close()

    # Fresh test memories are exempt by policy.is_exempt's own "recent
    # buffer" grace (created within RECENT_LIVED_HOURS) -- bypass that so
    # the loop actually reaches the salience/score step this test measures,
    # rather than `continue`-ing past it (and past the should_pause check,
    # which sits after the state-transition block, not before it).
    monkeypatch.setattr("brain.forgetting.policy.is_exempt", lambda *a, **kw: False)
    monkeypatch.setattr(
        "brain.forgetting.policy.is_within_import_grace", lambda *a, **kw: False
    )

    scored: list[str] = []
    import brain.forgetting.salience as salience_mod

    def _counting_score(memory, **kw):
        scored.append(memory.id)
        return 5.0  # a fixed mid-range score: no fade/unfade/lose transition needed for this test

    monkeypatch.setattr(salience_mod, "score", _counting_score)
    # brain.forgetting imports salience as a module (`from brain.forgetting
    # import ... salience`) and calls salience.score(...), so patching the
    # module attribute above is the real call site.

    pause = _pause_on_call(1)
    bus = EventBus()
    summary = run_pass(persona_dir, event_bus=bus, should_pause=pause)
    assert scored == [ids[0]]
    assert summary["total"] == 3, "total reflects the whole backlog, not just this call's slice"

    progress = job_progress.load_progress(persona_dir, "forgetting")
    assert progress.get("last_id") == ids[0], "C31: the new cursor names the last item done"

    # Resume: idle now -> the rest run, item 0 never re-scored.
    run_pass(persona_dir, event_bus=bus)
    assert scored == [ids[0], ids[1], ids[2]]
    # A clean finish clears the cursor.
    assert job_progress.load_progress(persona_dir, "forgetting") == {}


def test_c8_forgetting_resume_cursor_survives_the_anchor_item_being_lost(
    tmp_path: Path, monkeypatch
) -> None:
    """Stage-6 red-team MAJOR, regression guard: when the item the cursor
    names underwent a LOSE transition (hard_delete -- the memory row is
    GONE from the next pass's re-queried set), an exact `m.id == last_id`
    search would never match, silently leaving resume_idx at 0 and
    reprocessing the WHOLE backlog. The fix makes the cursor a keyset
    position (first remaining id > last_id), correct whether or not that
    exact row still exists."""
    from brain.bridge.events import EventBus
    from brain.forgetting import run_pass
    from brain.memory.store import Memory, MemoryStore

    # 4 memories: item 0 survives untouched (processed BEFORE the pause);
    # item 1 is the one that gets LOST (also processed before the pause --
    # it IS the cursor anchor); items 2/3 are only reached on resume. This
    # shape is what actually distinguishes the fix from the bug: an
    # exact-match cursor search that fails to find the (now-deleted) anchor
    # falls back to resume_idx=0, which would WRONGLY re-score the
    # already-done item 0 too -- not merely "process the same set either
    # way", which a 2-memory version of this test can't tell apart (both
    # the buggy and fixed logic happen to process the same remaining set
    # when there's nothing processed before the lost anchor).
    persona_dir = _persona(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    ids = []
    for i in range(4):
        m = Memory.create_new(content=f"lose content {i}", memory_type="conversation", domain="us")
        store.create(m)
        ids.append(m.id)
    ids.sort()
    # Seed memory 1 as already "fading" with one prior low-salience pass
    # (LOST_PASS_COUNT=2, brain/forgetting/policy.py) so THIS pass's
    # low score pushes it straight to LOSE (hard_delete).
    store._conn.execute("UPDATE memories SET state='fading' WHERE id=?", (ids[1],))
    store._conn.commit()
    store.close()
    (persona_dir / "forgetting_state.json").write_text(json.dumps({ids[1]: 1}))

    monkeypatch.setattr("brain.forgetting.policy.is_exempt", lambda *a, **kw: False)
    monkeypatch.setattr("brain.forgetting.policy.is_within_import_grace", lambda *a, **kw: False)

    scored: list[str] = []
    import brain.forgetting.salience as salience_mod

    def _score(memory, **kw):
        scored.append(memory.id)
        # Below LOST_THRESHOLD (0.10) for memory 1 -> LOSE this pass;
        # comfortably above it (no transition) for the others.
        return 0.0 if memory.id == ids[1] else 5.0

    monkeypatch.setattr(salience_mod, "score", _score)

    pause = _pause_on_call(2)  # pause right after item 1 (the one that gets LOST)
    bus = EventBus()
    run_pass(persona_dir, event_bus=bus, should_pause=pause)
    assert scored == [ids[0], ids[1]]

    store2 = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    remaining = {r["id"] for r in store2._conn.execute("SELECT id FROM memories").fetchall()}
    store2.close()
    assert ids[1] not in remaining, "item 1 must have been hard-deleted (LOSE) before the pause"

    # Resume: item 1 (the cursor's own anchor) is GONE from the re-queried
    # set -- the keyset cursor must still correctly skip past BOTH item 0
    # (already done) and item 1 (deleted), processing only items 2/3.
    run_pass(persona_dir, event_bus=bus)
    assert scored == [ids[0], ids[1], ids[2], ids[3]], (
        "resume must process exactly items 2 and 3 once each -- item 0 must "
        "NOT be re-scored (the bug this guards: an unresolvable exact-match "
        "cursor silently restarts the whole backlog from index 0)"
    )


# ---------------------------------------------------------------------------
# C8 — compaction (S32 row 8): pause between sessions.
# ---------------------------------------------------------------------------


def test_c8_compaction_pauses_between_sessions_and_resumes(tmp_path: Path, monkeypatch) -> None:
    """Resuming re-lists ALL active sessions and re-applies cascade_conversation
    to each (by design — 2-plan §4.1 row 8's "per-session buffer state is
    already persisted per fold" means re-visiting an already-cascaded session
    is a safe no-op in production, since ITS OWN age gate finds nothing new).
    This double models that: cascade_conversation only records a NEW effect
    the first time it is called for a given session (an age-gated no-op on a
    revisit), so the observable-effects assertion below proves the resume
    did no new work on s1 while still genuinely re-visiting it."""
    from brain.bridge.supervisor import _run_compaction_tick

    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(
        "brain.ingest.buffer.list_active_sessions", lambda _pd: ["s1", "s2", "s3"]
    )
    cascaded: list[str] = []
    already_done: set[str] = set()

    def _fake_cascade(_pd, sid, **kw):
        if sid in already_done:
            return  # age-gated no-op, mirrors the real cascade_conversation
        already_done.add(sid)
        cascaded.append(sid)

    monkeypatch.setattr("brain.chat.compaction.cascade_conversation", _fake_cascade)
    monkeypatch.setattr("brain.chat.rollover.maybe_weekly_rollover", lambda *a, **kw: None)

    pause = _pause_on_call(1)
    paused = _run_compaction_tick(persona_dir, provider=None, should_pause=pause)
    assert paused is True
    assert cascaded == ["s1"]

    paused2 = _run_compaction_tick(persona_dir, provider=None)
    assert paused2 is False
    assert cascaded == ["s1", "s2", "s3"], "s1's cascade must have no NEW effect on resume"


# ---------------------------------------------------------------------------
# C8 — finalize (S32 row 13): pause between finalized sessions.
# ---------------------------------------------------------------------------


def test_c8_finalize_pauses_between_sessions_and_resumes(tmp_path: Path, monkeypatch) -> None:
    from brain.ingest.pipeline import finalize_stale_sessions
    from brain.ingest.types import IngestReport

    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(
        "brain.ingest.pipeline.list_active_sessions", lambda _pd: ["s1", "s2"]
    )
    monkeypatch.setattr("brain.ingest.pipeline.read_session", lambda _pd, sid: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr("brain.ingest.pipeline.session_silence_minutes", lambda turns: 9999.0)
    extracted: list[str] = []
    already_done: set[str] = set()

    def _fake_extract(_pd, sid, **kw):
        if sid not in already_done:
            already_done.add(sid)
            extracted.append(sid)
        return IngestReport(session_id=sid)

    monkeypatch.setattr("brain.ingest.pipeline.extract_session_snapshot", _fake_extract)
    monkeypatch.setattr("brain.ingest.pipeline.read_backoff", lambda _pd, sid: None)

    pause = _pause_on_call(1)
    paused_out: list[bool] = []
    reports = finalize_stale_sessions(
        persona_dir,
        finalize_after_hours=24.0,
        store=None,
        hebbian=None,
        provider=None,
        should_pause=pause,
        paused_out=paused_out,
    )
    assert [r.session_id for r in reports] == ["s1"]
    assert paused_out == [True]

    finalize_stale_sessions(
        persona_dir, finalize_after_hours=24.0, store=None, hebbian=None, provider=None
    )
    assert extracted == ["s1", "s2"], "s1 must have no NEW extraction effect on resume"


# ---------------------------------------------------------------------------
# C8 — session snapshot (S32 row 2, snapshot half): pause between sessions.
# ---------------------------------------------------------------------------


def test_c8_snapshot_pauses_between_sessions_and_resumes(tmp_path: Path, monkeypatch) -> None:
    from brain.ingest.pipeline import snapshot_stale_sessions
    from brain.ingest.types import IngestReport

    persona_dir = _persona(tmp_path)
    monkeypatch.setattr(
        "brain.ingest.pipeline.list_active_sessions", lambda _pd: ["s1", "s2"]
    )
    monkeypatch.setattr("brain.ingest.pipeline.read_session", lambda _pd, sid: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr("brain.ingest.pipeline.session_silence_minutes", lambda turns: 9999.0)
    extracted: list[str] = []
    already_done: set[str] = set()

    def _fake_extract(_pd, sid, **kw):
        if sid not in already_done:
            already_done.add(sid)
            extracted.append(sid)
        return IngestReport(session_id=sid)

    monkeypatch.setattr("brain.ingest.pipeline.extract_session_snapshot", _fake_extract)

    pause = _pause_on_call(1)
    paused_out: list[bool] = []
    reports = snapshot_stale_sessions(
        persona_dir,
        silence_minutes=0.0,
        store=None,
        hebbian=None,
        provider=None,
        should_pause=pause,
        paused_out=paused_out,
    )
    assert [r.session_id for r in reports] == ["s1"]
    assert paused_out == [True]

    snapshot_stale_sessions(
        persona_dir, silence_minutes=0.0, store=None, hebbian=None, provider=None
    )
    assert extracted == ["s1", "s2"], "s1 must have no NEW extraction effect on resume"


# ---------------------------------------------------------------------------
# C8 — initiate review (S32 row 14): pause between candidates.
# ---------------------------------------------------------------------------


def test_c8_initiate_review_pauses_between_candidates_and_resumes(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from brain.initiate.emit import emit_initiate_candidate, read_candidates
    from brain.initiate.review import run_initiate_review_tick
    from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

    persona_dir = _persona(tmp_path)

    def _promote_all_reflection_run(candidates, *, deps):
        from brain.initiate.d_call_schema import DCallRow, make_d_call_id
        from brain.initiate.reflection import DDecision, DReflectionResult

        decisions = [DDecision(i, "promote", "test stub", "high") for i in range(len(candidates))]
        result = DReflectionResult(decisions=decisions, tick_note=None)
        dcall = DCallRow(
            d_call_id=make_d_call_id(deps.now), ts=deps.now.isoformat(), tick_id=deps.tick_id,
            model_tier_used="haiku", candidates_in=len(candidates), promoted_out=len(candidates),
            filtered_out=0, latency_ms=0, tokens_input=0, tokens_output=0,
        )
        return result, dcall

    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all_reflection_run)

    snap = EmotionalSnapshot(
        vector={"longing": 7}, rolling_baseline_mean=5.0, rolling_baseline_stdev=1.0,
        current_resonance=7.4, delta_sigma=2.4,
    )
    ctx = SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])
    for i in range(2):
        emit_initiate_candidate(
            persona_dir, kind="message", source="dream", source_id=f"dream_{i}",
            emotional_snapshot=snap, semantic_context=ctx,
        )

    provider = MagicMock()
    canned = ["subject", "tone", '{"decision": "send_quiet", "reasoning": "x"}'] * 2
    provider.complete = MagicMock(side_effect=canned)

    pause = _pause_on_call(1)
    paused_out: list[bool] = []
    run_initiate_review_tick(
        persona_dir, provider=provider, voice_template="x", cap_per_tick=3,
        should_pause=pause, paused_out=paused_out,
    )
    assert paused_out == [True]
    remaining = read_candidates(persona_dir)
    assert len(remaining) == 1, "one candidate processed and removed, one still queued"

    provider2 = MagicMock()
    provider2.complete = MagicMock(side_effect=["subject", "tone", '{"decision": "send_quiet", "reasoning": "x"}'])
    run_initiate_review_tick(persona_dir, provider=provider2, voice_template="x", cap_per_tick=3)
    assert read_candidates(persona_dir) == [], "the remaining candidate is processed on resume"
