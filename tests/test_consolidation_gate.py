"""Tests for the temporary memory-consolidation gate (Root 2 stopgap).

Covers the gating acceptance criteria C1-C14 + C18 from
changes/tempgate-memory-consolidation/1.5-criteria.md. Pass-2 decisions are
driven by an injected STUB classifier so the mechanism is verified without a
live model (C16 — real Haiku decision quality — is advisory/live, not here).

Naming: synthetic user = Bob, persona = Canary, model = Claude. No Phoebe.
"""

from __future__ import annotations

import threading

import pytest

from brain.engines.consolidation import (
    _ARCHIVE_FILENAME,
    _GATE_LOCK_FILENAME,
    Decision,
    run_consolidation,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import (
    GATE_BYPASS_TYPES,
    SALIENCE_ELIGIBLE_TYPES,
    PendingQueue,
    route_write,
)
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import file_lock


@pytest.fixture
def persona(tmp_path):
    """A persona dir with a MemoryStore + HebbianMatrix + PendingQueue."""
    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    queue = PendingQueue(tmp_path)
    yield tmp_path, store, hebbian, queue
    store.close()
    hebbian.close()


def _mem(content, mtype="dream", *, emotions=None, importance=None, domain="us"):
    return Memory.create_new(
        content=content,
        memory_type=mtype,
        domain=domain,
        emotions=emotions or {},
        importance=importance,
    )


def _stub(mapping):
    """Classifier returning a fixed Decision per candidate content."""

    def _c(cand, _context):
        return mapping.get(cand.content, Decision("new"))

    return _c


# --------------------------------------------------------------------------- C1
def test_c1_automatic_gated_write_enqueues_not_in_db(persona):
    tmp, store, _heb, queue = persona
    m = _mem("Canary had a flat dream", "dream")
    route_write(store, m, source="dream")
    # In the queue, NOT in memories.db.
    assert queue.read_recent("dream", limit=10)
    assert store.search_text("flat dream") == []


# --------------------------------------------------------------------------- C2
def test_c2_candidate_never_in_recall_but_promoted_is(persona):
    from brain.chat.prompt import _build_recall_block

    tmp, store, _heb, queue = persona
    queue.enqueue(_mem("Bob mentioned the harbor", "monologue"), source="x")
    # Candidate is invisible to the real recall block (it is not a memories.db row).
    assert store.search_text("harbor") == []
    assert "harbor" not in _build_recall_block(store, "harbor")
    # A genuine memory with matching content DOES surface in recall.
    store.create(_mem("the harbor at dawn", "conversation"))
    assert any("harbor" in m.content for m in store.search_text("harbor"))
    assert "harbor" in _build_recall_block(store, "harbor")


# --------------------------------------------------------------------------- C3
def test_c3_interior_continuity_reads_queue(persona):
    from brain.monologue.ambient import _AMBIENT_LIMIT, build_interior_continuity_block
    from brain.monologue.recall import recall_monologue
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    tmp, store, hebbian, queue = persona
    for i in range(3):
        queue.enqueue(
            _mem(f"Canary thought number {i}", MONOLOGUE_TRACE_TYPE, importance=0.3),
            source="trace",
        )
    block = build_interior_continuity_block(store)
    assert "Canary thought number" in block
    assert _AMBIENT_LIMIT == 5  # owner directive 3: unchanged
    # recall_monologue also reads the queue.
    res = recall_monologue("number", store=store, hebbian=hebbian, persona_dir=tmp)
    assert res["count"] >= 1


# --------------------------------------------------------------------------- C4
def test_c4_reject_discards_no_side_effects(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("dup content", "dream"), source="x")
    before = store.count(active_only=False)
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({"dup content": Decision("duplicate")}),
    )
    assert store.count(active_only=False) == before  # nothing created
    assert not (tmp / _ARCHIVE_FILENAME).exists()  # no archive
    assert not (tmp / "forgotten_memories.jsonl").exists()  # no grief graveyard entry
    edge_count = hebbian._conn.execute("SELECT COUNT(*) FROM hebbian_edges").fetchone()[0]
    assert edge_count == 0  # no edges created at all
    assert queue.drain() == []  # candidate consumed, not re-queued


# --------------------------------------------------------------------------- C5
def test_c5_promote_creates_row_and_round_trips_fields(persona):
    tmp, store, hebbian, queue = persona
    m = _mem("distinct memory", "dream", emotions={"joy": 4.0}, importance=6.5)
    m.tags = ["t1", "t2"]
    m.metadata = {"k": "v"}
    queue.enqueue(m, source="x")
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({"distinct memory": Decision("new")}),
    )
    hits = store.search_text("distinct memory")
    assert len(hits) == 1
    got = hits[0]
    assert got.emotions == {"joy": 4.0}
    assert got.importance == 6.5
    assert set(got.tags) == {"t1", "t2"}
    assert got.metadata.get("k") == "v"


# --------------------------------------------------------------------------- C6
def test_c6_pass1_exact_dup_only_never_non_identical(persona):
    tmp, store, hebbian, queue = persona
    # (a) within-batch exact repeat (normalized): one dropped
    queue.enqueue(_mem("The Dog Ate.", "dream"), source="x")
    queue.enqueue(_mem("the dog   ate.", "dream"), source="x")  # same after normalize
    # (b) high-overlap non-identical pair: both survive
    queue.enqueue(_mem("the dog ate cat food", "dream"), source="x")
    queue.enqueue(_mem("the dog ate the cat as food", "dream"), source="x")
    res = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian, classifier=_stub({}),  # all "new"
    )
    assert res.exact_dropped == 1  # exactly one exact repeat dropped
    # both non-identical survivors promoted; the surviving exact + the two = 3 rows
    contents = [m.content for m in store.list_active()]
    assert "the dog ate cat food" in contents
    assert "the dog ate the cat as food" in contents


def test_c6_exact_dup_vs_existing_db(persona):
    tmp, store, hebbian, queue = persona
    store.create(_mem("already committed here", "conversation"))
    queue.enqueue(_mem("already committed here", "dream"), source="x")
    res = run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=_stub({}))
    assert res.exact_dropped == 1
    assert store.count(active_only=False) == 1  # nothing new added


# --------------------------------------------------------------------------- C7
def test_c7_merge_archives_surgical_no_graveyard(persona):
    tmp, store, hebbian, queue = persona
    target_id = store.create(_mem("Bob likes tea", "conversation"))
    queue.enqueue(_mem("Bob likes tea with honey", "dream"), source="x")
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({
            "Bob likes tea with honey": Decision(
                "merge", target_id=target_id,
                merged_content="Bob likes tea with honey",
            )
        }),
    )
    # (a) pre-image archived to the PLAIN archive (not the grief graveyard)
    assert (tmp / _ARCHIVE_FILENAME).exists()
    archive = (tmp / _ARCHIVE_FILENAME).read_text()
    assert "Bob likes tea" in archive
    assert not (tmp / "forgotten_memories.jsonl").exists()  # grief graveyard untouched
    # (b) target edited to contain the new fact; (d) candidate not also promoted
    got = store.get(target_id)
    assert "honey" in got.content
    assert store.count(active_only=False) == 1  # only the (edited) target, no new row


def test_c7_merge_defers_when_target_missing(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("orphan merge", "dream"), source="x")
    res = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({"orphan merge": Decision("merge", target_id="nonexistent")}),
    )
    assert res.deferred == 1
    assert queue.read_recent("dream", limit=5)  # re-enqueued for next tick


# --------------------------------------------------------------------------- C8
def test_c8_correction_promotes_and_links_weight5(persona):
    tmp, store, hebbian, queue = persona
    ref_id = store.create(_mem("the meeting is at 3pm", "conversation"))
    queue.enqueue(_mem("actually the meeting is at 4pm", "dream"), source="x")
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({
            "actually the meeting is at 4pm": Decision("correction", target_id=ref_id)
        }),
    )
    hits = store.search_text("at 4pm")
    assert len(hits) == 1  # kept as its own memory
    new_id = hits[0].id
    assert hits[0].metadata.get("correction_of") == ref_id
    weight = dict(hebbian.neighbors(new_id)).get(ref_id)
    assert weight == 5.0


def test_c8_weight5_over_preexisting_weak_edge(persona):
    tmp, store, hebbian, queue = persona
    ref_id = store.create(_mem("part A of the story", "conversation"))
    # Promote first so both ids exist, then seed a weak edge, then correct.
    cand = _mem("part B continues the story", "dream")
    queue.enqueue(cand, source="x")
    # pre-seed a weak edge between the (soon-to-exist) candidate id and ref
    hebbian.strengthen(cand.id, ref_id, delta=0.5)
    assert dict(hebbian.neighbors(cand.id)).get(ref_id) == 0.5
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({
            "part B continues the story": Decision("continuation", target_id=ref_id)
        }),
    )
    assert dict(hebbian.neighbors(cand.id)).get(ref_id) == 5.0  # raised, not left at 0.5


# --------------------------------------------------------------------------- C9
def test_c9_gate_runs_before_reflex_in_run_tick():
    import inspect

    from brain.engines import heartbeat

    src = inspect.getsource(heartbeat.HeartbeatEngine.run_tick)
    assert "run_consolidation(" in src
    assert src.index("run_consolidation(") < src.index("_try_fire_reflex(")


# -------------------------------------------------------------------------- C10
def test_c10_no_candidate_lost_under_enqueue_drain_race(persona):
    tmp, store, _heb, queue = persona
    n = 60
    drained: list[dict] = []
    lock = threading.Lock()
    start = threading.Event()

    def enqueue_worker():
        start.wait()
        for i in range(n):
            queue.enqueue(_mem(f"race {i}", "dream"), source="x")

    def drain_worker():
        start.wait()
        for _ in range(30):
            got = queue.drain()
            if got:
                with lock:
                    drained.extend(got)

    threads = [threading.Thread(target=enqueue_worker), threading.Thread(target=drain_worker)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()
    remaining = queue.drain()
    total = len(drained) + len(remaining)
    assert total == n  # every enqueued candidate is accounted for — none lost


def test_file_lock_non_blocking_contract(tmp_path):
    """file_lock(blocking=False) yields True when acquired, False when contended,
    and never holds the lock on a False (the gate's skip contract; the path the
    Windows branch must also honour)."""
    p = tmp_path / "x.dat"
    with file_lock(p, blocking=False) as first:
        assert first is True
        with file_lock(p, blocking=False) as second:
            assert second is False  # a second holder is refused, not blocked
    # released on exit — acquirable again
    with file_lock(p, blocking=False) as third:
        assert third is True


# -------------------------------------------------------------------------- C11
def test_c11_overlapping_gate_run_skips(persona):
    tmp, store, hebbian, queue = persona
    queue.enqueue(_mem("only once", "dream"), source="x")
    # Hold the gate lock, then a concurrent run must SKIP (not drain/act).
    with file_lock(tmp / _GATE_LOCK_FILENAME, blocking=False) as acquired:
        assert acquired
        res = run_consolidation(store, persona_dir=tmp, hebbian=hebbian, classifier=_stub({}))
        assert res.skipped is True
    # queue untouched by the skipped run
    assert queue.read_recent("dream", limit=5)


def test_c11_merge_folded_exactly_once(persona):
    tmp, store, hebbian, queue = persona
    target_id = store.create(_mem("base fact", "conversation"))
    queue.enqueue(_mem("base fact plus extra", "dream"), source="x")
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({
            "base fact plus extra": Decision(
                "merge", target_id=target_id, merged_content="base fact plus extra"
            )
        }),
    )
    # exactly one archive record (one fold)
    lines = (tmp / _ARCHIVE_FILENAME).read_text().strip().splitlines()
    assert len(lines) == 1


# -------------------------------------------------------------------------- C12
def test_c12_deliberate_bypass_types_write_direct(persona):
    tmp, store, _heb, queue = persona
    for bypass in sorted(GATE_BYPASS_TYPES):
        m = _mem(f"deliberate {bypass}", bypass)
        route_write(store, m, source="deliberate")
        assert store.search_text(f"deliberate {bypass}")  # in memories.db
    assert queue.drain() == []  # nothing enqueued


# -------------------------------------------------------------------------- C14
def test_c14_temp_marker_present():
    from pathlib import Path

    for f in ("brain/memory/pending.py", "brain/engines/consolidation.py"):
        text = Path(f).read_text()
        assert "TEMP (Root 2 stopgap" in text


# -------------------------------------------------------------------------- C18
def test_c18_salience_scoped_legit_content_survives(persona):
    tmp, store, hebbian, queue = persona
    # exempt types with low/zero importance — must SURVIVE a floor of 1.0
    queue.enqueue(_mem("a flat dream", "dream", emotions={}), source="x")  # importance 0.0
    queue.enqueue(_mem("some research note", "research", emotions={"x": 2.0}), source="x")
    from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

    queue.enqueue(_mem("a trace thought", MONOLOGUE_TRACE_TYPE, importance=0.3), source="x")
    # an ELIGIBLE episode below the floor — must be DROPPED
    queue.enqueue(_mem("low value episode", "monologue", importance=0.5), source="x")

    res = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian, salience_floor=1.0, classifier=_stub({}),
    )
    survived = {m.content for m in store.list_active()}
    # dream/research promoted; trace stays in the queue (not salience-dropped)
    assert "a flat dream" in survived
    assert "some research note" in survived
    assert res.salience_dropped == 1  # only the eligible low episode
    assert "low value episode" not in survived
    # trace was exempt — it survived Pass 1 and (as monologue_trace) promoted
    assert "a trace thought" in survived


def test_c18_eligible_types_are_only_episode_types():
    assert SALIENCE_ELIGIBLE_TYPES == {
        "monologue", "monologue_emotion", "monologue_soul_candidate"
    }
    assert "dream" not in SALIENCE_ELIGIBLE_TYPES
    assert "monologue_trace" not in SALIENCE_ELIGIBLE_TYPES
