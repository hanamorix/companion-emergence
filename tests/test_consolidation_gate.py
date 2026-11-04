"""Tests for the temporary memory-consolidation gate (Root 2 stopgap).

Covers the gating acceptance criteria C1-C13 + C18 from
changes/tempgate-memory-consolidation/1.5-criteria.md. Pass-2 decisions are
driven by an injected STUB classifier so the mechanism is verified without a
live model (C16 — real Haiku decision quality — is advisory/live, not here).

Naming: synthetic user = Bob, persona = Canary, model = Claude. No real persona or user names.
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from brain import dev_constants
from brain.engines import consolidation
from brain.engines.consolidation import (
    _ARCHIVE_FILENAME,
    _GATE_LOCK_FILENAME,
    Decision,
    Reappraisal,
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


# --------------------------------------------------------------------------- F1 #259 step 4
def test_embed_on_write_promoted_memory_has_row_embedding_and_matrix_entry(
    persona, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 #259 step 4 (embed-on-write): a candidate promoted via
    `consolidation._dispatch` (the ``store.create(cand)`` call at the
    promote branch) must have its row `embedding` populated and its warm
    matrix entry set IMMEDIATELY — the steady-state path, not waiting for
    the next idle backfill tick. Fails pre-fix: without the
    `store.embed_row(cand.id, cand.content)` call right after
    `store.create(cand)`, the row's `embedding` column stays NULL and the
    matrix never sees the id."""
    tmp, store, hebbian, queue = persona
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod
    from brain.memory.embedding_matrix import build_embedding_matrix

    # Two separate alignments needed for this test's `matrix.get(...)` check
    # to actually find what `embed_row` wrote:
    #  (1) DIMENSION — the suite-wide autouse fixture fakes the provider to
    #      FakeEmbeddingProvider(dim=256). `EmbeddingMatrix._load_from_db`
    #      (#259 inc7 red-team F1) decodes each row to its OWN stored
    #      byte-length and no longer gates on any expected-dim constant, so
    #      a 256-dim embed WOULD now appear in the matrix — this override to
    #      a 384-dim fake provider is kept for realism/consistency with the
    #      rest of this test's production-shaped expectations, not because a
    #      256-dim row would be silently dropped.
    #  (2) MODEL ID — `embed_row` embeds via the process-cached provider,
    #      but the matrix's lazy-build filter is sourced from
    #      `model_tier.model_for_tier(TIER_EMBEDDING)` (F1 #259 step 0) — a
    #      SEPARATE lookup (see tests/unit/brain/memory/test_semantic_recall.py's
    #      `_align_embedding_tier` for the identical need). Align the two.
    fake_provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: fake_provider)
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, fake_provider.model_id())

    content = "a memory that should be embedded immediately at promotion"
    m = _mem(content, "dream")
    queue.enqueue(m, source="x")
    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({content: Decision("new")}),
    )

    hits = store.search_text("embedded immediately at promotion")
    assert len(hits) == 1
    promoted = hits[0]

    row = store._conn.execute(
        "SELECT embedding, embedding_model_id FROM memories WHERE id = ?", (promoted.id,)
    ).fetchone()
    assert row["embedding"] is not None, "row embedding must be populated at promotion time, not NULL"
    assert row["embedding_model_id"] == fake_provider.model_id()

    matrix = build_embedding_matrix(store.db_path)
    assert matrix.get(promoted.id) is not None, "the warm matrix must reflect the embed immediately"


# --------------------------------------------------------------------------- F1 #259 F3 (acceptance #7b)
def test_f3_cosine_near_dup_dedup_catches_same_batch_paraphrase(persona) -> None:
    """F1 #259 F3 / spec §4b (acceptance #7b): two PARAPHRASE candidates of
    the same fact, enqueued in the same batch, share no >3-char SUBSTRING
    (see the precondition note below — `_related_existing_lexical` probes
    via `store.search_text`, a plain `LIKE '%token%'` SUBSTRING match, not
    exact-token FTS, so this is a stricter bar than "no shared word") — the
    OLD lexical `_related_existing` would have returned empty context for
    the second candidate, so only the NEW cosine retrieval can surface the
    first as context for the second's Pass-2 judgment. Consolidation
    commits incrementally (`_run_locked`'s per-candidate loop calls
    `store.create` inside the loop), so the first candidate is already a
    committed, warm-matrix-visible row by the time the second candidate's
    `_related_existing` runs — same-batch coverage, no special-casing.

    Threshold-free (I3): the cosine retrieval never filters by a similarity
    cutoff, it just returns whatever ranks in the warm matrix's top-k (here
    trivially rank 1, since nothing else is embedded) — the REALISTIC
    classifier below is what actually decides duplicate-ness from that
    context, exactly like the real Haiku judge would. A promote-all stub
    would pass this test vacuously; `realistic_classifier` does not: it only
    rejects a candidate when its specific paraphrase partner is PRESENT in
    the context list `_related_existing` handed it, so this test fails (2
    rows, not 1) if the cosine retrieval regresses to lexical-only or to no
    context at all — proving the mechanism, not just the assertion. (Verified
    by hand during the F3 build: monkeypatching `_related_existing_cosine`
    to raise makes this test fail with 2 rows, confirming it genuinely
    bites — see the F3 build report.)"""
    tmp, store, hebbian, queue = persona

    content_a = "Bob's grandmother always made braunschweiger sandwiches on rye"
    content_b = "The old family recipe actually used liverwurst instead of ham"
    # Precondition (checked by hand, not asserted at runtime, to keep the
    # test focused on the mechanism): none of content_b's >3-char tokens
    # ("family", "recipe", "actually", "used", "liverwurst", "instead")
    # appear anywhere as a SUBSTRING of content_a — `search_text` is a
    # `LIKE '%token%'` substring probe, so even a partial overlap (e.g.
    # "sandwich" inside "sandwiches") would defeat this precondition; these
    # two strings were picked specifically to have none. The OLD lexical
    # probe could never have caught this near-dup, so the fix is genuinely
    # load-bearing here, not incidentally covered by substring luck.

    queue.enqueue(_mem(content_a, "dream"), source="x")
    queue.enqueue(_mem(content_b, "dream"), source="x")

    paraphrase_partner = {content_a: content_b, content_b: content_a}

    def realistic_classifier(cand, context):
        partner = paraphrase_partner.get(cand.content)
        if partner is not None and any(m.content == partner for m in context):
            return Decision("duplicate")
        return Decision("new")

    run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=realistic_classifier,
    )

    rows = store.list_active()
    assert len(rows) == 1, (
        f"expected exactly one durable row after same-batch near-dup dedup, "
        f"got {len(rows)}: {[r.content for r in rows]}"
    )


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
def test_c9_gate_runs_before_reflex_in_run_tick(monkeypatch, tmp_path):
    """C9: on every heartbeat tick that reaches the tick body, the
    consolidation gate must run before reflex evaluation, so each idle
    cycle consolidates the accumulated pending-candidate queue before the
    generative engines (reflex/dream) produce the next cycle's candidates.

    Behavioral proof, not a source-order grep: patches both
    `run_consolidation` and `HeartbeatEngine._try_fire_reflex` to record
    the order they're actually invoked in at runtime, across
    `HeartbeatEngine.run_tick`'s real dispatch (first-ever-tick init,
    then a fresh tick). ram-spike-fix INC-7 split run_tick's old
    monolithic body into `_run_tick_body` (fresh tick) / `_resume_decay_only`
    (resumes an interrupted decay pass and stops, per spec §5) — a source
    grep on `run_tick` itself no longer finds either call. The resumed
    path runs neither consolidation nor reflex at all (nothing else that
    tick), so it can't violate this ordering; only the fresh-tick path is
    exercised here.
    """
    from brain.bridge.provider import FakeProvider
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    # Keep dream/reflex/research/growth from constructing a real CLI
    # provider against a bare tmp_path persona (same guard
    # test_heartbeat_decay_batching.py uses).
    monkeypatch.setattr(
        "brain.engines.heartbeat.build_tier_provider", lambda *a, **k: FakeProvider()
    )

    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    engine = HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=tmp_path / "hb_state.json",
        config_path=tmp_path / "hb_config.json",
        dream_log_path=tmp_path / "dreams.log.jsonl",
        heartbeat_log_path=tmp_path / "heartbeats.log.jsonl",
        persona_name="Canary",
        persona_system_prompt="You are Canary.",
    )
    # First-ever tick only initializes state and defers all work — get it
    # out of the way so the second call exercises the real fresh-tick body.
    engine.run_tick()

    call_order: list[str] = []

    def _fake_run_consolidation(*_args, **_kwargs):
        call_order.append("consolidation")

    monkeypatch.setattr(
        "brain.engines.consolidation.run_consolidation", _fake_run_consolidation
    )

    orig_try_fire_reflex = HeartbeatEngine._try_fire_reflex

    def _spy_try_fire_reflex(self, *args, **kwargs):
        call_order.append("reflex")
        return orig_try_fire_reflex(self, *args, **kwargs)

    monkeypatch.setattr(HeartbeatEngine, "_try_fire_reflex", _spy_try_fire_reflex)

    engine.run_tick()

    assert call_order == ["consolidation", "reflex"], (
        f"consolidation gate must run before reflex evaluation, got: {call_order}"
    )


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


# --------------------------------------------------------------------------- Engine 5 (#580)
# The gate's merge combines both memories' emotions once the emotion engine is live; the temp-gate
# write-back writes faded values and a fresh created_at; the merge pre-read is not an access.
# `emotion_engine_stub` makes `engine.engine_for(...)` return one settable stand-in (`stub.live`).

_FIXED_NOW = datetime(2031, 5, 6, 7, 8, 9, tzinfo=UTC)


class _FixedDatetime(datetime):
    """`datetime` whose `now()` is `_FIXED_NOW`: the one stamp the write-back takes."""

    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


def _raw(store, memory_id, *columns):
    cols = ", ".join(columns)
    return store._conn.execute(f"SELECT {cols} FROM memories WHERE id = ?", (memory_id,)).fetchone()


_RUN_COUNTER = itertools.count()


def _merge_run(persona, target_emotions, cand_emotions, *, merged="Bob likes tea with honey", with_cand=False):
    """Make a target (no hook fired for it), queue a candidate, run a merge verdict. Returns (target_id, result), or
    (target_id, result, cand) with `with_cand` (the candidate as queued)."""
    tmp, store, hebbian, queue = persona
    # Distinct texts per call: a candidate whose text equals a stored row is dropped by the gate's exact-duplicate pass.
    n = next(_RUN_COUNTER)
    target = _mem(f"Bob likes tea number {n}", "conversation", emotions=target_emotions)
    target_id = store.create(target, fold=False)
    cand = _mem(f"Bob likes tea with honey, take {n}", "dream", emotions=cand_emotions)
    queue.enqueue(cand, source="x")
    result = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({cand.content: Decision("merge", target_id=target_id, merged_content=merged)}),
    )
    if with_cand:
        return target_id, result, cand
    return target_id, result


def _assert_probe_can_see_hooks(store, stub):
    """Positive control: a plain create (fold left True) records a hook call on the same stub."""
    stub.recorded_calls.clear()
    store.create(_mem("control row", "conversation", emotions={"joy": 1.0}))
    assert [c[0] for c in stub.recorded_calls] == ["on_memory_committed"]


_MERGE_TABLE = [
    pytest.param({"joy": 7, "trust": 4}, {"joy": 5, "awe": 6}, {"joy": 6.0, "trust": 4.0, "awe": 6.0}, id="shared-averaged-one-sided-copied"),
    pytest.param({"Joy": 5}, {"joy": 7}, {"joy": 6.0}, id="two-spellings-one-entry-average"),
    pytest.param({"Joy": 5}, {}, {"joy": 5.0}, id="lone-target-Joy-becomes-joy"),
    pytest.param({}, {"Joy": 5}, {"joy": 5.0}, id="lone-candidate-Joy-becomes-joy"),
    pytest.param({"Comfort-Seeking": 4}, {"comfort_seeking": 8}, {"comfort_seeking": 6.0}, id="hyphen-and-case"),
    pytest.param({"love_grief_blend": 2}, {"grief_love_blend": 6}, {"grief_love_blend": 4.0}, id="blend-pair-order"),
    pytest.param({"Joy": 5, "joy": 7}, {"joy": 8}, {"joy": 7.0}, id="within-one-memory-then-across"),
    pytest.param({"Joy": 5, "joy": 7}, {}, {"joy": 6.0}, id="within-one-memory-only"),
    pytest.param({"joy": 12.5}, {}, {"joy": 10.0}, id="one-sided-value-held-to-10"),
    pytest.param({"joy": -3}, {}, {"joy": 0.0}, id="one-sided-negative-held-to-0"),
    pytest.param({"joy": 12}, {"joy": 4}, {"joy": 7.0}, id="shared-out-of-range-held-then-averaged"),
    pytest.param({}, {"joy": 15}, {"joy": 10.0}, id="candidate-out-of-range-held"),
    pytest.param({"Joy": 20, "joy": 4}, {}, {"joy": 10.0}, id="spellings-averaged-then-held"),
    pytest.param({"Joy": 20, "joy": 4}, {"joy": 4}, {"joy": 7.0}, id="spellings-averaged-held-then-averaged-across"),
    pytest.param({"joy": 5}, {"joy": 5}, {"joy": 5.0}, id="equal-values"),
    pytest.param({}, {}, {}, id="both-empty"),
]


@pytest.mark.parametrize(("target", "cand", "expected"), _MERGE_TABLE)
def test_e5_merge_emotions_table(target, cand, expected):
    target_before, cand_before = dict(target), dict(cand)
    got = consolidation._merge_emotions(target, cand)
    assert got == expected
    assert list(got) == list(expected)
    assert got is not target
    assert got is not cand
    assert target == target_before
    assert cand == cand_before


def test_e5_merge_emotions_orders_the_target_names_first():
    got = consolidation._merge_emotions({"trust": 4, "joy": 7}, {"awe": 6, "Joy": 5})
    assert list(got) == ["trust", "joy", "awe"]
    assert got["joy"] == 6.0


_UNUSABLE = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param("5", id="string"),
    pytest.param(None, id="none"),
    pytest.param(True, id="bool"),
    pytest.param(10**400, id="int-too-big-for-a-float"),
]


@pytest.mark.parametrize("bad", _UNUSABLE)
@pytest.mark.parametrize("side", ["target", "candidate"])
@pytest.mark.parametrize("shared", [True, False])
def test_e5_merge_emotions_drops_a_value_that_is_not_a_finite_number(bad, side, shared, caplog):
    good = {"trust": 3.0}
    with_bad = {"joy" if shared else "awe": bad, **good}
    other = {"joy": 4.0}
    target, cand = (with_bad, other) if side == "target" else (other, with_bad)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine_utils"):
        got = consolidation._merge_emotions(target, cand)  # no exception
    # The bad value is dropped; on a shared name the other side's value is then the only one. The rest merges.
    assert got == {"trust": 3.0, "joy": 4.0}
    dropped = [r.getMessage() for r in caplog.records if "not a finite number" in r.getMessage()]
    assert len(dropped) == 1
    assert ("'joy'" if shared else "'awe'") in dropped[0]


def test_e5_merge_emotions_of_an_all_unusable_side_with_an_empty_side_is_empty():
    assert consolidation._merge_emotions({"joy": float("nan"), "awe": None}, {}) == {}
    assert consolidation._merge_emotions({}, {"joy": "5", "awe": True}) == {}


def test_e5_live_merge_averages_shared_and_copies_one_sided(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, res = _merge_run(persona, {"joy": 7.0, "trust": 4.0}, {"joy": 5.0, "awe": 6.0})
    assert res.merged == 1
    got = store.get(target_id, bump=False)
    assert got.emotions == {"joy": 6.0, "trust": 4.0, "awe": 6.0}
    assert got.content == "Bob likes tea with honey"
    assert store.count(active_only=False) == 1  # the candidate was not promoted


def test_e5_live_merge_joy_5_and_7_is_joy_6_under_the_standard_spelling(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"Joy": 5.0}, {"joy": 7.0})
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}


def test_e5_live_merge_lone_Joy_becomes_joy_with_the_same_value(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"Joy": 5.0}, {})
    assert store.get(target_id, bump=False).emotions == {"joy": 5.0}


def test_e5_live_merge_sends_one_commit_event_for_the_incoming_memory_and_none_for_the_target(
    persona, emotion_engine_stub
):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    # A target with NO emotions that gets some by the merge: with fold left True this is the "assigned" hook's own case.
    target_id, _res = _merge_run(persona, {}, {"joy": 5.0})
    assert store.get(target_id, bump=False).emotions == {"joy": 5.0}
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]  # the target's update fired none
    # The probe can see: the same empty-to-filled update with fold left True records the assigned hook.
    emotion_engine_stub.recorded_calls.clear()
    other_id = store.create(_mem("another row", "conversation"), fold=False)
    store.update(other_id, emotions={"joy": 5.0})
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_emotions_assigned"]
    _assert_probe_can_see_hooks(store, emotion_engine_stub)


def test_e5_the_commit_event_is_the_incoming_memorys_own_not_the_merged_emotions(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, res, cand = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0, "awe": 2.0}, with_cand=True)
    assert res.merged == 1
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0, "awe": 2.0}
    assert len(emotion_engine_stub.recorded_calls) == 1
    name, sent = emotion_engine_stub.recorded_calls[0]
    assert name == "on_memory_committed"
    assert sent.id == cand.id
    assert sent.emotions == {"joy": 7.0, "awe": 2.0}  # the candidate's own, not the merged {joy: 6, awe: 2}
    assert sent.created_at == cand.created_at
    assert sent.memory_type == cand.memory_type
    assert store.get(cand.id, bump=False) is None  # the merged candidate was never created as a row
    assert target_id != cand.id


def test_e5_the_commit_event_follows_the_write_not_before_it(persona, emotion_engine_stub, monkeypatch):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    log = []
    real_update = MemoryStore.update

    def spy(self, memory_id, **fields):
        real_update(self, memory_id, **fields)
        log.append(("update", memory_id))

    monkeypatch.setattr(MemoryStore, "update", spy)

    def hook(memory):
        # At the moment of the hook the target row already holds the merged content.
        log.append(("on_memory_committed", store.get(target_ids[0], bump=False).content))

    monkeypatch.setattr(emotion_engine_stub, "on_memory_committed", hook)
    target_ids = []
    real_create = store.create

    def remember_target(memory, **kw):
        out = real_create(memory, **kw)
        target_ids.append(out)
        return out

    monkeypatch.setattr(store, "create", remember_target)
    target_id, _res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    assert [entry[0] for entry in log] == ["update", "on_memory_committed"]
    assert log[0][1] == target_id
    assert log[1][1] == "Bob likes tea with honey"


def test_e5_live_merge_leaves_score_and_raises_peak_only_by_max(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    # A candidate-only emotion above the old peak raises the peak (update()'s own MAX); score is a snapshot.
    target_id, _res = _merge_run(persona, {"joy": 3.0}, {"joy": 3.0, "awe": 9.0})
    got = store.get(target_id, bump=False)
    assert got.score == pytest.approx(3.0)
    assert got.peak_emotion_intensity == pytest.approx(9.0)
    assert got.emotions == {"joy": 3.0, "awe": 9.0}


def test_e5_live_merge_never_lowers_the_peak(persona, emotion_engine_stub):
    tmp, store, hebbian, queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"joy": 9.0}, {"joy": 1.0})
    got = store.get(target_id, bump=False)
    assert got.emotions == {"joy": 5.0}
    assert got.peak_emotion_intensity == pytest.approx(9.0)


def test_e5_not_live_merge_is_todays_content_only_call(persona, emotion_engine_stub, monkeypatch):
    from brain.memory.store import MemoryStore

    _tmp, store, _heb, _queue = persona
    calls = []
    real_update = MemoryStore.update

    def spy(self, memory_id, **fields):
        calls.append((memory_id, fields))
        return real_update(self, memory_id, **fields)

    monkeypatch.setattr(MemoryStore, "update", spy)

    # Live first, the control: the same rows, a live merge, DO write the emotions (so the not-live arm below is a real contrast).
    emotion_engine_stub.live = True
    live_id, _ = _merge_run(persona, {"Joy": 5.0}, {"joy": 7.0, "awe": 6.0})
    assert store.get(live_id, bump=False).emotions == {"joy": 6.0, "awe": 6.0}
    assert {"emotions", "fold"} <= set(next(f for mid, f in calls if mid == live_id))
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]  # the live merge's event
    assert store.get(live_id, bump=False).content == "Bob likes tea with honey"  # a merge adds no date to the text

    emotion_engine_stub.live = False
    emotion_engine_stub.recorded_calls.clear()
    target_id, res = _merge_run(persona, {"Joy": 5.0}, {"joy": 7.0, "awe": 6.0})
    assert res.merged == 1
    assert emotion_engine_stub.recorded_calls == []  # not live: no event either
    assert [fields for mid, fields in calls if mid == target_id] == [{"content": "Bob likes tea with honey"}]
    got = store.get(target_id, bump=False)
    assert got.emotions == {"Joy": 5.0}  # untouched, spelling included
    assert got.peak_emotion_intensity == pytest.approx(5.0)
    assert got.content == "Bob likes tea with honey"


def test_e5_live_merge_of_a_target_deleted_before_the_write_defers_the_candidate(
    persona, emotion_engine_stub, monkeypatch
):
    from brain.memory.store import MemoryStore

    tmp, store, _heb, queue = persona
    emotion_engine_stub.live = True
    real_update = MemoryStore.update
    failing_call_fields = []

    def delete_then_update(self, memory_id, **fields):
        failing_call_fields.append(fields)
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return real_update(self, memory_id, **fields)  # raises KeyError: the row is gone

    monkeypatch.setattr(MemoryStore, "update", delete_then_update)
    _target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    # The live write is the one that failed: it carried the merged emotions and fold=False.
    assert failing_call_fields == [
        {"content": "Bob likes tea with honey", "emotions": {"joy": 6.0}, "fold": False}
    ]
    assert res.deferred == 1
    assert res.merged == 0
    assert queue.read_recent("dream", limit=5)  # the candidate is back in the queue
    assert emotion_engine_stub.recorded_calls == []  # a merge that did not happen sends no event


def _run_merge_verdict(persona, cand, target_id, *, merged="Bob likes tea with honey"):
    tmp, store, hebbian, queue = persona
    return run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({cand.content: Decision("merge", target_id=target_id, merged_content=merged)}),
    )


def test_e5_a_deferred_merge_sends_no_event_and_the_tick_that_writes_it_sends_one(persona, emotion_engine_stub):
    tmp, store, hebbian, queue = persona
    emotion_engine_stub.live = True
    target = _mem("Bob likes tea number 900", "conversation", emotions={"joy": 5.0})
    target_id = store.create(target, fold=False)
    cand = _mem("Bob likes tea with honey, take 900", "dream", emotions={"joy": 7.0})
    # Tick 1: the verdict names an id the store does not know (a missing target): deferred.
    queue.enqueue(cand, source="x")
    first = _run_merge_verdict(persona, cand, "no-such-id")
    assert first.deferred == 1
    assert first.merged == 0
    assert emotion_engine_stub.recorded_calls == []
    # Tick 2: the target is mid-fade (state 'fading'): deferred again.
    store._conn.execute("UPDATE memories SET state = 'fading' WHERE id = ?", (target_id,))
    store._conn.commit()
    second = _run_merge_verdict(persona, cand, target_id)
    assert second.deferred == 1
    assert second.merged == 0
    assert emotion_engine_stub.recorded_calls == []
    assert store.get(target_id, bump=False).emotions == {"joy": 5.0}
    # Tick 3: the target is active again, the verdict is the same: the merge is written and the event is sent, once.
    store._conn.execute("UPDATE memories SET state = 'active' WHERE id = ?", (target_id,))
    store._conn.commit()
    third = _run_merge_verdict(persona, cand, target_id)
    assert third.merged == 1
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]
    assert emotion_engine_stub.recorded_calls[0][1].id == cand.id
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}
    # The probe can see.
    _assert_probe_can_see_hooks(store, emotion_engine_stub)


def _set_raw_emotions(store, memory_id, emotions):
    """Put emotions on a row with a direct UPDATE (`create` drops non-finite values; the column takes anything)."""
    import json

    store._conn.execute(
        "UPDATE memories SET emotions_json = ? WHERE id = ?", (json.dumps(emotions, allow_nan=True), memory_id)
    )
    store._conn.commit()


def _archived_target_emotions(tmp, target_id):
    import json

    records = [json.loads(line) for line in (tmp / _ARCHIVE_FILENAME).read_text().splitlines() if line.strip()]
    mine = [r for r in records if r["target"]["id"] == target_id]
    assert len(mine) == 1
    return mine[0]["target"]["emotions"]


# What the archive writes for each unusable stored value (it writes JSON with `nonfinite="null"`).
_UNUSABLE_ARCHIVED = [
    pytest.param(float("nan"), None, id="nan"),
    pytest.param(float("inf"), None, id="inf"),
    pytest.param("5", "5", id="string"),
    pytest.param(None, None, id="none"),
    pytest.param(True, True, id="bool"),
    pytest.param(10**400, 10**400, id="int-too-big-for-a-float"),
]


@pytest.mark.parametrize(("bad", "archived"), _UNUSABLE_ARCHIVED)
def test_e5_live_merge_drops_a_stored_value_that_is_not_a_finite_number_and_merges_the_rest(
    persona, emotion_engine_stub, caplog, bad, archived
):
    tmp, store, hebbian, queue = persona
    emotion_engine_stub.live = True
    target_id = store.create(_mem("Bob likes tea", "conversation", emotions={"joy": 7.0}), fold=False)
    cand = _mem("Bob likes tea with honey", "dream", emotions={"awe": 6.0})
    queue.enqueue(cand, source="x")
    _set_raw_emotions(store, target_id, {"joy": bad, "trust": 3.0})
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine_utils"):
        res = run_consolidation(
            store, persona_dir=tmp, hebbian=hebbian,
            classifier=_stub({cand.content: Decision("merge", target_id=target_id, merged_content="merged text")}),
        )
    assert res.merged == 1
    got = store.get(target_id, bump=False)
    assert got.content == "merged text"  # the content merge happens
    assert got.emotions == {"trust": 3.0, "awe": 6.0}  # the unusable value is dropped, the rest merged
    assert sum("not a finite number" in r.getMessage() for r in caplog.records) == 1
    # The pre-merge row is in the archive (NaN and inf are written as null there).
    assert _archived_target_emotions(tmp, target_id) == {"joy": archived, "trust": 3.0}


def test_e5_live_merge_of_an_all_unusable_target_and_an_empty_candidate_writes_empty_emotions(
    persona, emotion_engine_stub
):
    tmp, store, hebbian, queue = persona
    emotion_engine_stub.live = True
    target_id = store.create(_mem("Bob likes tea", "conversation", emotions={"joy": 7.0}), fold=False)
    cand = _mem("Bob likes tea with honey", "dream")
    queue.enqueue(cand, source="x")
    _set_raw_emotions(store, target_id, {"joy": float("nan"), "awe": None})
    res = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({cand.content: Decision("merge", target_id=target_id, merged_content="merged text")}),
    )
    assert res.merged == 1
    assert store.get(target_id, bump=False).emotions == {}


def test_e5_live_merge_holds_an_out_of_range_value_to_0_10(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"joy": 12.5}, {})
    assert store.get(target_id, bump=False).emotions == {"joy": 10.0}
    target_id, _res = _merge_run(persona, {}, {"joy": 15.0})
    assert store.get(target_id, bump=False).emotions == {"joy": 10.0}


def test_e5_live_merge_drops_a_bool_value_on_the_candidate_with_a_log_line(persona, emotion_engine_stub, caplog):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine_utils"):
        target_id, res = _merge_run(persona, {}, {"joy": True, "awe": 6.0})
    assert res.merged == 1
    assert store.get(target_id, bump=False).emotions == {"awe": 6.0}
    assert sum("not a finite number" in r.getMessage() for r in caplog.records) == 1


def test_e5_live_merge_two_spellings_in_one_memory_then_across(persona, emotion_engine_stub):
    # Planning's ruling (Q1): the target's Joy 5 and joy 7 are one entry, 6; across the two memories 6 and the candidate's 8 give 7.
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"Joy": 5.0, "joy": 7.0}, {"joy": 8.0})
    assert store.get(target_id, bump=False).emotions == {"joy": 7.0}


def test_e5_live_merge_leaves_the_target_created_at_alone(persona, emotion_engine_stub):
    # Planning's ruling (Q4): the merge writes content, emotions and fold=False only.
    tmp, store, hebbian, queue = persona
    emotion_engine_stub.live = True
    old = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    target = replace(_mem("Bob likes tea", "conversation", emotions={"joy": 5.0}), created_at=old)
    target_id = store.create(target, fold=False)
    cand = _mem("Bob likes tea with honey", "dream", emotions={"joy": 7.0})
    queue.enqueue(cand, source="x")
    before = _raw(store, target_id, "created_at")[0]
    res = run_consolidation(
        store, persona_dir=tmp, hebbian=hebbian,
        classifier=_stub({cand.content: Decision("merge", target_id=target_id, merged_content="merged text")}),
    )
    assert res.merged == 1
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}  # the live merge ran
    assert _raw(store, target_id, "created_at")[0] == before == old.isoformat()


def _assert_preread_is_not_an_access(store, target_id, res):
    assert res.merged == 1
    got = store.get(target_id, bump=False)
    assert got.recall_count == 0.0
    assert got.last_accessed_at is None
    # The probe can see a bump: an ordinary read counts.
    store.get(target_id)
    assert store.get(target_id, bump=False).recall_count == 1.0


def test_e5_live_merge_preread_is_not_an_access(persona, emotion_engine_stub):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    _assert_preread_is_not_an_access(store, target_id, res)


def test_e5_not_live_merge_preread_is_not_an_access(persona):
    # No stub: the real engine with no switch file, so this fails on the bump alone on any base.
    _tmp, store, _heb, _queue = persona
    target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    _assert_preread_is_not_an_access(store, target_id, res)


def test_e5_live_merge_archives_the_pre_merge_emotions(persona, emotion_engine_stub):
    import json

    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    target_id, _res = _merge_run(persona, {"Joy": 5.0, "trust": 4.0}, {"joy": 7.0})
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0, "trust": 4.0}
    records = [json.loads(line) for line in (tmp / _ARCHIVE_FILENAME).read_text().splitlines() if line.strip()]
    mine = [r for r in records if r["target"]["id"] == target_id]
    assert len(mine) == 1
    assert mine[0]["target"]["emotions"] == {"Joy": 5.0, "trust": 4.0}  # the old emotions are kept in the archive


# --- the temp-gate write-back ---------------------------------------------------------------------------


def _reappraise_run(tmp, store, mem, *, new_importance=7.25, reappraiser=None):
    from brain.memory.pending import PendingQueue

    PendingQueue(tmp).enqueue_reappraisal(mem.id, source="recall")
    fake = reappraiser or (lambda m: Reappraisal(new_importance))
    return run_consolidation(store, persona_dir=tmp, reappraiser=fake)


def _count_refresh_calls(monkeypatch):
    """Replace `refresh_faded_emotions` with a recorder (it writes nothing). Returns the list of recorded calls."""
    from brain.memory.store import MemoryStore

    calls = []
    monkeypatch.setattr(MemoryStore, "refresh_faded_emotions", lambda self, *a, **k: calls.append(a))
    return calls


def _control_write_back_is_seen(tmp, store, stub, calls):
    """Positive control: a live write-back of a second row records exactly one call, so an empty recorder means something."""
    was_live = stub.live
    stub.live = True
    control = _mem("a control fact", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(control, fold=False)
    _reappraise_run(tmp, store, control)
    stub.live = was_live
    assert [call[0] for call in calls] == [control.id]
    calls.clear()


@pytest.mark.parametrize("memory_type", ["conversation", "dream", "making"])
def test_e5_write_back_writes_faded_values_and_a_new_created_at(
    persona, emotion_engine_stub, monkeypatch, memory_type
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = _mem("Bob's address is 12 Elm Street", memory_type, emotions={"joy": 8.0, "trust": 4.0}, importance=3.0)
    store.create(mem, fold=False)
    emotion_engine_stub.faded[(mem.id, "joy")] = 3.5
    emotion_engine_stub.faded[(mem.id, "trust")] = 1.25
    seen_importance = []
    real_faded = emotion_engine_stub.faded_value

    def spy(memory, emotion):
        seen_importance.append(memory.importance)
        return real_faded(memory, emotion)

    monkeypatch.setattr(emotion_engine_stub, "faded_value", spy)

    res = _reappraise_run(tmp, store, mem)

    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.emotions == {"joy": 3.5, "trust": 1.25}
    assert got.created_at == _FIXED_NOW
    assert _raw(store, mem.id, "created_at")[0] == "2031-05-06T07:08:09+00:00"
    assert got.importance == pytest.approx(7.25)
    assert got.peak_emotion_intensity == pytest.approx(8.0)
    # The write-back's own read of the row is a maintenance read, not an access.
    assert got.recall_count == 0.0
    assert got.last_accessed_at is None
    # Faded from the memory as it was loaded: the importance BEFORE this re-appraisal.
    assert seen_importance == [3.0, 3.0]
    assert emotion_engine_stub.recorded_calls == []  # no hook
    _assert_probe_can_see_hooks(store, emotion_engine_stub)


@pytest.mark.parametrize(("column", "value"), [("state", "fading"), ("active", 0)])
def test_e5_write_back_includes_rows_of_every_state(persona, emotion_engine_stub, monkeypatch, column, value):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = _mem("an old fact", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    store._conn.execute(f"UPDATE memories SET {column} = ? WHERE id = ?", (value, mem.id))
    store._conn.commit()
    emotion_engine_stub.faded[(mem.id, "joy")] = 2.0
    res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.emotions == {"joy": 2.0}
    assert got.created_at == _FIXED_NOW


def test_e5_write_back_for_a_memory_with_empty_emotions_resets_created_at_only(
    persona, emotion_engine_stub, monkeypatch
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = _mem("a quiet fact", "conversation", importance=3.0)
    store.create(mem, fold=False)
    res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.emotions == {}
    assert got.created_at == _FIXED_NOW
    assert got.peak_emotion_intensity == 0.0
    assert got.importance == pytest.approx(7.25)


def test_e5_not_live_writes_nothing_back(persona, emotion_engine_stub, monkeypatch):
    tmp, store, _heb, _queue = persona
    refreshed = _count_refresh_calls(monkeypatch)
    _control_write_back_is_seen(tmp, store, emotion_engine_stub, refreshed)
    emotion_engine_stub.live = False
    mem = _mem("a stable fact", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    before = store.get(mem.id, bump=False)
    emotion_engine_stub.faded[(mem.id, "joy")] = 1.0
    faded_asked = []
    real_faded = emotion_engine_stub.faded_value
    monkeypatch.setattr(
        emotion_engine_stub, "faded_value", lambda m, e: faded_asked.append(e) or real_faded(m, e)
    )
    res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.importance == pytest.approx(7.25)  # the importance update still happens
    assert got.emotions == before.emotions
    assert got.created_at == before.created_at
    assert got.content == before.content  # no date added when not live
    assert refreshed == []
    assert faded_asked == []


def test_e5_a_raising_write_back_is_logged_and_leaves_the_row_alone(
    persona, emotion_engine_stub, monkeypatch, caplog
):
    from brain.memory.store import MemoryStore

    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    before = store.get(mem.id, bump=False)
    emotion_engine_stub.faded[(mem.id, "joy")] = 1.0
    attempts = []

    def boom(self, memory_id, emotions, created_at):
        attempts.append(memory_id)
        raise RuntimeError("disk said no")

    monkeypatch.setattr(MemoryStore, "refresh_faded_emotions", boom)
    with caplog.at_level(logging.ERROR, logger="brain.engines.consolidation"):
        res = _reappraise_run(tmp, store, mem)
    assert attempts == [mem.id]  # the probe was reached
    assert any(r.levelno == logging.ERROR and mem.id in r.getMessage() for r in caplog.records)
    got = store.get(mem.id, bump=False)
    assert got.emotions == before.emotions
    assert got.created_at == before.created_at
    assert got.content == before.content
    assert got.importance == pytest.approx(7.25)  # the importance update stands
    assert res.reappraised == 1
    assert res.failed == 0
    assert not list(tmp.glob("*quarantine*"))
    # Settled, not requeued: a second pass has nothing to do.
    again = run_consolidation(store, persona_dir=tmp, reappraiser=lambda m: Reappraisal(1.0))
    assert again.reappraised == 0
    assert attempts == [mem.id]


def test_e5_a_failing_faded_value_is_logged_and_nothing_is_written(
    persona, emotion_engine_stub, monkeypatch, caplog
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0, "trust": 4.0}, importance=3.0)
    store.create(mem, fold=False)
    before = store.get(mem.id, bump=False)
    asked = []

    def faded_value(memory, emotion):
        asked.append(emotion)
        raise KeyError(emotion)

    monkeypatch.setattr(emotion_engine_stub, "faded_value", faded_value)
    with caplog.at_level(logging.ERROR, logger="brain.engines.consolidation"):
        res = _reappraise_run(tmp, store, mem)
    assert asked  # the probe was reached
    assert any(r.levelno == logging.ERROR and mem.id in r.getMessage() for r in caplog.records)
    got = store.get(mem.id, bump=False)
    assert got.emotions == before.emotions
    assert got.created_at == before.created_at
    assert got.content == before.content
    assert got.importance == pytest.approx(7.25)
    assert res.failed == 0


def test_e5_no_write_back_when_the_importance_update_fails(persona, emotion_engine_stub, monkeypatch):
    from brain.memory.store import MemoryStore

    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    refreshed = _count_refresh_calls(monkeypatch)
    _control_write_back_is_seen(tmp, store, emotion_engine_stub, refreshed)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    real_update = MemoryStore.update

    def failing_importance_update(self, memory_id, **fields):
        if memory_id == mem.id and "importance" in fields:
            raise RuntimeError("the importance update failed")  # not a KeyError: the row still exists
        return real_update(self, memory_id, **fields)

    monkeypatch.setattr(MemoryStore, "update", failing_importance_update)
    res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 0
    assert refreshed == []  # the write-back never ran: the row survives, so only the ORDER keeps it from writing


def test_e5_no_write_back_when_the_row_is_gone_before_the_importance_update(
    persona, emotion_engine_stub, monkeypatch
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    refreshed = _count_refresh_calls(monkeypatch)
    _control_write_back_is_seen(tmp, store, emotion_engine_stub, refreshed)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)

    def deleting_reappraiser(m):
        store._conn.execute("DELETE FROM memories WHERE id = ?", (m.id,))
        store._conn.commit()
        return Reappraisal(5.0)

    res = _reappraise_run(tmp, store, mem, reappraiser=deleting_reappraiser)
    assert res.reappraised == 0
    assert refreshed == []


def test_e5_a_row_deleted_right_after_the_importance_update_writes_nothing(
    persona, emotion_engine_stub, monkeypatch, caplog
):
    from brain.memory.store import MemoryStore

    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    refreshed = _count_refresh_calls(monkeypatch)
    _control_write_back_is_seen(tmp, store, emotion_engine_stub, refreshed)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    real_update = MemoryStore.update

    def update_then_delete(self, memory_id, **fields):
        real_update(self, memory_id, **fields)
        if memory_id == mem.id:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()

    monkeypatch.setattr(MemoryStore, "update", update_then_delete)
    with caplog.at_level(logging.ERROR, logger="brain.engines.consolidation"):
        res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1
    assert res.failed == 0
    assert refreshed == []  # the re-read found no row: nothing to write
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_e5_an_emotions_write_during_the_reappraiser_call_is_kept(
    persona, emotion_engine_stub, monkeypatch
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = _mem("Bob's address", "conversation", importance=3.0)  # loaded with no emotions
    store.create(mem, fold=False)

    def reappraiser_while_backfill_writes(m):
        assert m.emotions == {}  # the gate holds the stale row
        store.update(m.id, emotions={"joy": 6.0}, fold=False)  # the emotion backfill lands meanwhile
        return Reappraisal(7.25)

    res = _reappraise_run(tmp, store, mem, reappraiser=reappraiser_while_backfill_writes)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.emotions == {"joy": 6.0}  # not erased by a fade of the stale empty dict
    assert got.created_at == _FIXED_NOW


def _importance_sensitive_fade(memory, emotion):
    """A fade whose answer moves with the memory's importance, so the importance a write-back fades under shows in its output."""
    return memory.emotions[emotion] * (1 - memory.importance / 20)


@pytest.mark.parametrize("new_importance", [7.25, 9.5])
def test_e5_the_fade_in_a_write_back_does_not_depend_on_the_reappraised_importance(
    persona, emotion_engine_stub, monkeypatch, new_importance
):
    # Planning's ruling (Q3, spec 3.2): the fade uses the importance the memory was loaded with, not the re-appraised one.
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(emotion_engine_stub, "faded_value", _importance_sensitive_fade)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    res = _reappraise_run(tmp, store, mem, new_importance=new_importance)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.importance == pytest.approx(new_importance)  # the re-appraisal itself is stored
    assert got.emotions["joy"] == pytest.approx(8.0 * (1 - 3.0 / 20))  # 6.8: faded under the loaded 3.0


def test_e5_the_fade_takes_emotions_from_the_reread_row_and_importance_from_the_loaded_memory(
    persona, emotion_engine_stub, monkeypatch
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(emotion_engine_stub, "faded_value", _importance_sensitive_fade)
    mem = _mem("Bob's address", "conversation", importance=3.0)  # loaded with no emotions
    store.create(mem, fold=False)

    def reappraiser_while_backfill_writes(m):
        store.update(m.id, emotions={"joy": 6.0}, fold=False)  # emotions land during the model call
        return Reappraisal(9.5)

    _reappraise_run(tmp, store, mem, reappraiser=reappraiser_while_backfill_writes)
    got = store.get(mem.id, bump=False)
    assert got.importance == pytest.approx(9.5)
    assert got.emotions["joy"] == pytest.approx(6.0 * (1 - 3.0 / 20))  # 5.1: fresh emotions, loaded importance


def test_e5_a_later_fade_reads_the_reappraised_importance_and_the_reset_created_at(
    persona, emotion_engine_stub, monkeypatch
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    old_created_at = store.get(mem.id, bump=False).created_at
    emotion_engine_stub.faded[(mem.id, "joy")] = 6.0
    _reappraise_run(tmp, store, mem, new_importance=9.0)

    row = store.get(mem.id, bump=False)
    # The two inputs a later fade takes from the row are the new ones (the engine-level flip of `faded_value` is Engine 6's).
    assert row.importance == pytest.approx(9.0)
    assert row.created_at == _FIXED_NOW
    assert row.created_at != old_created_at
    probe = datetime(2031, 5, 16, 7, 8, 9, tzinfo=UTC)  # ten days after the write-back
    age_days = (probe - row.created_at).total_seconds() / 86400.0
    assert age_days == pytest.approx(10.0)
    stretched = 14.0 * (1.0 + dev_constants.HORIZON_IMPORTANCE_GAIN * 9.0 / 10.0)
    from brain.emotion.engine_utils import fade

    assert fade(row.emotions["joy"], age_days, 14.0, row.importance) == pytest.approx(6.0 * 0.5 ** (10.0 / stretched))


# --- the real engine and a real switch file (the stub ignores which directory it is asked about) ---------


def _switch_on(persona_dir):
    from brain.emotion import engine_files

    assert engine_files.write_switch(persona_dir, datetime(2030, 1, 1, tzinfo=UTC))


def test_e5_real_engine_merge_is_live_only_with_a_switch_file(persona):
    tmp, store, _heb, _queue = persona
    off_id, _ = _merge_run(persona, {"Joy": 5.0}, {"joy": 7.0})
    assert store.get(off_id, bump=False).emotions == {"Joy": 5.0}  # no switch file: today's content-only merge
    _switch_on(tmp)
    on_id, _ = _merge_run(persona, {"Joy": 5.0}, {"joy": 7.0})
    assert store.get(on_id, bump=False).emotions == {"joy": 6.0}  # switch file present: the merged emotions


def test_e5_real_engine_write_back_is_live_only_with_a_switch_file(persona):
    tmp, store, _heb, _queue = persona
    long_ago = datetime.now(UTC) - timedelta(days=30)
    off = replace(_mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0), created_at=long_ago)
    store.create(off, fold=False)
    before = store.get(off.id, bump=False)
    _reappraise_run(tmp, store, off)
    assert store.get(off.id, bump=False).created_at == before.created_at  # no switch file: nothing written back
    _switch_on(tmp)
    on = replace(_mem("Bob's phone", "conversation", emotions={"joy": 8.0}, importance=3.0), created_at=long_ago)
    store.create(on, fold=False)
    on_before = store.get(on.id, bump=False)
    _reappraise_run(tmp, store, on)
    after = store.get(on.id, bump=False)
    assert after.created_at > on_before.created_at  # switch file present: created_at reset to the gate's now
    assert after.emotions == {"joy": 8.0}  # the engine's own faded_value (today: the stored value)
    assert after.importance == pytest.approx(7.25)


# --- Engine 5, 2026-11-04: the merge's mood event on the real engine, and how the gate asks about liveness ----


def _record_real_engine_hook(monkeypatch, tmp, *, live):
    """Patch the REAL engine class: `is_live` answers `live`, `on_memory_committed` only records (it never runs the real hook, so
    the cell does not depend on what sits behind the hook). Returns the list of (persona_dir, memory) it was called with."""
    from brain.emotion import engine

    cls = type(engine.engine_for(tmp))
    seen = []
    monkeypatch.setattr(cls, "is_live", lambda self: live)
    monkeypatch.setattr(cls, "on_memory_committed", lambda self, memory: seen.append((self.persona_dir, memory)))
    return seen


def test_e5_real_engine_live_merge_sends_the_commit_event_for_the_persona_dir_it_was_given(persona, monkeypatch):
    tmp, store, _heb, _queue = persona
    seen = _record_real_engine_hook(monkeypatch, tmp, live=True)
    target_id, res, cand = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0}, with_cand=True)
    assert res.merged == 1
    assert [(path, m.id) for path, m in seen] == [(tmp.resolve(), cand.id)]
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}


def test_e5_real_engine_not_live_merge_sends_no_event(persona, monkeypatch):
    tmp, store, _heb, _queue = persona
    live_seen = _record_real_engine_hook(monkeypatch, tmp, live=True)
    _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    assert len(live_seen) == 1  # the probe can see: a live merge records its event
    off_seen = _record_real_engine_hook(monkeypatch, tmp, live=False)
    target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    assert res.merged == 1
    assert off_seen == []
    assert store.get(target_id, bump=False).emotions == {"joy": 5.0}  # not live: content only


def test_e5_the_gate_asks_is_live_once_per_merge_and_it_gates_the_write_and_the_event_together(
    persona, emotion_engine_stub, monkeypatch
):
    _tmp, store, _heb, _queue = persona
    asked = []

    def flipping_is_live():
        asked.append(1)
        return len(asked) == 1  # True the first time, False after

    monkeypatch.setattr(emotion_engine_stub, "is_live", flipping_is_live)
    target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    assert res.merged == 1
    assert len(asked) == 1
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}  # the emotions write happened
    assert [c[0] for c in emotion_engine_stub.recorded_calls] == ["on_memory_committed"]  # and so did the event


def test_e5_a_raising_commit_event_is_not_swallowed_by_the_merge(persona, emotion_engine_stub, monkeypatch):
    _tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True

    def raising_hook(memory):
        raise RuntimeError("the engine's guard is the contract; this hook broke it")

    monkeypatch.setattr(emotion_engine_stub, "on_memory_committed", raising_hook)
    target_id, res = _merge_run(persona, {"joy": 5.0}, {"joy": 7.0})
    # The merge had already committed; the failure reaches the per-item handler (item scope: quarantined), it is not caught
    # inside the merge branch. The retry edge (M06) is what this records.
    assert res.merged == 0
    assert res.failed == 1
    assert res.quarantined == 1
    assert store.get(target_id, bump=False).emotions == {"joy": 6.0}


# --- Engine 5, 2026-11-04: the date the write-back puts at the start of a recalled memory's text (sfa S31(c)) --------


_OLD_CREATED_AT = datetime(2031, 3, 4, 12, 0, 0, tzinfo=UTC)


def _patch_local_day(monkeypatch, hours=0):
    """Make the local-day conversion the store's date prefix goes through (`engine_utils.day_of` reads `engine_utils.to_local`) a
    fixed offset, so no cell depends on the host's zone. Returns the store module, whose placeholder the cells read."""
    import brain.emotion.engine_utils as engine_utils
    import brain.memory.store as store_module

    zone = timezone(timedelta(hours=hours))
    monkeypatch.setattr(engine_utils, "to_local", lambda dt: dt.astimezone(zone))
    return store_module


def _expected_prefix(store_module, iso):
    return store_module.DATE_PREFIX_FORMAT_PLACEHOLDER.replace("{date}", iso)


@pytest.mark.parametrize("memory_type", ["conversation", "dream", "making"])
def test_e5_write_back_puts_the_old_date_at_the_start_of_the_text_once(
    persona, emotion_engine_stub, monkeypatch, memory_type
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    store_module = _patch_local_day(monkeypatch)
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = replace(
        _mem("Bob's address is 12 Elm Street", memory_type, emotions={"joy": 8.0}, importance=3.0),
        created_at=_OLD_CREATED_AT,
    )
    store.create(mem, fold=False)
    emotion_engine_stub.faded[(mem.id, "joy")] = 3.5
    prefix = _expected_prefix(store_module, "2031-03-04")

    res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.content == prefix + "Bob's address is 12 Elm Street"
    assert got.created_at == _FIXED_NOW  # reset, and the date written is the one it had BEFORE the reset

    # A second re-appraisal: the date is already there, so it is not added again.
    again = _reappraise_run(tmp, store, got, new_importance=8.0)
    assert again.reappraised == 1
    assert store.get(mem.id, bump=False).content == prefix + "Bob's address is 12 Elm Street"


def test_e5_write_back_dates_a_memory_with_empty_emotions_too(persona, emotion_engine_stub, monkeypatch):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    store_module = _patch_local_day(monkeypatch)
    monkeypatch.setattr(consolidation, "datetime", _FixedDatetime)
    mem = replace(_mem("a quiet fact", "conversation", importance=3.0), created_at=_OLD_CREATED_AT)
    store.create(mem, fold=False)
    _reappraise_run(tmp, store, mem)
    got = store.get(mem.id, bump=False)
    assert got.content == _expected_prefix(store_module, "2031-03-04") + "a quiet fact"
    assert got.emotions == {}
    assert got.created_at == _FIXED_NOW


def test_e5_a_content_edit_during_the_reappraiser_call_is_kept_and_dated(persona, emotion_engine_stub, monkeypatch):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    store_module = _patch_local_day(monkeypatch)
    mem = replace(_mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0), created_at=_OLD_CREATED_AT)
    store.create(mem, fold=False)

    def reappraiser_while_another_writer_edits(m):
        assert m.content == "Bob's address"  # the gate holds the stale text
        store.update(m.id, content="Bob's new address")  # a content edit lands meanwhile
        return Reappraisal(7.25)

    _reappraise_run(tmp, store, mem, reappraiser=reappraiser_while_another_writer_edits)
    got = store.get(mem.id, bump=False)
    assert got.content == _expected_prefix(store_module, "2031-03-04") + "Bob's new address"  # not erased by the gate's copy


def test_e5_a_real_key_error_from_the_store_is_logged_and_not_requeued(
    persona, emotion_engine_stub, monkeypatch, caplog
):
    tmp, store, _heb, _queue = persona
    emotion_engine_stub.live = True
    _patch_local_day(monkeypatch)
    mem = _mem("Bob's address", "conversation", emotions={"joy": 8.0}, importance=3.0)
    store.create(mem, fold=False)
    real_refresh = MemoryStore.refresh_faded_emotions

    def delete_then_refresh(self, memory_id, emotions, created_at):
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))  # gone between the re-read and the UPDATE
        self._conn.commit()
        return real_refresh(self, memory_id, emotions, created_at)  # the real method: raises KeyError

    monkeypatch.setattr(MemoryStore, "refresh_faded_emotions", delete_then_refresh)
    with caplog.at_level(logging.ERROR, logger="brain.engines.consolidation"):
        res = _reappraise_run(tmp, store, mem)
    assert res.reappraised == 1  # the importance update stood
    assert res.failed == 0
    assert any(r.levelno == logging.ERROR and mem.id in r.getMessage() for r in caplog.records)
    again = run_consolidation(store, persona_dir=tmp, reappraiser=lambda m: Reappraisal(1.0))
    assert again.reappraised == 0  # settled, not requeued
