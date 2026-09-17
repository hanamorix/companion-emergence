"""Tests for brain.memory.semantic_recall — Stage 3 (semantic-PRIMARY
retrieval) + #231 RERANKER RE-ARCHITECTURE (floor-gated standout selection,
replacing the cosine-era per-persona calibration/shape classifier).

Covers the pure floor-gating/surfacing-tier logic directly (cheap, exact
boundary control at 5/6/9/10 candidates) and the candidate-pool builder's
warm-up/scoring-safety/state-filter contract. The end-to-end #88 case, the
surfacing tiers wired through the real recall block, and the recall-counter
tick semantics are covered as integration tests through
`brain.chat.prompt._build_recall_block` in
`tests/unit/brain/chat/test_semantic_primary_recall.py` — this file is the
unit layer underneath that.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import brain.memory.semantic_recall as semantic_recall_mod
from brain.bridge import model_tier
from brain.memory.embedding_matrix import EmbeddingMatrix
from brain.memory.reranker import FakeRerankerProvider
from brain.memory.semantic_recall import (
    FULL_INJECT_STANDOUT_MAX,
    MAX_STANDOUT_COUNT,
    RERANK_FLOOR,
    build_semantic_candidate_pool,
    run_semantic_recall,
    select_standouts,
)
from brain.memory.store import Memory, MemoryStore

_TEST_MODEL_ID = "fake-test-model"


def _seed_row_vector(store: MemoryStore, memory_id: str, vec: np.ndarray, *, model_id: str = _TEST_MODEL_ID) -> None:
    """F1 #259 step 2/3: write a vector directly onto a row's `embedding` /
    `embedding_model_id` columns — the matrix reads THIS (the memories.db
    row), not the old content-hash `embeddings.db` cache these tests used to
    seed. Bypasses the real provider entirely: these tests exercise the pool
    builder / fail-soft plumbing against a KNOWN vector, not a real embed."""
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (np.asarray(vec, dtype=np.float32).tobytes(), model_id, memory_id),
    )
    store._conn.commit()  # noqa: SLF001


def _align_embedding_tier(monkeypatch: pytest.MonkeyPatch, model_id: str = _TEST_MODEL_ID) -> None:
    """`run_semantic_recall` sources its matrix via `build_embedding_matrix`,
    which derives the matrix's filter model_id from
    `model_tier.model_for_tier(TIER_EMBEDDING)` (F1 #259 step 0) — NOT from
    whatever provider `build_embedding_provider()` is monkeypatched to for a
    given test. A test that seeds rows under `_TEST_MODEL_ID` and drives the
    REAL `run_semantic_recall` (as opposed to constructing its own
    `EmbeddingMatrix` directly, like the pool-builder tests below) must align
    the two so the matrix's lazy-build filter actually matches the seeded
    rows — otherwise the first `matrix.get`/`snapshot()` reloads from disk
    filtered to the (unrelated) real production model id, finds nothing, and
    silently discards whatever this test put there."""
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, model_id)


def _mem(store: MemoryStore, content: str, *, state: str = "active") -> Memory:
    m = Memory(
        id=str(uuid.uuid4()),
        content=content,
        memory_type="event",
        domain="d",
        created_at=datetime.now(UTC),
        importance=1.0,
        state=state,
    )
    store.create(m)
    return m


# ---------------------------------------------------------------------------
# select_standouts — floor filtering
# ---------------------------------------------------------------------------


def test_nothing_clears_the_floor_returns_none() -> None:
    scored = [("a", RERANK_FLOOR - 5.0), ("b", RERANK_FLOOR - 1.0)]
    assert select_standouts(scored) is None


def test_a_below_floor_candidate_never_enters_the_standout_set() -> None:
    scored = [("a", RERANK_FLOOR + 1.0), ("b", RERANK_FLOOR - 0.01)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == ["a"]
    assert tiers.snippet_ids == []


def test_a_score_exactly_at_the_floor_clears_it() -> None:
    """The floor comparison is >=, not > — a candidate scoring exactly
    RERANK_FLOOR counts as a standout, matching the pre-#231 classifier's
    own `< calibration.floor` (strict) exclusion rule."""
    scored = [("a", RERANK_FLOOR)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == ["a"]


# ---------------------------------------------------------------------------
# select_standouts — boundary counts (5 / 6 / 9 / 10+)
# ---------------------------------------------------------------------------


def _above_floor_scores(n: int, *, top: float = 5.0, step: float = 0.5) -> list[float]:
    """n scores, each clearing RERANK_FLOOR (the floor no longer cares about
    the GAP between them — every above-floor candidate is a standout)."""
    return [top - i * step for i in range(n)]


@pytest.mark.parametrize("n", [1, 5])
def test_le_5_standouts_all_full(n: int) -> None:
    scores = _above_floor_scores(n)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == [f"m{i}" for i in range(n)]
    assert tiers.snippet_ids == []


@pytest.mark.parametrize("n", [6, 9])
def test_6_to_9_standouts_top5_full_rest_snippet(n: int) -> None:
    scores = _above_floor_scores(n)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == [f"m{i}" for i in range(FULL_INJECT_STANDOUT_MAX)]
    assert tiers.snippet_ids == [f"m{i}" for i in range(FULL_INJECT_STANDOUT_MAX, n)]


def test_10_or_more_above_floor_caps_at_max_standout_count_not_lexical() -> None:
    """#231 correction: the old cosine-era '10+ = clump -> lexical' bucket
    is DROPPED. A trustworthy per-candidate reranker floor means 10+
    above-floor candidates are 10+ genuinely relevant results -- capped at
    MAX_STANDOUT_COUNT (top 5 full + 4 snippet), never demoted to the
    lexical fallback."""
    scores = _above_floor_scores(12)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == [f"m{i}" for i in range(FULL_INJECT_STANDOUT_MAX)]
    assert tiers.snippet_ids == [f"m{i}" for i in range(FULL_INJECT_STANDOUT_MAX, MAX_STANDOUT_COUNT)]
    assert len(tiers.full_ids) + len(tiers.snippet_ids) == MAX_STANDOUT_COUNT


def test_surfacing_tier_ids_are_in_reranker_selection_order() -> None:
    """full/snippet ids come out highest-reranker-score-first — presentation
    re-ordering is the CALLER's job (prompt.py), not this function's."""
    scores = _above_floor_scores(6)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    tiers = select_standouts(scored)
    assert tiers is not None
    assert tiers.full_ids == sorted(tiers.full_ids, key=lambda mid: -dict(scored)[mid])
    assert tiers.snippet_ids == sorted(tiers.snippet_ids, key=lambda mid: -dict(scored)[mid])


# ---------------------------------------------------------------------------
# build_semantic_candidate_pool — warm-up + bump-free scoring + #231
# fold-in fix (b): state=='active' filter.
# ---------------------------------------------------------------------------


def test_empty_cache_yields_empty_pool_no_active_scan(tmp_path: Path) -> None:
    """Warm-up / cold-start: a matrix with nothing embedded yet must
    short-circuit to an empty pool without even scanning list_active()."""
    store = MemoryStore(tmp_path / "memories.db")
    _mem(store, "some memory nobody has embedded yet")
    matrix = EmbeddingMatrix(store.db_path, model_id=_TEST_MODEL_ID)
    pool = build_semantic_candidate_pool(store, matrix)
    assert pool == {}


def test_pool_only_includes_memories_with_a_cached_vector(tmp_path: Path) -> None:
    """A memory whose row has no embedding yet (embed-on-write / backfill
    hasn't reached it) is simply absent from the pool — the pool builder
    reads only what the matrix already has, it never computes a new embed
    itself (the matrix primitive has no embed capability at all)."""
    store = MemoryStore(tmp_path / "memories.db")
    embedded = _mem(store, "this one is already embedded")
    not_embedded = _mem(store, "this one is NOT embedded yet")
    _seed_row_vector(store, embedded.id, np.zeros(384, dtype=np.float32))

    matrix = EmbeddingMatrix(store.db_path, model_id=_TEST_MODEL_ID)
    pool = build_semantic_candidate_pool(store, matrix)

    assert embedded.id in pool
    assert not_embedded.id not in pool


def test_pool_excludes_fading_state_memories_even_if_cached(tmp_path: Path) -> None:
    """#231 fold-in fix (b): a memory in state='fading' is still active=1
    (list_active() filters only the deactivation flag), so without this
    filter it could enter the semantic pool and double-render/double-bump
    alongside the separately-computed fading partition. The pool must
    filter to state=='active'.

    NOTE: ``store.create()`` does not persist an arbitrary ``Memory.state``
    passed to it (schema column ``state`` isn't in its INSERT column list;
    every row lands ``state='active'`` regardless of the dataclass value) —
    the only real way a memory transitions to ``state='fading'`` is the
    production path, ``store.fade(id, summary=...)``, which is what every
    other fading-memory test in this suite uses (see e.g.
    tests/unit/brain/memory/test_store.py, tests/unit/brain/chat/
    test_prompt.py). Mirror that here rather than constructing a Memory with
    state='fading' directly, which would silently exercise a state create()
    can never actually produce.

    Seeds vectors directly onto the rows AFTER `fade()` runs — `fade()` now
    synchronously re-embeds as an F1 #259 step 5 side effect (via the real,
    suite-faked provider), so seeding afterward pins the exact vector +
    model_id this test controls rather than depending on that side effect's
    output."""
    store = MemoryStore(tmp_path / "memories.db")
    active_mem = _mem(store, "an active memory")
    fading_mem = _mem(store, "the original content before it faded")
    fading_summary = "a softened fading memory"
    store.fade(fading_mem.id, summary=fading_summary)

    _seed_row_vector(store, active_mem.id, np.zeros(384, dtype=np.float32))
    _seed_row_vector(store, fading_mem.id, np.ones(384, dtype=np.float32))

    matrix = EmbeddingMatrix(store.db_path, model_id=_TEST_MODEL_ID)
    pool = build_semantic_candidate_pool(store, matrix)

    assert active_mem.id in pool
    assert fading_mem.id not in pool, "a state='fading' memory must never enter the semantic pool"


def test_pool_build_never_bumps_recall_count(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    m = _mem(store, "scored but maybe not surfaced")
    before = store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (m.id,)
    ).fetchone()[0]

    _seed_row_vector(store, m.id, np.zeros(384, dtype=np.float32))
    matrix = EmbeddingMatrix(store.db_path, model_id=_TEST_MODEL_ID)
    build_semantic_candidate_pool(store, matrix)

    after = store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (m.id,)
    ).fetchone()[0]
    assert after == before, "scoring/pool-building must never tick recall_count"


# ---------------------------------------------------------------------------
# run_semantic_recall — fail-soft contract (module docstring: "ANY failure
# ... must never break recall"). Regression coverage for the defect where
# only the cache-open and query-embed steps were individually wrapped in
# try/except; everything after (pool build, cosine scoring, reranking,
# floor-gating) sat inside a bare try/finally with NO except, so an
# exception there propagated straight out of run_semantic_recall instead of
# demoting the turn to the lexical fallback.
# ---------------------------------------------------------------------------


def test_run_semantic_recall_is_fail_soft_when_pool_build_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure inside build_semantic_candidate_pool (e.g. a real
    `sqlite3.OperationalError: database is locked` from the store, plausible
    given the supervisor's background-thread embedding backfill racing this
    call) must not propagate -- run_semantic_recall must catch it and return
    None, exactly like the already-caught cache-open/query-embed failures."""

    def _raise(store: MemoryStore, matrix: object) -> dict:
        raise RuntimeError("sqlite3.OperationalError: database is locked (simulated)")

    monkeypatch.setattr(semantic_recall_mod, "build_semantic_candidate_pool", _raise)

    store = MemoryStore(":memory:")
    _mem(store, "something the pool build never gets a chance to see")

    result = run_semantic_recall(store, tmp_path, "any query")

    assert result is None, "a pool-build failure must demote this turn to the lexical fallback, not raise"


def test_run_semantic_recall_is_fail_soft_when_scoring_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same contract, but the failure is AFTER a non-empty pool is built —
    inside cosine scoring — proving the catch covers the whole body, not
    just the pool-build call specifically. Seeds the vector directly onto
    the row and aligns `model_tier`'s embedding tier to the seeded model id
    (`_align_embedding_tier`) so `run_semantic_recall`'s internal
    `build_embedding_matrix` lazy-build actually finds it — a mismatched
    model id would make the pool come back empty and never reach scoring at
    all (see `_align_embedding_tier`'s docstring)."""

    def _raise_cosine(*args: object, **kwargs: object) -> float:
        raise RuntimeError("simulated scoring failure")

    monkeypatch.setattr(semantic_recall_mod, "cosine_similarity", _raise_cosine)
    _align_embedding_tier(monkeypatch)

    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem(store, "a memory that DOES have a cached vector")
    _seed_row_vector(store, mem.id, np.zeros(384, dtype=np.float32))

    result = run_semantic_recall(store, tmp_path, "any query")

    assert result is None, "a scoring failure must demote this turn to the lexical fallback, not raise"


def test_run_semantic_recall_is_fail_soft_when_reranker_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#231: a reranker failure must demote to the LEXICAL fallback, never
    raise and never fall back to raw cosine ranking (the unreliable signal
    the reranker replaces)."""

    class _BoomReranker(FakeRerankerProvider):
        def rerank(self, query: str, documents: list[str]):
            raise RuntimeError("simulated reranker failure")

    # semantic_recall.py calls `reranker_mod.build_reranker_provider()` — a
    # dynamic attribute lookup on the imported `reranker` MODULE at call
    # time (`from brain.memory import reranker as reranker_mod`), not a
    # name bound directly into semantic_recall's own namespace. Patch the
    # attribute on the reranker module itself (mirrors how conftest.py's
    # own `_fake_reranker_provider_by_default` fixture patches it).
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider", lambda: _BoomReranker()
    )
    _align_embedding_tier(monkeypatch)

    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem(store, "a memory that DOES have a cached vector")
    _seed_row_vector(store, mem.id, np.zeros(384, dtype=np.float32))

    result = run_semantic_recall(store, tmp_path, "any query")

    assert result is None, "a reranker failure must demote this turn to the lexical fallback, not raise"


# ---------------------------------------------------------------------------
# calibration-log write (F2a #250 inc4, spec Section 4 / acceptance #5) —
# every conclusive recall turn logs the REAL query + already-computed
# candidate ids/scores, and a logging failure must never demote a good
# semantic result to the lexical fallback.
# ---------------------------------------------------------------------------


def test_run_semantic_recall_writes_one_calibration_log_row_with_real_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A conclusive recall turn writes exactly one `calibration_log` row:
    the literal raw query string (byte-identical, never reconstructed —
    acceptance #5), the turn's candidate ids + reranker scores (the
    ALREADY-COMPUTED rerank output, reused as-is), and the reranker's
    model_id. The tie-break label columns (Section 6/F2c) start null."""
    content = "a memory that clears the floor"
    query = "  query with odd\twhitespace and Ünicode  "

    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda: FakeRerankerProvider(scores={content: RERANK_FLOOR + 1.0}),
    )
    _align_embedding_tier(monkeypatch)

    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem(store, content)
    _seed_row_vector(store, mem.id, np.zeros(384, dtype=np.float32))

    result = run_semantic_recall(store, tmp_path, query)

    assert result is not None
    rows = store._conn.execute(  # noqa: SLF001
        "SELECT query, candidate_ids, reranker_scores, reranker_model_id, "
        "day_bucket, local_judge_label, haiku_label FROM calibration_log"
    ).fetchall()
    assert len(rows) == 1, "exactly one calibration row per recall turn"
    row = rows[0]
    assert row["query"] == query, "logged query must be byte-identical to the literal recall-call argument"
    candidate_ids = json.loads(row["candidate_ids"])
    scores = json.loads(row["reranker_scores"])
    assert mem.id in candidate_ids
    assert scores[candidate_ids.index(mem.id)] == pytest.approx(RERANK_FLOOR + 1.0)
    assert row["reranker_model_id"] == "fake-reranker"  # FakeRerankerProvider.model_id()
    assert row["day_bucket"]  # populated, non-empty
    assert row["local_judge_label"] is None
    assert row["haiku_label"] is None


def test_calibration_log_write_failure_does_not_break_recall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A logging failure must never demote a good semantic result to the
    lexical fallback — it only loses that one turn's calibration row. The
    write is wrapped in its OWN try/except, separate from the outer
    fail-soft catch that governs the rest of the recall path."""
    content = "a memory that clears the floor"
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda: FakeRerankerProvider(scores={content: RERANK_FLOOR + 1.0}),
    )
    _align_embedding_tier(monkeypatch)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated calibration-log write failure")

    monkeypatch.setattr(MemoryStore, "log_calibration_sample", _boom)

    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem(store, content)
    _seed_row_vector(store, mem.id, np.zeros(384, dtype=np.float32))

    result = run_semantic_recall(store, tmp_path, "any query")

    assert result is not None, "a calibration-log write failure must not fall back to lexical"
    assert mem.id in [m.id for m in result.full]


