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

import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import brain.memory.semantic_recall as semantic_recall_mod
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
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
    """Warm-up / cold-start: an embeddings.db with nothing cached yet must
    short-circuit to an empty pool without even scanning list_active()."""
    store = MemoryStore(":memory:")
    _mem(store, "some memory nobody has embedded yet")
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    try:
        pool = build_semantic_candidate_pool(store, cache)
        assert pool == {}
    finally:
        cache.close()


def test_pool_only_includes_memories_with_a_cached_vector(tmp_path: Path) -> None:
    """A memory whose content was never embedded (backfill hasn't reached
    it) is simply absent from the pool — never triggers a new embed."""
    store = MemoryStore(":memory:")
    embedded = _mem(store, "this one is already embedded")
    not_embedded = _mem(store, "this one is NOT embedded yet")

    provider = FakeEmbeddingProvider(dim=8)
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    try:
        cache.get_or_compute(embedded.content)  # seed only ONE vector

        call_count = {"n": 0}
        real_embed = provider.embed

        def counting_embed(text: str) -> np.ndarray:
            call_count["n"] += 1
            return real_embed(text)

        provider.embed = counting_embed  # type: ignore[method-assign]

        pool = build_semantic_candidate_pool(store, cache)

        assert embedded.id in pool
        assert not_embedded.id not in pool
        assert call_count["n"] == 0, "pool build must never compute a new embedding"
    finally:
        cache.close()


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
    can never actually produce."""
    store = MemoryStore(":memory:")
    active_mem = _mem(store, "an active memory")
    fading_mem = _mem(store, "the original content before it faded")
    fading_summary = "a softened fading memory"
    store.fade(fading_mem.id, summary=fading_summary)

    provider = FakeEmbeddingProvider(dim=8)
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    try:
        cache.get_or_compute(active_mem.content)
        cache.get_or_compute(fading_summary)

        pool = build_semantic_candidate_pool(store, cache)

        assert active_mem.id in pool
        assert fading_mem.id not in pool, "a state='fading' memory must never enter the semantic pool"
    finally:
        cache.close()


def test_pool_build_never_bumps_recall_count(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _mem(store, "scored but maybe not surfaced")
    before = store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (m.id,)
    ).fetchone()[0]

    provider = FakeEmbeddingProvider(dim=8)
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    try:
        cache.get_or_compute(m.content)
        build_semantic_candidate_pool(store, cache)
    finally:
        cache.close()

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

    def _raise(store: MemoryStore, embeddings_cache: object) -> dict:
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
    just the pool-build call specifically. Seeds the vector through
    build_embedding_cache() (the same production helper run_semantic_recall
    itself uses) so it shares the conftest-faked provider's model_id — a
    manually-constructed cache under a different model_id would make the
    pool come back empty and never reach scoring at all."""

    def _raise_cosine(*args: object, **kwargs: object) -> float:
        raise RuntimeError("simulated scoring failure")

    monkeypatch.setattr(semantic_recall_mod, "cosine_similarity", _raise_cosine)

    store = MemoryStore(":memory:")
    mem = _mem(store, "a memory that DOES have a cached vector")

    from brain.memory.embeddings import build_embedding_cache

    cache = build_embedding_cache(tmp_path)
    try:
        cache.get_or_compute(mem.content)
    finally:
        cache.close()

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

    store = MemoryStore(":memory:")
    mem = _mem(store, "a memory that DOES have a cached vector")

    from brain.memory.embeddings import build_embedding_cache

    cache = build_embedding_cache(tmp_path)
    try:
        cache.get_or_compute(mem.content)
    finally:
        cache.close()

    result = run_semantic_recall(store, tmp_path, "any query")

    assert result is None, "a reranker failure must demote this turn to the lexical fallback, not raise"


def test_run_semantic_recall_is_fail_soft_when_close_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#231 Fix 4: the `finally: embeddings_cache.close()` sat OUTSIDE the
    inner `except Exception` clause (only the two earlier steps and the
    pool-build/scoring/reranking/floor-gating block were guarded), so a
    pathological `close()` error could escape this function's own
    documented "never raises" contract — even on an otherwise-SUCCESSFUL
    turn, since the return value is built before `finally` runs but a
    raising `finally` replaces it. Wraps a real, working EmbeddingCache in
    a proxy whose close() blows up; run_semantic_recall must still not
    raise."""
    store = MemoryStore(":memory:")
    mem = _mem(store, "a memory that DOES have a cached vector")

    from brain.memory.embeddings import build_embedding_cache

    real_cache = build_embedding_cache(tmp_path)
    real_cache.get_or_compute(mem.content)

    class _BoomOnClose:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def close(self) -> None:
            raise RuntimeError("simulated close() failure")

    monkeypatch.setattr(
        semantic_recall_mod,
        "build_embedding_cache",
        lambda persona_dir: _BoomOnClose(real_cache),
    )

    try:
        result = run_semantic_recall(store, tmp_path, mem.content)
    finally:
        real_cache.close()

    # Reaching this line at all proves close()'s own failure did not
    # propagate. The actual result (None or a SemanticRecallResult) is
    # incidental to this test.
    assert result is None or hasattr(result, "full")
