"""Tests for brain.memory.semantic_recall — Stage 3 (semantic-PRIMARY
retrieval + option-4 surfacing) of the local semantic-retrieval build.

Covers the pure shape-classification/surfacing-tier logic directly (cheap,
exact boundary control at 5/6/9/10 candidates) and the candidate-pool
builder's warm-up/scoring-safety contract. The end-to-end #88 case, the
three surfacing tiers wired through the real recall block, and the
recall-counter tick semantics are covered as integration tests through
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
from brain.memory.semantic_recall import (
    SEMANTIC_FLOOR_BOOTSTRAP,
    SEMANTIC_GAP_BOOTSTRAP,
    SemanticCalibration,
    build_semantic_candidate_pool,
    classify_semantic_shape,
    run_semantic_recall,
    surfacing_tiers,
)
from brain.memory.store import Memory, MemoryStore


def _cal() -> SemanticCalibration:
    return SemanticCalibration.bootstrap()


def _mem(store: MemoryStore, content: str) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()),
        content=content,
        memory_type="event",
        domain="d",
        created_at=datetime.now(UTC),
        importance=1.0,
    )
    store.create(m)
    return m


# ---------------------------------------------------------------------------
# SemanticCalibration
# ---------------------------------------------------------------------------


def test_bootstrap_calibration_matches_module_constants() -> None:
    cal = SemanticCalibration.bootstrap()
    assert cal.floor == SEMANTIC_FLOOR_BOOTSTRAP
    assert cal.gap == SEMANTIC_GAP_BOOTSTRAP


def test_calibration_is_pluggable_not_hardcoded() -> None:
    """Stage 4's plug-in seam: classify_semantic_shape takes a calibration
    parameter rather than reading the module constants directly — a
    different SemanticCalibration instance changes classification."""
    scored = [("a", 0.9), ("b", 0.5)]
    loose = SemanticCalibration(floor=0.0, gap=1.0)  # gap too big to ever cliff
    tight = SemanticCalibration(floor=0.0, gap=0.1)  # 0.4 gap clears this easily
    assert classify_semantic_shape(scored, calibration=loose).kind == "clump"
    assert classify_semantic_shape(scored, calibration=tight).kind == "standouts"


# ---------------------------------------------------------------------------
# classify_semantic_shape — floor filtering
# ---------------------------------------------------------------------------


def test_no_candidate_passes_floor_is_shape_none() -> None:
    scored = [("a", 0.1), ("b", 0.2)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "none"
    assert shape.standout_count == 0


def test_single_floor_passing_candidate_is_a_trivial_standout() -> None:
    scored = [("a", 0.9), ("b", 0.1)]  # only "a" passes the 0.45 floor
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "standouts"
    assert shape.standout_count == 1


def test_below_floor_candidates_never_enter_the_window() -> None:
    """A candidate below the floor cannot extend the standout cluster even
    if it happens to sit close in score to the last real standout."""
    scored = [("a", 0.9), ("b", 0.46), ("c", 0.44)]  # c fails the 0.45 floor
    shape = classify_semantic_shape(scored, calibration=_cal())
    # a->b gap = 0.44 (cliff at 0), b never compared against c (c excluded).
    assert shape.kind == "standouts"
    assert shape.standout_count == 1


# ---------------------------------------------------------------------------
# classify_semantic_shape — boundary counts (5 / 6 / 9 / 10)
# ---------------------------------------------------------------------------


def _stepped_scores(n: int, *, top: float = 0.90, step: float = 0.05, floor: float = 0.30) -> list[float]:
    """n scores each `step` apart (below the bootstrap gap of 0.08, so no
    internal cliff), followed by one score far below `floor` (a clean cliff
    right after the nth item)."""
    scores = [top - i * step for i in range(n)]
    scores.append(min(scores) - 1.0)  # far below any floor
    return scores


@pytest.mark.parametrize("n", [1, 5])
def test_le_5_standouts_shape(n: int) -> None:
    scores = _stepped_scores(n)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "standouts"
    assert shape.standout_count == n


@pytest.mark.parametrize("n", [6, 9])
def test_6_to_9_standouts_shape(n: int) -> None:
    scores = _stepped_scores(n)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "standouts"
    assert shape.standout_count == n


def test_10_bunched_candidates_with_no_cliff_in_scan_window_is_a_clump() -> None:
    """10 candidates, each only `step` apart all the way down — no cliff
    anywhere in the top-9 scan window, so the shape is INCONCLUSIVE (a
    clump), matching the spec's 'bunched, ~>=10' bucket."""
    scores = [0.90 - i * 0.05 for i in range(10)]
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "clump"
    assert shape.standout_count == 0


# ---------------------------------------------------------------------------
# surfacing_tiers
# ---------------------------------------------------------------------------


def test_surfacing_tiers_none_for_inconclusive_shape() -> None:
    scored = [("a", 0.5), ("b", 0.49)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    assert shape.kind == "clump"
    assert surfacing_tiers(scored, shape) is None


def test_surfacing_tiers_all_full_for_le_5() -> None:
    scores = _stepped_scores(5)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    tiers = surfacing_tiers(scored, shape)
    assert tiers is not None
    assert tiers.full_ids == [f"m{i}" for i in range(5)]
    assert tiers.snippet_ids == []


def test_surfacing_tiers_top5_full_rest_snippet_for_6_to_9() -> None:
    scores = _stepped_scores(9)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    tiers = surfacing_tiers(scored, shape)
    assert tiers is not None
    assert tiers.full_ids == [f"m{i}" for i in range(5)]
    assert tiers.snippet_ids == [f"m{i}" for i in range(5, 9)]


def test_surfacing_tier_ids_are_in_cosine_selection_order() -> None:
    """full/snippet ids come out highest-cosine-first — presentation
    re-ordering is the CALLER's job (prompt.py), not this function's."""
    scores = _stepped_scores(6)
    scored = [(f"m{i}", s) for i, s in enumerate(scores)]
    shape = classify_semantic_shape(scored, calibration=_cal())
    tiers = surfacing_tiers(scored, shape)
    assert tiers is not None
    assert tiers.full_ids == sorted(tiers.full_ids, key=lambda mid: -dict(scored)[mid])
    assert tiers.snippet_ids == sorted(tiers.snippet_ids, key=lambda mid: -dict(scored)[mid])


# ---------------------------------------------------------------------------
# build_semantic_candidate_pool — warm-up + bump-free scoring
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
# try/except; everything after (pool build, cosine scoring, shape
# classification, surfacing) sat inside a bare try/finally with NO except,
# so an exception there propagated straight out of run_semantic_recall
# instead of demoting the turn to the lexical fallback.
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


def test_run_semantic_recall_is_fail_soft_when_close_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#231 Fix 4: the `finally: embeddings_cache.close()` sat OUTSIDE the
    inner `except Exception` clause (only the two earlier steps and the
    pool-build/scoring/classification block were guarded), so a
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
