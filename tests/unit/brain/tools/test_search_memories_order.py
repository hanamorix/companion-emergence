"""search_memories `order` toggle (#231, Change 2): caller-selectable ordering
that composes with the existing `mode` (semantic/lexical) toggle.

`order="relevance"` (default) must be BYTE-IDENTICAL to today's behavior —
purely additive, no change when unused. `order="age"` WIDENS the internal
fetch to `CANDIDATE_POOL` for BOTH modes (lexical: `rank_memories(...,
limit=CANDIDATE_POOL)`; semantic: reranks up to `CANDIDATE_POOL`
floor-clearing candidates in `_semantic_top_k` — #231's `RERANK_FLOOR` gate
still applies, "age" only widens the FETCH, it never skips the floor),
then sorts that wider matched set by `created_at` DESC, then slices to the
caller's real `limit` — proving a naive re-sort of an already-`limit`-capped
set would have missed the whole point: an "age" ordering must be able to
surface a genuinely recent match, not just the newest of the few candidates
`mode` already ranked highest.

Uses the real `dispatch` path, mirroring `test_search_memories_mode.py`'s
style (scripted embedding provider for the semantic cases).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from brain.memory.embeddings import EmbeddingCache, EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.semantic_recall import RERANK_FLOOR
from brain.memory.store import Memory, MemoryStore
from brain.tools.dispatch import dispatch


class _ScriptedProvider(EmbeddingProvider):
    """Returns a HAND-CHOSEN vector for each scripted text; anything else
    embeds to an all-zero vector (cosine 0.0 against everything)."""

    def __init__(self, vectors: dict[str, np.ndarray], *, dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        if text in self._vectors:
            return self._vectors[text].astype(np.float32)
        return np.zeros(self._dim, dtype=np.float32)

    def embedding_dim(self) -> int:
        return self._dim

    def model_id(self) -> str:
        return "scripted-test"


def _unit_vec_with_cosine(score: float) -> np.ndarray:
    """A 2-D unit vector whose cosine similarity against [1.0, 0.0] is
    exactly `score` (for |score| <= 1)."""
    return np.array([score, math.sqrt(max(0.0, 1.0 - score * score))], dtype=np.float32)


def _seed(
    store: MemoryStore,
    content: str,
    *,
    created_at: datetime | None = None,
    importance: float | None = None,
) -> Memory:
    m = Memory.create_new(content=content, memory_type="event", domain="d", importance=importance)
    if created_at is not None:
        object.__setattr__(m, "created_at", created_at)
    store.create(m)
    return m


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def _seed_vectors(persona_dir: Path, vectors: dict[str, np.ndarray], *, dim: int, contents: list[str]) -> None:
    cache = EmbeddingCache(persona_dir / "embeddings.db", _ScriptedProvider(vectors, dim=dim))
    try:
        for content in contents:
            cache.get_or_compute(content)
    finally:
        cache.close()


def _patch_provider(monkeypatch: pytest.MonkeyPatch, vectors: dict[str, np.ndarray], *, dim: int) -> None:
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _ScriptedProvider(vectors, dim=dim),
    )


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, scores: dict[str, float]) -> None:
    """#231: `_semantic_top_k` floor-gates on the RERANKER score (not
    cosine) for BOTH `order` values — `order="age"` widens the internal
    fetch, it does not skip the floor gate. conftest.py's autouse fixture
    already forces `build_reranker_provider` to a scoreless
    `FakeRerankerProvider` (every unscripted document defaults far below
    `RERANK_FLOOR`), so a test that wants a CONCLUSIVE semantic result must
    script the specific memory contents it expects to clear the floor."""
    from brain.memory.reranker import FakeRerankerProvider

    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda: FakeRerankerProvider(scores=scores),
    )


_NOW = datetime.now(UTC)
_OLD = _NOW - timedelta(days=60)
_RECENT = _NOW - timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Default order="relevance" is byte-identical to pre-toggle behavior.
# ---------------------------------------------------------------------------


def test_default_order_is_relevance_and_matches_omitted_order(tmp_path: Path) -> None:
    # Bump-free + read-only: safe to issue both calls against the SAME store,
    # so the two results are directly comparable (no re-created, re-uuid'd
    # rows to account for).
    ctx = _ctx(tmp_path)
    _seed(ctx["store"], "workshop workshop workshop notes from the sunrise workshop", created_at=_OLD, importance=9.0)
    _seed(ctx["store"], "a fleeting workshop mention, nothing more", created_at=_RECENT, importance=0.0)

    res_omitted = dispatch("search_memories", {"query": "workshop", "mode": "lexical", "limit": 1}, **ctx)
    assert res_omitted["resolved_order"] == "relevance"

    res_explicit = dispatch(
        "search_memories", {"query": "workshop", "mode": "lexical", "limit": 1, "order": "relevance"}, **ctx
    )

    # Explicitly requesting "relevance" must produce the exact same result as
    # omitting `order` entirely — purely additive, no behavior change.
    assert res_omitted["memories"] == res_explicit["memories"]
    assert res_omitted["mode"] == res_explicit["mode"] == "lexical"


# ---------------------------------------------------------------------------
# order="age" widens the fetch + age-sorts — LEXICAL mode.
# ---------------------------------------------------------------------------


def test_order_age_lexical_widens_and_surfaces_a_recent_low_rank_match(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    strong_old = _seed(
        ctx["store"],
        "workshop workshop workshop notes from the sunrise workshop",
        created_at=_OLD,
        importance=9.0,
    )
    weak_recent = _seed(
        ctx["store"],
        "a fleeting workshop mention, nothing more",
        created_at=_RECENT,
        importance=0.0,
    )

    # Under default relevance ordering with limit=1, the strong/old match
    # wins — the weak/recent match ranks outside the top-1 by BM25+importance.
    relevance_res = dispatch("search_memories", {"query": "workshop", "mode": "lexical", "limit": 1}, **ctx)
    assert [m["id"] for m in relevance_res["memories"]] == [strong_old.id]

    # order="age" widens the fetch to CANDIDATE_POOL (both memories are still
    # matches — matching is unchanged), sorts by created_at DESC, THEN slices
    # to limit=1 — so the recent-but-weak match now surfaces instead.
    age_res = dispatch(
        "search_memories", {"query": "workshop", "mode": "lexical", "limit": 1, "order": "age"}, **ctx
    )
    assert age_res["resolved_order"] == "age"
    assert age_res["mode"] == "lexical"
    assert [m["id"] for m in age_res["memories"]] == [weak_recent.id]


def test_order_age_lexical_still_matches_by_mode_first(tmp_path: Path) -> None:
    """An unrelated memory that never matched the lexical query must NOT
    surface under order="age" just because it is the newest row in the
    store — matching by `mode` happens first; `order` only re-sorts the
    matched set."""
    ctx = _ctx(tmp_path)
    _seed(ctx["store"], "workshop notes from the sunrise session", created_at=_OLD)
    unrelated_but_newest = _seed(ctx["store"], "an entirely unrelated grocery list", created_at=_NOW)

    res = dispatch("search_memories", {"query": "workshop", "mode": "lexical", "limit": 5, "order": "age"}, **ctx)
    ids = {m["id"] for m in res["memories"]}
    assert unrelated_but_newest.id not in ids


# ---------------------------------------------------------------------------
# order="age" widens the fetch + age-sorts — SEMANTIC mode (top-K reranked,
# RERANK_FLOOR still gates which candidates count as matches).
# ---------------------------------------------------------------------------


def test_order_age_semantic_widens_and_age_sorts_over_top_k_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "how do I calm down when everything feels like too much"
    strong_old_text = "slow controlled breathing eases panic and racing thoughts"
    weak_recent_text = "a loosely related note about feeling overwhelmed sometimes"

    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        strong_old_text: _unit_vec_with_cosine(0.95),
        weak_recent_text: _unit_vec_with_cosine(0.4),
    }

    ctx = _ctx(tmp_path)
    strong_old = _seed(ctx["store"], strong_old_text, created_at=_OLD)
    weak_recent = _seed(ctx["store"], weak_recent_text, created_at=_RECENT)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[strong_old_text, weak_recent_text])
    _patch_provider(monkeypatch, vectors, dim=dim)
    # Both clear RERANK_FLOOR (order="age" widens the fetch, it does not
    # skip the floor gate — see _patch_reranker's docstring) — scored to
    # preserve the cosine-era relative strength (0.95 vs 0.4) this test's
    # docstring describes.
    _patch_reranker(
        monkeypatch,
        scores={strong_old_text: RERANK_FLOOR + 5.0, weak_recent_text: RERANK_FLOOR + 1.0},
    )

    # Default relevance, limit=1: the higher-cosine memory wins.
    relevance_res = dispatch("search_memories", {"query": query, "mode": "semantic", "limit": 1}, **ctx)
    assert relevance_res["mode"] == "semantic"
    assert [m["id"] for m in relevance_res["memories"]] == [strong_old.id]

    # order="age", limit=1: both scripted scores clear RERANK_FLOOR, so both
    # are still top-CANDIDATE_POOL reranked matches, and widening surfaces
    # the weaker one too — sorted by created_at DESC, the recent-but-weaker
    # match now wins the limit=1 slice.
    age_res = dispatch("search_memories", {"query": query, "mode": "semantic", "limit": 1, "order": "age"}, **ctx)
    assert age_res["mode"] == "semantic"
    assert age_res["resolved_order"] == "age"
    assert [m["id"] for m in age_res["memories"]] == [weak_recent.id]


# ---------------------------------------------------------------------------
# resolved_order reporting + garbage-value normalization (same advisory-enum
# posture as `mode` / `resolved_mode`).
# ---------------------------------------------------------------------------


def test_resolved_order_reports_age_and_relevance_correctly(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    _seed(ctx["store"], "workshop notes")

    res_rel = dispatch("search_memories", {"query": "workshop", "mode": "lexical"}, **ctx)
    assert res_rel["resolved_order"] == "relevance"

    res_age = dispatch("search_memories", {"query": "workshop", "mode": "lexical", "order": "age"}, **ctx)
    assert res_age["resolved_order"] == "age"


def test_order_garbage_value_normalizes_to_relevance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    strong_old = _seed(
        ctx["store"],
        "workshop workshop workshop notes from the sunrise workshop",
        created_at=_OLD,
        importance=9.0,
    )
    _seed(
        ctx["store"],
        "a fleeting workshop mention, nothing more",
        created_at=_RECENT,
        importance=0.0,
    )

    res = dispatch(
        "search_memories", {"query": "workshop", "mode": "lexical", "limit": 1, "order": "garbage"}, **ctx
    )

    # Garbage `order` must normalize to "relevance", never echo the invalid
    # value and never silently behave like "age" (no widening).
    assert res["resolved_order"] == "relevance"
    assert res["resolved_order"] != "garbage"
    assert [m["id"] for m in res["memories"]] == [strong_old.id]
