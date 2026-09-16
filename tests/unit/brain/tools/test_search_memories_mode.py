"""search_memories mode toggle (#231): semantic (default, local cosine) vs
lexical (today's BM25/rank_memories blend), driven through the real
`dispatch` path.

Uses a small scripted `EmbeddingProvider` (deterministic, hand-chosen cosine
relationships), mirroring `tests/unit/brain/chat/test_semantic_primary_recall.py`
— the suite-wide `FakeEmbeddingProvider` is hash-seeded and ~orthogonal for
any two distinct strings, which cannot exercise a SPECIFIC semantic
relationship like "paraphrase beats keyword-overlap decoy".
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import FakeRerankerProvider
from brain.memory.semantic_recall import RERANK_FLOOR
from brain.memory.store import Memory, MemoryStore
from brain.tools.dispatch import dispatch

_SCRIPTED_MODEL_ID = "scripted-test"
# Row vectors now flow through EmbeddingMatrix, which enforces a FIXED
# 384-dim blob width (`brain.memory.embedding_matrix._EXPECTED_DIM`) and
# silently skips any row whose blob is a different length — a 2-dim test
# vector would simply never appear in a `matrix.snapshot()`. All scripted
# vectors here are padded to this width (see `_unit_vec_with_cosine` /
# `_query_unit_vec`) so they survive the matrix read path.
_EMBED_DIM = 384


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
        return _SCRIPTED_MODEL_ID


def _unit_vec_with_cosine(score: float) -> np.ndarray:
    """A `_EMBED_DIM`-wide vector whose cosine similarity against
    `_query_unit_vec()` is exactly `score` (for |score| <= 1) — only the
    first two components are non-zero; the zero padding contributes nothing
    to either the dot product or the norm, so it never perturbs the
    hand-chosen cosine relationship."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = score
    vec[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return vec


def _query_unit_vec() -> np.ndarray:
    """The `_EMBED_DIM`-wide vector `_unit_vec_with_cosine`'s cosine scores
    are measured against — the scripted "query" vector."""
    vec = np.zeros(_EMBED_DIM, dtype=np.float32)
    vec[0] = 1.0
    return vec


def _seed(store: MemoryStore, content: str) -> Memory:
    m = Memory.create_new(content=content, memory_type="event", domain="d")
    store.create(m)
    return m


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(tmp_path / "memories.db"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def _seed_vectors(store: MemoryStore, vectors: dict[str, np.ndarray], *, contents_by_id: dict[str, str]) -> None:
    """Write a vector directly onto each memory row's `embedding` /
    `embedding_model_id` columns (F1 #259: `_semantic_top_k` now sources the
    candidate pool from the warm matrix over these row columns, not the old
    content-hash `embeddings.db` cache). `contents_by_id` maps memory id ->
    its content, used to look the right vector up in `vectors` (keyed by
    content, matching `_patch_provider`'s scripting)."""
    for memory_id, content in contents_by_id.items():
        vec = vectors[content]
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
            (np.asarray(vec, dtype=np.float32).tobytes(), _SCRIPTED_MODEL_ID, memory_id),
        )
    store._conn.commit()  # noqa: SLF001


def _patch_provider(monkeypatch: pytest.MonkeyPatch, vectors: dict[str, np.ndarray], *, dim: int) -> None:
    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _ScriptedProvider(vectors, dim=dim),
    )
    # `_semantic_top_k` sources its candidate pool via `build_embedding_matrix`,
    # which derives the matrix's filter model id from
    # `model_tier.model_for_tier(TIER_EMBEDDING)` (F1 #259 step 0) — NOT from
    # whichever provider `build_embedding_provider` is patched to above. Align
    # the two so the matrix's lazy-build filter matches what `_seed_vectors`
    # stamped on the rows; otherwise the first matrix read reloads from disk
    # filtered to the real production model id, finds nothing, and silently
    # discards the seeded vectors.
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _SCRIPTED_MODEL_ID)


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, scores: dict[str, float]) -> None:
    """#231: `_semantic_top_k` floor-gates on the RERANKER score, not cosine
    — conftest.py's autouse fixture already forces `build_reranker_provider`
    to a `FakeRerankerProvider()` with no scripted scores (every unscripted
    document defaults far below `RERANK_FLOOR`), so a test that wants a
    CONCLUSIVE (floor-clearing) semantic result must script the specific
    memory contents it expects to surface, same pattern as `_patch_provider`
    above for the embedding side."""
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda: FakeRerankerProvider(scores=scores),
    )


def _rc(store: MemoryStore, mid: str) -> int:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Default (no mode arg) uses semantic.
# ---------------------------------------------------------------------------


def test_default_mode_is_semantic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    query = "how do I calm down when everything feels like too much"
    target = "deep breathing helps when you are feeling anxious"

    dim = _EMBED_DIM
    vectors = {
        query: _query_unit_vec(),
        target: _unit_vec_with_cosine(0.95),
    }

    ctx = _ctx(tmp_path)
    m_target = _seed(ctx["store"], target)

    _seed_vectors(ctx["store"], vectors, contents_by_id={m_target.id: target})
    _patch_provider(monkeypatch, vectors, dim=dim)
    _patch_reranker(monkeypatch, scores={target: RERANK_FLOOR + 5.0})

    res = dispatch("search_memories", {"query": query}, **ctx)

    assert res["mode"] == "semantic", "no mode arg must resolve to semantic"
    ids = {m["id"] for m in res["memories"]}
    assert m_target.id in ids


# ---------------------------------------------------------------------------
# mode="lexical" returns the existing BM25/search_with_loss behavior,
# unchanged — and does NOT pick up a semantic-only (no-shared-keyword) match.
# ---------------------------------------------------------------------------


def test_mode_lexical_matches_existing_keyword_behavior(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    m1 = _seed(ctx["store"], "henryk drinks coffee")
    m2 = _seed(ctx["store"], "her preferences about tea")

    res = dispatch("search_memories", {"query": "henryk preferences", "mode": "lexical"}, **ctx)

    assert res["mode"] == "lexical"
    ids = {m["id"] for m in res["memories"]}
    assert m1.id in ids and m2.id in ids, "disjoint multi-term query must union, not AND (unchanged C23 behavior)"


def test_mode_lexical_never_touches_the_embedding_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """lexical mode must not construct an embedding cache/provider at all —
    forcing that construction to raise proves the lexical path never even
    tries it."""

    def _boom() -> None:
        raise AssertionError("lexical mode must never build an embedding provider")

    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", _boom)

    ctx = _ctx(tmp_path)
    _seed(ctx["store"], "henryk likes long walks")

    res = dispatch("search_memories", {"query": "henryk", "mode": "lexical"}, **ctx)

    assert res["mode"] == "lexical"
    assert res["memories"]


# ---------------------------------------------------------------------------
# mode="semantic" returns cosine-ranked results — the #88 flavor: a
# paraphrase with no shared keyword surfaces over a keyword-overlap decoy
# that lexical search would instead have favored.
# ---------------------------------------------------------------------------


def test_mode_semantic_paraphrase_beats_keyword_overlap_decoy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `target` shares NO token (>=3 chars) with `query`, so a keyword search
    # could never find it; `decoy` shares "too"/"much"/"down" with `query`
    # but is semantically unrelated. Proves semantic surfaces the meaning
    # match a keyword search structurally cannot reach.
    query = "how do I calm down when everything feels like too much"
    target = "slow controlled breathing eases panic and racing thoughts"
    decoy = "too much rain fell down all afternoon"

    dim = _EMBED_DIM
    vectors = {
        query: _query_unit_vec(),
        target: _unit_vec_with_cosine(0.95),  # clear semantic match
        decoy: _unit_vec_with_cosine(0.10),  # semantically unrelated despite shared words
    }

    ctx = _ctx(tmp_path)
    m_target = _seed(ctx["store"], target)
    m_decoy = _seed(ctx["store"], decoy)

    _seed_vectors(ctx["store"], vectors, contents_by_id={m_target.id: target, m_decoy.id: decoy})
    _patch_provider(monkeypatch, vectors, dim=dim)
    # Both clear RERANK_FLOOR (so the assertion actually exercises the
    # reranker's ORDERING, not just floor-based exclusion of the decoy) —
    # target scored clearly higher, matching the cosine-era hand-chosen
    # relationship (0.95 vs 0.10) this test's docstring describes.
    _patch_reranker(
        monkeypatch,
        scores={target: RERANK_FLOOR + 5.0, decoy: RERANK_FLOOR + 0.5},
    )

    semantic_res = dispatch("search_memories", {"query": query, "mode": "semantic"}, **ctx)
    assert semantic_res["mode"] == "semantic"
    semantic_ranked_ids = [m["id"] for m in semantic_res["memories"]]
    assert semantic_ranked_ids[0] == m_target.id, "semantic ranks the meaning-match first"

    # Same query, lexical mode: the keyword-sharing decoy is what a plain
    # keyword search would surface (proves lexical would have MISSED the
    # meaning match — the #88 point of running semantic in the first place).
    lexical_res = dispatch("search_memories", {"query": query, "mode": "lexical"}, **ctx)
    lexical_ids = {m["id"] for m in lexical_res["memories"]}
    assert m_decoy.id in lexical_ids
    assert m_target.id not in lexical_ids, "lexical has no shared keyword with the paraphrase target"


def test_mode_semantic_result_shape_matches_existing_snippet_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Semantic-mode result rows use the SAME snippet shape the tool already
    formats (id, content, snippet: true, etc.) — the output contract is
    unchanged, only the ranking path differs."""
    query = "quiet evening"
    target = "a quiet evening with nothing much happening"
    dim = _EMBED_DIM
    vectors = {
        query: _query_unit_vec(),
        target: _unit_vec_with_cosine(0.9),
    }

    ctx = _ctx(tmp_path)
    m_target = _seed(ctx["store"], target)
    _seed_vectors(ctx["store"], vectors, contents_by_id={m_target.id: target})
    _patch_provider(monkeypatch, vectors, dim=dim)

    res = dispatch("search_memories", {"query": query, "mode": "semantic"}, **ctx)
    assert res["memories"], "expected at least one semantic hit"
    for m in res["memories"]:
        assert m.get("snippet") is True
        assert "id" in m
        assert "content" in m
        assert "emotions" in m
        assert "tags" in m


def test_mode_semantic_stays_bump_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    query = "quiet evening"
    target = "a quiet evening with nothing much happening"
    dim = _EMBED_DIM
    vectors = {
        query: _query_unit_vec(),
        target: _unit_vec_with_cosine(0.9),
    }

    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], target)
    _seed_vectors(ctx["store"], vectors, contents_by_id={m.id: target})
    _patch_provider(monkeypatch, vectors, dim=dim)

    before = _rc(ctx["store"], m.id)
    res = dispatch("search_memories", {"query": query, "mode": "semantic"}, **ctx)
    assert res["memories"]
    after = _rc(ctx["store"], m.id)
    assert after == before, "search_memories (either mode) must never bump recall_count"


# ---------------------------------------------------------------------------
# Fail-soft: semantic falls back to lexical without erroring.
# ---------------------------------------------------------------------------


def test_semantic_falls_back_to_lexical_when_no_vectors_cached(tmp_path: Path) -> None:
    """Cold-start: no embeddings.db, no cached vectors at all — the default
    "semantic" request must still return a usable (lexical) result, never
    raise."""
    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], "henryk untouched by any embedding backfill yet")

    res = dispatch("search_memories", {"query": "henryk"}, **ctx)

    assert res["mode"] == "lexical", "empty candidate pool must fall back to lexical"
    ids = {mm["id"] for mm in res["memories"]}
    assert m.id in ids


def test_semantic_falls_back_to_lexical_when_query_embed_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cached pool exists (so build_semantic_candidate_pool succeeds) but
    the query embed itself blows up — must still fall back cleanly rather
    than error out."""
    target = "a memory that does have a cached vector"
    dim = _EMBED_DIM
    vectors = {target: _query_unit_vec()}

    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], target)
    _seed_vectors(ctx["store"], vectors, contents_by_id={m.id: target})
    # Align model_tier's embedding tier to the seeded rows' model id — see
    # `_patch_provider`'s docstring for why (this test scripts its own
    # provider directly rather than going through `_patch_provider`, so it
    # must do the alignment itself).
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _SCRIPTED_MODEL_ID)

    class _BoomProvider(_ScriptedProvider):
        def embed(self, text: str) -> np.ndarray:
            raise RuntimeError("simulated embed failure")

    monkeypatch.setattr(
        "brain.memory.embeddings.build_embedding_provider",
        lambda: _BoomProvider(vectors, dim=dim),
    )

    # Query shares a token ("cached") with `target` so the lexical fallback
    # actually has something to find — proving the fallback returns a real,
    # usable result, not just an empty-but-non-erroring response.
    res = dispatch("search_memories", {"query": "cached", "mode": "semantic"}, **ctx)

    assert res["mode"] == "lexical"
    ids = {mm["id"] for mm in res["memories"]}
    assert m.id in ids


def test_semantic_falls_back_to_lexical_when_close_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#231 Fix 5 (mirrors semantic_recall.run_semantic_recall's Fix 4 /
    ``test_run_semantic_recall_is_fail_soft_when_close_raises`` in
    tests/unit/brain/memory/test_semantic_recall.py): the
    `finally: embeddings_cache.close()` in `_semantic_top_k` sat OUTSIDE the
    inner `except Exception` clause, so a pathological `close()` error could
    escape this function's own documented "Returns None (never raises) ...
    falls back to the lexical path" contract and crash `search_memories`
    instead of demoting the turn to lexical. Wraps a real, working
    EmbeddingCache in a proxy whose close() blows up; search_memories must
    still not raise, and must still return a usable (lexical) result."""
    import brain.tools.impls.search_memories as search_memories_mod
    from brain.memory.embeddings import build_embedding_cache

    target = "a memory that does have a cached vector"
    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], target)

    real_cache = build_embedding_cache(tmp_path)
    real_cache.get_or_compute(target)

    class _BoomOnClose:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def close(self) -> None:
            raise RuntimeError("simulated close() failure")

    monkeypatch.setattr(
        search_memories_mod,
        "build_embedding_cache",
        lambda persona_dir: _BoomOnClose(real_cache),
    )

    try:
        # Query shares a token ("cached") with `target` so the lexical
        # fallback actually has something to find — proving the fallback
        # returns a real, usable result, not just an empty-but-non-erroring
        # response.
        res = dispatch("search_memories", {"query": "cached", "mode": "semantic"}, **ctx)
    finally:
        real_cache.close()

    assert res["mode"] == "lexical", (
        "a close() failure must demote this turn to the lexical fallback, not raise"
    )
    ids = {mm["id"] for mm in res["memories"]}
    assert m.id in ids


# ---------------------------------------------------------------------------
# #231 Fix 3 — resolved_mode must never lie. An invalid `mode` value isn't
# blocked by `dispatch` (the "enum" in the schema is advisory, not enforced
# at the call boundary — see brain/tools/dispatch.py, which only checks
# `required`), so it reaches the impl as-is. Pre-fix, that garbage value
# skipped the `mode == "semantic"` branch entirely (retrieval correctly ran
# lexical) but `resolved_mode` was seeded from the raw `mode` argument and
# only ever corrected inside that branch — so the output "mode" field kept
# reporting "garbage" even though lexical is what actually ran.
# ---------------------------------------------------------------------------


def test_mode_garbage_value_runs_lexical_and_reports_lexical_honestly(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], "henryk likes long walks")

    res = dispatch("search_memories", {"query": "henryk", "mode": "garbage"}, **ctx)

    # Retrieval actually ran lexical (a keyword hit surfaces with no
    # embedding infra involved at all).
    ids = {mm["id"] for mm in res["memories"]}
    assert m.id in ids
    # The output "mode" field must report the path that actually ran, never
    # echo the invalid input back.
    assert res["mode"] == "lexical"
    assert res["mode"] != "garbage"
