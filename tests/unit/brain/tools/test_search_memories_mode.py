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

from brain.memory.embeddings import EmbeddingCache, EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
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


def _seed(store: MemoryStore, content: str) -> Memory:
    m = Memory.create_new(content=content, memory_type="event", domain="d")
    store.create(m)
    return m


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def _seed_vectors(persona_dir: Path, vectors: dict[str, np.ndarray], *, dim: int, contents: list[str]) -> None:
    """Pre-populate embeddings.db as if the idle backfill already embedded
    `contents` (memory bodies only, never the query text itself)."""
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

    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.95),
    }

    ctx = _ctx(tmp_path)
    m_target = _seed(ctx["store"], target)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target])
    _patch_provider(monkeypatch, vectors, dim=dim)

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

    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.95),  # clear semantic match
        decoy: _unit_vec_with_cosine(0.10),  # semantically unrelated despite shared words
    }

    ctx = _ctx(tmp_path)
    m_target = _seed(ctx["store"], target)
    m_decoy = _seed(ctx["store"], decoy)

    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target, decoy])
    _patch_provider(monkeypatch, vectors, dim=dim)

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
    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.9),
    }

    ctx = _ctx(tmp_path)
    _seed(ctx["store"], target)
    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target])
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
    dim = 2
    vectors = {
        query: np.array([1.0, 0.0], dtype=np.float32),
        target: _unit_vec_with_cosine(0.9),
    }

    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], target)
    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target])
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
    dim = 2
    vectors = {target: np.array([1.0, 0.0], dtype=np.float32)}

    ctx = _ctx(tmp_path)
    m = _seed(ctx["store"], target)
    _seed_vectors(tmp_path, vectors, dim=dim, contents=[target])

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
