"""Decisive end-to-end proof for #259 inc7 cold red-team F1: the embedding
dimension is genuinely ONE-TOUCH-swappable, derived from the REAL model
output at every layer — never from the hand-maintained
`model_tier.MODEL_EMBEDDING_DIM` constant.

Before this fix, swapping to a different-dim embedding model (e.g. a
1024-dim bge-m3 in place of bge-small's 384) without ALSO updating
`MODEL_EMBEDDING_DIM` failed SILENTLY: `EmbeddingMatrix._load_from_db`
skipped every row whose blob didn't match that stale constant, so the warm
matrix ended up empty and the whole corpus degraded to lexical-only recall
with no error, no crash, nothing — see `brain/memory/embedding_matrix.py`.

This test simulates exactly that swap: it points the embedding provider at
a 1024-dim `FakeEmbeddingProvider` and DELIBERATELY NEVER TOUCHES
`model_tier.MODEL_EMBEDDING_DIM` (still 384 throughout). It then proves the
whole pipeline works anyway: `MemoryStore.embed_row` persists the real
1024-dim vector, the warm `EmbeddingMatrix` builds/decodes/serves it,
`EmbeddingMatrix.put()` accepts a matching-dim vector (and rejects a
mismatched one), a brute-force cosine scan over the matrix snapshot (the
same shape of operation `semantic_recall`'s candidate pool performs) runs
without error, and `clustering.run_clustering_pass` completes without
crashing on `np.stack` over the resulting vectors.

Fails pre-fix: `_load_from_db` used to gate on
`model_tier.MODEL_EMBEDDING_DIM` (unchanged at 384), so every 1024-dim row
would be silently skipped and the matrix/clustering assertions below would
fail (empty matrix, clustering sparse-skips instead of running).
"""

from __future__ import annotations

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.memory import embeddings as embeddings_mod
from brain.memory.clustering import MIN_VECTORS_TO_CLUSTER, run_clustering_pass
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.embeddings import cosine_similarity
from brain.memory.store import Memory, MemoryStore

_SWAPPED_DIM = 1024  # deliberately NOT 384 — a stand-in for a real bge-m3-style swap


@pytest.fixture
def swapped_provider(monkeypatch: pytest.MonkeyPatch) -> embeddings_mod.FakeEmbeddingProvider:
    """Simulate a `MODEL_EMBEDDING` swap to a different-dim model WITHOUT
    updating `model_tier.MODEL_EMBEDDING_DIM` (still 384) — the exact
    "forgot to update the constant" scenario F1 exists to make harmless."""
    provider = embeddings_mod.FakeEmbeddingProvider(dim=_SWAPPED_DIM)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())
    assert model_tier.MODEL_EMBEDDING_DIM == 384, "sanity: the constant must stay stale/untouched"
    return provider


def test_one_touch_dimension_swap_works_end_to_end_without_editing_model_embedding_dim(
    tmp_path, swapped_provider: embeddings_mod.FakeEmbeddingProvider
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    try:
        # -- embed_row stores the REAL (1024-dim) vector -------------------
        ids: list[str] = []
        for i in range(MIN_VECTORS_TO_CLUSTER):
            m = Memory.create_new(
                content=f"one-touch swap test memory number {i}",
                memory_type="conversation",
                domain="us",
            )
            store.create(m)
            store.embed_row(m.id, m.content)
            ids.append(m.id)

        for mem_id in ids:
            row = store._conn.execute(  # noqa: SLF001
                "SELECT embedding, embedding_model_id FROM memories WHERE id = ?",
                (mem_id,),
            ).fetchone()
            assert row["embedding"] is not None
            assert len(row["embedding"]) == _SWAPPED_DIM * 4, (
                "embed_row must persist the model's REAL byte-length, not "
                "MODEL_EMBEDDING_DIM's stale 384"
            )
            assert row["embedding_model_id"] == swapped_provider.model_id()

        # -- the warm matrix builds + decodes + serves it ------------------
        matrix = build_embedding_matrix(store.db_path)
        snap = matrix.snapshot()
        assert set(snap.keys()) == set(ids), (
            "pre-fix, every 1024-dim row would be silently skipped by "
            "_load_from_db's stale-384 gate, leaving this empty"
        )
        for vec in snap.values():
            assert vec.shape == (_SWAPPED_DIM,)
            assert vec.dtype == np.float32

        # -- put() accepts a matching-dim vector ---------------------------
        matrix.put("mem-put-ok", np.full(_SWAPPED_DIM, 0.42, dtype=np.float32))
        assert "mem-put-ok" in matrix
        got = matrix.get("mem-put-ok")
        assert got is not None and got.shape == (_SWAPPED_DIM,)

        # -- put() rejects a mismatched-dim vector (the OLD 384) -----------
        matrix.put("mem-put-bad", np.full(384, 0.1, dtype=np.float32))
        assert "mem-put-bad" not in matrix

        # -- recall/candidate-pool: a brute-force cosine scan over the
        # matrix snapshot (the same operation semantic_recall's candidate
        # pool performs) must run cleanly over the real-dim vectors.
        query_vec = swapped_provider.embed("a query about the swap test memories")
        assert query_vec.shape == (_SWAPPED_DIM,)
        scored = sorted(
            ((cosine_similarity(query_vec, vec), mid) for mid, vec in matrix.snapshot().items()),
            reverse=True,
        )
        assert len(scored) == MIN_VECTORS_TO_CLUSTER + 1  # the 8 rows + mem-put-ok
        assert all(-1.0 <= s <= 1.0 for s, _ in scored)

        # -- clustering does NOT crash on np.stack over the real-dim vectors
        result = run_clustering_pass(store)
        assert result.ran is True, f"clustering unexpectedly skipped: {result.reason}"
        assert result.n_vectors == MIN_VECTORS_TO_CLUSTER + 1
        assert result.k >= 2
    finally:
        store.close()
