"""Tests for brain.memory.embeddings — provider + cache."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from brain.memory.embeddings import (
    EmbeddingCache,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    FastEmbedProvider,
    build_embedding_cache,
    build_embedding_provider,
    cosine_similarity,
)


@pytest.fixture
def provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def cache(provider: FakeEmbeddingProvider) -> EmbeddingCache:
    return EmbeddingCache(db_path=":memory:", provider=provider)


def test_fake_provider_produces_unit_vector(provider: FakeEmbeddingProvider) -> None:
    """FakeEmbeddingProvider returns a unit-norm vector."""
    vec = provider.embed("anything")
    assert isinstance(vec, np.ndarray)
    assert math.isclose(float(np.linalg.norm(vec)), 1.0, rel_tol=1e-6)


def test_fake_provider_embedding_dim_is_256(provider: FakeEmbeddingProvider) -> None:
    """Default embedding dim is 256."""
    vec = provider.embed("x")
    assert vec.shape == (256,)
    assert provider.embedding_dim() == 256


def test_fake_provider_deterministic_same_text(provider: FakeEmbeddingProvider) -> None:
    """Same text → identical vector every time."""
    a = provider.embed("the cold coffee")
    b = provider.embed("the cold coffee")
    np.testing.assert_array_equal(a, b)


def test_fake_provider_different_text_different_vectors(
    provider: FakeEmbeddingProvider,
) -> None:
    """Different text produces different vectors (not identical)."""
    a = provider.embed("hello")
    b = provider.embed("goodbye")
    assert not np.array_equal(a, b)


def test_cache_get_or_compute_returns_vector(cache: EmbeddingCache) -> None:
    """get_or_compute returns a numpy array for new content."""
    vec = cache.get_or_compute("fresh content")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (256,)


def test_cache_hit_avoids_recomputation(provider: FakeEmbeddingProvider) -> None:
    """Second call for the same content hits cache (provider.embed called once)."""
    cache = EmbeddingCache(db_path=":memory:", provider=provider)

    call_count = {"n": 0}
    real_embed = provider.embed

    def counting_embed(text: str) -> np.ndarray:
        call_count["n"] += 1
        return real_embed(text)

    provider.embed = counting_embed  # type: ignore[method-assign]

    cache.get_or_compute("once")
    cache.get_or_compute("once")
    cache.get_or_compute("once")

    assert call_count["n"] == 1


def test_cache_different_content_produces_separate_cache_entries(
    cache: EmbeddingCache,
) -> None:
    """Different content strings produce different cached vectors."""
    a = cache.get_or_compute("first")
    b = cache.get_or_compute("second")
    assert not np.array_equal(a, b)


def test_cache_count_reflects_stored_entries(cache: EmbeddingCache) -> None:
    """count() returns the number of stored embedding entries."""
    assert cache.count() == 0
    cache.get_or_compute("a")
    cache.get_or_compute("b")
    cache.get_or_compute("a")  # duplicate
    assert cache.count() == 2


def test_cosine_similarity_self_is_one() -> None:
    """cosine_similarity(v, v) == 1.0."""
    v = np.array([1.0, 0.0, 0.0])
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-6)


def test_cosine_similarity_orthogonal_is_zero() -> None:
    """Orthogonal vectors have cosine similarity 0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-6)


def test_cosine_similarity_antiparallel_is_negative_one() -> None:
    """Anti-parallel vectors have cosine similarity -1."""
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-6)


def test_cosine_similarity_zero_vector_returns_zero() -> None:
    """Zero-norm input returns 0.0 without dividing by zero."""
    zero = np.zeros(3)
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(zero, v) == 0.0
    assert cosine_similarity(v, zero) == 0.0
    assert cosine_similarity(zero, zero) == 0.0


def test_cache_roundtrip_vector_values_match(
    cache: EmbeddingCache, provider: FakeEmbeddingProvider
) -> None:
    """Stored blob decodes back to the exact same float32 values across calls."""
    expected = provider.embed("roundtrip").astype(np.float32)
    cache.get_or_compute("roundtrip")  # store
    actual = cache.get_or_compute("roundtrip")  # read from cache
    np.testing.assert_array_equal(actual, expected)


def test_embedding_cache_sets_wal_and_busy_timeout(tmp_path: Path) -> None:
    """Concurrent-write safety: EmbeddingCache must enable WAL and a
    busy_timeout to match the rest of the brain's SQLite stores."""
    from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider

    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    try:
        journal_mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = cache._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        cache.close()
    assert journal_mode.lower() == "wal", f"expected WAL, got {journal_mode}"
    assert busy_timeout >= 5000, f"expected busy_timeout >= 5000, got {busy_timeout}"


# ---------------------------------------------------------------------------
# model_id — the swap-staleness guard (Stage 1 of the semantic-retrieval
# build). A cached vector is only ever served back to the provider/model that
# produced it; a different provider recomputes rather than reading a
# dimensionally- or semantically-incompatible row.
# ---------------------------------------------------------------------------


def test_fake_provider_model_id_is_dim_qualified() -> None:
    """Two Fake providers of different dims must never share a model_id —
    otherwise their cache rows would collide under the (content_hash,
    model_id) key even though the vectors are incompatible shapes."""
    assert FakeEmbeddingProvider(dim=256).model_id() != FakeEmbeddingProvider(dim=384).model_id()


def test_fresh_embedding_cache_schema_has_model_id_column(tmp_path: Path) -> None:
    """A brand-new DB already has model_id (CREATE TABLE path, not just the
    ALTER-guard for pre-existing DBs)."""
    cache = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=8))
    try:
        cols = {row[1] for row in cache._conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
    finally:
        cache.close()
    assert "model_id" in cols


def test_embedding_cache_migrates_legacy_db_missing_model_id_column(tmp_path: Path) -> None:
    """A DB created before the model_id column existed (the pre-Stage-1
    schema, content_hash/vector/dim/created_at only) gets the column added
    via the ALTER-guard on open — mirrors store.py's recall_count pattern."""
    db_path = tmp_path / "embeddings.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE embedding_cache (
            content_hash TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    legacy_hash = EmbeddingCache._hash("legacy content")  # noqa: SLF001
    legacy_vec = np.zeros(256, dtype=np.float32)
    conn.execute(
        "INSERT INTO embedding_cache (content_hash, vector, dim) VALUES (?, ?, ?)",
        (legacy_hash, legacy_vec.tobytes(), 256),
    )
    conn.commit()
    conn.close()

    cache = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=384))
    try:
        cols = {row[1] for row in cache._conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
        assert "model_id" in cols
        # The pre-existing row backfills to the column default (''), which
        # never matches a real provider's model_id — see the next test.
        row = cache._conn.execute(
            "SELECT model_id FROM embedding_cache WHERE content_hash = ?", (legacy_hash,)
        ).fetchone()
        assert row[0] == ""
    finally:
        cache.close()


def test_get_or_compute_never_serves_a_different_models_vector(tmp_path: Path) -> None:
    """The swap-staleness guard itself: a 256-dim vector cached under one
    provider must never come back for a query issued by a different
    (e.g. 384-dim) provider against the same content + db file — it must
    recompute at the new provider's dim instead."""
    db_path = tmp_path / "embeddings.db"
    old_cache = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=256))
    old_cache.get_or_compute("shared content")
    old_cache.close()

    new_cache = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=384))
    try:
        vec = new_cache.get_or_compute("shared content")
    finally:
        new_cache.close()
    assert vec.shape == (384,)  # not the stale 256-dim row


def test_get_or_compute_overwrites_stale_row_on_model_swap(tmp_path: Path) -> None:
    """After the new-model recompute above, the row in the db is the NEW
    model's vector (INSERT OR REPLACE) — the stale row doesn't linger
    alongside it under the same content_hash (PK is content_hash alone)."""
    db_path = tmp_path / "embeddings.db"
    EmbeddingCache(db_path, FakeEmbeddingProvider(dim=256)).get_or_compute("x")

    new_provider = FakeEmbeddingProvider(dim=384)
    new_cache = EmbeddingCache(db_path, new_provider)
    new_cache.get_or_compute("x")
    row = new_cache._conn.execute(  # noqa: SLF001
        "SELECT dim, model_id FROM embedding_cache WHERE content_hash = ?",
        (EmbeddingCache._hash("x"),),  # noqa: SLF001
    ).fetchone()
    new_cache.close()
    assert row == (384, new_provider.model_id())


def test_get_or_compute_same_provider_still_cache_hits(tmp_path: Path) -> None:
    """Non-regression: two caches opened against the SAME provider/model_id
    (e.g. across a process restart) still hit the cache, not recompute."""
    provider = FakeEmbeddingProvider(dim=32)
    db_path = tmp_path / "embeddings.db"
    EmbeddingCache(db_path, provider).get_or_compute("stable content")

    calls = {"n": 0}
    real_embed = provider.embed

    def counting(text: str) -> np.ndarray:
        calls["n"] += 1
        return real_embed(text)

    provider.embed = counting  # type: ignore[method-assign]
    cache2 = EmbeddingCache(db_path, provider)
    cache2.get_or_compute("stable content")
    cache2.close()
    assert calls["n"] == 0


def test_embedding_cache_exposes_its_provider_model_id(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider(dim=16)
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    try:
        assert cache.model_id == provider.model_id()
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# FastEmbedProvider — real local provider. Construction wired against a stub
# fastembed.TextEmbedding so these tests never touch the network or download
# a model file; the real download/inference path is covered by a manual
# smoke check (see the Stage-1 report), not the automated suite.
# ---------------------------------------------------------------------------


class _StubTextEmbedding:
    """Stand-in for fastembed.TextEmbedding — records constructor args,
    returns a deterministic zero vector so shape/plumbing can be asserted
    without any model download or ONNX inference."""

    def __init__(self, model_name: str, cache_dir: str, lazy_load: bool = False, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.lazy_load = lazy_load

    def embed(self, texts):
        for _ in texts:
            yield np.ones(384, dtype=np.float32)


def test_fastembed_provider_wires_model_name_and_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    provider = FastEmbedProvider(model_id="BAAI/bge-small-en-v1.5", cache_dir=tmp_path, dim=384)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_id() == "BAAI/bge-small-en-v1.5"
    assert provider.embedding_dim() == 384
    assert provider._model.model_name == "BAAI/bge-small-en-v1.5"  # noqa: SLF001
    assert provider._model.cache_dir == str(tmp_path)  # noqa: SLF001
    assert provider._model.lazy_load is True  # noqa: SLF001 — never blocks construction on a download


def test_fastembed_provider_embed_returns_the_declared_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    provider = FastEmbedProvider(model_id="some/model", cache_dir=tmp_path, dim=384)
    vec = provider.embed("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


def test_build_embedding_provider_resolves_model_id_from_model_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """build_embedding_provider() must source the model id from
    model_tier.model_for_tier(TIER_EMBEDDING) — never a hardcoded literal in
    embeddings.py — and cache into the shared get_cache_dir()."""
    from brain.bridge.model_tier import MODEL_EMBEDDING, MODEL_EMBEDDING_DIM

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    provider = build_embedding_provider()
    assert provider.model_id() == MODEL_EMBEDDING
    assert provider.embedding_dim() == MODEL_EMBEDDING_DIM
    assert provider._model.cache_dir == str(tmp_path)  # noqa: SLF001


def test_build_embedding_provider_repointed_by_reassigning_model_tier_constant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """N-model-extensible: repointing MODEL_EMBEDDING (or TIER_MODEL's
    embedding entry) is a model_tier.py-only edit — build_embedding_provider
    picks it up with no change to embeddings.py itself."""
    from brain.bridge import model_tier

    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "some/other-model")

    provider = build_embedding_provider()
    assert provider.model_id() == "some/other-model"


@pytest.mark.requires_network
def test_build_embedding_cache_targets_embeddings_db_under_persona_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Marked requires_network purely to opt OUT of conftest's suite-wide
    fake-embedding-provider override (this test still stubs fastembed itself
    and touches no real network) — build_embedding_cache() calls
    build_embedding_provider() by its own module-global name, which that
    autouse fixture patches; this test wants the REAL build_embedding_cache
    -> build_embedding_provider wiring exercised, with only fastembed itself
    stubbed out."""
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path / "cache")

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    cache = build_embedding_cache(persona_dir)
    try:
        assert (persona_dir / "embeddings.db").exists()
        vec = cache.get_or_compute("hello")
        assert vec.shape == (384,)
    finally:
        cache.close()
