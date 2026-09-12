"""Tests for brain.ingest.dedupe — DEDUPE stage."""

from __future__ import annotations

import pytest

from brain.ingest.dedupe import DEFAULT_DEDUP_THRESHOLD, is_duplicate
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


@pytest.fixture
def embedding_cache() -> EmbeddingCache:
    provider = FakeEmbeddingProvider(dim=64)
    return EmbeddingCache(":memory:", provider)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_is_duplicate_returns_false_when_embeddings_is_none(store: MemoryStore) -> None:
    """When embeddings=None, dedupe is skipped and returns False."""
    result = is_duplicate("some text here", store=store, embeddings=None)
    assert result is False


def test_is_duplicate_returns_false_when_no_embeddings_exist(
    store: MemoryStore, embedding_cache: EmbeddingCache
) -> None:
    """When the cache is empty (no stored vectors), returns False."""
    assert embedding_cache.count() == 0
    result = is_duplicate("fresh memory", store=store, embeddings=embedding_cache)
    assert result is False


def test_is_duplicate_returns_true_when_similarity_above_threshold(
    store: MemoryStore,
) -> None:
    """When a stored vector is near-identical (same text), similarity >= threshold."""
    text = "Nell loves writing and spending time with Hana"
    provider = FakeEmbeddingProvider(dim=64)
    cache = EmbeddingCache(":memory:", provider)

    # Store the embedding of the exact text we'll check against.
    cache.get_or_compute(text)
    assert cache.count() == 1

    # Check with the same text — FakeEmbeddingProvider is deterministic,
    # so same text → same vector → cosine similarity = 1.0.
    result = is_duplicate(text, store=store, threshold=DEFAULT_DEDUP_THRESHOLD, embeddings=cache)
    assert result is True


def test_is_duplicate_returns_false_when_similarity_below_threshold(
    store: MemoryStore,
) -> None:
    """When stored vectors are sufficiently different, returns False."""
    provider = FakeEmbeddingProvider(dim=64)
    cache = EmbeddingCache(":memory:", provider)

    # Populate with an unrelated text; FakeEmbeddingProvider gives different
    # random vectors for different texts (hash-seeded).
    cache.get_or_compute("the quick brown fox jumps over the lazy dog")

    # Check with a completely different string — low similarity expected.
    result = is_duplicate(
        "Nell is a sweater-wearing novelist",
        store=store,
        threshold=DEFAULT_DEDUP_THRESHOLD,
        embeddings=cache,
    )
    # With deterministic hash-based vectors for completely different texts,
    # similarity should be well below 0.88.
    assert result is False


def test_is_duplicate_ignores_rows_from_a_different_model_id(
    store: MemoryStore, tmp_path
) -> None:
    """The model_id swap-staleness guard reaches dedupe's raw scan too: a
    vector cached under a DIFFERENT provider/model_id (e.g. left over from a
    prior FakeEmbeddingProvider dim, or a stale non-production model) must
    never enter the cosine comparison, even though it shares a content_hash-
    scoped table with the current provider's rows."""
    db_path = tmp_path / "embeddings.db"
    text = "Nell loves writing and spending time with Hana"

    # Seed a row under an OLD provider/model_id for this exact text.
    old_cache = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=64))
    old_cache.get_or_compute(text)
    old_cache.close()

    # A cache opened with a DIFFERENT provider (different model_id) must not
    # see that row as a match, even for the identical text — same as
    # get_or_compute's own guard, but exercised through is_duplicate's raw
    # SELECT ... WHERE model_id = ? scan.
    new_cache = EmbeddingCache(db_path, FakeEmbeddingProvider(dim=128))
    result = is_duplicate(text, store=store, threshold=DEFAULT_DEDUP_THRESHOLD, embeddings=new_cache)
    new_cache.close()
    assert result is False
