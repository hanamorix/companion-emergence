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
