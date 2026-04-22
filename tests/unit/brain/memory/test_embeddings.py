"""Tests for brain.memory.embeddings — provider + cache."""

from __future__ import annotations

import math

import numpy as np
import pytest

from brain.memory.embeddings import (
    EmbeddingCache,
    FakeEmbeddingProvider,
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
