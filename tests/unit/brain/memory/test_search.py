"""Tests for brain.memory.search — combined memory queries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.search import MemorySearch
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def search() -> MemorySearch:
    store = MemoryStore(db_path=":memory:")
    hebbian = HebbianMatrix(db_path=":memory:")
    cache = EmbeddingCache(db_path=":memory:", provider=FakeEmbeddingProvider())
    return MemorySearch(store=store, hebbian=hebbian, embeddings=cache)


def _mem(content: str, **kw: object) -> Memory:
    defaults: dict[str, object] = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    return Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]


def test_semantic_search_returns_memories_ordered_by_similarity(
    search: MemorySearch,
) -> None:
    """semantic_search returns (memory, similarity) tuples ordered desc."""
    m1 = _mem("the cold coffee")
    m2 = _mem("warm hana")
    m3 = _mem("the cold coffee")  # duplicate content → highest similarity
    for m in (m1, m2, m3):
        search.store.create(m)

    results = search.semantic_search("the cold coffee", limit=3)
    assert len(results) == 3
    # Top result should have similarity 1.0 (exact content match)
    top_memory, top_score = results[0]
    assert top_score == pytest.approx(1.0, abs=1e-5)


def test_semantic_search_respects_limit(search: MemorySearch) -> None:
    """limit caps the number of results."""
    for i in range(5):
        search.store.create(_mem(f"m{i}"))
    results = search.semantic_search("query", limit=2)
    assert len(results) == 2


def test_emotional_search_matches_by_emotion_overlap(search: MemorySearch) -> None:
    """Memories with matching high-intensity emotions score higher."""
    m1 = _mem("a", emotions={"love": 9.0, "tenderness": 6.0})
    m2 = _mem("b", emotions={"anger": 8.0})
    m3 = _mem("c", emotions={"love": 5.0})
    for m in (m1, m2, m3):
        search.store.create(m)

    results = search.emotional_search({"love": 9.0, "tenderness": 5.0}, limit=3)
    assert len(results) >= 1
    # m1 should be first (strongest overlap)
    assert results[0].id == m1.id


def test_temporal_search_filters_by_time_range(search: MemorySearch) -> None:
    """temporal_search returns memories within [after, before]."""
    now = datetime.now(UTC)
    old = Memory(
        id="old",
        content="yesterday",
        memory_type="conversation",
        domain="us",
        created_at=now - timedelta(days=10),
    )
    recent = Memory(
        id="recent",
        content="today",
        memory_type="conversation",
        domain="us",
        created_at=now - timedelta(hours=1),
    )
    search.store.create(old)
    search.store.create(recent)

    results = search.temporal_search(after=now - timedelta(days=1), before=now)
    assert len(results) == 1
    assert results[0].id == "recent"


def test_spreading_search_returns_connected_memories_with_activation(
    search: MemorySearch,
) -> None:
    """spreading_search from a seed returns connected memories ordered by activation."""
    m1 = _mem("seed")
    m2 = _mem("close friend")
    m3 = _mem("distant")
    for m in (m1, m2, m3):
        search.store.create(m)

    search.hebbian.strengthen(m1.id, m2.id, delta=0.8)
    search.hebbian.strengthen(m2.id, m3.id, delta=0.8)

    results = search.spreading_search(m1.id, depth=2, decay_per_hop=0.5)
    ids = [m.id for m, _ in results]
    # Seed not included; closer neighbor appears before distant one
    assert m1.id not in ids
    assert ids.index(m2.id) < ids.index(m3.id)


def test_combined_search_text_only(search: MemorySearch) -> None:
    """combined_search with only query returns semantic results."""
    search.store.create(_mem("exact match query"))
    search.store.create(_mem("unrelated content"))
    results = search.combined_search(query="exact match query", limit=2)
    assert len(results) >= 1
    assert results[0][0].content == "exact match query"


def test_combined_search_emotion_only(search: MemorySearch) -> None:
    """combined_search with only emotions returns emotional-overlap results."""
    m_love = _mem("loved", emotions={"love": 9.0})
    m_angry = _mem("angry", emotions={"anger": 8.0})
    search.store.create(m_love)
    search.store.create(m_angry)

    results = search.combined_search(emotions={"love": 8.0}, limit=2)
    assert len(results) >= 1
    assert results[0][0].id == m_love.id


def test_combined_search_domain_filter(search: MemorySearch) -> None:
    """Domain filter narrows the candidate pool before scoring."""
    m1 = _mem("work note", domain="work")
    m2 = _mem("us moment", domain="us")
    search.store.create(m1)
    search.store.create(m2)

    results = search.combined_search(query="work note", domain="work", limit=5)
    returned_ids = [m.id for m, _ in results]
    assert m1.id in returned_ids
    assert m2.id not in returned_ids


def test_combined_search_empty_inputs_returns_empty(search: MemorySearch) -> None:
    """combined_search with no filters returns empty list (nothing to score)."""
    search.store.create(_mem("x"))
    results = search.combined_search(limit=5)
    assert results == []


def test_combined_search_seed_id_surfaces_connected_memories(
    search: MemorySearch,
) -> None:
    """combined_search with only seed_id returns spreading-activation neighbours."""
    seed = _mem("seed memory")
    near = _mem("near neighbour")
    far = _mem("unrelated")
    for m in (seed, near, far):
        search.store.create(m)
    search.hebbian.strengthen(seed.id, near.id, delta=0.8)

    results = search.combined_search(seed_id=seed.id, limit=5)
    returned_ids = [m.id for m, _ in results]
    assert near.id in returned_ids
    assert seed.id not in returned_ids  # seed always excluded
    assert far.id not in returned_ids  # no edge to far


def test_combined_search_seed_id_honours_domain_filter(search: MemorySearch) -> None:
    """combined_search post-filters spreading results by domain (graph is
    domain-agnostic, so the filter runs after activation).
    """
    seed = _mem("seed", domain="us")
    us_neighbour = _mem("us neighbour", domain="us")
    work_neighbour = _mem("work neighbour", domain="work")
    for m in (seed, us_neighbour, work_neighbour):
        search.store.create(m)
    search.hebbian.strengthen(seed.id, us_neighbour.id, delta=0.7)
    search.hebbian.strengthen(seed.id, work_neighbour.id, delta=0.7)

    results = search.combined_search(seed_id=seed.id, domain="us", limit=5)
    returned_ids = [m.id for m, _ in results]
    assert us_neighbour.id in returned_ids
    assert work_neighbour.id not in returned_ids


def test_combined_search_blends_query_and_emotions(search: MemorySearch) -> None:
    """When both query and emotions are given, scores from both sub-queries
    accumulate on matching memories — a memory hitting both filters outranks
    one hitting only one.
    """
    # m1 matches both the query content AND the emotion filter
    m1 = _mem("strong love memory", emotions={"love": 9.0})
    # m2 matches only the query content
    m2 = _mem("strong love memory", emotions={"anger": 8.0})
    # m3 matches only the emotion filter
    m3 = _mem("unrelated", emotions={"love": 9.0})
    for m in (m1, m2, m3):
        search.store.create(m)

    results = search.combined_search(query="strong love memory", emotions={"love": 9.0}, limit=5)
    assert len(results) >= 1
    # m1 accumulates both signals → top rank
    top_id = results[0][0].id
    assert top_id == m1.id
