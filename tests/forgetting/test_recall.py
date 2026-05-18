"""Test recall.py — graveyard-augmented search partitioning."""

from brain.forgetting import graveyard
from brain.forgetting.recall import search_with_loss
from brain.forgetting.salience import SalienceInputs
from brain.memory.store import Memory, MemoryStore


def _make_memory(*, content="x") -> Memory:
    """Minimal Memory for test fixtures."""
    return Memory.create_new(content=content, memory_type="episodic", domain="chat", emotions={})


def test_search_with_loss_partitions_active_fading_lost(tmp_path):
    """search_with_loss partitions results by state."""
    store = MemoryStore(":memory:")
    # One active, one fading
    m1 = _make_memory(content="apple active")
    store.create(m1)
    m2 = _make_memory(content="apple fading")
    store.create(m2)
    store.fade(m2.id, summary="apple brief")
    # One lost (graveyard)
    m3 = _make_memory(content="apple lost")
    graveyard.append(
        tmp_path,
        memory=m3,
        salience_at_drop=0.05,
        inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
        lived_age_hours=100.0,
        reason="test",
    )
    result = search_with_loss(tmp_path, store, "apple")
    assert len(result.active) == 1
    assert result.active[0].id == m1.id
    assert len(result.fading) == 1
    assert result.fading[0].id == m2.id
    assert len(result.lost) == 1
    store.close()


def test_search_with_loss_empty_buckets(tmp_path):
    """Empty query returns empty SearchResult."""
    store = MemoryStore(":memory:")
    result = search_with_loss(tmp_path, store, "")
    assert result.active == []
    assert result.fading == []
    assert result.lost == []
    store.close()


def test_search_with_loss_respects_limit(tmp_path):
    """search_with_loss respects limit parameter."""
    store = MemoryStore(":memory:")
    # Create 3 active memories matching "test"
    for i in range(3):
        m = _make_memory(content=f"test item {i}")
        store.create(m)
    result = search_with_loss(tmp_path, store, "test", limit=2)
    assert len(result.active) <= 2
    store.close()


def test_search_with_loss_lost_sorted_most_recent_first(tmp_path):
    """Lost results are sorted by forgotten_at_iso, most recent first."""
    # Graveyard.search handles the ordering; verify it passes through.
    store = MemoryStore(":memory:")
    # Append 3 entries in order: one, two, three
    for content in ["apple one", "apple two", "apple three"]:
        m = _make_memory(content=content)
        graveyard.append(
            tmp_path,
            memory=m,
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )
    result = search_with_loss(tmp_path, store, "apple")
    # graveyard.search returns most-recent first
    assert [e["summary"] for e in result.lost] == ["apple three", "apple two", "apple one"]
    store.close()


def test_search_with_loss_fading_returns_summary_body(tmp_path):
    """Fading memories carry the summary as body, with state='fading'."""
    store = MemoryStore(":memory:")
    m = _make_memory(content="original detailed text about widgets")
    store.create(m)
    store.fade(m.id, summary="widgets, briefly")
    result = search_with_loss(tmp_path, store, "widgets")
    assert len(result.fading) == 1
    assert result.fading[0].content == "widgets, briefly"
    assert result.fading[0].state == "fading"
    # The original body lives in content_snapshot, not surfaced in recall results.
    assert result.fading[0].content_snapshot == "original detailed text about widgets"
    store.close()
