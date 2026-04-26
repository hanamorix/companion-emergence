"""Tests for brain.ingest.commit — COMMIT stage."""

from __future__ import annotations

import pytest

from brain.ingest.commit import commit_item
from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(":memory:")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_commit_item_creates_memory_with_correct_type_domain_tags(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """commit_item writes a memory with memory_type=label, domain='brain', and expected tags."""
    item = ExtractedItem(text="Nell prefers black coffee", label="fact", importance=5)
    mem_id = commit_item(item, session_id="sess_x", store=store, hebbian=hebbian)

    assert mem_id is not None
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.content == "Nell prefers black coffee"
    assert memory.memory_type == "fact"
    assert memory.domain == "brain"
    assert "auto_ingest" in memory.tags
    assert "conversation" in memory.tags
    assert "fact" in memory.tags


def test_commit_item_bypasses_write_gate_low_importance(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """commit_item commits even importance=1 items — no add_memory gate in ingest."""
    item = ExtractedItem(text="she paused briefly", label="observation", importance=1)
    mem_id = commit_item(item, session_id="sess_low", store=store, hebbian=hebbian)

    assert mem_id is not None
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.importance == 1.0


def test_commit_item_auto_hebbian_links_related_memories(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """commit_item strengthens Hebbian edges to top-3 textually related memories."""
    # Pre-populate the store with memories that share keywords.
    for label in ("fact", "observation", "feeling"):
        related = Memory.create_new(
            content="Nell writes fiction and poetry",
            memory_type=label,
            domain="brain",
        )
        store.create(related)

    item = ExtractedItem(text="Nell writes dark fiction", label="fact", importance=6)
    mem_id = commit_item(item, session_id="sess_hebb", store=store, hebbian=hebbian)

    assert mem_id is not None
    # The new memory should have Hebbian edges to the pre-existing related memories.
    neighbors = hebbian.neighbors(mem_id)
    assert len(neighbors) >= 1
    weights = [w for _, w in neighbors]
    assert all(w == 0.5 for w in weights)
