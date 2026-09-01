"""Tests for brain.ingest.commit — COMMIT stage."""

from __future__ import annotations

import pytest

from brain.ingest.commit import commit_item
from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    # Real on-disk db so store.persona_dir points at a real dir: the write gate
    # enqueues gated ingest labels (all of VALID_LABELS) to the sibling
    # pending_candidates.jsonl, which a ":memory:" store cannot host.
    return MemoryStore(tmp_path / "memories.db")


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(":memory:")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_commit_item_creates_memory_with_correct_type_domain_tags(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """commit_item enqueues a candidate with memory_type=label, domain='brain', and expected tags."""
    item = ExtractedItem(text="Nell prefers black coffee", label="fact", importance=5)
    mem_id = commit_item(item, session_id="sess_x", store=store, hebbian=hebbian)

    assert mem_id is not None
    # A gated ingest label is now enqueued as a candidate, not written to
    # memories.db; assert the candidate carries the right fields.
    assert store.get(mem_id) is None
    memory = PendingQueue(store.persona_dir).read_recent("fact", limit=1)[0]
    assert memory.id == mem_id
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
    memory = PendingQueue(store.persona_dir).read_recent("observation", limit=1)[0]
    assert memory.importance == 1.0


def test_commit_item_defers_auto_hebbian_for_gated_label(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """A gated ingest label is enqueued (not a memories.db row), so auto-Hebbian
    linking is DEFERRED to the consolidation gate: commit_item creates no edges.

    (Migration rule 3: every VALID_LABELS label is now gated, so ingest-time
    linking would dangle against a non-existent row. The gate seeds weight-5
    edges for corrections/continuations on promote instead.)
    """
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
    # Candidate lives in the queue, not the DB…
    assert store.get(mem_id) is None
    assert PendingQueue(store.persona_dir).read_recent("fact", limit=1)[0].id == mem_id
    # …and no Hebbian edge was created at ingest (linking deferred to the gate).
    assert hebbian.neighbors(mem_id) == []


# ---------------------------------------------------------------------------
# image_shas — multimodal metadata
# ---------------------------------------------------------------------------


def test_commit_item_persists_image_shas_in_metadata(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Image-bearing memory carries the sha list in metadata."""
    item = ExtractedItem(
        text="Hana shared a photo of her mirror selfie", label="fact", importance=7
    )
    mem_id = commit_item(
        item,
        session_id="sess_img",
        store=store,
        hebbian=hebbian,
        image_shas=["a" * 64, "b" * 64],
    )
    assert mem_id is not None
    memory = PendingQueue(store.persona_dir).read_recent("fact", limit=1)[0]
    assert memory.metadata.get("image_shas") == ["a" * 64, "b" * 64]


def test_commit_item_no_image_shas_metadata_when_unset(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Existing text-only flow leaves metadata as before — no image_shas key."""
    item = ExtractedItem(text="Plain text fact", label="fact", importance=4)
    mem_id = commit_item(item, session_id="sess_text", store=store, hebbian=hebbian)
    assert mem_id is not None
    memory = PendingQueue(store.persona_dir).read_recent("fact", limit=1)[0]
    assert "image_shas" not in memory.metadata


def test_commit_item_empty_image_shas_treated_as_unset(
    store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """An empty list is dropped — keeps metadata clean for downstream consumers."""
    item = ExtractedItem(text="Plain text", label="fact", importance=4)
    mem_id = commit_item(item, session_id="s", store=store, hebbian=hebbian, image_shas=[])
    assert mem_id is not None
    memory = PendingQueue(store.persona_dir).read_recent("fact", limit=1)[0]
    assert "image_shas" not in memory.metadata
