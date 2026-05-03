"""Tests for brain.growth.crystallizers.vocabulary."""

from __future__ import annotations

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.memory.store import Memory, MemoryStore


def _mem(content: str, emotions: dict[str, float], *, domain: str = "us") -> Memory:
    return Memory.create_new(
        content=content,
        memory_type="conversation",
        domain=domain,
        emotions=emotions,
        tags=[],
        importance=8.0,
    )


def test_crystallizer_returns_empty_when_there_is_not_enough_evidence() -> None:
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("one tender complicated moment", {"love": 8, "grief": 7}))

        result = crystallize_vocabulary(store, current_vocabulary_names=set())

        assert result == []
    finally:
        store.close()


def test_crystallizer_proposes_recurring_unnamed_emotional_configuration() -> None:
    store = MemoryStore(":memory:")
    try:
        first = _mem("love and grief braided through the promise", {"love": 9, "grief": 8})
        second = _mem("another love-grief evening", {"love": 8, "grief": 7})
        third = _mem("grief softened by love", {"grief": 9, "love": 7})
        store.create(first)
        store.create(second)
        store.create(third)
        store.create(_mem("plain joy", {"joy": 9}))

        result = crystallize_vocabulary(store, current_vocabulary_names={"love", "grief", "joy"})

        assert len(result) == 1
        proposal = result[0]
        assert proposal.name == "love_grief_blend"
        assert "love" in proposal.description
        assert "grief" in proposal.description
        assert proposal.evidence_memory_ids == (first.id, second.id, third.id)
        assert proposal.score >= 0.7
        assert proposal.relational_context == "recurring emotional configuration in us memories"

    finally:
        store.close()


def test_crystallizer_skips_configuration_when_name_already_exists() -> None:
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("love and grief braided through the promise", {"love": 9, "grief": 8}))
        store.create(_mem("another love-grief evening", {"love": 8, "grief": 7}))
        store.create(_mem("grief softened by love", {"grief": 9, "love": 7}))

        result = crystallize_vocabulary(
            store,
            current_vocabulary_names={"love", "grief", "love_grief_blend"},
        )

        assert result == []
    finally:
        store.close()
