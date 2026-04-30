"""Tests for brain.health.reconstruct — vocabulary reconstruction from memories."""

from __future__ import annotations

from brain.health.reconstruct import reconstruct_vocabulary_from_memories
from brain.memory.store import Memory, MemoryStore


def test_empty_store_returns_baseline_only() -> None:
    store = MemoryStore(":memory:")
    try:
        result = reconstruct_vocabulary_from_memories(store)
        names = {e["name"] for e in result["emotions"]}
        # 25 framework baseline emotions (11 core + 10 complex + 4 body)
        assert "love" in names
        assert "joy" in names
        assert "belonging" in names
        assert "climax" in names  # body
        # No persona extensions
        for e in result["emotions"]:
            assert e["category"] in {"core", "complex", "body"}
    finally:
        store.close()


def test_reconstructs_persona_extensions_from_memories() -> None:
    store = MemoryStore(":memory:")
    try:
        # Seed memories that reference custom emotions not in baseline.
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"body_grief": 8.0, "love": 9.0},
            )
        )
        store.create(
            Memory.create_new(
                content="y",
                memory_type="conversation",
                domain="us",
                emotions={"creative_hunger": 7.0},
            )
        )

        result = reconstruct_vocabulary_from_memories(store)
        names = {e["name"] for e in result["emotions"]}
        assert "body_grief" in names
        assert "creative_hunger" in names
        # love is baseline, should also be present
        assert "love" in names

        # Extensions have placeholder description + conservative decay
        body_grief = next(e for e in result["emotions"] if e["name"] == "body_grief")
        assert "reconstructed from memory" in body_grief["description"]
        assert body_grief["category"] == "persona_extension"
        assert body_grief["decay_half_life_days"] == 1.0
    finally:
        store.close()


def test_baseline_names_in_memories_not_duplicated() -> None:
    """If a baseline emotion name appears in memories, it doesn't get duplicated as extension."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"love": 9.0},
            )
        )
        result = reconstruct_vocabulary_from_memories(store)
        love_entries = [e for e in result["emotions"] if e["name"] == "love"]
        assert len(love_entries) == 1
        assert love_entries[0]["category"] == "core"  # baseline, not extension
    finally:
        store.close()


def test_returned_shape_matches_persona_loader_expectation() -> None:
    """Output is loadable by load_persona_vocabulary."""
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"x_emotion": 5.0},
            )
        )
        result = reconstruct_vocabulary_from_memories(store)
        assert "version" in result
        assert isinstance(result["emotions"], list)
        for e in result["emotions"]:
            assert "name" in e
            assert "description" in e
            assert "category" in e
            assert "decay_half_life_days" in e
    finally:
        store.close()
