"""Tests for brain.health.reconstruct — vocabulary reconstruction from memories."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from brain.health.reconstruct import PLACEHOLDER_DESCRIPTION, reconstruct_vocabulary_from_memories
from brain.memory.store import Memory, MemoryStore


def test_empty_store_returns_baseline_only() -> None:
    store = MemoryStore(":memory:")
    try:
        result = reconstruct_vocabulary_from_memories(store)
        names = {e["name"] for e in result["emotions"]}
        # the framework baseline emotions (12 core + 14 complex + 4 body)
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
                emotions={"emergence": 8.0, "love": 9.0},
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
        assert "emergence" in names
        assert "creative_hunger" in names
        # love is baseline, should also be present
        assert "love" in names

        # Extensions have placeholder description + persona-typical decay (14d)
        emergence = next(e for e in result["emotions"] if e["name"] == "emergence")
        assert "reconstructed from memory" in emergence["description"]
        assert emergence["category"] == "persona_extension"
        assert emergence["decay_half_life_days"] == 14.0
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


def test_reconstructed_entry_gets_persona_typical_half_life() -> None:
    """Reconstructed persona-extension entries carry decay_half_life_days == 14.0.

    The 1.0 fast-decay default caused the stub-flood bug (2026-06-10): 42
    reconstructed/healed emotion stubs decayed 14x faster than proper persona
    entries (persona-typical half-life is 14.0) and were erased from memories
    within ~9-13 days.
    """
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="x",
                memory_type="conversation",
                domain="us",
                emotions={"emergence": 8.0},
            )
        )
        result = reconstruct_vocabulary_from_memories(store)
        emergence = next(e for e in result["emotions"] if e["name"] == "emergence")
        assert emergence["decay_half_life_days"] == 14.0
    finally:
        store.close()


def test_non_finite_valued_emotion_name_still_gets_a_placeholder(caplog) -> None:
    """A stored emotion whose value is not finite (json.dumps writes NaN) still names an emotion the brain has used.

    The rebuild reads names only, so it neither loses that name nor logs "dropping emotion" on every rebuild.
    `create_new` drops non-finite values, so the memory is built the way a loaded row is.
    """
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.from_stored_row(
                id="nan-valued-row",
                content="x",
                memory_type="conversation",
                domain="us",
                created_at=datetime.now(UTC),
                emotions={"nan_ache": float("nan"), "inf_ache": float("inf"), "love": 9.0},
            ),
            fold=False,
        )
        with caplog.at_level(logging.DEBUG):
            result = reconstruct_vocabulary_from_memories(store)
        by_name = {e["name"]: e for e in result["emotions"]}
        assert by_name["nan_ache"]["category"] == "persona_extension"
        assert by_name["nan_ache"]["description"] == PLACEHOLDER_DESCRIPTION
        assert by_name["inf_ache"]["category"] == "persona_extension"
        assert not [r for r in caplog.records if "dropping emotion" in r.getMessage()]
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
