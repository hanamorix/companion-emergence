"""Tests for brain.growth.crystallizers.vocabulary — Phase 2a stub."""

from __future__ import annotations

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.memory.store import MemoryStore


def test_phase_2a_stub_returns_empty_list() -> None:
    """Phase 2a: crystallizer is a no-op. Phase 2b populates with pattern matchers."""
    store = MemoryStore(":memory:")
    try:
        result = crystallize_vocabulary(store, current_vocabulary_names=set())
        assert result == []
    finally:
        store.close()


def test_phase_2a_stub_ignores_inputs() -> None:
    """Stub returns [] regardless of input — verifies signature accepts the
    arguments Phase 2b will use."""
    store = MemoryStore(":memory:")
    try:
        result = crystallize_vocabulary(
            store,
            current_vocabulary_names={"love", "joy", "grief"},
        )
        assert result == []
    finally:
        store.close()
