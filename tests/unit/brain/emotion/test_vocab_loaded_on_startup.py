"""Test that ensure_persona_vocabulary_loaded() loads the persona
emotion vocabulary into the process-global registry at bridge startup,
so aggregate_state never silently drops extension emotions right after
launch (the ~15-min window bug).

Regression: before this fix, the vocabulary was only loaded inside the
supervisor heartbeat/soul-review ticks, leaving the chat path flatlined
to 26 baseline emotions until the first tick fired.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.emotion import vocabulary
from brain.emotion.aggregate import aggregate_state
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Helper — minimal valid emotion_vocabulary.json payload
# ---------------------------------------------------------------------------

_EXTENSION_EMOTION_NAME = "devotion_test_startup"

_VOCAB_JSON = json.dumps(
    {
        "version": 1,
        "emotions": [
            {
                "name": _EXTENSION_EMOTION_NAME,
                "description": "test-only extension emotion for startup test",
                "category": "persona_extension",
                "decay_half_life_days": 14.0,
                "intensity_clamp": 10.0,
            }
        ],
    }
)


def _make_memory_with_extension_emotion() -> Memory:
    return Memory.create_new(
        content="test memory carrying extension emotion",
        memory_type="conversation",
        domain="us",
        emotions={_EXTENSION_EMOTION_NAME: 8.0},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_extension_emotion():
    """Remove the test extension emotion from registry before + after each test."""
    vocabulary._unregister(_EXTENSION_EMOTION_NAME)
    yield
    vocabulary._unregister(_EXTENSION_EMOTION_NAME)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ensure_persona_vocabulary_loaded_registers_extension_emotion(tmp_path: Path):
    """ensure_persona_vocabulary_loaded() registers extension emotions so
    aggregate_state includes them on the first chat request after startup.
    """
    from brain.bridge.server import ensure_persona_vocabulary_loaded

    # Write a vocab file with the extension emotion.
    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(_VOCAB_JSON, encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        # Confirm extension emotion is NOT registered before the call.
        assert vocabulary.get(_EXTENSION_EMOTION_NAME) is None

        # --- call under test ---
        ensure_persona_vocabulary_loaded(tmp_path, store=store)

        # After the call the extension emotion must be in the registry.
        assert vocabulary.get(_EXTENSION_EMOTION_NAME) is not None

        # And aggregate_state now surfaces it.
        mem = _make_memory_with_extension_emotion()
        result = aggregate_state([mem])
        assert _EXTENSION_EMOTION_NAME in result.emotions
        assert result.emotions[_EXTENSION_EMOTION_NAME] == 8.0
    finally:
        store.close()


def test_without_startup_load_extension_emotion_is_dropped(tmp_path: Path):
    """Baseline: with no startup load, aggregate_state drops the extension emotion."""
    # Sanity-check: the extension emotion is NOT in the registry (the autouse
    # fixture just removed it).
    assert vocabulary.get(_EXTENSION_EMOTION_NAME) is None

    mem = _make_memory_with_extension_emotion()
    result = aggregate_state([mem])

    # The extension emotion is unknown → silently dropped from aggregation.
    assert _EXTENSION_EMOTION_NAME not in result.emotions


def test_ensure_persona_vocabulary_loaded_is_idempotent(tmp_path: Path):
    """Calling ensure_persona_vocabulary_loaded() twice does not raise."""
    from brain.bridge.server import ensure_persona_vocabulary_loaded

    vocab_path = tmp_path / "emotion_vocabulary.json"
    vocab_path.write_text(_VOCAB_JSON, encoding="utf-8")

    store = MemoryStore(":memory:")
    try:
        ensure_persona_vocabulary_loaded(tmp_path, store=store)
        # Second call — re-registering is a no-op, must not raise.
        ensure_persona_vocabulary_loaded(tmp_path, store=store)
        assert vocabulary.get(_EXTENSION_EMOTION_NAME) is not None
    finally:
        store.close()


def test_ensure_persona_vocabulary_loaded_missing_file_is_silent(tmp_path: Path):
    """Missing emotion_vocabulary.json must not raise — fresh personas have no file."""
    from brain.bridge.server import ensure_persona_vocabulary_loaded

    store = MemoryStore(":memory:")
    try:
        # No vocab file written — must not raise.
        ensure_persona_vocabulary_loaded(tmp_path, store=store)
    finally:
        store.close()
