"""Tests for brain.emotion.state — EmotionalState dataclass."""

from __future__ import annotations

import pytest

from brain.emotion.state import EmotionalState


def test_empty_state_has_no_dominant() -> None:
    """An EmotionalState with no emotions has dominant=None."""
    state = EmotionalState()
    assert state.emotions == {}
    assert state.dominant is None


def test_set_intensity_updates_dominant() -> None:
    """Setting an intensity makes it (or the highest) dominant."""
    state = EmotionalState()
    state.set("love", 9.0)
    assert state.emotions["love"] == 9.0
    assert state.dominant == "love"

    state.set("grief", 7.0)
    assert state.dominant == "love"

    state.set("grief", 10.0)
    assert state.dominant == "grief"


def test_set_intensity_rejects_unknown_emotion() -> None:
    """Setting an intensity on an unknown emotion raises KeyError."""
    state = EmotionalState()
    with pytest.raises(KeyError, match="nonsense"):
        state.set("nonsense", 5.0)


def test_set_intensity_respects_clamp() -> None:
    """Setting an intensity above the clamp raises ValueError."""
    state = EmotionalState()
    with pytest.raises(ValueError, match="clamp"):
        state.set("love", 11.0)


def test_set_intensity_rejects_negative() -> None:
    """Negative intensities are rejected."""
    state = EmotionalState()
    with pytest.raises(ValueError, match="negative"):
        state.set("love", -1.0)


def test_set_zero_removes_emotion() -> None:
    """Setting intensity=0 removes the emotion from the state."""
    state = EmotionalState()
    state.set("love", 5.0)
    assert "love" in state.emotions
    state.set("love", 0.0)
    assert "love" not in state.emotions


def test_dominant_ties_broken_by_insertion_order() -> None:
    """If two emotions have equal intensity, the one set first wins."""
    state = EmotionalState()
    state.set("love", 7.0)
    state.set("grief", 7.0)
    assert state.dominant == "love"


def test_copy_returns_independent_state() -> None:
    """copy() returns a new EmotionalState — mutations don't affect the original."""
    original = EmotionalState()
    original.set("love", 9.0)

    clone = original.copy()
    clone.set("grief", 8.0)

    assert "grief" not in original.emotions
    assert clone.emotions["love"] == 9.0
    assert clone.emotions["grief"] == 8.0


def test_to_dict_round_trips() -> None:
    """to_dict() → from_dict() reproduces the state."""
    original = EmotionalState()
    original.set("tenderness", 8.0)
    original.set("desire", 5.0)

    data = original.to_dict()
    restored = EmotionalState.from_dict(data)

    assert restored.emotions == original.emotions
    assert restored.dominant == original.dominant


def test_from_dict_ignores_legacy_residue_key() -> None:
    """An old dict's residue keys are ignored on read and never written back."""
    state = EmotionalState.from_dict(
        {"emotions": {"joy": 3.0}, "residue": [{"x": 1}], "residue_max": 16}
    )
    assert state.emotions == {"joy": 3.0}
    assert "residue" not in state.to_dict()
