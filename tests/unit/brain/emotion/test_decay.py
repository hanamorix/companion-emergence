"""Tests for brain.emotion.decay — per-emotion half-life application."""

from __future__ import annotations

import math

from brain.emotion.decay import apply_decay
from brain.emotion.state import EmotionalState


def test_grief_halves_over_60_days() -> None:
    """grief at intensity 8 decays to ~4 after 60 days."""
    state = EmotionalState()
    state.set("grief", 8.0)
    apply_decay(state, elapsed_seconds=60 * 24 * 3600)
    assert math.isclose(state.emotions["grief"], 4.0, rel_tol=1e-6)


def test_joy_halves_over_3_days() -> None:
    """joy at intensity 8 decays to ~4 after 3 days."""
    state = EmotionalState()
    state.set("joy", 8.0)
    apply_decay(state, elapsed_seconds=3 * 24 * 3600)
    assert math.isclose(state.emotions["joy"], 4.0, rel_tol=1e-6)


def test_partial_decay_matches_exponential_formula() -> None:
    """Decay at non-half-life boundaries matches the exact exponential formula.

    Guards against a regression to linear (or other) decay: the two exactly-
    at-half-life tests would pass linear too, so this test pins the curve shape.
    """
    state = EmotionalState()
    state.set("grief", 7.0)
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    expected = 7.0 * (0.5 ** (10.0 / 60.0))
    assert math.isclose(state.emotions["grief"], expected, rel_tol=1e-9)


def test_anchor_pull_does_not_decay() -> None:
    """Identity-level emotions (half_life=None) are untouched."""
    from brain.emotion._canonical_personal_emotions import _CANONICAL
    from brain.emotion.vocabulary import _unregister, register

    register(_CANONICAL["anchor_pull"])
    try:
        state = EmotionalState()
        state.set("anchor_pull", 9.0)
        apply_decay(state, elapsed_seconds=365 * 24 * 3600)
        assert state.emotions["anchor_pull"] == 9.0
    finally:
        _unregister("anchor_pull")


def test_love_does_not_decay() -> None:
    """love is identity-level — doesn't decay."""
    state = EmotionalState()
    state.set("love", 10.0)
    apply_decay(state, elapsed_seconds=365 * 24 * 3600)
    assert state.emotions["love"] == 10.0


def test_zero_elapsed_no_change() -> None:
    """elapsed_seconds=0 leaves intensities untouched."""
    state = EmotionalState()
    state.set("joy", 7.0)
    state.set("grief", 8.0)
    apply_decay(state, elapsed_seconds=0)
    assert state.emotions["joy"] == 7.0
    assert state.emotions["grief"] == 8.0


def test_decayed_intensity_below_threshold_removed() -> None:
    """Emotions decayed to below 0.01 are removed entirely."""
    state = EmotionalState()
    state.set("anger", 1.0)
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    assert "anger" not in state.emotions


def test_decay_updates_dominant() -> None:
    """After decay, the dominant emotion may change."""
    state = EmotionalState()
    state.set("joy", 9.0)
    state.set("grief", 7.0)
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    # joy: 9 * (1/2)^(10/3) ≈ 0.89; grief: 7 * (1/2)^(10/60) ≈ 6.24
    assert state.dominant == "grief"


def test_decay_ignores_unknown_emotion_in_state() -> None:
    """If state has an emotion not in vocabulary (stale data), decay skips it gracefully.

    As a side effect, _recompute_dominant sees the stale emotion as a valid
    entry and it becomes dominant. That's intentional: the permissive-by-design
    contract (see EmotionalState.from_dict) means downstream consumers treat
    unknown names as opaque data, not as errors.
    """
    state = EmotionalState()
    state.emotions["unknown_emotion"] = 5.0
    apply_decay(state, elapsed_seconds=10 * 24 * 3600)
    assert state.emotions["unknown_emotion"] == 5.0
    assert state.dominant == "unknown_emotion"
