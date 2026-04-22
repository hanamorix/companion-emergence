"""Tests for brain.emotion.expression — state → face/voice parameter vector."""

from __future__ import annotations

from brain.emotion.arousal import TIER_CHARGED, TIER_DORMANT, TIER_REACHING
from brain.emotion.expression import ExpressionVector, compute_expression
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_empty_state_returns_neutral_vector() -> None:
    """Empty state → neutral expression (all params at ~0.5 baseline)."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert isinstance(vec, ExpressionVector)
    assert 0.4 <= vec.facial["mouth_curve"] <= 0.6
    assert 0.4 <= vec.facial["eye_openness"] <= 0.8


def test_joy_opens_mouth_and_eyes() -> None:
    """High joy raises mouth_curve and eye_openness above baseline."""
    vec = compute_expression(_with(joy=9.0), arousal_tier=TIER_DORMANT, energy=8)
    assert vec.facial["mouth_curve"] > 0.6
    assert vec.facial["eye_openness"] > 0.6


def test_grief_lowers_mouth_and_brow() -> None:
    """High grief pushes mouth_curve down and brow_furrow up."""
    vec = compute_expression(_with(grief=9.0), arousal_tier=TIER_DORMANT, energy=4)
    assert vec.facial["mouth_curve"] < 0.5
    assert vec.facial["brow_furrow"] > 0.4


def test_tenderness_raises_blush() -> None:
    """Tenderness increases blush opacity."""
    vec = compute_expression(_with(tenderness=9.0), arousal_tier=TIER_DORMANT, energy=7)
    assert vec.facial["blush_opacity"] > 0.3


def test_high_arousal_pushes_face_and_body() -> None:
    """Charged arousal tier raises blush, opens mouth further, tenses arms."""
    vec = compute_expression(_with(desire=9.0, tenderness=7.0), arousal_tier=TIER_CHARGED, energy=7)
    assert vec.facial["blush_opacity"] > 0.5
    assert vec.arm_hand["arm_tension"] > 0.5


def test_anger_narrows_eyes_furrows_brow() -> None:
    """High anger narrows eyes and raises brow furrow sharply."""
    vec = compute_expression(_with(anger=9.0), arousal_tier=TIER_DORMANT, energy=7)
    assert vec.facial["eye_openness"] < 0.5
    assert vec.facial["brow_furrow"] > 0.6


def test_expression_vector_includes_arousal_tier() -> None:
    """ExpressionVector carries the arousal tier through to NellFace."""
    vec = compute_expression(_with(), arousal_tier=TIER_REACHING, energy=7)
    assert vec.arousal_tier == TIER_REACHING


def test_expression_vector_has_24_facial_params() -> None:
    """The facial dict has the 24 params the Tier 7 spec names."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert len(vec.facial) == 24


def test_expression_vector_has_8_arm_hand_params() -> None:
    """The arm/hand dict has 8 params."""
    vec = compute_expression(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert len(vec.arm_hand) == 8


def test_all_params_in_zero_to_one_range() -> None:
    """All params stay in [0, 1] even at extreme emotional inputs."""
    vec = compute_expression(
        _with(anger=10.0, fear=10.0, grief=10.0), arousal_tier=TIER_DORMANT, energy=2
    )
    for value in vec.facial.values():
        assert 0.0 <= value <= 1.0
    for value in vec.arm_hand.values():
        assert isinstance(value, (float, str))


def test_to_dict_round_trips() -> None:
    """to_dict produces a serialisable snapshot."""
    vec = compute_expression(_with(joy=8.0), arousal_tier=TIER_DORMANT, energy=7)
    data = vec.to_dict()
    assert "facial" in data
    assert "arm_hand" in data
    assert data["arousal_tier"] == TIER_DORMANT
