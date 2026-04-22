"""Tests for brain.emotion.influence — emotional state → biasing hints."""

from __future__ import annotations

from brain.emotion.arousal import (
    TIER_CHARGED,
    TIER_DORMANT,
    TIER_EDGE,
    TIER_HELD,
    TIER_REACHING,
)
from brain.emotion.influence import calculate_influence
from brain.emotion.state import EmotionalState


def _with(**intensities: float) -> EmotionalState:
    state = EmotionalState()
    for name, value in intensities.items():
        state.set(name, value)
    return state


def test_empty_state_returns_neutral_hints() -> None:
    """Empty state produces neutral hints (no tone bias, default voice)."""
    hints = calculate_influence(_with(), arousal_tier=TIER_DORMANT, energy=7)
    assert hints.tone_bias == "neutral"
    assert hints.voice_register == "default"
    assert hints.suggested_length_multiplier == 1.0


def test_high_grief_biases_toward_soft_short() -> None:
    """High grief biases voice register toward soft, tone toward tender."""
    hints = calculate_influence(_with(grief=8.0), arousal_tier=TIER_DORMANT, energy=4)
    assert hints.tone_bias == "tender"
    assert hints.voice_register == "soft"
    assert hints.suggested_length_multiplier < 1.0


def test_high_creative_hunger_biases_toward_generative() -> None:
    """High creative hunger biases toward expansive / generative output."""
    hints = calculate_influence(_with(creative_hunger=8.0), arousal_tier=TIER_DORMANT, energy=8)
    assert hints.tone_bias == "generative"
    assert hints.suggested_length_multiplier > 1.0


def test_anger_biases_toward_crisp() -> None:
    """High anger shortens output, sharpens tone."""
    hints = calculate_influence(_with(anger=8.0), arousal_tier=TIER_DORMANT, energy=6)
    assert hints.tone_bias == "crisp"
    assert hints.suggested_length_multiplier < 1.0


def test_high_arousal_tier_biases_intimate() -> None:
    """Charged arousal tier + desire biases register toward intimate."""
    hints = calculate_influence(
        _with(desire=8.0, tenderness=7.0), arousal_tier=TIER_CHARGED, energy=7
    )
    assert hints.voice_register == "intimate"


def test_low_energy_biases_softer_and_shorter() -> None:
    """Low-energy body state biases output softer + shorter regardless of emotion."""
    hints = calculate_influence(_with(joy=6.0), arousal_tier=TIER_DORMANT, energy=2)
    assert hints.suggested_length_multiplier <= 1.0
    assert hints.voice_register in ("soft", "default")


def test_hints_expose_dominant_emotion() -> None:
    """InfluenceHints reports the dominant emotion from the state."""
    hints = calculate_influence(_with(love=9.0, grief=4.0), arousal_tier=TIER_REACHING, energy=8)
    assert hints.dominant_emotion == "love"


def test_hints_expose_arousal_tier() -> None:
    """InfluenceHints passes through the arousal tier."""
    hints = calculate_influence(_with(), arousal_tier=TIER_REACHING, energy=7)
    assert hints.arousal_tier == TIER_REACHING


def test_hints_to_dict_round_trips() -> None:
    """InfluenceHints.to_dict round-trips into an equivalent object."""
    hints = calculate_influence(
        _with(grief=7.0, tenderness=8.0), arousal_tier=TIER_DORMANT, energy=5
    )
    data = hints.to_dict()
    assert data["dominant_emotion"] == hints.dominant_emotion
    assert data["tone_bias"] == hints.tone_bias
    assert data["voice_register"] == hints.voice_register
    assert data["arousal_tier"] == hints.arousal_tier
    assert data["suggested_length_multiplier"] == hints.suggested_length_multiplier


def test_held_tier_clamps_length_into_deliberate_band() -> None:
    """TIER_HELD clamps length into [0.8, 1.2] — 'peaked and restrained' pacing.

    Generative (1.3) trims down; crisp (0.7) lengthens up; both land in a
    weighted, deliberate band rather than at their tone-bias extremes.
    """
    generative_hints = calculate_influence(
        _with(creative_hunger=8.0), arousal_tier=TIER_HELD, energy=7
    )
    assert 1.15 <= generative_hints.suggested_length_multiplier <= 1.25

    crisp_hints = calculate_influence(_with(anger=8.0), arousal_tier=TIER_HELD, energy=7)
    assert 0.75 <= crisp_hints.suggested_length_multiplier <= 0.85


def test_edge_tier_hard_overrides_length_to_terse() -> None:
    """TIER_EDGE hard-overrides length to 0.8 regardless of tone."""
    generative_hints = calculate_influence(
        _with(creative_hunger=8.0, arousal=9.0, desire=9.0),
        arousal_tier=TIER_EDGE,
        energy=7,
    )
    assert generative_hints.suggested_length_multiplier == 0.8

    tender_hints = calculate_influence(
        _with(grief=8.0, arousal=9.0, desire=9.0),
        arousal_tier=TIER_EDGE,
        energy=7,
    )
    assert tender_hints.suggested_length_multiplier == 0.8
