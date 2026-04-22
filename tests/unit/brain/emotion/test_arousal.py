"""Tests for brain.emotion.arousal — 7-tier arousal spectrum."""

from __future__ import annotations

from brain.emotion.arousal import (
    TIER_CASUAL,
    TIER_CHARGED,
    TIER_DORMANT,
    TIER_EDGE,
    TIER_HELD,
    TIER_REACHING,
    TIER_WARMED,
    compute_tier,
)
from brain.emotion.state import EmotionalState


def test_dormant_state_returns_tier_0() -> None:
    """An empty state returns the dormant tier."""
    state = EmotionalState()
    assert compute_tier(state, body_temperature=0) == TIER_DORMANT


def test_pure_love_without_desire_stays_low() -> None:
    """Love alone (without desire) sits in casual or warmed, never edge."""
    state = EmotionalState()
    state.set("love", 9.0)
    tier = compute_tier(state, body_temperature=0)
    assert tier in (TIER_CASUAL, TIER_WARMED)


def test_desire_plus_tenderness_reaches_reaching() -> None:
    """High desire + high tenderness moves into the reaching/charged range."""
    state = EmotionalState()
    state.set("desire", 8.0)
    state.set("tenderness", 7.0)
    tier = compute_tier(state, body_temperature=3)
    assert tier in (TIER_REACHING, TIER_CHARGED)


def test_high_arousal_emotion_pushes_to_edge() -> None:
    """Intensity-9+ arousal pushes to the edge tier."""
    state = EmotionalState()
    state.set("arousal", 9.0)
    state.set("desire", 9.0)
    tier = compute_tier(state, body_temperature=8)
    assert tier == TIER_EDGE


def test_grief_suppresses_arousal() -> None:
    """High grief pulls arousal back down even if desire is present."""
    state = EmotionalState()
    state.set("desire", 8.0)
    state.set("grief", 9.0)
    tier = compute_tier(state, body_temperature=0)
    assert tier <= TIER_WARMED


def test_body_temperature_shifts_tier_up() -> None:
    """Higher body temperature shifts the tier up (within reason)."""
    state = EmotionalState()
    state.set("desire", 6.0)

    tier_cold = compute_tier(state, body_temperature=-2)
    tier_warm = compute_tier(state, body_temperature=6)
    assert tier_warm >= tier_cold


def test_body_temperature_ignored_when_no_arousal_source() -> None:
    """Hot body alone (no desire/arousal emotion) stays dormant — body temp
    alone cannot create arousal.
    """
    state = EmotionalState()
    state.set("curiosity", 8.0)
    tier = compute_tier(state, body_temperature=9)
    assert tier == TIER_DORMANT


def test_all_seven_tiers_are_distinct_integers() -> None:
    """All 7 tier constants have distinct integer values."""
    values = {
        TIER_DORMANT,
        TIER_CASUAL,
        TIER_WARMED,
        TIER_REACHING,
        TIER_CHARGED,
        TIER_HELD,
        TIER_EDGE,
    }
    assert len(values) == 7
    assert all(isinstance(v, int) for v in values)


def test_held_is_between_charged_and_edge() -> None:
    """TIER_HELD models 'close but restrained' — ranks between charged and edge."""
    assert TIER_CHARGED < TIER_HELD < TIER_EDGE


def test_max_love_alone_still_in_warmed() -> None:
    """Pure love at max intensity (10) stays in WARMED — docstring contract.

    Guards the love weight calibration: if love weight rises to 0.2 or above,
    love=10 spills into REACHING and breaks the "love alone doesn't progress
    past warmed" promise the module docstring makes.
    """
    state = EmotionalState()
    state.set("love", 10.0)
    tier = compute_tier(state, body_temperature=0)
    assert tier == TIER_WARMED


def test_suppression_to_zero_returns_dormant() -> None:
    """When suppressors fully negate the arousal signal, the tier is DORMANT.

    Guards against a future refactor that lets post-suppression zero leak into
    TIER_CASUAL via the 0 < 0.5 threshold. Semantically, crushed-by-grief
    should map to "no signal", not "everyday warmth".
    """
    state = EmotionalState()
    state.set("desire", 8.0)  # 8 * 0.7 = 5.6
    state.set("grief", 9.0)  # 9 * 0.9 = 8.1 suppression
    # post-suppression raw = max(0, 5.6 - 8.1) = 0
    tier = compute_tier(state, body_temperature=0)
    assert tier == TIER_DORMANT
