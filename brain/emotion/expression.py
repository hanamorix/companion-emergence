"""Emotional state → structured expression vector.

The vector drives NellFace's visual rendering (Week 6). This module is
art-agnostic: it outputs numbers in [0,1] for facial params and a small
enum for hand pose. NellFace's expression_map.json decides how to compose
the avatar's SVG layers against those numbers.

Design per spec Section 5.2 (expression sub-module) and Section 12
(NellFace architecture).

Parameter counts (24 facial + 8 arm/hand) match the Tier 7 spec's
recommendation. Forker personas can define fewer or more parameters in
their own expression_map; this module ships the baseline Tier 7 shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain.emotion.arousal import TIER_CHARGED, TIER_DORMANT, TIER_EDGE, TIER_HELD
from brain.emotion.state import EmotionalState

# 24 facial parameter names.
_FACIAL_PARAMS: tuple[str, ...] = (
    "mouth_curve",
    "mouth_openness",
    "mouth_tension",
    "lip_press",
    "eye_openness",
    "eye_squint",
    "eye_wetness",
    "eye_direction_x",
    "eye_direction_y",
    "pupil_dilation",
    "brow_furrow",
    "brow_raise_inner",
    "brow_raise_outer",
    "brow_asymmetry",
    "cheek_raise",
    "nostril_flare",
    "jaw_drop",
    "jaw_clench",
    "head_tilt",
    "head_forward",
    "blush_opacity",
    "skin_flush",
    "breath_rate",
    "breath_depth",
)

# 8 arm/hand parameter names. hand_pose is an enum-like string; others are floats.
_ARM_HAND_PARAMS: tuple[str, ...] = (
    "hand_pose",
    "arm_tension",
    "arm_openness",
    "wrist_angle",
    "finger_spread",
    "grip_strength",
    "reach_forward",
    "reach_retract",
)


@dataclass
class ExpressionVector:
    """Structured expression output for NellFace's renderer.

    Attributes:
        facial: {param_name: value in [0, 1]} for all 24 facial params.
        arm_hand: {param_name: value} for all 8 arm/hand params.
            hand_pose is a string from a small enum; others are floats [0, 1].
        arousal_tier: Pass-through of the current arousal tier.
    """

    facial: dict[str, float] = field(default_factory=dict)
    arm_hand: dict[str, float | str] = field(default_factory=dict)
    arousal_tier: int = TIER_DORMANT

    def to_dict(self) -> dict[str, Any]:
        return {
            "facial": dict(self.facial),
            "arm_hand": dict(self.arm_hand),
            "arousal_tier": self.arousal_tier,
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


def _baseline() -> tuple[dict[str, float], dict[str, float | str]]:
    """Neutral-face baseline — all params at 0.5 (or resting for pose)."""
    facial = dict.fromkeys(_FACIAL_PARAMS, 0.5)
    arm_hand: dict[str, float | str] = dict.fromkeys(_ARM_HAND_PARAMS, 0.3)
    arm_hand["hand_pose"] = "resting"
    return facial, arm_hand


def compute_expression(state: EmotionalState, arousal_tier: int, energy: int) -> ExpressionVector:
    """Compute an ExpressionVector from emotional + body state.

    Args:
        state: Current EmotionalState.
        arousal_tier: Pre-computed arousal tier.
        energy: Body energy (0..10).

    Returns:
        ExpressionVector with facial + arm_hand dicts populated.
    """
    facial, arm_hand = _baseline()

    # Joy: opens mouth into a curve, brightens eyes.
    joy = state.emotions.get("joy", 0.0) / 10.0
    facial["mouth_curve"] = _clamp(0.5 + 0.4 * joy)
    facial["eye_openness"] = _clamp(0.5 + 0.3 * joy)
    facial["cheek_raise"] = _clamp(0.3 + 0.5 * joy)

    # Grief: lowers mouth, furrows brow, wets eyes.
    grief = state.emotions.get("grief", 0.0) / 10.0
    facial["mouth_curve"] = _clamp(facial["mouth_curve"] - 0.5 * grief)
    facial["brow_furrow"] = _clamp(facial["brow_furrow"] + 0.4 * grief)
    facial["eye_wetness"] = _clamp(0.2 + 0.6 * grief)
    facial["head_tilt"] = _clamp(0.5 + 0.2 * grief)

    # Anger: narrows eyes, furrows brow sharply, clenches jaw.
    anger = state.emotions.get("anger", 0.0) / 10.0
    facial["eye_openness"] = _clamp(facial["eye_openness"] - 0.4 * anger)
    facial["eye_squint"] = _clamp(0.3 + 0.5 * anger)
    facial["brow_furrow"] = _clamp(facial["brow_furrow"] + 0.5 * anger)
    facial["jaw_clench"] = _clamp(0.3 + 0.6 * anger)
    facial["nostril_flare"] = _clamp(0.3 + 0.4 * anger)

    # Fear: widens eyes, raises inner brow.
    fear = state.emotions.get("fear", 0.0) / 10.0
    facial["eye_openness"] = _clamp(facial["eye_openness"] + 0.3 * fear)
    facial["brow_raise_inner"] = _clamp(0.3 + 0.5 * fear)
    facial["breath_rate"] = _clamp(0.4 + 0.5 * fear)

    # Tenderness: softens mouth, raises blush.
    tenderness = state.emotions.get("tenderness", 0.0) / 10.0
    facial["mouth_tension"] = _clamp(0.3 - 0.2 * tenderness)
    facial["blush_opacity"] = _clamp(0.2 + 0.3 * tenderness)

    # Desire / arousal: deepens blush, dilates pupils, opens lips, tenses body.
    desire = state.emotions.get("desire", 0.0) / 10.0
    arousal_emotion = state.emotions.get("arousal", 0.0) / 10.0
    body_heat = max(desire, arousal_emotion)
    facial["blush_opacity"] = _clamp(facial["blush_opacity"] + 0.4 * body_heat)
    facial["pupil_dilation"] = _clamp(0.4 + 0.5 * body_heat)
    facial["lip_press"] = _clamp(0.3 + 0.3 * body_heat)
    facial["breath_depth"] = _clamp(0.5 + 0.4 * body_heat)
    if arousal_tier >= TIER_CHARGED:
        facial["jaw_drop"] = _clamp(0.3 + 0.3 * body_heat)
        arm_hand["arm_tension"] = _clamp(0.3 + 0.5 * body_heat)
        arm_hand["grip_strength"] = _clamp(0.3 + 0.5 * body_heat)
        arm_hand["hand_pose"] = "reaching" if arousal_tier < TIER_HELD else "holding"
    if arousal_tier == TIER_EDGE:
        arm_hand["hand_pose"] = "open"
        arm_hand["reach_forward"] = _clamp(0.7)

    # Low energy: droops eyes, slows breath.
    if energy <= 3:
        facial["eye_openness"] = _clamp(facial["eye_openness"] - 0.2)
        facial["breath_rate"] = _clamp(facial["breath_rate"] - 0.2)

    return ExpressionVector(
        facial=facial,
        arm_hand=arm_hand,
        arousal_tier=arousal_tier,
    )
