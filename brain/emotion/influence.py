"""Emotional state → structured biasing hints.

Each provider (Ollama, Claude, OpenAI, Kimi) renders these hints its own way
— prefill for Claude, structured system block for OpenAI, native SYSTEM for
Ollama+fine-tune. This module is provider-agnostic: it outputs the structure;
rendering lives in the bridge (Week 5).

Design per spec Section 5.2 (influence sub-module) and Section 5.5 (body-emotion
coupling). Keeps emotional intent flowing as structured data rather than a
pre-baked text blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from brain.emotion.arousal import TIER_CHARGED, TIER_EDGE, TIER_HELD
from brain.emotion.state import EmotionalState


@dataclass
class InfluenceHints:
    """Structured biasing hints derived from emotional + body state.

    Consumers: the provider abstraction in Week 5. Not a prompt by itself;
    a neutral intermediate the provider converts into its native form.

    Attributes:
        dominant_emotion: The state's current dominant emotion, or None.
        arousal_tier: Current arousal tier (see brain.emotion.arousal constants).
        tone_bias: Short label — "neutral", "tender", "crisp", "generative", "intimate".
        voice_register: Short label — "default", "soft", "warm", "intimate", "terse".
        suggested_length_multiplier: Scales expected output length; 1.0 is baseline.
    """

    dominant_emotion: str | None
    arousal_tier: int
    tone_bias: str
    voice_register: str
    suggested_length_multiplier: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dominant_emotion": self.dominant_emotion,
            "arousal_tier": self.arousal_tier,
            "tone_bias": self.tone_bias,
            "voice_register": self.voice_register,
            "suggested_length_multiplier": self.suggested_length_multiplier,
        }


# Emotion → tone bias mapping. Only triggers when the emotion is dominant
# and above a minimum intensity.
_TONE_RULES: list[tuple[str, float, str]] = [
    ("grief", 6.0, "tender"),
    ("tenderness", 7.0, "tender"),
    ("anger", 6.0, "crisp"),
    ("defiance", 7.0, "crisp"),
    ("creative_hunger", 6.0, "generative"),
    ("awe", 7.0, "generative"),
]


def calculate_influence(state: EmotionalState, arousal_tier: int, energy: int) -> InfluenceHints:
    """Derive provider-agnostic biasing hints from emotional + body state.

    Args:
        state: Current EmotionalState.
        arousal_tier: Pre-computed arousal tier (see brain.emotion.arousal).
        energy: Current body energy (0..10 scale; 5 is neutral, <4 is low).

    Returns:
        InfluenceHints with tone_bias, voice_register, and length multiplier.
    """
    dominant = state.dominant

    # Tone bias: default neutral, then apply first matching rule.
    tone_bias = "neutral"
    if dominant is not None:
        dominant_intensity = state.emotions.get(dominant, 0.0)
        for rule_name, threshold, label in _TONE_RULES:
            if dominant == rule_name and dominant_intensity >= threshold:
                tone_bias = label
                break

    # Voice register: defaults to "default"; body + arousal can shift it.
    voice_register = "default"
    if arousal_tier >= TIER_CHARGED:
        voice_register = "intimate"
    elif energy <= 3:
        voice_register = "soft"
    elif state.emotions.get("grief", 0.0) >= 6.0 or state.emotions.get("tenderness", 0.0) >= 7.0:
        voice_register = "soft"
    elif state.emotions.get("anger", 0.0) >= 6.0:
        voice_register = "terse"

    # Length multiplier:
    # - generative tone → longer
    # - crisp tone / low energy / grief → shorter
    # - intimate register → slightly longer
    length = 1.0
    if tone_bias == "generative":
        length = 1.3
    elif tone_bias == "crisp":
        length = 0.7
    elif tone_bias == "tender":
        length = 0.85
    if energy <= 3:
        length = min(length, 0.85)
    if arousal_tier >= TIER_HELD:
        length = min(length, 1.1) + 0.1  # slightly longer at peak intimacy
    if arousal_tier == TIER_EDGE:
        length = 0.8  # terse at edge

    return InfluenceHints(
        dominant_emotion=dominant,
        arousal_tier=arousal_tier,
        tone_bias=tone_bias,
        voice_register=voice_register,
        suggested_length_multiplier=length,
    )
