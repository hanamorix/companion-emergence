"""Tests for brain.initiate.reach_emotion — source → emotion map (W8)."""
from __future__ import annotations

from brain.emotion.vocabulary import get as vocab_get
from brain.initiate.reach_emotion import reach_emotions_for


def test_dream_is_tenderness_plus_vulnerability():
    assert reach_emotions_for("dream") == {"tenderness": 0.15, "vulnerability": 0.10}


def test_crystallization_accent_is_pride():
    assert reach_emotions_for("crystallization") == {"tenderness": 0.15, "pride": 0.10}


def test_reflex_firing_is_tenderness_only():
    assert reach_emotions_for("reflex_firing") == {"tenderness": 0.15}


def test_unknown_source_is_tenderness_only_fail_safe():
    assert reach_emotions_for("some_future_source") == {"tenderness": 0.15}


def test_all_magnitudes_are_low():
    for src in (
        "dream", "emotion_spike", "crystallization", "voice_reflection",
        "research_completion", "recall_resonance", "reflex_firing", "unknown",
    ):
        for v in reach_emotions_for(src).values():
            assert 0.0 < v <= 0.3


def test_all_channels_are_registered():
    for src in (
        "dream", "emotion_spike", "crystallization", "voice_reflection",
        "research_completion", "recall_resonance", "reflex_firing",
    ):
        for name in reach_emotions_for(src):
            assert vocab_get(name) is not None, f"{name} not registered"
