"""Tests for monologue + reply prompt frames."""
from __future__ import annotations

from brain.chat.monologue_prompts import (
    build_monologue_frame,
    build_reply_frame,
)


def test_monologue_frame_includes_persona_name():
    frame = build_monologue_frame(
        persona_name="Nell",
        emotion_summary="curious(0.7), warm(0.4)",
        voice_excerpt="sweater-wearing novelist",
        soul_hints=(),
        narrative_hints=(),
    )
    assert "Nell" in frame
    # Structural invariant: monologue frame must not be the reply frame.
    assert "compose the visible reply" not in frame.lower()


def test_monologue_frame_names_tangents_as_belonging_here():
    frame = build_monologue_frame(
        persona_name="Nell",
        emotion_summary="",
        voice_excerpt="",
        soul_hints=(),
        narrative_hints=(),
    )
    assert "tangent" in frame.lower() or "drift" in frame.lower()


def test_monologue_frame_includes_emotion_when_present():
    frame = build_monologue_frame(
        persona_name="Nell",
        emotion_summary="curious(0.7)",
        voice_excerpt="",
        soul_hints=(),
        narrative_hints=(),
    )
    assert "curious(0.7)" in frame


def test_monologue_frame_omits_emotion_when_empty():
    frame = build_monologue_frame(
        persona_name="Nell",
        emotion_summary="",
        voice_excerpt="",
        soul_hints=(),
        narrative_hints=(),
    )
    # No empty "current emotions:" header.
    assert "current emotions:" not in frame


def test_reply_frame_signals_tangents_handled():
    frame = build_reply_frame(persona_name="Nell")
    assert "Nell" in frame
    # Reply frame: tangents are already handled.
    assert "tangent" in frame.lower() or "already handled" in frame.lower()
    # Structural discriminator: reply frame contains this; monologue frame does not.
    assert "compose the visible reply" in frame.lower()
    # Structural invariant: reply frame must not be the monologue frame.
    assert "inner monologue" not in frame.lower()


def test_reply_frame_does_not_impose_hard_token_cap():
    """Per spec §1 non-goals: prompt-shape mechanism, not a hard cap."""
    frame = build_reply_frame(persona_name="Nell")
    assert "max tokens" not in frame.lower()
    assert "characters" not in frame.lower()


def test_monologue_frame_renders_soul_and_narrative_hints():
    """Hints render with '; ' separator under labelled lines."""
    frame = build_monologue_frame(
        persona_name="Nell",
        emotion_summary="",
        voice_excerpt="",
        soul_hints=("late-night writing", "tea over coffee"),
        narrative_hints=("Loopy-the-cat thread", "Hana's apartment"),
    )
    assert "soul threads: late-night writing; tea over coffee" in frame
    assert "narrative threads: Loopy-the-cat thread; Hana's apartment" in frame


def test_reply_frame_reboots_second_person_address():
    frame = build_reply_frame(persona_name="Nell", user_name="Hana")
    assert "private thought" in frame
    assert "speaking to Hana directly in second person" in frame
    assert "'you', never 'she'/'her'/'they'" in frame


def test_reply_frame_user_name_defaults():
    frame = build_reply_frame(persona_name="Nell")
    assert "speaking to the user directly in second person" in frame
