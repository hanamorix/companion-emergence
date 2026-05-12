"""Tests for brain.initiate.compose — three-prompt composition pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from brain.initiate.compose import (
    compose_decision,
    compose_decision_voice_edit,
    compose_subject,
    compose_tone,
)
from brain.initiate.schemas import (
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
)


def _candidate(kind: str = "message") -> InitiateCandidate:
    return InitiateCandidate(
        candidate_id="ic_001",
        ts="2026-05-11T14:32:04+00:00",
        kind=kind,
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0,
            rolling_baseline_stdev=1.0,
            current_resonance=7.4,
            delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_xyz"],
            topic_tags=["dream", "workshop"],
        ),
    )


def test_compose_subject_excludes_emotion_from_prompt() -> None:
    """The subject prompt must NOT see emotional state — only candidate facts."""
    provider = MagicMock()
    provider.complete = MagicMock(return_value="the dream from this morning")
    cand = _candidate()
    result = compose_subject(provider, cand, semantic_memory_excerpts=["the workshop"])
    # Inspect what was sent to the provider.
    args, kwargs = provider.complete.call_args
    prompt_text = args[0] if args else kwargs.get("prompt", "")
    assert "longing" not in prompt_text.lower()
    assert "resonance" not in prompt_text.lower()
    assert "delta_sigma" not in prompt_text.lower()
    assert result == "the dream from this morning"


def test_compose_subject_includes_linked_memories() -> None:
    provider = MagicMock(complete=MagicMock(return_value="x"))
    cand = _candidate()
    compose_subject(provider, cand, semantic_memory_excerpts=["the workshop bench"])
    args, _ = provider.complete.call_args
    assert "workshop bench" in args[0]


def test_compose_tone_receives_subject_immutable() -> None:
    """Tone prompt sees the subject as input but must not change it."""
    provider = MagicMock(complete=MagicMock(return_value="the dream from this morning landed somewhere"))
    cand = _candidate()
    result = compose_tone(
        provider,
        subject="the dream from this morning",
        candidate=cand,
        voice_template="be warm and direct",
    )
    args, _ = provider.complete.call_args
    assert "the dream from this morning" in args[0]
    assert "be warm and direct" in args[0]
    assert "longing" in args[0].lower() or "emotional" in args[0].lower()
    assert result.startswith("the dream from this morning")


def test_compose_tone_handles_none_snapshot_gracefully() -> None:
    """v0.0.9: when emotional_snapshot is None (voice-reflection), tone prompt
    must not crash and must signal absence honestly."""
    provider = MagicMock(complete=MagicMock(return_value="rendered"))
    cand = InitiateCandidate(
        candidate_id="ic_002",
        ts="2026-05-11T14:32:04+00:00",
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id="vr_001",
        emotional_snapshot=None,
        semantic_context=SemanticContext(),
        proposal={"old_text": "a", "new_text": "b"},
    )
    compose_tone(
        provider,
        subject="subject text",
        candidate=cand,
        voice_template="voice",
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    # Should not raise AttributeError; should make absence visible.
    assert "no moment-in-time" in prompt_text or "no emotional" in prompt_text.lower()


def test_compose_decision_excludes_candidate_metadata() -> None:
    """Decision prompt sees the rendered message but NOT the candidate metadata."""
    provider = MagicMock()
    provider.complete = MagicMock(
        return_value='{"decision": "send_quiet", "reasoning": "real but late"}'
    )
    result = compose_decision(
        provider,
        rendered_message="the dream from this morning",
        recent_send_history=[],
        current_local_time=datetime(2026, 5, 11, 22, 30, tzinfo=UTC),
        voice_edit_acceptance_rate=None,
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    # The decision prompt must NOT carry source_id, emotional_snapshot, etc.
    assert "dream_abc" not in prompt_text
    assert "delta_sigma" not in prompt_text.lower()
    assert result.decision == "send_quiet"
    assert result.reasoning == "real but late"


def test_compose_decision_parses_all_four_outcomes() -> None:
    for canned, expected in [
        ('{"decision": "send_notify", "reasoning": "x"}', "send_notify"),
        ('{"decision": "send_quiet", "reasoning": "x"}', "send_quiet"),
        ('{"decision": "hold", "reasoning": "x"}', "hold"),
        ('{"decision": "drop", "reasoning": "x"}', "drop"),
    ]:
        provider = MagicMock(complete=MagicMock(return_value=canned))
        result = compose_decision(
            provider,
            rendered_message="x",
            recent_send_history=[],
            current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=UTC),
            voice_edit_acceptance_rate=None,
        )
        assert result.decision == expected


def test_compose_decision_handles_malformed_json_as_hold() -> None:
    """A garbage LLM response defaults to 'hold' — never accidentally send."""
    provider = MagicMock(complete=MagicMock(return_value="this is not json"))
    result = compose_decision(
        provider,
        rendered_message="x",
        recent_send_history=[],
        current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=UTC),
        voice_edit_acceptance_rate=None,
    )
    assert result.decision == "hold"
    assert "malformed" in result.reasoning.lower() or "parse" in result.reasoning.lower()


def test_compose_decision_voice_edit_carries_gravity_framing() -> None:
    """Voice-edit decision prompt must include the gravity instruction."""
    provider = MagicMock(complete=MagicMock(
        return_value='{"decision": "send_quiet", "reasoning": "evidence is strong"}'
    ))
    result = compose_decision_voice_edit(
        provider,
        proposal={
            "old_text": "a",
            "new_text": "b",
            "rationale": "x",
            "evidence": ["e1", "e2", "e3"],
        },
        current_voice_template="full voice template content",
        recent_voice_evolutions=[],
        current_local_time=datetime(2026, 5, 11, 12, 0, tzinfo=UTC),
    )
    args, _ = provider.complete.call_args
    prompt_text = args[0]
    assert "change who you are" in prompt_text.lower()
    assert "usually `hold`" in prompt_text or "usually 'hold'" in prompt_text
    assert "full voice template content" in prompt_text
    assert result.decision == "send_quiet"
