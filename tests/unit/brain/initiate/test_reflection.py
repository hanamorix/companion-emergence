"""Unit tests for the D-reflection module."""
from __future__ import annotations

import pytest

from brain.initiate.reflection import (
    DDecision,
    DReflectionResult,
    parse_structured_response,
)


def test_d_decision_dataclass_shape():
    d = DDecision(
        candidate_index=1,
        decision="promote",
        reason="genuinely surprising memory return",
        confidence="high",
    )
    assert d.candidate_index == 1
    assert d.decision == "promote"
    assert d.confidence == "high"


def test_parse_structured_response_happy_path():
    raw = """
    {
      "decisions": [
        {"candidate_index": 1, "decision": "promote",
         "reason": "worth saying", "confidence": "high"},
        {"candidate_index": 2, "decision": "filter",
         "reason": "private weather", "confidence": "medium"}
      ],
      "tick_note": "one worth surfacing today"
    }
    """
    result = parse_structured_response(raw)
    assert isinstance(result, DReflectionResult)
    assert len(result.decisions) == 2
    assert result.decisions[0].decision == "promote"
    assert result.decisions[1].decision == "filter"
    assert result.tick_note == "one worth surfacing today"


def test_parse_structured_response_extracts_from_text_with_prose():
    """Models sometimes wrap JSON in prose. Parser should still find it."""
    raw = 'Here is my decision:\n```json\n{"decisions": [], "tick_note": null}\n```\nThanks.'
    result = parse_structured_response(raw)
    assert result.decisions == []
    assert result.tick_note is None


def test_parse_structured_response_raises_on_malformed():
    with pytest.raises(ValueError):
        parse_structured_response("not even close to json")


def test_build_system_message_substitutes_names(tmp_path):
    from brain.initiate.reflection import build_system_message

    voice_template_path = tmp_path / "voice.md"
    voice_template_path.write_text(
        "# Voice\n\nSweater-wearing novelist; southern english flair.\n"
    )
    msg = build_system_message(
        companion_name="Nell",
        user_name="Hana",
        voice_template_path=voice_template_path,
    )
    assert "Nell's own physiology" in msg
    assert "something to Hana" in msg
    # Voice anchor appended:
    assert "=== Your voice ===" in msg
    assert "Sweater-wearing novelist" in msg


def test_build_system_message_no_voice_template_omits_anchor(tmp_path):
    from brain.initiate.reflection import build_system_message

    msg = build_system_message(
        companion_name="Aria",
        user_name="Sam",
        voice_template_path=tmp_path / "missing.md",
    )
    assert "Aria's own physiology" in msg
    assert "something to Sam" in msg
    # No voice template => no anchor section.
    assert "=== Your voice ===" not in msg


def test_build_user_message_renders_candidates_and_time():
    from datetime import UTC, datetime

    from brain.initiate.reflection import build_user_message

    now = datetime(2026, 5, 12, 9, 30, 0, tzinfo=UTC)
    rendered = build_user_message(
        user_name="Hana",
        now=now,
        outbound_recall_block="(no recent outbound)",
        candidate_summaries=[
            "source: dream  ·  ts: 12 min ago  ·  Δσ: 1.8\n"
            "  semantic_context: linked m_a / m_b\n"
            "  fragment-of-self: there was something quieter beneath...",
        ],
    )
    assert "Current time (Hana's local)" in rendered
    assert "[1] source: dream" in rendered
    assert "(no recent outbound)" in rendered
    assert "Promote at most 2" in rendered


def test_reflection_run_happy_path_haiku(tmp_path):
    """All high-confidence Haiku decisions parsed; no escalation."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a",
            ts=(now - timedelta(minutes=5)).isoformat(),
            kind="message",
            source="dream",
            source_id="d1",
            semantic_context=SemanticContext(),
        ),
        InitiateCandidate(
            candidate_id="ic_b",
            ts=(now - timedelta(minutes=2)).isoformat(),
            kind="message",
            source="emotion_spike",
            source_id="e1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = """
    {"decisions":[
      {"candidate_index":1,"decision":"promote","reason":"surprising return","confidence":"high"},
      {"candidate_index":2,"decision":"filter","reason":"weather","confidence":"high"}
    ],"tick_note":"one worth saying"}
    """

    calls: list[str] = []

    def fake_haiku_call(*, system: str, user: str) -> tuple[str, int, int, int]:
        calls.append("haiku")
        return haiku_response, 200, 500, 180  # raw, latency_ms, tokens_in, tokens_out

    def fake_sonnet_call(*, system: str, user: str) -> tuple[str, int, int, int]:
        raise AssertionError("should not escalate on all-high-confidence")

    deps = ReflectionDeps(
        companion_name="Nell",
        user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=fake_haiku_call,
        sonnet_call=fake_sonnet_call,
        now=now,
        tick_id="tick_001",
    )

    result, dcall = run(candidates, deps=deps)
    assert len(result.decisions) == 2
    assert result.decisions[0].decision == "promote"
    assert dcall.model_tier_used == "haiku"
    assert dcall.candidates_in == 2
    assert dcall.promoted_out == 1
    assert dcall.filtered_out == 1
    assert dcall.failure_type is None
    assert dcall.retry_count == 0
    assert calls == ["haiku"]  # Sonnet not called
