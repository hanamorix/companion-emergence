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


def test_reflection_run_escalates_on_low_confidence(tmp_path):
    """Any 'low' confidence in Haiku response triggers Sonnet re-call."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = (
        '{"decisions":[{"candidate_index":1,"decision":"filter",'
        '"reason":"unsure","confidence":"low"}],"tick_note":null}'
    )
    sonnet_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"resonant","confidence":"high"}],"tick_note":"yes"}'
    )

    def haiku_call(*, system, user):
        return haiku_response, 200, 400, 100

    def sonnet_call(*, system, user):
        return sonnet_response, 700, 400, 100

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.retry_count == 1
    assert result.decisions[0].decision == "promote"
    assert result.tick_note == "yes"


def test_reflection_run_escalates_on_malformed_haiku(tmp_path):
    """Malformed JSON from Haiku triggers Sonnet re-call."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        return "I cannot comply with this request.", 200, 400, 50

    def sonnet_call(*, system, user):
        return (
            '{"decisions":[{"candidate_index":1,"decision":"filter",'
            '"reason":"ok","confidence":"high"}],"tick_note":null}',
            700, 400, 100,
        )

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.retry_count == 1
    assert result.decisions[0].decision == "filter"


def test_reflection_run_filters_when_both_tiers_low_confidence(tmp_path):
    """If Sonnet's confidence is also low, decision is forced to filter."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    haiku_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"maybe","confidence":"low"}],"tick_note":null}'
    )
    sonnet_response = (
        '{"decisions":[{"candidate_index":1,"decision":"promote",'
        '"reason":"still maybe","confidence":"low"}],"tick_note":null}'
    )

    def haiku_call(*, system, user):
        return haiku_response, 200, 400, 100

    def sonnet_call(*, system, user):
        return sonnet_response, 700, 400, 100

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.model_tier_used == "sonnet"
    assert dcall.failure_type == "both_low_confidence"
    # Decision forced to filter despite Sonnet saying promote.
    assert result.decisions[0].decision == "filter"
    assert "ambivalent" in result.decisions[0].reason.lower()


def test_reflection_run_records_timeout_failure(tmp_path):
    """Timeout raised inside the LLM call is captured into DCallRow.failure_type."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import DTimeoutError, ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DTimeoutError("haiku timed out at 30s")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on timeout — passthrough retry")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "timeout"
    assert dcall.retry_count == 0
    assert dcall.model_tier_used == "haiku"
    # No decisions on passthrough (caller decides what to do — see Task 14).
    assert result.decisions == []


def test_reflection_run_records_rate_limit_failure(tmp_path):
    """Rate-limit error captured as 'rate_limit' failure type."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import DRateLimitError, ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DRateLimitError("429")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on rate_limit")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    result, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "rate_limit"
    assert result.decisions == []


def test_reflection_run_records_provider_error(tmp_path):
    """Generic provider error captured as 'provider_error'."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import DProviderError, ReflectionDeps, run
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidates = [
        InitiateCandidate(
            candidate_id="ic_a", ts=(now - timedelta(minutes=1)).isoformat(),
            kind="message", source="dream", source_id="d1",
            semantic_context=SemanticContext(),
        ),
    ]

    def haiku_call(*, system, user):
        raise DProviderError("500")

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate on provider_error")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="t1",
    )
    _, dcall = run(candidates, deps=deps)
    assert dcall.failure_type == "provider_error"


def test_demote_to_draft_space_writes_fragment(tmp_path):
    """A filtered candidate becomes a draft-space fragment with D-frontmatter."""
    from datetime import UTC, datetime, timedelta

    from brain.initiate.reflection import DDecision, demote_to_draft_space
    from brain.initiate.schemas import InitiateCandidate, SemanticContext

    persona = tmp_path / "p"
    persona.mkdir()
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)
    candidate = InitiateCandidate(
        candidate_id="ic_x",
        ts=(now - timedelta(minutes=5)).isoformat(),
        kind="message",
        source="reflex_firing",
        source_id="rfx_001",
        semantic_context=SemanticContext(source_meta={"pattern_id": "p1"}),
    )
    decision = DDecision(
        candidate_index=1,
        decision="filter",
        reason="private weather; passing through",
        confidence="high",
    )
    demote_to_draft_space(persona, candidate=candidate, decision=decision, now=now)

    draft_path = persona / "draft_space.md"
    text = draft_path.read_text(encoding="utf-8")
    assert "demoted_by: d_reflection" in text
    assert 'd_reason: "private weather; passing through"' in text
    assert "source: reflex_firing" in text
    assert "source_id: rfx_001" in text
