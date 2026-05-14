"""End-to-end: source event → emitter → queue → D-tick → outcome."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.initiate.audit import (
    append_d_call_row,
    read_recent_d_calls,
)
from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.new_sources import (
    emit_reflex_firing_candidate,
    emit_research_completion_candidate,
)
from brain.initiate.reflection import (
    ReflectionDeps,
    demote_to_draft_space,
)
from brain.initiate.reflection import run as reflection_run
from brain.initiate.schemas import SemanticContext


def test_e2e_five_candidates_d_promotes_one_filters_four(tmp_path):
    """A realistic cohort: 5 candidates from 5 sources, D promotes 1 / filters 4."""
    persona = tmp_path / "persona"
    now = datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC)

    # Seed 5 candidates across 5 source types.
    emit_initiate_candidate(
        persona, kind="message", source="dream", source_id="d1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=10),
    )
    emit_initiate_candidate(
        persona, kind="message", source="crystallization", source_id="c1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=8),
    )
    emit_initiate_candidate(
        persona, kind="message", source="emotion_spike", source_id="e1",
        semantic_context=SemanticContext(), now=now - timedelta(minutes=6),
    )

    class FakeFiring:
        pattern_id = "pE2E"
        confidence = 0.85
        flinch_intensity = 0.75
        linked_memory_ids = ["m_a"]
        triggered_by_companion_outbound = False
        ts = now - timedelta(minutes=4)

    emit_reflex_firing_candidate(
        persona, firing=FakeFiring(), firing_log_id="rfx_e2e", now=now - timedelta(minutes=4),
    )

    class FakeThread:
        thread_id = "trE2E"
        topic = "quiet rivers"
        maturity_score = 0.85
        summary_excerpt = "..."
        linked_memory_ids = []
        completed_at = now - timedelta(minutes=2)
        previously_linked_to_audit = False

    emit_research_completion_candidate(
        persona, thread=FakeThread(), topic_overlap_score=0.50, now=now - timedelta(minutes=2),
    )

    candidates = read_candidates(persona)
    assert len(candidates) == 5
    sources = sorted(c.source for c in candidates)
    assert sources == [
        "crystallization", "dream", "emotion_spike",
        "reflex_firing", "research_completion",
    ]

    def haiku_call(*, system, user):
        return (
            '{"decisions":['
            '{"candidate_index":1,"decision":"filter","reason":"old weather","confidence":"high"},'
            '{"candidate_index":2,"decision":"filter","reason":"already settled","confidence":"high"},'
            '{"candidate_index":3,"decision":"promote","reason":"genuine spike","confidence":"high"},'
            '{"candidate_index":4,"decision":"filter","reason":"reflex echo","confidence":"high"},'
            '{"candidate_index":5,"decision":"filter","reason":"interesting but private","confidence":"high"}'
            '],"tick_note":"only one worth Hana hearing"}'
        ), 250, 600, 200

    def sonnet_call(*, system, user):
        raise AssertionError("should not escalate")

    deps = ReflectionDeps(
        companion_name="Nell", user_name="Hana",
        voice_template_path=tmp_path / "voice.md",
        outbound_recall_block="(none)",
        haiku_call=haiku_call, sonnet_call=sonnet_call,
        now=now, tick_id="tick_e2e",
    )
    result, dcall = reflection_run(candidates, deps=deps)
    append_d_call_row(persona, dcall)

    # Dispatch.
    for d, c in zip(result.decisions, candidates, strict=True):
        if d.decision == "filter":
            demote_to_draft_space(persona, candidate=c, decision=d, now=now)
        # (composition handoff stubbed in this test; verify-by-side-effect below)

    # Assertions:
    rows = list(read_recent_d_calls(persona, window_hours=1, now=now + timedelta(minutes=1)))
    assert len(rows) == 1
    assert rows[0].candidates_in == 5
    assert rows[0].promoted_out == 1
    assert rows[0].filtered_out == 4
    assert rows[0].failure_type is None
    # 4 candidates demoted to draft space.
    text = (persona / "draft_space.md").read_text()
    assert text.count("demoted_by: d_reflection") == 4
    # tick_note captured.
    assert rows[0].tick_note == "only one worth Hana hearing"
