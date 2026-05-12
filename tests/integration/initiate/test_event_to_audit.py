# tests/integration/initiate/test_event_to_audit.py
"""End-to-end: dream emits candidate → review tick processes → audit row exists."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.audit import read_recent_audit
from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def test_full_pipeline_dream_to_audit(tmp_path: Path, monkeypatch) -> None:
    """Simulate: dream emits candidate; review tick composes + writes audit.

    D-reflection is stubbed to promote the candidate unconditionally so the
    test remains focused on the composition + audit pipeline rather than the
    D editorial gate (which has its own dedicated unit tests).
    """
    from brain.initiate.d_call_schema import DCallRow, make_d_call_id
    from brain.initiate.reflection import DDecision, DReflectionResult
    from datetime import UTC, datetime

    def _promote_all(candidates, *, deps):
        decisions = [
            DDecision(i, "promote", "test stub", "high")
            for i in range(len(candidates))
        ]
        result = DReflectionResult(decisions=decisions, tick_note=None)
        dcall = DCallRow(
            d_call_id=make_d_call_id(deps.now),
            ts=deps.now.isoformat(),
            tick_id=deps.tick_id,
            model_tier_used="haiku",
            candidates_in=len(candidates),
            promoted_out=len(candidates),
            filtered_out=0,
            latency_ms=0,
            tokens_input=0,
            tokens_output=0,
        )
        return result, dcall

    monkeypatch.setattr("brain.initiate.review.reflection_run", _promote_all)

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("be warm and direct.\n")

    # Stage 1: dream completion emits a candidate.
    emit_initiate_candidate(
        persona_dir,
        kind="message", source="dream", source_id="dream_abc",
        emotional_snapshot=EmotionalSnapshot(
            vector={"longing": 7},
            rolling_baseline_mean=5.0, rolling_baseline_stdev=1.0,
            current_resonance=7.4, delta_sigma=2.4,
        ),
        semantic_context=SemanticContext(
            linked_memory_ids=["m_workshop"],
            topic_tags=["dream", "workshop"],
        ),
    )
    assert len(read_candidates(persona_dir)) == 1

    # Stage 2: review tick processes the candidate.
    provider = MagicMock()
    provider.complete = MagicMock(side_effect=[
        "the dream from this morning",  # subject
        "the dream from this morning landed somewhere",  # tone
        '{"decision": "send_quiet", "reasoning": "real but late"}',  # decision
    ])
    run_initiate_review_tick(
        persona_dir, provider=provider, voice_template="be warm",
    )

    # Stage 3: audit row exists with current state=delivered.
    rows = list(read_recent_audit(persona_dir, window_hours=24))
    assert len(rows) == 1
    assert rows[0].decision == "send_quiet"
    assert rows[0].delivery is not None
    assert rows[0].delivery["current_state"] == "delivered"
    assert "the dream" in rows[0].subject

    # Stage 4: candidate removed from queue.
    assert read_candidates(persona_dir) == []
