"""Tests for brain.initiate.review — orchestrator tick."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.review import run_initiate_review_tick
from brain.initiate.schemas import EmotionalSnapshot, SemanticContext


def _snap() -> EmotionalSnapshot:
    return EmotionalSnapshot(
        vector={"longing": 7},
        rolling_baseline_mean=5.0,
        rolling_baseline_stdev=1.0,
        current_resonance=7.4,
        delta_sigma=2.4,
    )


def _ctx() -> SemanticContext:
    return SemanticContext(linked_memory_ids=["m_xyz"], topic_tags=["dream"])


def _fake_provider(decision: str = "send_quiet") -> MagicMock:
    """Provider that returns canned outputs for subject/tone/decision."""
    provider = MagicMock()
    responses = [
        "the dream from this morning",  # subject
        "the dream from this morning landed somewhere",  # tone
        f'{{"decision": "{decision}", "reasoning": "x"}}',  # decision
    ]
    provider.complete = MagicMock(side_effect=responses)
    return provider


def test_review_tick_processes_queued_candidate_writes_audit(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path,
        provider=_fake_provider("send_quiet"),
        voice_template="be warm",
        cap_per_tick=3,
    )
    audit_path = tmp_path / "initiate_audit.jsonl"
    assert audit_path.exists()
    lines = audit_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"decision": "send_quiet"' in lines[0]


def test_review_tick_removes_processed_candidate_from_queue(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    run_initiate_review_tick(
        tmp_path, provider=_fake_provider(), voice_template="x", cap_per_tick=3,
    )
    assert read_candidates(tmp_path) == []


def test_review_tick_respects_cap_per_tick(tmp_path: Path) -> None:
    """Only `cap_per_tick` candidates are processed in one call."""
    for i in range(5):
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id=f"dream_{i}",
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    provider = MagicMock()
    canned = ["subject", "tone", '{"decision": "send_quiet", "reasoning": "x"}'] * 3
    provider.complete = MagicMock(side_effect=canned)
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    # 5 emitted, 3 processed, 2 remaining
    assert len(read_candidates(tmp_path)) == 2
    audit_lines = (tmp_path / "initiate_audit.jsonl").read_text().splitlines()
    assert len(audit_lines) == 3


def test_review_tick_gate_blocks_send_records_hold(tmp_path: Path) -> None:
    """When decision = send_notify but gate denies (blackout), audit shows hold."""
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = _fake_provider("send_notify")
    blackout_time = datetime(2026, 5, 11, 1, 30, tzinfo=UTC)
    with patch("brain.initiate.review.datetime") as mock_dt:
        mock_dt.now = MagicMock(return_value=blackout_time)
        mock_dt.fromisoformat = datetime.fromisoformat
        run_initiate_review_tick(
            tmp_path,
            provider=provider,
            voice_template="x",
            cap_per_tick=3,
            now=blackout_time,
        )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "hold"' in audit_line
    assert "blackout" in audit_line


def test_review_tick_handles_compose_exception_as_error_decision(
    tmp_path: Path,
) -> None:
    """A composition failure produces decision=error, candidate not requeued."""
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    provider = MagicMock(complete=MagicMock(side_effect=RuntimeError("boom")))
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    audit_line = (tmp_path / "initiate_audit.jsonl").read_text().strip()
    assert '"decision": "error"' in audit_line
    # Candidate is dropped from the queue (the fresh emission next event
    # will rejoin if still relevant).
    assert read_candidates(tmp_path) == []


def test_review_tick_no_op_when_queue_empty(tmp_path: Path) -> None:
    """Empty queue → no audit writes, no errors, no LLM calls."""
    provider = MagicMock(complete=MagicMock())
    run_initiate_review_tick(
        tmp_path, provider=provider, voice_template="x", cap_per_tick=3,
    )
    assert not (tmp_path / "initiate_audit.jsonl").exists()
    provider.complete.assert_not_called()
