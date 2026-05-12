"""Tests for brain.initiate.emit — deterministic candidate emission, no LLM."""

from __future__ import annotations

from pathlib import Path

from brain.initiate.emit import emit_initiate_candidate, read_candidates
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


def test_emit_appends_candidate_to_queue(tmp_path: Path) -> None:
    emit_initiate_candidate(
        tmp_path,
        kind="message",
        source="dream",
        source_id="dream_abc",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    queue_path = tmp_path / "initiate_candidates.jsonl"
    assert queue_path.exists()
    candidates = read_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].source_id == "dream_abc"
    assert candidates[0].kind == "message"


def test_emit_is_idempotent_on_source_id(tmp_path: Path) -> None:
    """Re-emission of the same source_id is a no-op (dedupes)."""
    for _ in range(3):
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id="dream_abc",
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    candidates = read_candidates(tmp_path)
    assert len(candidates) == 1


def test_emit_creates_queue_when_absent(tmp_path: Path) -> None:
    """If the persona dir doesn't exist, emit creates the queue file under it."""
    persona = tmp_path / "fresh-persona"
    persona.mkdir()  # but no queue file yet
    emit_initiate_candidate(
        persona,
        kind="message",
        source="crystallization",
        source_id="cryst_001",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
    )
    assert (persona / "initiate_candidates.jsonl").exists()


def test_emit_voice_edit_proposal_carries_proposal_payload(tmp_path: Path) -> None:
    proposal = {
        "old_text": "old line",
        "new_text": "new line",
        "rationale": "feels truer",
        "evidence": ["dream_a", "cryst_b"],
    }
    emit_initiate_candidate(
        tmp_path,
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id="vr_001",
        emotional_snapshot=_snap(),
        semantic_context=_ctx(),
        proposal=proposal,
    )
    candidates = read_candidates(tmp_path)
    assert candidates[0].proposal == proposal
    assert candidates[0].kind == "voice_edit_proposal"


def test_read_candidates_returns_empty_when_missing(tmp_path: Path) -> None:
    assert read_candidates(tmp_path) == []


def test_emit_with_none_emotional_snapshot_round_trips(tmp_path: Path) -> None:
    """v0.0.9: emotional_snapshot is Optional — None must round-trip cleanly."""
    emit_initiate_candidate(
        tmp_path,
        kind="voice_edit_proposal",
        source="voice_reflection",
        source_id="vr_none_snap",
        semantic_context=_ctx(),
        proposal={"old_text": "a", "new_text": "b", "evidence": ["e1", "e2", "e3"]},
    )
    candidates = read_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0].emotional_snapshot is None
    assert candidates[0].kind == "voice_edit_proposal"


def test_remove_candidate_drops_specific_id(tmp_path: Path) -> None:
    from brain.initiate.emit import remove_candidate

    for sid in ["dream_a", "dream_b", "dream_c"]:
        emit_initiate_candidate(
            tmp_path,
            kind="message",
            source="dream",
            source_id=sid,
            emotional_snapshot=_snap(),
            semantic_context=_ctx(),
        )
    candidates = read_candidates(tmp_path)
    target_id = candidates[1].candidate_id
    remove_candidate(tmp_path, target_id)
    after = read_candidates(tmp_path)
    assert len(after) == 2
    assert all(c.candidate_id != target_id for c in after)
