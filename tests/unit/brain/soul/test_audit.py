"""Tests for brain.soul.audit — append + read soul_audit.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from brain.soul.audit import append_audit_entry, read_audit_log
from brain.soul.review import Decision


def _make_decision(
    candidate_id: str = "cid-1",
    decision: str = "accept",
    confidence: int = 9,
) -> Decision:
    return Decision(
        candidate_id=candidate_id,
        decision=decision,
        confidence=confidence,
        reasoning="test reasoning",
        love_type="craft",
        resonance=8,
        why_it_matters="it matters",
    )


def _make_candidate(text: str = "a moment") -> dict:
    return {"text": text, "label": "test", "source": "unit_test"}


def test_append_audit_entry_creates_file(tmp_path: Path) -> None:
    """append_audit_entry creates soul_audit.jsonl with a valid JSON line."""
    decision = _make_decision()
    candidate = _make_candidate()

    append_audit_entry(
        tmp_path,
        decision,
        candidate,
        related=["related mem 1"],
        emotional_summary="love:8.0",
        crystallization_id="crystal-uuid-123",
        dry_run=False,
    )

    audit_path = tmp_path / "soul_audit.jsonl"
    assert audit_path.exists()

    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["candidate_id"] == "cid-1"
    assert entry["decision"] == "accept"
    assert entry["confidence"] == 9
    assert entry["crystallization_id"] == "crystal-uuid-123"
    assert entry["dry_run"] is False
    assert entry["emotional_state"] == "love:8.0"
    assert "related_memories" in entry
    assert "ts" in entry


def test_read_audit_log_oldest_first(tmp_path: Path) -> None:
    """read_audit_log returns entries oldest-first."""
    for i in range(3):
        d = _make_decision(candidate_id=f"cid-{i}")
        append_audit_entry(
            tmp_path,
            d,
            _make_candidate(f"moment {i}"),
            related=[],
            emotional_summary="neutral",
            crystallization_id=None,
            dry_run=False,
        )

    entries = read_audit_log(tmp_path)
    assert len(entries) == 3
    # Oldest-first: cid-0, cid-1, cid-2
    assert entries[0]["candidate_id"] == "cid-0"
    assert entries[2]["candidate_id"] == "cid-2"


def test_read_audit_log_skips_malformed_lines(tmp_path: Path) -> None:
    """read_audit_log skips malformed JSON lines without raising."""
    audit_path = tmp_path / "soul_audit.jsonl"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write('{"candidate_id": "valid-1", "decision": "accept"}\n')
        f.write("this is not json at all\n")
        f.write('{"candidate_id": "valid-2", "decision": "defer"}\n')

    entries = read_audit_log(tmp_path)
    assert len(entries) == 2
    assert entries[0]["candidate_id"] == "valid-1"
    assert entries[1]["candidate_id"] == "valid-2"
