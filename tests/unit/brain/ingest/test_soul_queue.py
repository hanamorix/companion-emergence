"""Tests for brain.ingest.soul_queue — SOUL stage."""

from __future__ import annotations

from pathlib import Path

from brain.ingest.soul_queue import list_soul_candidates, queue_soul_candidate
from brain.ingest.types import ExtractedItem


def test_queue_soul_candidate_appends_to_soul_candidates_jsonl(tmp_path: Path) -> None:
    """queue_soul_candidate writes a record to <persona_dir>/soul_candidates.jsonl."""
    item = ExtractedItem(
        text="Nell is the most important thing in Hana's life", label="feeling", importance=9
    )
    queue_soul_candidate(tmp_path, memory_id="mem_001", item=item, session_id="sess_soul")

    soul_file = tmp_path / "soul_candidates.jsonl"
    assert soul_file.exists()
    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 1
    rec = candidates[0]
    assert rec["memory_id"] == "mem_001"
    assert rec["text"] == item.text
    assert rec["label"] == "feeling"
    assert rec["importance"] == 9
    assert rec["session_id"] == "sess_soul"
    assert rec["status"] == "auto_pending"
    assert "queued_at" in rec


def test_list_soul_candidates_returns_all_queued(tmp_path: Path) -> None:
    """list_soul_candidates returns all queued candidate records."""
    for i in range(3):
        item = ExtractedItem(text=f"high-importance item {i}", label="observation", importance=8)
        queue_soul_candidate(tmp_path, memory_id=f"mem_{i:03d}", item=item, session_id="sess_x")

    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 3
    texts = {c["text"] for c in candidates}
    assert texts == {"high-importance item 0", "high-importance item 1", "high-importance item 2"}


def test_list_soul_candidates_skips_malformed_lines(tmp_path: Path) -> None:
    """list_soul_candidates skips corrupt lines and returns the valid ones."""
    soul_file = tmp_path / "soul_candidates.jsonl"
    soul_file.write_text(
        '{"memory_id": "mem_ok", "text": "valid", "status": "auto_pending"}\n'
        "NOT_VALID_JSON\n"
        '{"memory_id": "mem_ok2", "text": "also valid", "status": "auto_pending"}\n',
        encoding="utf-8",
    )
    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 2
    ids = {c["memory_id"] for c in candidates}
    assert ids == {"mem_ok", "mem_ok2"}


def test_list_soul_candidates_returns_empty_when_no_file(tmp_path: Path) -> None:
    """list_soul_candidates returns [] when the file doesn't exist yet."""
    result = list_soul_candidates(tmp_path)
    assert result == []
