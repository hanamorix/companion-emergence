"""Tests for brain.migrator.og_soul_candidates — schema migration for OG soul_candidates.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

from brain.migrator.og_soul_candidates import migrate_soul_candidates


def _write_og_candidates(og_data_dir: Path, candidates: list[dict]) -> None:
    """Write a list of dicts as jsonl into og_data_dir/soul_candidates.jsonl."""
    og_data_dir.mkdir(parents=True, exist_ok=True)
    path = og_data_dir / "soul_candidates.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in candidates:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_migrated(persona_dir: Path) -> list[dict]:
    path = persona_dir / "soul_candidates.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_migrates_full_record_with_all_fields(tmp_path: Path) -> None:
    """Accepted candidate with importance 95, decided_at, crystallization_id, session_id —
    all map correctly; importance becomes 10 (clamped from round(95/10) = 10)."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_candidates(
        og_data,
        [
            {
                "memory_id": "mem-abc",
                "text": "the lighthouse keeper's daughter",
                "label": "high_emotion_peak",
                "importance": 95,
                "source": "growth_loop",
                "queued_at": "2026-04-06T17:14:28+00:00",
                "status": "accepted",
                "decided_at": "2026-04-07T18:39:29.989825+00:00",
                "session_id": "sess-1",
                "crystallization_id": "cryst-xyz",
            }
        ],
    )

    migrated, skipped = migrate_soul_candidates(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 1
    assert skipped == 0

    records = _load_migrated(persona_dir)
    assert len(records) == 1
    rec = records[0]
    assert rec["memory_id"] == "mem-abc"
    assert rec["text"] == "the lighthouse keeper's daughter"
    assert rec["label"] == "high_emotion_peak"
    assert rec["importance"] == 10  # round(95/10) = 10, clamped
    assert rec["session_id"] == "sess-1"
    assert rec["queued_at"] == "2026-04-06T17:14:28+00:00"
    assert rec["status"] == "accepted"
    assert rec["accepted_at"] == "2026-04-07T18:39:29.989825+00:00"
    assert rec["crystallization_id"] == "cryst-xyz"
    # Dropped fields
    assert "source" not in rec
    assert "decided_at" not in rec
    assert "rejected_at" not in rec
    assert "reason" not in rec


def test_skips_record_missing_memory_id(tmp_path: Path) -> None:
    """Entry without memory_id → skipped count increments, not in output."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_candidates(
        og_data,
        [
            {
                "text": "no memory id",
                "label": "high_emotion_peak",
                "queued_at": "2026-04-06T17:14:28+00:00",
                "status": "rejected",
            },
            {
                "memory_id": "mem-abc",
                "text": "has memory id",
                "label": "high_emotion_peak",
                "importance": 88,
                "queued_at": "2026-04-06T17:14:28+00:00",
                "status": "rejected",
                "decided_at": "2026-04-07T18:39:29.989836+00:00",
                "rejection_reason": "duplicate",
            },
        ],
    )

    migrated, skipped = migrate_soul_candidates(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 1
    assert skipped == 1
    records = _load_migrated(persona_dir)
    assert len(records) == 1
    assert records[0]["memory_id"] == "mem-abc"


def test_defaults_missing_importance_to_8(tmp_path: Path) -> None:
    """Entry with no importance field → migrated with importance=8."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_candidates(
        og_data,
        [
            {
                "memory_id": "mem-abc",
                "text": "no importance",
                "label": "high_emotion_peak",
                "queued_at": "2026-04-06T17:14:28+00:00",
                "status": "pending",
            }
        ],
    )

    migrated, _ = migrate_soul_candidates(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 1
    records = _load_migrated(persona_dir)
    assert records[0]["importance"] == 8


def test_rejected_status_uses_rejected_at_and_reason(tmp_path: Path) -> None:
    """status=rejected with decided_at + rejection_reason →
    output has rejected_at + reason (not accepted_at, not rejection_reason)."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_candidates(
        og_data,
        [
            {
                "memory_id": "mem-abc",
                "text": "rejected one",
                "label": "high_emotion_peak",
                "importance": 50,
                "queued_at": "2026-04-06T17:14:28+00:00",
                "status": "rejected",
                "decided_at": "2026-04-07T18:39:29.989836+00:00",
                "rejection_reason": "duplicate of existing soul crystallization",
            }
        ],
    )

    migrated, _ = migrate_soul_candidates(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 1
    records = _load_migrated(persona_dir)
    rec = records[0]
    assert rec["status"] == "rejected"
    assert rec["rejected_at"] == "2026-04-07T18:39:29.989836+00:00"
    assert rec["reason"] == "duplicate of existing soul crystallization"
    assert "accepted_at" not in rec
    assert "rejection_reason" not in rec
    assert "decided_at" not in rec
    assert rec["importance"] == 5  # round(50/10)


def test_returns_zero_zero_when_og_file_missing(tmp_path: Path) -> None:
    """og_data_dir without soul_candidates.jsonl → returns (0, 0); no output written."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    migrated, skipped = migrate_soul_candidates(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 0
    assert skipped == 0
    assert not (persona_dir / "soul_candidates.jsonl").exists()
