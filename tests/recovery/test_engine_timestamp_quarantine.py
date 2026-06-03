"""Tests that graveyard entries with missing or unparseable created_at_iso are
quarantined (skipped), not admitted with a fabricated datetime.now() timestamp.

Fabricating now() corrupts forgetting salience, felt-time, and narrative
ordering — a recovered memory would look brand-new instead of carrying its
true lived age.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.store import Memory, MemoryStore
from brain.recovery.engine import _build_restore_plan, run_recovery


def _seed_db(persona: Path) -> None:
    """Write a minimal memories.db so _build_restore_plan can open the store."""
    persona.mkdir(parents=True, exist_ok=True)
    s = MemoryStore(persona / "memories.db")
    s.create(Memory(
        id="existing",
        content="already here",
        memory_type="conversation",
        domain="us",
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    ))
    s.close()


def _write_graveyard(persona: Path, entries: list[dict]) -> None:
    lines = "\n".join(json.dumps(e) for e in entries) + "\n"
    (persona / "forgotten_memories.jsonl").write_text(lines, encoding="utf-8")


def test_missing_created_at_iso_is_quarantined(tmp_path):
    """Entry with no created_at_iso field must be skipped, not admitted."""
    persona = tmp_path / "Nova"
    _seed_db(persona)
    _write_graveyard(persona, [
        {
            "memory_id": "no-ts",
            "summary": "a lost thought",
            "domain": "us",
            "memory_type": "conversation",
            # created_at_iso intentionally absent
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        }
    ])

    plan = _build_restore_plan(persona, source_dir=None)

    # Must NOT be in missing_summaries — fabricating now() is forbidden
    assert "no-ts" not in plan.missing_summaries, (
        "Entry with missing created_at_iso was admitted with a fabricated timestamp"
    )
    # The skipped counter must be incremented
    assert plan.skipped_no_timestamp == 1, (
        f"Expected skipped_no_timestamp=1, got {plan.skipped_no_timestamp}"
    )


def test_valid_timestamp_still_restores(tmp_path):
    """Regression: an entry WITH a valid timestamp must still be admitted normally."""
    persona = tmp_path / "Nova"
    _seed_db(persona)
    valid_ts = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC).isoformat()
    _write_graveyard(persona, [
        {
            "memory_id": "good-ts",
            "summary": "a properly-timestamped memory",
            "domain": "us",
            "memory_type": "conversation",
            "created_at_iso": valid_ts,
            "emotion_at_ingest": {"warmth": 0.7},
            "hebbian_neighbors": [],
        }
    ])

    plan = _build_restore_plan(persona, source_dir=None)

    assert "good-ts" in plan.missing_summaries, "Valid entry must still be admitted"
    assert plan.missing_summaries["good-ts"].created_at == datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
    assert plan.skipped_no_timestamp == 0


def test_garbage_created_at_iso_is_quarantined(tmp_path):
    """Entry with unparseable created_at_iso must be skipped, not admitted."""
    persona = tmp_path / "Nova"
    _seed_db(persona)
    _write_graveyard(persona, [
        {
            "memory_id": "bad-ts",
            "summary": "another lost thought",
            "domain": "us",
            "memory_type": "conversation",
            "created_at_iso": "not-a-date",
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        }
    ])

    plan = _build_restore_plan(persona, source_dir=None)

    assert "bad-ts" not in plan.missing_summaries, (
        "Entry with garbage created_at_iso was admitted with a fabricated timestamp"
    )
    assert plan.skipped_no_timestamp == 1, (
        f"Expected skipped_no_timestamp=1, got {plan.skipped_no_timestamp}"
    )


def test_mixed_entries_counts_correctly(tmp_path):
    """Two bad entries + one good: skipped_no_timestamp == 2, one admitted."""
    persona = tmp_path / "Nova"
    _seed_db(persona)
    valid_ts = datetime(2026, 3, 1, tzinfo=UTC).isoformat()
    _write_graveyard(persona, [
        {
            "memory_id": "bad-1",
            "summary": "missing ts",
            "domain": "us",
            "memory_type": "conversation",
            # no created_at_iso
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        },
        {
            "memory_id": "bad-2",
            "summary": "garbage ts",
            "domain": "us",
            "memory_type": "conversation",
            "created_at_iso": "2026-99-99T00:00:00",
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        },
        {
            "memory_id": "good-1",
            "summary": "valid ts",
            "domain": "us",
            "memory_type": "conversation",
            "created_at_iso": valid_ts,
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        },
    ])

    plan = _build_restore_plan(persona, source_dir=None)

    assert "bad-1" not in plan.missing_summaries
    assert "bad-2" not in plan.missing_summaries
    assert "good-1" in plan.missing_summaries
    assert plan.skipped_no_timestamp == 2


def test_run_recovery_surfaces_skipped_count_in_report(tmp_path):
    """run_recovery must propagate skipped_no_timestamp to the RecoveryReport."""
    persona = tmp_path / "Nova"
    _seed_db(persona)
    (persona / "source-manifest.json").write_text(
        '{"migrated_at_utc":"2026-05-01T00:00:00Z","lived_age_hours_at_migration":1.0}'
    )
    _write_graveyard(persona, [
        {
            "memory_id": "bad-ts",
            "summary": "bad",
            "domain": "us",
            "memory_type": "conversation",
            # no created_at_iso
            "emotion_at_ingest": {},
            "hebbian_neighbors": [],
        }
    ])

    report = run_recovery(persona, source_dir=None, dry_run=False)

    assert report.memories_skipped_no_timestamp == 1
