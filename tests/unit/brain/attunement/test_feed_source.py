"""Tests for the attunement feed source.

Uses the real FeedEntry shape from brain.bridge.feed — `type` field, no
`color` field (color lives in the frontend TYPE_DOT lookup).
"""
from __future__ import annotations

import json
from pathlib import Path

from brain.attunement.schemas import SCHEMA_VERSION, LearnedPattern
from brain.attunement.store import _append_pattern
from brain.bridge.feed import FeedEntry


def _write_backfill_complete(persona_dir: Path) -> None:
    p = persona_dir / "attunement"
    p.mkdir(parents=True, exist_ok=True)
    (p / "backfill_state.json").write_text(
        json.dumps(
            {
                "started_at": "2026-05-31T10:00:00Z",
                "total_windows": 100,
                "sampled_windows": 20,
                "processed_windows": 20,
                "patterns_emitted": 5,
                "status": "complete",
                "last_cursor": "window-19",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )


def test_feed_source_emits_backfill_complete_event(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _write_backfill_complete(tmp_path)
    entries = build_attunement_entries(tmp_path)
    backfill_entries = [e for e in entries if e.type == "attunement_backfill"]
    assert len(backfill_entries) == 1


def test_feed_source_skips_when_backfill_not_complete(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    p = tmp_path / "attunement"
    p.mkdir(parents=True, exist_ok=True)
    (p / "backfill_state.json").write_text(
        json.dumps(
            {
                "started_at": "2026-05-31T10:00:00Z",
                "total_windows": 100,
                "sampled_windows": 20,
                "processed_windows": 5,
                "patterns_emitted": 1,
                "status": "running",
                "last_cursor": "window-4",
                "schema_version": SCHEMA_VERSION,
            }
        )
    )
    entries = build_attunement_entries(tmp_path)
    backfill_entries = [e for e in entries if e.type == "attunement_backfill"]
    assert backfill_entries == []


def test_feed_source_emits_crystallisation_for_known_patterns(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _append_pattern(
        tmp_path,
        LearnedPattern(
            id="p1",
            category="tone",
            canonical_key="tone:warm-when-dog",
            description="softens about the dog",
            evidence_count=12,
            maturity="known",
            first_seen_at="2026-04-01T00:00:00Z",
            last_confirmed_at="2026-05-31T12:00:00Z",
            last_addressed_at=None,
            crystallised_at="2026-05-30T12:00:00Z",
            falsified_at=None,
            examples=[],
            schema_version=SCHEMA_VERSION,
        ),
    )
    entries = build_attunement_entries(tmp_path)
    crystal_entries = [e for e in entries if e.type == "attunement_crystal"]
    assert len(crystal_entries) == 1
    assert "softens about the dog" in crystal_entries[0].body


def test_feed_source_skips_uncrystallised_patterns(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _append_pattern(
        tmp_path,
        LearnedPattern(
            id="p1",
            category="tone",
            canonical_key="key",
            description="x",
            evidence_count=5,
            maturity="forming",
            first_seen_at="2026-04-01T00:00:00Z",
            last_confirmed_at="2026-05-31T12:00:00Z",
            last_addressed_at=None,
            crystallised_at=None,
            falsified_at=None,
            examples=[],
            schema_version=SCHEMA_VERSION,
        ),
    )
    entries = build_attunement_entries(tmp_path)
    assert all(e.type != "attunement_crystal" for e in entries)


def test_feed_entries_are_feed_entry_instances(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _write_backfill_complete(tmp_path)
    _append_pattern(
        tmp_path,
        LearnedPattern(
            id="p1",
            category="tone",
            canonical_key="tone:warm-when-dog",
            description="softens about the dog",
            evidence_count=12,
            maturity="known",
            first_seen_at="2026-04-01T00:00:00Z",
            last_confirmed_at="2026-05-31T12:00:00Z",
            last_addressed_at=None,
            crystallised_at="2026-05-30T12:00:00Z",
            falsified_at=None,
            examples=[],
            schema_version=SCHEMA_VERSION,
        ),
    )
    entries = build_attunement_entries(tmp_path)
    assert all(isinstance(e, FeedEntry) for e in entries)
    types = {e.type for e in entries}
    assert "attunement_backfill" in types
    assert "attunement_crystal" in types


def test_backfill_entry_ts_matches_started_at(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _write_backfill_complete(tmp_path)
    entries = build_attunement_entries(tmp_path)
    backfill_entries = [e for e in entries if e.type == "attunement_backfill"]
    assert backfill_entries[0].ts == "2026-05-31T10:00:00Z"


def test_crystal_entry_ts_matches_crystallised_at(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    _append_pattern(
        tmp_path,
        LearnedPattern(
            id="p1",
            category="tone",
            canonical_key="tone:warm-when-dog",
            description="softens about the dog",
            evidence_count=12,
            maturity="known",
            first_seen_at="2026-04-01T00:00:00Z",
            last_confirmed_at="2026-05-31T12:00:00Z",
            last_addressed_at=None,
            crystallised_at="2026-05-30T12:00:00Z",
            falsified_at=None,
            examples=[],
            schema_version=SCHEMA_VERSION,
        ),
    )
    entries = build_attunement_entries(tmp_path)
    crystal_entries = [e for e in entries if e.type == "attunement_crystal"]
    assert crystal_entries[0].ts == "2026-05-30T12:00:00Z"


def test_corrupt_backfill_state_is_skipped(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    p = tmp_path / "attunement"
    p.mkdir(parents=True, exist_ok=True)
    (p / "backfill_state.json").write_text("NOT VALID JSON }{")
    entries = build_attunement_entries(tmp_path)
    assert all(e.type != "attunement_backfill" for e in entries)


def test_returns_empty_when_no_attunement_dir(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    entries = build_attunement_entries(tmp_path)
    assert entries == []


def test_multiple_crystallised_patterns_each_get_entry(tmp_path: Path) -> None:
    from brain.attunement.feed_source import build_attunement_entries

    for i, desc in enumerate(["desc A", "desc B", "desc C"]):
        _append_pattern(
            tmp_path,
            LearnedPattern(
                id=f"p{i}",
                category="tone",
                canonical_key=f"tone:key{i}",
                description=desc,
                evidence_count=12,
                maturity="known",
                first_seen_at="2026-04-01T00:00:00Z",
                last_confirmed_at="2026-05-31T12:00:00Z",
                last_addressed_at=None,
                crystallised_at=f"2026-05-30T1{i}:00:00Z",
                falsified_at=None,
                examples=[],
                schema_version=SCHEMA_VERSION,
            ),
        )
    entries = build_attunement_entries(tmp_path)
    crystal_entries = [e for e in entries if e.type == "attunement_crystal"]
    assert len(crystal_entries) == 3
