"""brain.behavioral.log — append-only JSONL for creative_dna + journal lifecycle changes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.behavioral.log import (
    append_behavioral_event,
    read_behavioral_log,
)


def test_append_and_read_creative_dna_event(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    append_behavioral_event(
        log_path,
        kind="creative_dna_emerging_added",
        name="intentional sentence fragments",
        timestamp=datetime(2026, 4, 29, 10, 15, 0, tzinfo=UTC),
        reasoning="appeared in 3 recent fiction sessions",
        evidence_memory_ids=("mem_xyz", "mem_uvw"),
    )
    entries = read_behavioral_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "creative_dna_emerging_added"
    assert e["name"] == "intentional sentence fragments"
    assert e["reasoning"] == "appeared in 3 recent fiction sessions"
    assert e["evidence_memory_ids"] == ["mem_xyz", "mem_uvw"]
    assert e["timestamp"].endswith("Z")  # iso UTC


def test_append_journal_entry_event(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    append_behavioral_event(
        log_path,
        kind="journal_entry_added",
        name="mem_journal_abc",
        timestamp=datetime(2026, 4, 29, 11, 0, 0, tzinfo=UTC),
        source="brain_authored",
        reflex_arc_name=None,
        emotional_state={"vulnerability": 7.5, "gratitude": 5.0},
    )
    entries = read_behavioral_log(log_path)
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "journal_entry_added"
    assert e["source"] == "brain_authored"
    assert e["reflex_arc_name"] is None
    assert e["emotional_state"]["vulnerability"] == 7.5


def test_multiple_appends_preserve_order(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    base = datetime(2026, 4, 29, tzinfo=UTC)
    for i in range(3):
        append_behavioral_event(
            log_path,
            kind="creative_dna_emerging_added",
            name=f"tendency_{i}",
            timestamp=base.replace(hour=i),
            reasoning=f"reason {i}",
            evidence_memory_ids=(),
        )
    entries = read_behavioral_log(log_path)
    assert [e["name"] for e in entries] == ["tendency_0", "tendency_1", "tendency_2"]


def test_corrupt_line_is_skipped(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    log_path.write_text(
        '{"kind":"creative_dna_emerging_added","name":"valid1","timestamp":"2026-04-29T00:00:00Z","reasoning":"r","evidence_memory_ids":[]}\n'
        "this is not json\n"
        '{"kind":"creative_dna_emerging_added","name":"valid2","timestamp":"2026-04-29T01:00:00Z","reasoning":"r","evidence_memory_ids":[]}\n'
    )
    entries = read_behavioral_log(log_path)
    assert [e["name"] for e in entries] == ["valid1", "valid2"]


def test_read_missing_file_returns_empty(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    assert read_behavioral_log(log_path) == []


def test_filter_by_window(tmp_path: Path):
    log_path = tmp_path / "behavioral_log.jsonl"
    base = datetime(2026, 4, 29, tzinfo=UTC)
    for days_ago in (1, 5, 10, 20):
        append_behavioral_event(
            log_path,
            kind="creative_dna_emerging_added",
            name=f"d{days_ago}",
            timestamp=datetime(2026, 4, 29, tzinfo=UTC).replace(day=29 - days_ago),
            reasoning="r",
            evidence_memory_ids=(),
        )
    # last 7 days = day 22..29 inclusive => d1, d5
    entries = read_behavioral_log(log_path, since=base.replace(day=22))
    names = sorted(e["name"] for e in entries)
    assert names == ["d1", "d5"]
