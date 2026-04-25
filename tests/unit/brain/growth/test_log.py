# tests/unit/brain/growth/test_log.py
"""Tests for brain.growth.log — append-only biography of brain growth."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.growth.log import (
    GrowthLogEvent,
    append_growth_event,
    read_growth_log,
)


def _event(name: str = "lingering", **overrides) -> GrowthLogEvent:
    base = {
        "timestamp": datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        "type": "emotion_added",
        "name": name,
        "description": "test description",
        "decay_half_life_days": 7.0,
        "reason": "test reason",
        "evidence_memory_ids": ("mem_a", "mem_b"),
        "score": 0.78,
        "relational_context": "test relational",
    }
    base.update(overrides)
    return GrowthLogEvent(**base)  # type: ignore[arg-type]


def test_growth_log_event_is_frozen() -> None:
    e = _event()
    with pytest.raises(FrozenInstanceError):
        e.name = "mutated"  # type: ignore[misc]


def test_append_creates_log_file_when_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["name"] == "lingering"
    assert parsed["type"] == "emotion_added"


def test_append_is_append_only_across_calls(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(name="first"))
    append_growth_event(log_path, _event(name="second"))
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["name"] == "first"
    assert json.loads(lines[1])["name"] == "second"


def test_append_writes_iso_utc_timestamp(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["timestamp"].endswith("Z")  # tz-aware UTC ISO format


def test_append_serializes_evidence_ids_as_list(tmp_path: Path) -> None:
    """evidence_memory_ids is a tuple in Python; JSON must serialize as list."""
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event())
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["evidence_memory_ids"] == ["mem_a", "mem_b"]


def test_append_serializes_none_relational_context(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(relational_context=None))
    parsed = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["relational_context"] is None


def test_read_growth_log_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_growth_log(tmp_path / "missing.jsonl") == []


def test_read_growth_log_returns_oldest_first(tmp_path: Path) -> None:
    log_path = tmp_path / "growth.log.jsonl"
    e1 = _event(name="first", timestamp=datetime(2026, 4, 25, tzinfo=UTC))
    e2 = _event(name="second", timestamp=datetime(2026, 4, 26, tzinfo=UTC))
    append_growth_event(log_path, e1)
    append_growth_event(log_path, e2)
    events = read_growth_log(log_path)
    assert len(events) == 2
    assert events[0].name == "first"
    assert events[1].name == "second"


def test_read_growth_log_with_limit_returns_most_recent(tmp_path: Path) -> None:
    """`limit=N` returns the N most-recent events (last N lines)."""
    log_path = tmp_path / "growth.log.jsonl"
    for i in range(5):
        append_growth_event(log_path, _event(name=f"e{i}"))
    events = read_growth_log(log_path, limit=2)
    assert len(events) == 2
    assert events[0].name == "e3"  # second-most recent
    assert events[1].name == "e4"  # most recent


def test_read_growth_log_skips_corrupt_lines(tmp_path: Path, caplog) -> None:
    """A partial-write or hand-edited bad line is skipped, others still parse."""
    log_path = tmp_path / "growth.log.jsonl"
    append_growth_event(log_path, _event(name="good"))
    # Append a corrupt line manually
    with log_path.open("a", encoding="utf-8") as f:
        f.write("{not valid json\n")
    append_growth_event(log_path, _event(name="also_good"))
    events = read_growth_log(log_path)
    assert len(events) == 2
    assert {e.name for e in events} == {"good", "also_good"}


def test_read_growth_log_round_trips_all_fields(tmp_path: Path) -> None:
    """Every field on GrowthLogEvent makes it through write+read intact."""
    log_path = tmp_path / "growth.log.jsonl"
    e = _event(
        name="x",
        description="d",
        decay_half_life_days=None,
        reason="r",
        evidence_memory_ids=("a", "b", "c"),
        score=0.5,
        relational_context="ctx",
    )
    append_growth_event(log_path, e)
    [restored] = read_growth_log(log_path)
    assert restored.name == "x"
    assert restored.description == "d"
    assert restored.decay_half_life_days is None
    assert restored.reason == "r"
    assert restored.evidence_memory_ids == ("a", "b", "c")
    assert restored.score == 0.5
    assert restored.relational_context == "ctx"
