"""Tests for brain.health.anomaly — BrainAnomaly + AlarmEntry frozen dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from brain.health.anomaly import AlarmEntry, BrainAnomaly


def test_brain_anomaly_construction() -> None:
    a = BrainAnomaly(
        timestamp=datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        file="emotion_vocabulary.json",
        kind="json_parse_error",
        action="restored_from_bak1",
        quarantine_path="emotion_vocabulary.json.corrupt-2026-04-25T18:30:00Z",
        likely_cause="user_edit",
        detail="Expecting ',' delimiter: line 12 column 5",
    )
    assert a.file == "emotion_vocabulary.json"
    assert a.kind == "json_parse_error"
    assert a.likely_cause == "user_edit"


def test_brain_anomaly_is_frozen() -> None:
    a = BrainAnomaly(
        timestamp=datetime.now(UTC),
        file="x.json",
        kind="json_parse_error",
        action="reset_to_default",
        quarantine_path=None,
        likely_cause="unknown",
        detail="",
    )
    with pytest.raises(FrozenInstanceError):
        a.file = "mutated"  # type: ignore[misc]


def test_brain_anomaly_to_dict_serialises_iso_utc() -> None:
    a = BrainAnomaly(
        timestamp=datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        file="x.json",
        kind="schema_mismatch",
        action="reset_to_default",
        quarantine_path=None,
        likely_cause="unknown",
        detail="missing 'emotions' key",
    )
    d = a.to_dict()
    assert d["timestamp"].endswith("Z")
    assert d["file"] == "x.json"
    assert d["quarantine_path"] is None


def test_alarm_entry_is_frozen() -> None:
    e = AlarmEntry(
        file="memories.db",
        kind="sqlite_integrity_fail",
        first_seen_at=datetime.now(UTC),
        occurrences_in_window=1,
    )
    with pytest.raises(FrozenInstanceError):
        e.file = "x"  # type: ignore[misc]
