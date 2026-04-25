"""Tests for brain.health.alarm."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.alarm import compute_pending_alarms


def _audit_entry(
    file: str, when: datetime, action: str = "restored_from_bak1", kind: str = "json_parse_error"
) -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": action,
                "quarantine_path": None,
                "likely_cause": "unknown",
                "detail": "",
            }
        ],
    }


def _seed(persona_dir: Path, entries: list[dict]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_no_anomalies_no_alarms(tmp_path: Path) -> None:
    assert compute_pending_alarms(tmp_path) == []


def test_reset_to_default_on_identity_file_alarms(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [_audit_entry("emotion_vocabulary.json", datetime.now(UTC), action="reset_to_default")],
    )
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].file == "emotion_vocabulary.json"


def test_sqlite_integrity_fail_alarms(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        [
            _audit_entry(
                "memories.db",
                datetime.now(UTC),
                kind="sqlite_integrity_fail",
                action="alarmed_unrecoverable",
            )
        ],
    )
    alarms = compute_pending_alarms(tmp_path)
    assert len(alarms) == 1
    assert alarms[0].kind == "sqlite_integrity_fail"


def test_acknowledged_alarm_suppressed(tmp_path: Path) -> None:
    when = datetime.now(UTC)
    entries = [
        _audit_entry("emotion_vocabulary.json", when, action="reset_to_default"),
        # User acknowledged after the reset
        {
            "timestamp": (when + timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
            "user_acknowledged": ["emotion_vocabulary.json"],
        },
    ]
    _seed(tmp_path, entries)
    alarms = compute_pending_alarms(tmp_path)
    assert alarms == []
