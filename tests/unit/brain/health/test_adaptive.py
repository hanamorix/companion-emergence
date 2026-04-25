"""Tests for brain.health.adaptive — backup-depth + verify-after-write driven by audit log."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.adaptive import FileTreatment, compute_treatment


def _audit_entry(file: str, when: datetime, kind: str = "json_parse_error") -> dict:
    return {
        "timestamp": when.isoformat().replace("+00:00", "Z"),
        "anomalies": [
            {
                "timestamp": when.isoformat().replace("+00:00", "Z"),
                "file": file,
                "kind": kind,
                "action": "restored_from_bak1",
                "quarantine_path": None,
                "likely_cause": "unknown",
                "detail": "test",
            }
        ],
    }


def _seed_audit(persona_dir: Path, entries: list[dict]) -> None:
    persona_dir.mkdir(parents=True, exist_ok=True)
    p = persona_dir / "heartbeats.log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def test_no_audit_log_returns_default(tmp_path: Path) -> None:
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_one_anomaly_within_window_returns_default(tmp_path: Path) -> None:
    _seed_audit(tmp_path, [_audit_entry("x.json", datetime.now(UTC) - timedelta(days=2))])
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_three_anomalies_within_window_bumps_treatment(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("x.json", now - timedelta(days=1)),
            _audit_entry("x.json", now - timedelta(days=3)),
            _audit_entry("x.json", now - timedelta(days=5)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=6, verify_after_write=True)


def test_anomalies_for_other_file_not_counted(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("y.json", now - timedelta(days=1)),
            _audit_entry("y.json", now - timedelta(days=2)),
            _audit_entry("y.json", now - timedelta(days=3)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)


def test_anomalies_outside_window_not_counted(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    _seed_audit(
        tmp_path,
        [
            _audit_entry("x.json", now - timedelta(days=10)),
            _audit_entry("x.json", now - timedelta(days=12)),
            _audit_entry("x.json", now - timedelta(days=15)),
        ],
    )
    t = compute_treatment(tmp_path, "x.json")
    assert t == FileTreatment(backup_count=3, verify_after_write=False)
