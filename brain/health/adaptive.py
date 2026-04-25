"""Adaptive treatment — backup depth + verify-after-write computed from audit log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import parse_iso_utc

WINDOW_DAYS = 7
BUMP_THRESHOLD = 3
ELEVATED_BACKUP_COUNT = 6
DEFAULT_BACKUP_COUNT = 3


@dataclass(frozen=True)
class FileTreatment:
    backup_count: int
    verify_after_write: bool


def compute_treatment(persona_dir: Path, file: str) -> FileTreatment:
    """Read audit log; if `file` saw ≥3 anomalies in last 7 days, return elevated treatment."""
    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    count = 0
    for entry in read_jsonl_skipping_corrupt(audit_path):
        for a in entry.get("anomalies") or []:
            if not isinstance(a, dict):
                continue
            if a.get("file") != file:
                continue
            try:
                ts = parse_iso_utc(a["timestamp"])
            except (KeyError, ValueError, TypeError):
                continue
            if ts >= cutoff:
                count += 1

    if count >= BUMP_THRESHOLD:
        return FileTreatment(backup_count=ELEVATED_BACKUP_COUNT, verify_after_write=True)
    return FileTreatment(backup_count=DEFAULT_BACKUP_COUNT, verify_after_write=False)
