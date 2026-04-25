"""Pending-alarm computation — derived from audit log; no separate state file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.anomaly import AlarmEntry
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import parse_iso_utc

_IDENTITY_FILES = frozenset(
    {
        "emotion_vocabulary.json",
        "interests.json",
        "reflex_arcs.json",
        # future: "soul.json"
    }
)

WINDOW_DAYS = 7


def compute_pending_alarms(persona_dir: Path) -> list[AlarmEntry]:
    """Walk recent audit log; return alarms not yet acknowledged."""
    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    # First pass: collect all anomalies + acknowledgments in window.
    anomalies: list[dict] = []
    acknowledged_files: set[str] = set()

    for entry in read_jsonl_skipping_corrupt(audit_path):
        try:
            ts = parse_iso_utc(entry["timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        if ts < cutoff:
            continue

        ack = entry.get("user_acknowledged") or []
        if isinstance(ack, list):
            for f in ack:
                if isinstance(f, str):
                    acknowledged_files.add(f)

        for a in entry.get("anomalies") or []:
            if isinstance(a, dict):
                anomalies.append(a)

    # Second pass: count per file + classify alarmable.
    by_file: dict[str, list[dict]] = {}
    for a in anomalies:
        f = a.get("file")
        if not isinstance(f, str):
            continue
        by_file.setdefault(f, []).append(a)

    alarms: list[AlarmEntry] = []
    for f, anoms in by_file.items():
        if f in acknowledged_files:
            continue

        is_alarm = False
        for a in anoms:
            if a.get("action") == "reset_to_default" and f in _IDENTITY_FILES:
                is_alarm = True
                break
            if a.get("kind") == "sqlite_integrity_fail":
                is_alarm = True
                break
        # ≥6 anomalies in window = recurring-after-adaptation
        if len(anoms) >= 6:
            is_alarm = True

        if is_alarm:
            first_seen = min(parse_iso_utc(a["timestamp"]) for a in anoms)
            kind = anoms[-1].get("kind", "json_parse_error")
            alarms.append(
                AlarmEntry(
                    file=f,
                    kind=kind,
                    first_seen_at=first_seen,
                    occurrences_in_window=len(anoms),
                )
            )

    return alarms
