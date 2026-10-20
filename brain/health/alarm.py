"""Pending-alarm computation — derived from audit log; no separate state file."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.anomaly import AlarmEntry
from brain.health.json_io import Recovery, iter_jsonl, read_jsonl_tail
from brain.utils.time import parse_iso_utc

_IDENTITY_FILES = frozenset(
    {
        "emotion_vocabulary.json",
        "interests.json",
        "reflex_arcs.json",
        "voice.md",
        # When the soul module lands, add "soul.json" here so its
        # reset_to_default raises an alarm. See spec §9.1.
    }
)

WINDOW_DAYS = 7
# This many anomalies (or kept-aside rows) in the window = recurring after adaptation: an alarm.
RECURRING_THRESHOLD = 6

# The two files the consolidation gate sets rows aside in (#470 review item 5). Nothing else reads them, so without this a
# gate bug that quarantines every candidate is invisible. The names are the producers' (`engines.consolidation` for the
# quarantine, `memory.pending` for the damaged lines); a test pins that they are equal. They are literals here, not imports,
# so `health` does not depend on `memory` and `scripts/organ_wireback_audit.py` (one file at a time) sees the read.
QUARANTINE_FILENAME = "consolidation_quarantine.jsonl"
DAMAGED_FILENAME = "pending_candidates.damaged.jsonl"


def _stamps_after(path: Path, field: str, since: datetime) -> list[datetime]:
    """The stamps of the rows of an append-only, time-ordered JSONL file that are newer than ``since``.

    Read backwards in bounded chunks, starting from the alarm threshold and doubling, until a row at or before ``since`` turns up
    or the file is exhausted: the cost follows the rows inside the window, not the history of the file (the quarantine has no
    rotation and ``/health`` reads this on its restart poll). A row without a usable stamp, a damaged line and an unreadable
    file count for nothing and raise nothing (``read_jsonl_tail`` skips and reports them).
    """
    n = RECURRING_THRESHOLD
    previous = -1
    while True:
        stamps: list[datetime] = []
        boundary = False
        rows = read_jsonl_tail(path, n)
        for row in rows:
            try:
                stamp = parse_iso_utc(row[field])
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
            if stamp > since:
                stamps.append(stamp)
            else:
                boundary = True
        # Stop at the boundary, or when a doubled read found no more records than the last one (the start of the file).
        if boundary or len(rows) == previous:
            return stamps
        previous = len(rows)
        n *= 2


def compute_pending_alarms(persona_dir: Path) -> list[AlarmEntry]:
    """Walk recent audit log; return alarms not yet acknowledged."""
    audit_path = persona_dir / "heartbeats.log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)

    # First pass: collect all anomalies + acknowledgments in window.
    anomalies: list[dict] = []
    acknowledged_files: set[str] = set()
    acknowledged_at: dict[str, datetime] = {}  # file name -> the latest acknowledgement in the window

    for entry in iter_jsonl(audit_path, recover=Recovery.RAISE):
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
                    acknowledged_at[f] = max(ts, acknowledged_at.get(f, ts))

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
        if len(anoms) >= RECURRING_THRESHOLD:
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

    # The gate's kept-aside files: an alarm when the rows in the window reach the recurring threshold. An acknowledgement
    # counts only the rows after it, so a gate bug that keeps quarantining after the user looked raises it again.
    kept_aside = (  # (file, the field that stamps a row, the alarm kind)
        (QUARANTINE_FILENAME, "quarantined_at", "quarantined_candidates"),
        (DAMAGED_FILENAME, "kept_at", "damaged_queue_lines"),
    )
    for name, field, kind in kept_aside:
        stamps = _stamps_after(persona_dir / name, field, max(cutoff, acknowledged_at.get(name, cutoff)))
        if len(stamps) >= RECURRING_THRESHOLD:
            alarms.append(
                AlarmEntry(file=name, kind=kind, first_seen_at=min(stamps), occurrences_in_window=len(stamps))
            )

    return alarms
