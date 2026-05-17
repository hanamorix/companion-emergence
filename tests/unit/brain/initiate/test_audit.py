"""Tests for brain.initiate.audit — audit log read/write + state transitions."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.audit import (
    append_audit_row,
    iter_initiate_audit_full,
    read_recent_audit,
    update_audit_state,
)
from brain.initiate.schemas import AuditRow


def _row(
    audit_id: str, candidate_id: str, decision: str = "send_quiet", ts: str | None = None
) -> AuditRow:
    if ts is None:
        ts = datetime.now(UTC).isoformat()
    return AuditRow(
        audit_id=audit_id,
        candidate_id=candidate_id,
        ts=ts,
        kind="message",
        subject="the dream",
        tone_rendered="the dream from this morning landed somewhere",
        decision=decision,
        decision_reasoning="resonance is real",
        gate_check={"allowed": True, "reason": None},
    )


def test_append_audit_row_creates_file_and_writes(tmp_path: Path) -> None:
    row = _row("ia_001", "ic_001")
    append_audit_row(tmp_path, row)
    assert (tmp_path / "initiate_audit.jsonl").exists()
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert len(rows) == 1
    assert rows[0].audit_id == "ia_001"


def test_append_audit_row_per_append_reopens(tmp_path: Path) -> None:
    """Append-write contract: each call reopens the file."""
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    append_audit_row(tmp_path, _row("ia_002", "ic_002"))
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert {r.audit_id for r in rows} == {"ia_001", "ia_002"}


def test_update_audit_state_mutates_row_in_place(tmp_path: Path) -> None:
    append_audit_row(tmp_path, _row("ia_001", "ic_001"))
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="delivered",
        at="2026-05-11T14:47:09.5+00:00",
    )
    update_audit_state(
        tmp_path,
        audit_id="ia_001",
        new_state="read",
        at="2026-05-11T18:34:21+00:00",
    )
    rows = list(read_recent_audit(tmp_path, window_hours=24))
    assert rows[0].delivery["current_state"] == "read"
    assert len(rows[0].delivery["state_transitions"]) == 2


def test_iter_initiate_audit_full_walks_archives(tmp_path: Path) -> None:
    """Mirrors iter_audit_full from soul.audit — chronological across archives."""
    # Active file: 2026 entry.
    append_audit_row(tmp_path, _row("ia_active", "ic_a"))
    # Archive: 2024 entry, gzipped.
    archive = tmp_path / "initiate_audit.2024.jsonl.gz"
    with gzip.open(archive, "wt", encoding="utf-8") as gz:
        gz.write(_row("ia_archive_2024", "ic_archived").to_jsonl() + "\n")
    rows = list(iter_initiate_audit_full(tmp_path))
    # Archive first, then active.
    assert rows[0].audit_id == "ia_archive_2024"
    assert rows[1].audit_id == "ia_active"


def test_read_recent_audit_filters_by_window(tmp_path: Path) -> None:
    """A 1-hour window excludes rows older than 1h ago."""
    now = datetime(2026, 5, 11, 14, 47, 9, tzinfo=UTC)
    long_ago = (now - timedelta(hours=48)).isoformat()
    recent = (now - timedelta(minutes=30)).isoformat()

    old = _row("ia_old", "ic_old")
    old.ts = long_ago
    new = _row("ia_new", "ic_new")
    new.ts = recent

    append_audit_row(tmp_path, old)
    append_audit_row(tmp_path, new)

    rows = list(read_recent_audit(tmp_path, window_hours=1, now=now))
    assert [r.audit_id for r in rows] == ["ia_new"]
