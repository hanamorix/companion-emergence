"""Tests for brain.initiate.ambient — always-on verify slice builder."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.ambient import build_outbound_recall_block
from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow


def _row(audit_id: str, ts: str, decision: str, state: str = "delivered") -> AuditRow:
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts=ts,
        kind="message",
        subject="the dream from this morning",
        tone_rendered="the dream from this morning landed",
        decision=decision,
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
        delivery=None,
    )
    row.record_transition("delivered", ts)
    if state != "delivered":
        row.record_transition(state, ts)
    return row


def test_build_outbound_recall_block_empty_returns_none(tmp_path: Path) -> None:
    """No audit history → block is omitted (returns None or empty string)."""
    result = build_outbound_recall_block(tmp_path)
    assert result is None or result == ""


def test_build_outbound_recall_block_includes_recent_outbound(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
    recent_ts = (now - timedelta(hours=4)).isoformat()
    append_audit_row(tmp_path, _row("ia_1", recent_ts, "send_quiet"))
    block = build_outbound_recall_block(tmp_path, now=now)
    assert block is not None
    assert "the dream from this morning" in block
    assert "Recent outbound" in block


def test_build_outbound_recall_block_surfaces_acknowledged_unclear(
    tmp_path: Path,
) -> None:
    """acknowledged_unclear entries from last 24h get a 'Pending uncertainty' block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
    ts = (now - timedelta(hours=2)).isoformat()
    append_audit_row(
        tmp_path, _row("ia_1", ts, "send_quiet", state="acknowledged_unclear")
    )
    block = build_outbound_recall_block(tmp_path, now=now)
    assert "Pending uncertainty" in block
    assert "acknowledged_unclear" in block


def test_build_outbound_recall_block_caps_at_5_recent(tmp_path: Path) -> None:
    """Show at most 5 most-recent rows in the Recent block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
    for i in range(10):
        ts = (now - timedelta(hours=i + 1)).isoformat()
        append_audit_row(tmp_path, _row(f"ia_{i}", ts, "send_quiet"))
    block = build_outbound_recall_block(tmp_path, now=now)
    assert block.count("the dream") == 5  # cap


def test_build_outbound_recall_block_excludes_holds_and_drops(tmp_path: Path) -> None:
    """Only actual sends appear in the Recent block."""
    now = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
    ts = (now - timedelta(hours=1)).isoformat()
    append_audit_row(tmp_path, _row("ia_hold", ts, "hold"))
    append_audit_row(tmp_path, _row("ia_drop", ts, "drop"))
    block = build_outbound_recall_block(tmp_path, now=now)
    # Either block is None (no qualifying sends) or it doesn't include these.
    if block is not None:
        assert "ia_hold" not in block
        assert "ia_drop" not in block
