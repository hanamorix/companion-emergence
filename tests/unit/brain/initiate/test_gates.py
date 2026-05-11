"""Tests for brain.initiate.gates — cost-cap + cooldown + user-local time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from brain.initiate.audit import append_audit_row
from brain.initiate.gates import (
    check_send_allowed,
    count_recent_sends,
    in_blackout_window,
)
from brain.initiate.schemas import AuditRow


def _delivered_row(audit_id: str, ts: str, urgency: str = "send_quiet") -> AuditRow:
    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts=ts,
        kind="message",
        subject="x",
        tone_rendered="x",
        decision=urgency,
        decision_reasoning="x",
        gate_check={"allowed": True, "reason": None},
    )
    row.record_transition("delivered", ts)
    return row


def test_in_blackout_window_default_23_to_07():
    """Default blackout: 23:00-07:00 user-local."""
    tz = ZoneInfo("America/Los_Angeles")
    assert in_blackout_window(datetime(2026, 5, 11, 23, 30, tzinfo=tz)) is True
    assert in_blackout_window(datetime(2026, 5, 11, 6, 59, tzinfo=tz)) is True
    assert in_blackout_window(datetime(2026, 5, 11, 7, 0, tzinfo=tz)) is False
    assert in_blackout_window(datetime(2026, 5, 11, 22, 59, tzinfo=tz)) is False
    assert in_blackout_window(datetime(2026, 5, 11, 12, 0, tzinfo=tz)) is False


def test_count_recent_sends_filters_by_urgency_and_window(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 18, 0, tzinfo=UTC)
    # 1 notify 2h ago; 1 quiet 5h ago; 1 notify 30h ago (outside 24h window)
    append_audit_row(
        tmp_path,
        _delivered_row("ia_1", (now - timedelta(hours=2)).isoformat(), "send_notify"),
    )
    append_audit_row(
        tmp_path,
        _delivered_row("ia_2", (now - timedelta(hours=5)).isoformat(), "send_quiet"),
    )
    append_audit_row(
        tmp_path,
        _delivered_row("ia_3", (now - timedelta(hours=30)).isoformat(), "send_notify"),
    )
    assert count_recent_sends(tmp_path, urgency="notify", window_hours=24, now=now) == 1
    assert count_recent_sends(tmp_path, urgency="quiet", window_hours=24, now=now) == 1


def test_check_send_allowed_passes_when_under_cap(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
    allowed, reason = check_send_allowed(tmp_path, urgency="quiet", now=now)
    assert allowed is True
    assert reason is None


def test_check_send_allowed_blocks_notify_in_blackout(tmp_path: Path) -> None:
    """If user-local time is in 23:00-07:00, notify is denied."""
    tz = ZoneInfo("America/Los_Angeles")
    blackout_local = datetime(2026, 5, 11, 1, 30, tzinfo=tz)
    allowed, reason = check_send_allowed(
        tmp_path, urgency="notify", now=blackout_local
    )
    assert allowed is False
    assert reason is not None
    assert "blackout" in reason


def test_check_send_allowed_blocks_when_notify_cap_reached(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
    for i in range(3):
        append_audit_row(
            tmp_path,
            _delivered_row(
                f"ia_{i}",
                (now - timedelta(hours=2 * (i + 1))).isoformat(),
                "send_notify",
            ),
        )
    allowed, reason = check_send_allowed(tmp_path, urgency="notify", now=now)
    assert allowed is False
    assert "notify_cap_24h_reached" in reason


def test_check_send_allowed_blocks_when_min_gap_not_met(tmp_path: Path) -> None:
    now = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
    append_audit_row(
        tmp_path,
        _delivered_row(
            "ia_recent",
            (now - timedelta(hours=1)).isoformat(),
            "send_notify",
        ),
    )
    allowed, reason = check_send_allowed(tmp_path, urgency="notify", now=now)
    assert allowed is False
    assert "min_gap" in reason
