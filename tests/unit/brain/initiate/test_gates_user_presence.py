"""Tests for UserPresence adjustments in check_send_allowed."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.audit import append_audit_row
from brain.initiate.schemas import AuditRow
from brain.initiate.user_pattern import UserPresence


def _delivered_row(audit_id: str, ts: str, urgency: str = "send_notify") -> AuditRow:
    """Build a minimal valid AuditRow for gate tests."""
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


def _presence(
    *,
    silence_days: float = 0.0,
    ignore_streak: int = 0,
    likely_active: bool = True,
    response_lag_p50: float | None = None,
) -> UserPresence:
    return UserPresence(
        silence_days=silence_days,
        ignore_streak=ignore_streak,
        likely_active=likely_active,
        response_lag_p50=response_lag_p50,
    )


# UTC noon — outside the default 23-07 blackout
_SAFE_NOW = datetime(2026, 5, 29, 12, 0, 0, tzinfo=UTC)


def test_presence_none_no_effect(tmp_path: Path) -> None:
    """user_presence=None must not change existing behaviour."""
    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path, urgency="notify", now=_SAFE_NOW, user_presence=None
    )
    assert allowed is True
    assert reason is None


def test_likely_inactive_blocks_send(tmp_path: Path) -> None:
    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path, urgency="notify", now=_SAFE_NOW,
        user_presence=_presence(likely_active=False),
    )
    assert allowed is False
    assert reason == "user_likely_inactive"


def test_silence_loosens_gap(tmp_path: Path) -> None:
    """silence_days >= 3 with zero streak halves the notify min gap."""
    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path, urgency="notify", now=_SAFE_NOW,
        user_presence=_presence(silence_days=3.5, ignore_streak=0),
    )
    assert allowed is True


def test_silence_loosens_quota(tmp_path: Path) -> None:
    """silence_days >= 3 raises the notify cap by 1 (default 3 → 4)."""
    # Fill the default notify cap (3) with sends 3h ago — within 24h window but
    # outside the halved 2h silence-loosened gap, so only the cap drives the block.
    ts_recent = datetime(2026, 5, 29, 9, 0, 0, tzinfo=UTC).isoformat()
    for i in range(3):
        append_audit_row(tmp_path, _delivered_row(f"ia_{i}", ts_recent, "send_notify"))

    from brain.initiate.gates import check_send_allowed

    # Without presence: cap reached → blocked
    blocked, _ = check_send_allowed(tmp_path, urgency="notify", now=_SAFE_NOW)
    assert blocked is False

    # With silence loosening: cap raised to 4 → allowed
    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(silence_days=3.5, ignore_streak=0),
    )
    assert allowed is True


def test_ignore_streak_3_doubles_gap(tmp_path: Path) -> None:
    """ignore_streak >= 3 sets notify gap to 8h; a send 5h ago is blocked."""
    # Last send was 5h ago — passes default 4h gap but fails streak-raised 8h gap
    last_send = datetime(2026, 5, 29, 7, 0, 0, tzinfo=UTC)
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(ignore_streak=3),
    )
    assert allowed is False
    assert reason == "notify_min_gap_not_met"


def test_ignore_streak_6_sets_24h_gap(tmp_path: Path) -> None:
    """ignore_streak >= 6 sets notify gap to 24h; a send 10h ago is blocked."""
    # Last send was 10h ago — within the 24h gap
    last_send = datetime(2026, 5, 29, 2, 0, 0, tzinfo=UTC)
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(ignore_streak=6),
    )
    assert allowed is False
    assert reason == "notify_min_gap_not_met"


def test_response_lag_above_4h_raises_gap(tmp_path: Path) -> None:
    """lag_p50 >= 14400s (4h) raises gap to 8h; a send 5h ago is blocked."""
    # Last send was 5h ago — passes default 4h gap but fails lag-raised 8h gap
    last_send = datetime(2026, 5, 29, 7, 0, 0, tzinfo=UTC)
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(response_lag_p50=18000.0),  # 5h lag p50
    )
    assert allowed is False
    assert reason == "notify_min_gap_not_met"


def test_lag_below_600_is_no_op(tmp_path: Path) -> None:
    """lag_p50 < 600s must not affect gap at all — including not undoing silence loosening.

    silence_days=3.5 + streak=0 halves the notify gap to 2h.
    A send 3h ago is outside the 2h silence-loosened gap → allowed.
    lag=300 (< 600) must NOT push the gap back to 4h (which would block it).
    """
    # Last send was 3h ago — passes the 2h silence-loosened gap but would fail the
    # default 4h gap that the buggy else-branch restores.
    last_send = datetime(2026, 5, 29, 9, 0, 0, tzinfo=UTC)  # 3h before noon
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(silence_days=3.5, ignore_streak=0, response_lag_p50=300.0),
    )
    assert allowed is True  # silence loosening preserved — lag<600 is a no-op


def test_silence_streak_conflict_backoff_wins(tmp_path: Path) -> None:
    """silence_days >= 3 AND ignore_streak >= 3 → backoff wins, not silence loosening."""
    # Last send was 5h ago — would pass 2h silence-gap but blocked by 8h backoff gap.
    last_send = datetime(2026, 5, 29, 7, 0, 0, tzinfo=UTC)
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(silence_days=4.0, ignore_streak=3),
    )
    # ignore-backoff wins: gap is 8h, only 5h elapsed → blocked
    assert allowed is False
    assert reason == "notify_min_gap_not_met"


def test_lag_none_is_no_op(tmp_path: Path) -> None:
    """response_lag_p50=None must not affect gap — silence loosening preserved.

    silence_days=3.5 + streak=0 halves the gap to 2h.
    A send 3h ago is outside 2h → allowed. lag=None must not raise the gap.
    """
    last_send = datetime(2026, 5, 29, 9, 0, 0, tzinfo=UTC)  # 3h before noon
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(silence_days=3.5, ignore_streak=0, response_lag_p50=None),
    )
    assert allowed is True


def test_lag_proportional_band_raises_gap(tmp_path: Path) -> None:
    """600 <= lag_p50 < 14400 raises gap proportionally; 3600s lag → 1.5h gap.

    lag_p50=3600s (1h) → lag_hours = 1h * 1.5 = 1.5h.
    Last send 1h12min ago (1.2h) < 1.5h gap → blocked.
    """
    last_send = datetime(2026, 5, 29, 10, 48, 0, tzinfo=UTC)  # 1h12min before noon
    append_audit_row(tmp_path, _delivered_row("ia_1", last_send.isoformat(), "send_notify"))

    from brain.initiate.gates import check_send_allowed

    allowed, reason = check_send_allowed(
        tmp_path,
        urgency="notify",
        now=_SAFE_NOW,
        user_presence=_presence(response_lag_p50=3600.0),  # 1h lag → 1.5h gap
    )
    assert allowed is False
    assert reason == "notify_min_gap_not_met"
