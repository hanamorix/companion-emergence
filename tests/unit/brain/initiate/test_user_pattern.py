"""Tests for brain.initiate.user_pattern."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest


def _write_turn(conv_dir: Path, *, speaker: str, ts: datetime) -> None:
    """Helper: append one turn to a session file."""
    (conv_dir / "sess_test.jsonl").open("a").write(
        json.dumps({"session_id": "sess_test", "speaker": speaker, "text": "hi", "ts": ts.isoformat()}) + "\n"
    )


def test_compute_silence_days_no_buffer_returns_zero(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_silence_days

    assert _compute_silence_days(tmp_path) == pytest.approx(0.0)


def test_compute_silence_days_recent_user_turn(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_silence_days

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    twelve_hours_ago = datetime.now(UTC) - timedelta(hours=12)
    _write_turn(conv_dir, speaker="user", ts=twelve_hours_ago)

    result = _compute_silence_days(tmp_path)
    assert 0.4 < result < 0.6  # ~0.5 days


def test_compute_silence_days_skips_companion_turns(tmp_path: Path) -> None:
    """Turns from the companion (persona_dir.name) must not reset the silence clock."""
    from brain.initiate.user_pattern import _compute_silence_days

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # Companion spoke 5 minutes ago; user spoke 3 days ago
    _write_turn(conv_dir, speaker=tmp_path.name, ts=datetime.now(UTC) - timedelta(minutes=5))
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(days=3))

    result = _compute_silence_days(tmp_path)
    assert result > 2.9  # ~3 days — companion turn ignored


def _write_audit_rows(persona_dir: Path, rows: list[dict]) -> None:
    """Write rows to initiate_audit.jsonl."""
    path = persona_dir / "initiate_audit.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_compute_ignore_streak_no_file_returns_zero(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    assert _compute_ignore_streak(tmp_path) == 0


def test_compute_ignore_streak_consecutive_unanswered(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    _write_audit_rows(tmp_path, [
        {"audit_id": "3", "ts": "2026-05-29T08:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit"}},
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_quiet",
         "delivery": {"current_state": "dismissed"}},
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}},
    ])
    # Walking newest-first: unanswered (1), dismissed (1) = streak 2, then replied_explicit -> stop
    assert _compute_ignore_streak(tmp_path) == 2


def test_compute_ignore_streak_filters_non_send_decisions(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_ignore_streak

    _write_audit_rows(tmp_path, [
        {"audit_id": "2", "ts": "2026-05-29T09:00:00+00:00", "decision": "filtered_pre_compose",
         "delivery": {"current_state": "unanswered"}},
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}},
    ])
    # filtered_pre_compose does not count; only the send_notify row counts
    assert _compute_ignore_streak(tmp_path) == 1


def test_compute_likely_active_no_buffer_returns_true(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_likely_active

    assert _compute_likely_active(tmp_path) is True


def test_compute_likely_active_insufficient_history_returns_true(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_likely_active

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    # Write only 10 turns — below _SCHEDULE_MIN_TURNS = 50
    for i in range(10):
        _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=i))
    assert _compute_likely_active(tmp_path) is True


def test_compute_likely_active_peak_hour_true_offpeak_false(tmp_path: Path) -> None:
    """60 turns concentrated at UTC 14:00 → peak hour True, 12 hours away False."""
    from brain.initiate.user_pattern import _compute_likely_active

    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()

    # Write 60 turns all at UTC 14:00, spread over 30 days
    for i in range(60):
        ts = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC) - timedelta(days=i % 30)
        _write_turn(conv_dir, speaker="user", ts=ts)

    # Inject _now at UTC 14:00 — same local bucket as the turns
    now_peak = datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)
    assert _compute_likely_active(tmp_path, _now=now_peak) is True

    # Inject _now at UTC 02:00 — 12 hours away, that bucket has 0 turns
    now_off = datetime(2026, 1, 15, 2, 0, 0, tzinfo=UTC)
    # Guard: confirm these map to different local hours (always true since 12h apart)
    if now_off.astimezone().hour != now_peak.astimezone().hour:
        assert _compute_likely_active(tmp_path, _now=now_off) is False


def test_compute_response_lag_p50_no_file_returns_none(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    assert _compute_response_lag_p50(tmp_path) is None


def test_compute_response_lag_p50_below_cold_start_returns_none(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    # Only 2 replied_explicit rows — below cold-start minimum of 3
    rows = [
        {"audit_id": str(i), "ts": f"2026-05-29T10:0{i}:00+00:00",
         "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": f"2026-05-29T10:0{i}:30+00:00"}]}}
        for i in range(2)
    ]
    _write_audit_rows(tmp_path, rows)
    assert _compute_response_lag_p50(tmp_path) is None


def test_compute_response_lag_p50_computes_median(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import _compute_response_lag_p50

    # Three sends, lags = 60s, 120s, 300s → median = 120s
    rows = [
        {"audit_id": "1", "ts": "2026-05-29T09:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T09:01:00+00:00"}]}},  # 60s
        {"audit_id": "2", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T10:02:00+00:00"}]}},  # 120s
        {"audit_id": "3", "ts": "2026-05-29T11:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "replied_explicit",
                      "state_transitions": [{"to": "replied_explicit",
                                             "at": "2026-05-29T11:05:00+00:00"}]}},  # 300s
    ]
    _write_audit_rows(tmp_path, rows)
    assert _compute_response_lag_p50(tmp_path) == pytest.approx(120.0)


def test_compute_user_presence_cold_start_defaults(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import UserPresence, compute_user_presence

    presence = compute_user_presence(tmp_path)
    assert isinstance(presence, UserPresence)
    assert presence.silence_days == pytest.approx(0.0)
    assert presence.ignore_streak == 0
    assert presence.likely_active is True
    assert presence.response_lag_p50 is None


def test_compute_user_presence_assembles_all_signals(tmp_path: Path) -> None:
    from brain.initiate.user_pattern import compute_user_presence

    # Write one unanswered send to create a streak of 1
    _write_audit_rows(tmp_path, [
        {"audit_id": "1", "ts": "2026-05-29T10:00:00+00:00", "decision": "send_notify",
         "delivery": {"current_state": "unanswered"}}
    ])
    # Write a recent user turn (1 hour ago) → silence_days ~ 0.04
    conv_dir = tmp_path / "active_conversations"
    conv_dir.mkdir()
    _write_turn(conv_dir, speaker="user", ts=datetime.now(UTC) - timedelta(hours=1))

    presence = compute_user_presence(tmp_path)
    assert presence.ignore_streak == 1
    assert presence.silence_days < 0.1
    assert presence.likely_active is True  # < 50 turns → permissive
    assert presence.response_lag_p50 is None  # < 3 replied rows
