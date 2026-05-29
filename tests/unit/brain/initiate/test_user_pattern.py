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
