"""Unit tests for brain/body/session_hours.py — idle-threshold behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from brain.body.session_hours import compute_active_session_hours
from brain.ingest.buffer import ingest_turn


def test_stale_buffer_returns_zero(tmp_path):
    """Buffer with last activity > 5 min ago must return 0.0."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    old_ts = (now - timedelta(minutes=10)).isoformat()
    ingest_turn(persona_dir, {"speaker": "user", "text": "hi", "session_id": "s1", "ts": old_ts})

    result = compute_active_session_hours(persona_dir, now=now)
    assert result == 0.0, f"stale buffer should be 0.0, got {result}"


def test_active_buffer_returns_elapsed_hours(tmp_path):
    """Buffer with recent activity returns elapsed time since session start."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    start_ts = (now - timedelta(hours=2)).isoformat()
    recent_ts = (now - timedelta(minutes=1)).isoformat()
    ingest_turn(persona_dir, {"speaker": "user", "text": "first", "session_id": "s1", "ts": start_ts})
    ingest_turn(persona_dir, {"speaker": "nell", "text": "reply", "session_id": "s1", "ts": recent_ts})

    result = compute_active_session_hours(persona_dir, now=now)
    assert 1.9 <= result <= 2.1, f"expected ~2 hours for active session, got {result}"
