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
    """A genuinely continuous session (turns < 30 min apart spanning ~2h)
    returns the full elapsed time since the session started."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    # Turns every 20 minutes from 2h ago to 1 min ago — no gap >= 30 min, so
    # this is one continuous sitting.
    for minutes_ago in (120, 100, 80, 60, 40, 20, 1):
        ts = (now - timedelta(minutes=minutes_ago)).isoformat()
        ingest_turn(persona_dir, {"speaker": "user", "text": "t", "session_id": "s1", "ts": ts})

    result = compute_active_session_hours(persona_dir, now=now)
    assert 1.9 <= result <= 2.1, f"expected ~2 hours for continuous session, got {result}"


def test_multiline_buffer_stale_last_line_returns_zero(tmp_path):
    """Multi-line buffer whose LAST turn is >= 5 min old must return 0.0.

    Pins the 'orphan drain' shape: even when a buffer has many turns, if
    the last turn is stale the session is not currently live.
    """
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    # Several turns ending > 5 min ago.
    for minutes_ago in (60, 45, 30, 15, 10):
        ts = (now - timedelta(minutes=minutes_ago)).isoformat()
        ingest_turn(persona_dir, {"speaker": "user", "text": "t", "session_id": "s1", "ts": ts})

    result = compute_active_session_hours(persona_dir, now=now)
    assert result == 0.0, f"stale-last-line multiline buffer should be 0.0, got {result}"


def test_multiline_buffer_fresh_last_line_counts_contiguous_run(tmp_path):
    """Multi-line buffer with a fresh last turn returns the full contiguous span.

    All turns are < 30 min apart and the last turn is < 5 min ago → the
    engine counts the whole run from the earliest turn.
    """
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    # Five turns, each 20 min apart, last one 1 min ago — one 81-minute run.
    for minutes_ago in (81, 61, 41, 21, 1):
        ts = (now - timedelta(minutes=minutes_ago)).isoformat()
        ingest_turn(persona_dir, {"speaker": "user", "text": "t", "session_id": "s1", "ts": ts})

    result = compute_active_session_hours(persona_dir, now=now)
    # 81 min ≈ 1.35 h; allow a small float tolerance.
    assert 1.3 <= result <= 1.4, f"expected ~1.35h for 81-min contiguous run, got {result}"


def test_reactivated_stale_buffer_counts_only_current_run(tmp_path):
    """A buffer with a turn days ago and a fresh turn now is NOT one long
    session — the multi-day idle gap is a session boundary, so only the
    current contiguous run counts. Regression for the 69.7h session-hours
    energy-collapse bug (a reply into a 3-day-old buffer reported ~70h).
    """
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir(parents=True)
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    days_ago = (now - timedelta(hours=70)).isoformat()
    just_now = (now - timedelta(minutes=1)).isoformat()
    ingest_turn(persona_dir, {"speaker": "user", "text": "old", "session_id": "s1", "ts": days_ago})
    ingest_turn(persona_dir, {"speaker": "user", "text": "new", "session_id": "s1", "ts": just_now})

    result = compute_active_session_hours(persona_dir, now=now)
    assert result < 0.5, f"reactivated stale buffer should count only the current run, got {result}h"
