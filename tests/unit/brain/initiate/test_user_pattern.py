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
