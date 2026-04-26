"""Tests for brain.ingest.buffer — BUFFER + CLOSE stage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.ingest.buffer import (
    delete_session_buffer,
    ingest_turn,
    list_active_sessions,
    read_session,
    session_silence_minutes,
)


def test_ingest_turn_appends_to_session_file(tmp_path: Path) -> None:
    """ingest_turn writes a JSONL record to active_conversations/<session_id>.jsonl."""
    session_id = ingest_turn(
        tmp_path, {"session_id": "sess_abc", "speaker": "Hana", "text": "hello"}
    )
    assert session_id == "sess_abc"
    buf_file = tmp_path / "active_conversations" / "sess_abc.jsonl"
    assert buf_file.exists()
    line = buf_file.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["session_id"] == "sess_abc"
    assert record["speaker"] == "Hana"
    assert record["text"] == "hello"
    assert "ts" in record


def test_ingest_turn_auto_generates_session_id(tmp_path: Path) -> None:
    """When session_id is absent, ingest_turn generates one and returns it."""
    session_id = ingest_turn(tmp_path, {"speaker": "Nell", "text": "hi"})
    assert session_id.startswith("sess_")
    buf_file = tmp_path / "active_conversations" / f"{session_id}.jsonl"
    assert buf_file.exists()


def test_ingest_turn_multiple_turns_append(tmp_path: Path) -> None:
    """Multiple ingest_turn calls to the same session append multiple lines."""
    for text in ("first", "second", "third"):
        ingest_turn(tmp_path, {"session_id": "sess_multi", "speaker": "user", "text": text})
    buf_file = tmp_path / "active_conversations" / "sess_multi.jsonl"
    lines = [ln for ln in buf_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    texts = [json.loads(ln)["text"] for ln in lines]
    assert texts == ["first", "second", "third"]


def test_list_active_sessions_returns_existing_buffer_ids(tmp_path: Path) -> None:
    """list_active_sessions returns only session_ids that have buffer files."""
    ingest_turn(tmp_path, {"session_id": "sess_a", "speaker": "Hana", "text": "a"})
    ingest_turn(tmp_path, {"session_id": "sess_b", "speaker": "Nell", "text": "b"})
    sessions = list_active_sessions(tmp_path)
    assert set(sessions) == {"sess_a", "sess_b"}


def test_list_active_sessions_empty_when_no_dir(tmp_path: Path) -> None:
    """list_active_sessions returns [] when the active_conversations dir doesn't exist."""
    result = list_active_sessions(tmp_path / "nonexistent")
    assert result == []


def test_read_session_reads_turns_and_skips_malformed(tmp_path: Path) -> None:
    """read_session parses valid lines and skips corrupt ones."""
    buf_dir = tmp_path / "active_conversations"
    buf_dir.mkdir(parents=True)
    buf_file = buf_dir / "sess_test.jsonl"
    buf_file.write_text(
        '{"session_id": "sess_test", "speaker": "A", "text": "ok", "ts": "2026-04-25T10:00:00+00:00"}\n'
        "NOT_VALID_JSON\n"
        '{"session_id": "sess_test", "speaker": "B", "text": "also ok", "ts": "2026-04-25T10:01:00+00:00"}\n',
        encoding="utf-8",
    )
    turns = read_session(tmp_path, "sess_test")
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A"
    assert turns[1]["speaker"] == "B"


def test_session_silence_minutes_computes_from_last_turn(tmp_path: Path) -> None:
    """session_silence_minutes returns minutes since the last turn's ts."""
    past = (datetime.now(UTC) - timedelta(minutes=7)).isoformat(timespec="seconds")
    turns = [
        {"ts": (datetime.now(UTC) - timedelta(minutes=20)).isoformat(timespec="seconds")},
        {"ts": past},
    ]
    minutes = session_silence_minutes(turns)
    # Should be approximately 7 minutes — allow 1-minute tolerance for test timing.
    assert 6.0 <= minutes <= 8.0


def test_session_silence_minutes_returns_zero_for_empty(tmp_path: Path) -> None:
    """session_silence_minutes returns 0.0 when turns list is empty."""
    assert session_silence_minutes([]) == 0.0


def test_delete_session_buffer_is_idempotent(tmp_path: Path) -> None:
    """delete_session_buffer is a no-op when the file doesn't exist."""
    # Should not raise — file was never created.
    delete_session_buffer(tmp_path, "nonexistent_session")
    # Also works when file exists.
    ingest_turn(tmp_path, {"session_id": "sess_del", "speaker": "x", "text": "y"})
    delete_session_buffer(tmp_path, "sess_del")
    assert not (tmp_path / "active_conversations" / "sess_del.jsonl").exists()
    # Second call — still no error.
    delete_session_buffer(tmp_path, "sess_del")
