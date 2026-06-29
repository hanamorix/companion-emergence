"""Completeness contract: every active_conversations buffer reader must skip
speaker=="summary" rows (T2 deliverable).

A compaction `summary` row carries a compaction-time ts, NOT a live
conversational-turn ts. Any reader that treats it as a real turn produces
silent corruption:

  Reader                                     Corruption if not guarded
  ──────────────────────────────────────────────────────────────────────
  session_hours._entry_timestamps            old ts inflates session_hours -> energy drain
  felt_time.count_chat_turns_since           inflated turn count -> skewed rolling baseline
  ingest/extract.format_transcript           summary re-ingested as memory
  ingest/pipeline.extract_session_snapshot   cursor advanced past un-extracted turns
  initiate/user_pattern._compute_silence_days  compaction ts = "recent user msg"
  initiate/user_pattern._compute_likely_active compaction ts skews hour distribution

All 6 readers are covered here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

_NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)


def _write_buffer(persona_dir: Path, session_id: str, rows: list) -> Path:
    ac = persona_dir / "active_conversations"
    ac.mkdir(parents=True, exist_ok=True)
    buf = ac / f"{session_id}.jsonl"
    with buf.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return buf


def _summary_row(ts: datetime) -> dict:
    return {"speaker": "summary", "text": "Compacted summary of earlier conversation.", "ts": ts.isoformat()}


def _user_row(ts: datetime, text: str = "hello") -> dict:
    return {"speaker": "user", "text": text, "ts": ts.isoformat()}


def _assistant_row(ts: datetime, text: str = "hi there") -> dict:
    return {"speaker": "assistant", "text": text, "ts": ts.isoformat()}


# 1. session_hours

def test_session_hours_excludes_summary_ts(tmp_path):
    from brain.body.session_hours import compute_active_session_hours
    persona_dir = tmp_path / "nell"
    _write_buffer(persona_dir, "s1", [
        _summary_row(_NOW - timedelta(hours=8)),
        _user_row(_NOW - timedelta(minutes=2)),
        _assistant_row(_NOW - timedelta(seconds=30)),
    ])
    result = compute_active_session_hours(persona_dir, now=_NOW)
    assert result < 0.1, f"session_hours should reflect only the 2-minute session, got {result:.4f}h"


def test_session_hours_only_summary_row_returns_zero(tmp_path):
    from brain.body.session_hours import compute_active_session_hours
    persona_dir = tmp_path / "nell"
    _write_buffer(persona_dir, "s1", [_summary_row(_NOW - timedelta(seconds=10))])
    result = compute_active_session_hours(persona_dir, now=_NOW)
    assert result == 0.0, f"buffer with only a summary should give 0.0, got {result}"


# 2. felt_time count

def test_count_chat_turns_since_excludes_summary(tmp_path):
    from brain.felt_time.chat_log import count_chat_turns_since
    from brain.ingest.buffer import ingest_turn
    persona_dir = tmp_path / "nell"
    session_id = "s1"
    cutoff = (_NOW - timedelta(hours=1)).isoformat()
    for offset_s in (30 * 60, 10):
        ts = (_NOW - timedelta(seconds=offset_s)).isoformat()
        ingest_turn(persona_dir, {"speaker": "user", "text": "turn", "session_id": session_id, "ts": ts})
    ac = persona_dir / "active_conversations"
    with (ac / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(_summary_row(_NOW - timedelta(minutes=5))) + "\n")
    total = count_chat_turns_since(persona_dir, cutoff)
    assert total == 2, f"expected 2 real turns (not the summary), got {total}"


# 3. format_transcript

def test_format_transcript_excludes_summary_row(tmp_path):
    from brain.ingest.extract import format_transcript
    turns = [
        _summary_row(_NOW - timedelta(hours=3)),
        _user_row(_NOW - timedelta(minutes=5), "first real turn"),
        _assistant_row(_NOW - timedelta(minutes=4), "real response"),
    ]
    transcript = format_transcript(turns)
    assert "Compacted summary" not in transcript
    assert "first real turn" in transcript
    assert "real response" in transcript


def test_format_transcript_only_summary_is_empty(tmp_path):
    from brain.ingest.extract import format_transcript
    result = format_transcript([_summary_row(_NOW)])
    assert result.strip() == "", f"expected empty transcript, got: {result!r}"


# 4. extract_session_snapshot

def test_extract_session_snapshot_excludes_summary_from_transcript(tmp_path):
    from unittest.mock import MagicMock, patch

    from brain.ingest.buffer import ingest_turn, write_cursor
    from brain.ingest.extract import ExtractionOutcome
    from brain.ingest.pipeline import extract_session_snapshot
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    persona_dir = tmp_path / "nell"
    session_id = "s1"
    write_cursor(persona_dir, session_id, "2000-01-01T00:00:00+00:00")
    ingest_turn(persona_dir, {
        "speaker": "user", "text": "real message", "session_id": session_id,
        "ts": (_NOW - timedelta(minutes=2)).isoformat(),
    })
    ac = persona_dir / "active_conversations"
    with (ac / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(_summary_row(_NOW - timedelta(hours=6))) + "\n")
    recorded = []
    def fake_extract(transcript, **kwargs):
        recorded.append(transcript)
        return ExtractionOutcome(items=[])
    with patch("brain.ingest.pipeline.extract_items_with_status", side_effect=fake_extract):
        extract_session_snapshot(
            persona_dir, session_id,
            store=MemoryStore(persona_dir / "memories.db"),
            hebbian=HebbianMatrix(persona_dir / "hebbian.db"),
            provider=MagicMock(),
        )
    assert recorded, "extractor should have been called"
    for t in recorded:
        assert "Compacted summary" not in t, f"summary text must not reach extractor; got: {t!r}"
    assert any("real message" in t for t in recorded)


def test_extract_session_snapshot_cursor_unaffected_by_summary_ts(tmp_path):
    from unittest.mock import MagicMock, patch

    from brain.ingest.buffer import ingest_turn, read_cursor, write_cursor
    from brain.ingest.extract import ExtractionOutcome
    from brain.ingest.pipeline import extract_session_snapshot
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    persona_dir = tmp_path / "nell"
    session_id = "s1"
    write_cursor(persona_dir, session_id, "2000-01-01T00:00:00+00:00")
    real_ts = (_NOW - timedelta(minutes=2)).isoformat()
    ingest_turn(persona_dir, {"speaker": "user", "text": "hello", "session_id": session_id, "ts": real_ts})
    ac = persona_dir / "active_conversations"
    with (ac / f"{session_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"speaker": "summary", "text": "old", "ts": (_NOW - timedelta(hours=12)).isoformat()}) + "\n")
    with patch("brain.ingest.pipeline.extract_items_with_status", return_value=ExtractionOutcome(items=[])):
        extract_session_snapshot(
            persona_dir, session_id,
            store=MemoryStore(persona_dir / "memories.db"),
            hebbian=HebbianMatrix(persona_dir / "hebbian.db"),
            provider=MagicMock(),
        )
    cursor_after = read_cursor(persona_dir, session_id)
    assert cursor_after is not None, "cursor should have advanced"
    assert cursor_after >= real_ts, f"cursor should be >= real turn ts ({real_ts}), got {cursor_after}"


# 5. user_pattern._compute_silence_days

def test_silence_days_excludes_summary_row(tmp_path):
    from brain.initiate.user_pattern import _compute_silence_days
    persona_dir = tmp_path / "nell"
    _write_buffer(persona_dir, "s1", [_summary_row(_NOW - timedelta(seconds=10))])
    result = _compute_silence_days(persona_dir, _now=_NOW)
    assert result == 0.0, f"summary row should not count as user turn; expected 0.0, got {result}"


def test_silence_days_real_turn_still_counted(tmp_path):
    from brain.initiate.user_pattern import _compute_silence_days
    persona_dir = tmp_path / "nell"
    user_ts = _NOW - timedelta(hours=2)
    _write_buffer(persona_dir, "s1", [
        _summary_row(_NOW - timedelta(hours=10)),
        _user_row(user_ts, "a real message"),
    ])
    result = _compute_silence_days(persona_dir, _now=_NOW)
    assert 0.07 <= result <= 0.09, f"expected ~0.083 days (2h), got {result}"


# 6. user_pattern._compute_likely_active

def test_likely_active_summary_only_falls_back_to_permissive(tmp_path):
    from brain.initiate.user_pattern import _compute_likely_active
    persona_dir = tmp_path / "nell"
    _write_buffer(persona_dir, "s1", [_summary_row(_NOW - timedelta(minutes=5))])
    # No real turns -> total < _SCHEDULE_MIN_TURNS (50) -> permissive True
    result = _compute_likely_active(persona_dir, _now=_NOW)
    assert result is True, f"no real turns -> permissive True, got {result}"
