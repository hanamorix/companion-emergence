"""Tests for brain.utils.memory."""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory.store import Memory, MemoryStore
from brain.utils.memory import days_since_human


def _write_buffer(persona_dir: Path, rows: list[dict], *, sid: str = "sess") -> None:
    active = persona_dir / "active_conversations"
    active.mkdir(parents=True, exist_ok=True)
    (active / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _turn(speaker: str, ts: datetime) -> dict:
    return {"speaker": speaker, "text": "hi", "ts": ts.isoformat(timespec="seconds")}


def _conversation_memory(store: MemoryStore, *, created_at: datetime) -> Memory:
    """A memory shaped like the ingest pipeline's output (``metadata.source_summary = "conversation:<sid>"``)."""
    mem = Memory.create_new(
        content="Hana said hello",
        memory_type="observation",
        domain="brain",
        emotions={},
        metadata={"source_summary": "conversation:abc123"},
    )
    store.create(mem)
    store._conn.execute(  # type: ignore[attr-defined]
        "UPDATE memories SET created_at = ? WHERE id = ?", (created_at.isoformat(), mem.id)
    )
    store._conn.commit()  # type: ignore[attr-defined]
    return mem


def test_days_since_human_takes_no_store():
    """It reads the conversation files only: its parameters are the clock and the folder."""
    assert list(inspect.signature(days_since_human).parameters) == ["now", "persona_dir"]


def test_days_since_human_returns_999_when_no_signal_anywhere(tmp_path: Path):
    """No conversation folder at all: the fallback, and the call creates nothing."""
    assert days_since_human(datetime.now(UTC), persona_dir=tmp_path) == 999.0
    assert list(tmp_path.iterdir()) == []


def test_days_since_human_prefers_active_buffer_over_closed_memories(tmp_path: Path):
    """The answer is the days since the last user turn in the files. A conversation memory, old or new, and a presence
    file that claims the user was seen a minute ago move nothing."""
    now = datetime.now(UTC)
    _write_buffer(tmp_path, [_turn("user", now - timedelta(hours=5)), _turn("user", now - timedelta(hours=2))])
    (tmp_path / "presence_state.json").write_text(
        json.dumps({"last_seen_ts": (now - timedelta(minutes=1)).isoformat()}), encoding="utf-8"
    )
    store = MemoryStore(tmp_path / "memories.db")
    try:
        _conversation_memory(store, created_at=now - timedelta(days=3))
        _conversation_memory(store, created_at=now - timedelta(minutes=1))
        result = days_since_human(now, persona_dir=tmp_path)
        assert result == pytest.approx(2 / 24, abs=1e-3)
    finally:
        store.close()


def test_days_since_human_does_not_move_when_a_memorys_created_at_is_refreshed(tmp_path: Path):
    now = datetime.now(UTC)
    _write_buffer(tmp_path, [_turn("user", now - timedelta(days=2))])
    store = MemoryStore(tmp_path / "memories.db")
    try:
        mem = _conversation_memory(store, created_at=now - timedelta(days=9))
        before = days_since_human(now, persona_dir=tmp_path)
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?", (now.isoformat(), mem.id)
        )
        store._conn.commit()  # type: ignore[attr-defined]
        assert days_since_human(now, persona_dir=tmp_path) == before == pytest.approx(2.0, abs=1e-3)
    finally:
        store.close()


def test_days_since_human_ignores_assistant_turns_in_buffer(tmp_path: Path):
    """Only USER speaker turns count. Nell's own replies don't reset the
    'days since human contact' counter."""
    now = datetime.now(UTC)
    _write_buffer(tmp_path, [_turn("assistant", now - timedelta(hours=1))])
    assert days_since_human(now, persona_dir=tmp_path) == 999.0  # no user signal at all


def test_days_since_human_handles_corrupt_buffer_lines_gracefully(tmp_path: Path):
    """Malformed JSON lines or missing ts are skipped, not raised."""
    now = datetime.now(UTC)
    active = tmp_path / "active_conversations"
    active.mkdir()
    fresh_ts = (now - timedelta(hours=3)).isoformat(timespec="seconds")
    (active / "messy.jsonl").write_text(
        "not json at all\n"
        + json.dumps({"speaker": "user"})
        + "\n"  # no ts
        + json.dumps({"speaker": "user", "text": "hi", "ts": fresh_ts})
        + "\n"
    )
    assert days_since_human(now, persona_dir=tmp_path) == pytest.approx(3 / 24, abs=1e-3)


def test_days_since_human_reads_the_archive_when_no_active_file_holds_a_user_turn(tmp_path: Path):
    now = datetime.now(UTC)
    _write_buffer(tmp_path, [{"speaker": "summary", "text": "earlier", "ts": now.isoformat(timespec="seconds")}])
    archive = tmp_path / "archived_conversations"
    archive.mkdir()
    (archive / "sess.000.jsonl").write_text(
        json.dumps(_turn("user", now - timedelta(days=6))) + "\n", encoding="utf-8"
    )
    assert days_since_human(now, persona_dir=tmp_path) == pytest.approx(6.0, abs=1e-3)


def test_days_since_human_is_999_for_a_missing_folder_and_for_a_folder_where_a_session_file_should_be(tmp_path: Path):
    now = datetime.now(UTC)
    store = MemoryStore(tmp_path / "memories.db")
    try:
        _conversation_memory(store, created_at=now - timedelta(minutes=1))  # a recent memory is not a signal
        assert days_since_human(now, persona_dir=tmp_path / "missing") == 999.0
        (tmp_path / "active_conversations" / "s.jsonl").mkdir(parents=True)  # a folder where the file should be
        assert days_since_human(now, persona_dir=tmp_path) == 999.0
    finally:
        store.close()


def test_days_since_human_requires_persona_dir():
    with pytest.raises(TypeError):
        days_since_human(datetime.now(UTC))  # type: ignore[call-arg]
