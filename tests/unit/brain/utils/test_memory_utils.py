"""Tests for brain.utils.memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.memory.store import Memory, MemoryStore
from brain.utils.memory import days_since_human


def test_days_since_human_returns_999_when_no_signal_anywhere(tmp_path: Path):
    """Empty store, no persona dir → 999.0 fallback."""
    store = MemoryStore(":memory:")
    try:
        result = days_since_human(store, datetime.now(UTC))
        assert result == 999.0
    finally:
        store.close()


def test_days_since_human_uses_conversation_extraction_metadata(tmp_path: Path):
    """Memories tagged via the ingest pipeline carry
    metadata.source_summary='conversation:<sid>'. days_since_human picks
    them up — this is the closed-session signal that lives forever."""
    store = MemoryStore(":memory:")
    try:
        # Realistic shape: memory extracted from a chat buffer carries
        # source_summary in its metadata. Type is one of the extractor's
        # labels (observation/feeling/etc), NOT 'conversation'.
        mem = Memory.create_new(
            content="Hana said hello",
            memory_type="observation",
            domain="brain",
            emotions={},
            metadata={"source_summary": "conversation:abc123"},
        )
        store.create(mem)
        # Backdate 48h
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(hours=48)).isoformat(), mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]
        result = days_since_human(store, datetime.now(UTC))
        assert 1.9 < result < 2.1  # ~2 days
    finally:
        store.close()


def test_days_since_human_prefers_active_buffer_over_closed_memories(tmp_path: Path):
    """Active session JSONL buffer is fresher than any closed-session
    memory. days_since_human takes the most recent of the two sources.
    This was the bug Hana hit — chatting yesterday should drop
    days_since_contact under 1, but the function only saw the (older)
    closed-session memories until the active buffer was checked."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    active = persona_dir / "active_conversations"
    active.mkdir()

    store = MemoryStore(":memory:")
    try:
        # Old closed-session memory (3 days ago)
        old_mem = Memory.create_new(
            content="x",
            memory_type="observation",
            domain="brain",
            emotions={},
            metadata={"source_summary": "conversation:old_sid"},
        )
        store.create(old_mem)
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=3)).isoformat(), old_mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]

        # Fresh user-turn in an active buffer (2 hours ago)
        fresh_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat(timespec="seconds")
        (active / "fresh_session.jsonl").write_text(
            json.dumps({"speaker": "user", "text": "hi", "ts": fresh_ts}) + "\n"
        )

        result = days_since_human(store, datetime.now(UTC), persona_dir=persona_dir)
        assert 0.05 < result < 0.15  # ~2 hours = ~0.083 days
    finally:
        store.close()


def test_days_since_human_ignores_assistant_turns_in_buffer(tmp_path: Path):
    """Only USER speaker turns count. Nell's own replies don't reset the
    'days since human contact' counter."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    active = persona_dir / "active_conversations"
    active.mkdir()

    # Recent assistant turn, no user turns
    fresh_ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds")
    (active / "sess.jsonl").write_text(
        json.dumps({"speaker": "assistant", "text": "hi back", "ts": fresh_ts}) + "\n"
    )

    store = MemoryStore(":memory:")
    try:
        result = days_since_human(store, datetime.now(UTC), persona_dir=persona_dir)
        assert result == 999.0  # no user signal at all
    finally:
        store.close()


def test_days_since_human_handles_corrupt_buffer_lines_gracefully(tmp_path: Path):
    """Malformed JSON lines or missing ts are skipped, not raised."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    active = persona_dir / "active_conversations"
    active.mkdir()

    fresh_ts = (datetime.now(UTC) - timedelta(hours=3)).isoformat(timespec="seconds")
    (active / "messy.jsonl").write_text(
        "not json at all\n"
        + json.dumps({"speaker": "user"}) + "\n"  # no ts
        + json.dumps({"speaker": "user", "text": "hi", "ts": fresh_ts}) + "\n"
    )

    store = MemoryStore(":memory:")
    try:
        result = days_since_human(store, datetime.now(UTC), persona_dir=persona_dir)
        assert 0.1 < result < 0.2  # ~3 hours = ~0.125 days
    finally:
        store.close()


def test_days_since_human_legacy_call_without_persona_dir_still_works():
    """Backward compat: callers that don't pass persona_dir still get a
    valid answer from the closed-session memory scan."""
    store = MemoryStore(":memory:")
    try:
        mem = Memory.create_new(
            content="x",
            memory_type="observation",
            domain="brain",
            emotions={},
            metadata={"source_summary": "conversation:legacy"},
        )
        store.create(mem)
        store._conn.execute(  # type: ignore[attr-defined]
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(hours=12)).isoformat(), mem.id),
        )
        store._conn.commit()  # type: ignore[attr-defined]
        # No persona_dir kwarg
        result = days_since_human(store, datetime.now(UTC))
        assert 0.4 < result < 0.6  # ~12 hours = 0.5 days
    finally:
        store.close()
