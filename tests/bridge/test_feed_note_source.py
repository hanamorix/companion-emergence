"""Feed note source: note-state initiate memories surface in the inner-life feed
via build_note_entries (mirrors build_dream_entries — MemoryStore-backed)."""

from brain.bridge.feed import TYPE_OPENER, build_note_entries
from brain.initiate.memory import write_initiate_memory
from brain.memory.store import MemoryStore


def _seed_note_memory(persona_dir, *, subject="the sea", message="I left Hana a note in /x/Notes"):
    store = MemoryStore(persona_dir / "memories.db")
    try:
        write_initiate_memory(
            store,
            audit_id="aud_note_1",
            subject=subject,
            message=message,
            state="note",
            ts="2026-06-15T00:00:00+00:00",
            user_name="Hana",
        )
    finally:
        store.close()


def test_build_note_entries_surfaces_note_memories(tmp_path):
    _seed_note_memory(tmp_path)
    entries = build_note_entries(tmp_path, limit=10)
    assert len(entries) == 1
    e = entries[0]
    assert e.type == "note"
    assert e.opener == TYPE_OPENER["note"]
    assert "Notes" in e.body


def test_build_note_entries_ignores_non_note_initiate(tmp_path):
    # a delivered outbound (not a note) must NOT show in the note source
    store = MemoryStore(tmp_path / "memories.db")
    try:
        write_initiate_memory(
            store,
            audit_id="aud_out_1",
            subject="how is your book",
            message="how is your book coming along?",
            state="delivered",
            ts="2026-06-15T00:00:00+00:00",
            user_name="Hana",
        )
    finally:
        store.close()
    entries = build_note_entries(tmp_path, limit=10)
    assert entries == []


def test_build_note_entries_empty_when_no_db(tmp_path):
    assert build_note_entries(tmp_path, limit=10) == []
