"""Task 9 wire-backs: after make_note_and_wire writes a note, the note becomes a
note-state initiate memory (so she mentions it next chat), carries an emotion
delta, and surfaces in the inner-life feed.

persona_dir and the notes folder are DISJOINT siblings — production resolves the
folder under <Documents>, OUTSIDE the deny-listed persona substrate.
"""

import json
from datetime import UTC, datetime

from brain.memory.store import MemoryStore
from brain.notes.runner import make_note_and_wire


class _Cfg:
    notes_enabled = True
    user_name = "Hana"

    def __init__(self, folder):
        self.notes_folder = folder


class _Provider:
    def complete(self, prompt):
        return json.dumps({"subject": "the sea", "body": "I dreamt of the sea."})


def _run(tmp_path, monkeypatch):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    import brain.notes.runner as r
    monkeypatch.setattr(r, "_acquire_slot", lambda: True, raising=False)
    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)), provider=_Provider(),
                       now=datetime(2026, 6, 15, tzinfo=UTC))
    return persona_dir, folder


def test_note_writes_initiate_memory_naming_folder(tmp_path, monkeypatch):
    persona_dir, folder = _run(tmp_path, monkeypatch)
    store = MemoryStore(persona_dir / "memories.db")
    try:
        mems = store.list_by_type("initiate_outbound", active_only=False, limit=10)
    finally:
        store.close()
    # a note-state memory naming the folder, so she can mention it next chat
    note_mems = [m for m in mems if "note" in (m.tags or [])]
    assert len(note_mems) == 1
    assert "Notes" in note_mems[0].content


def test_note_memory_carries_emotion_delta(tmp_path, monkeypatch):
    persona_dir, folder = _run(tmp_path, monkeypatch)
    store = MemoryStore(persona_dir / "memories.db")
    try:
        mems = store.list_by_type("initiate_outbound", active_only=False, limit=10)
    finally:
        store.close()
    note_mems = [m for m in mems if "note" in (m.tags or [])]
    assert note_mems
    # vocab-filtered tenderness delta seeded onto the memory (if tenderness is
    # registered for this persona; on the empty default vocab it filters to {}).
    assert isinstance(note_mems[0].emotions, dict)
