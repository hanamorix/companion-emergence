import json
from datetime import UTC, datetime

from brain.notes.runner import make_note_and_wire


class _Cfg:
    notes_enabled = True
    user_name = "Hana"

    def __init__(self, folder):
        self.notes_folder = folder


class _Provider:
    def complete(self, prompt):
        return json.dumps({"subject": "the sea", "body": "I dreamt of the sea and thought of you."})


def test_make_note_and_wire_writes_a_file(tmp_path, monkeypatch):
    # persona_dir and the notes folder are disjoint siblings — production resolves
    # the folder under <Documents>, OUTSIDE the persona substrate (deny-listed).
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Notes"
    folder.mkdir()
    # avoid the real throttle/budget blocking the test
    import brain.notes.runner as r
    monkeypatch.setattr(r, "_acquire_slot", lambda: True, raising=False)
    make_note_and_wire(persona_dir=persona_dir, config=_Cfg(str(folder)), provider=_Provider(),
                       now=datetime(2026, 6, 15, tzinfo=UTC))
    notes = list(folder.glob("*.md"))
    assert len(notes) == 1
    assert "the sea" in notes[0].read_text().lower()
