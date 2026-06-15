from pathlib import Path

from brain.notes.config import resolve_notes_folder


def test_resolve_notes_folder_uses_documents(monkeypatch):
    import brain.notes.config as cfg
    monkeypatch.setattr(cfg, "user_documents_dir", lambda: "/Users/x/Documents")
    folder = resolve_notes_folder("Nell")
    assert folder == Path("/Users/x/Documents/Nell Notes")
