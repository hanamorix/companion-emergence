from datetime import UTC, datetime
from pathlib import Path

from brain.notes.compose import Note
from brain.notes.write import commit_note


def test_commit_note_writes_inside_folder(tmp_path):
    # persona_dir and the notes folder are siblings — production resolves the
    # notes folder under <Documents>, OUTSIDE the persona substrate (which the
    # guard's deny-list protects). Keep them disjoint so check_write_target clears.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Nell Notes"
    folder.mkdir()
    n = Note(subject="the sea", body="I dreamt of the sea.")
    path = commit_note(persona_dir, folder, n, now=datetime(2026, 6, 15, tzinfo=UTC))
    assert path is not None
    assert Path(path).parent == folder
    assert "2026-06-15" in Path(path).name
    assert "I dreamt of the sea." in Path(path).read_text()


def test_commit_note_create_only_no_clobber(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Nell Notes"
    folder.mkdir()
    n = Note(subject="dup", body="first")
    p1 = commit_note(persona_dir, folder, n, now=datetime(2026, 6, 15, tzinfo=UTC))
    p2 = commit_note(persona_dir, folder, Note(subject="dup", body="second"), now=datetime(2026, 6, 15, tzinfo=UTC))
    assert p1 != p2  # collision-suffixed, never overwrites
    assert Path(p1).read_text().endswith("first") or "first" in Path(p1).read_text()


def test_commit_note_refuses_outside_folder(tmp_path):
    # Disjoint dirs so the deny-list clears — the ONLY thing forcing None here is
    # the monkeypatched is_within_authorized, proving the allowlist is consulted.
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    folder = tmp_path / "Nell Notes"
    folder.mkdir()
    import brain.notes.write as w
    orig = w.is_within_authorized
    w.is_within_authorized = lambda *a, **k: False
    try:
        path = commit_note(persona_dir, folder, Note(subject="x", body="y"), now=datetime(2026, 6, 15, tzinfo=UTC))
    finally:
        w.is_within_authorized = orig
    assert path is None
    assert list(folder.iterdir()) == []  # nothing written
