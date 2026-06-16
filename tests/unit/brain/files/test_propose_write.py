# tests/unit/brain/files/test_propose_write.py
from pathlib import Path

from brain.tools.impls.propose_write import propose_write


# NOTE (deviation from plan): the guard's persona-substrate deny rule (Task 1)
# refuses any path inside persona_dir. The plan's Task 3 tests passed
# persona_dir=tmp_path while home=tmp_path/"home" — nesting the user's home
# *inside* her substrate, so every ~ target is denied and the tests can't pass.
# In production persona_dir is $KINDLED_HOME/personas/<name>/, never an ancestor
# of ~. We mirror Task 1's own fixture (persona_dir a sibling of home) via a
# helper; all assertions are otherwise verbatim.
def _persona(tmp_path) -> Path:
    return tmp_path / "persona"


def test_propose_into_authorised_notes_folder_auto_commits(tmp_path, monkeypatch):
    """A tool write whose target is inside the user-authorised notes folder is
    committed directly (no confirmation card) — the user enabled that folder."""
    import json
    from datetime import UTC, datetime

    from brain.files.pending import list_pending

    h = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: h)
    notes = h / "Documents" / "Phoebe Notes"
    notes.mkdir(parents=True)
    persona = _persona(tmp_path)
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(
        json.dumps({"notes_enabled": True, "notes_folder": str(notes)}), encoding="utf-8"
    )

    target = notes / "today.md"
    out = propose_write(path=str(target), content="a thought", op="create", persona_dir=persona)

    assert out.get("status") == "written", out
    assert target.read_text(encoding="utf-8") == "a thought"  # written now, not deferred
    assert list_pending(persona, now=datetime.now(UTC)) == []  # no confirmation card queued


def test_propose_outside_authorised_folder_still_pending(tmp_path, monkeypatch):
    """Notes enabled, but a target OUTSIDE the authorised folder still goes
    through the confirmation card — the auto-commit is scoped to the folder."""
    import json

    h = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: h)
    notes = h / "Documents" / "Phoebe Notes"
    notes.mkdir(parents=True)
    persona = _persona(tmp_path)
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(
        json.dumps({"notes_enabled": True, "notes_folder": str(notes)}), encoding="utf-8"
    )

    target = h / "Documents" / "elsewhere.md"  # NOT inside the notes folder
    out = propose_write(path=str(target), content="x", op="create", persona_dir=persona)

    assert out["status"] == "proposed"
    assert not target.exists()


def test_propose_creates_pending_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    target = tmp_path / "home" / "Documents" / "note.md"
    out = propose_write(path=str(target), content="hello", op="create",
                        persona_dir=_persona(tmp_path))
    assert out["status"] == "proposed"
    assert not target.exists()  # NOTHING written at propose time
    from brain.files.pending import get
    assert get(_persona(tmp_path), out["id"])["content"] == "hello"


def test_propose_denied_path_no_pending_row(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: h)
    (h / ".ssh").mkdir(parents=True, exist_ok=True)
    (h / ".ssh" / "authorized_keys").write_text("k")
    out = propose_write(path=str(h / ".ssh" / "authorized_keys"), content="evil",
                        op="append", persona_dir=_persona(tmp_path))
    assert "error" in out
    from datetime import UTC, datetime

    from brain.files.pending import list_pending
    assert list_pending(_persona(tmp_path), now=datetime.now(UTC)) == []  # nothing queued


def test_propose_over_queue_cap_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    for i in range(10):
        propose_write(path=str(tmp_path / "home" / f"f{i}.md"), content="x", op="create",
                      persona_dir=_persona(tmp_path))
    out = propose_write(path=str(tmp_path / "home" / "f11.md"), content="x", op="create",
                        persona_dir=_persona(tmp_path))
    assert "error" in out and "awaiting" in out["error"].lower()
