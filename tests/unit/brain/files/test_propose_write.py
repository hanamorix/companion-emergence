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


def test_second_identical_proposal_is_deduped(tmp_path, monkeypatch):
    """#93/#101: one turn must not stage the same write twice."""
    from datetime import UTC, datetime

    from brain.files.pending import list_pending

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    target = tmp_path / "home" / "Documents" / "note.md"

    first = propose_write(path=str(target), content="block", op="create",
                          persona_dir=_persona(tmp_path))
    second = propose_write(path=str(target), content="block", op="create",
                           persona_dir=_persona(tmp_path))

    assert first["status"] == "proposed"
    assert second["status"] == "already_proposed"
    assert second["deduped"] is True
    assert second["id"] == first["id"]
    # The canary: ONE card, not two.
    assert len(list_pending(_persona(tmp_path), now=datetime.now(UTC))) == 1


def test_different_content_still_creates_a_second_card(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from brain.files.pending import list_pending

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    target = tmp_path / "home" / "Documents" / "note.md"

    propose_write(path=str(target), content="block one", op="create",
                  persona_dir=_persona(tmp_path))
    second = propose_write(path=str(target), content="block two", op="create",
                           persona_dir=_persona(tmp_path))

    assert second["status"] == "proposed"
    assert "deduped" not in second
    assert len(list_pending(_persona(tmp_path), now=datetime.now(UTC))) == 2


def _notes_persona(tmp_path, monkeypatch):
    """Notes enabled on an authorised folder. Returns (persona_dir, notes_dir)."""
    import json

    h = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: h)
    notes = h / "Documents" / "Notes"
    notes.mkdir(parents=True)
    persona = _persona(tmp_path)
    persona.mkdir(parents=True)
    (persona / "persona_config.json").write_text(
        json.dumps({"notes_enabled": True, "notes_folder": str(notes)}), encoding="utf-8"
    )
    return persona, notes


def test_notes_append_skips_a_block_already_present(tmp_path, monkeypatch):
    """#105: the autonomous notes branch must not write a block the file already has.

    It commits inside the same call, so the pending-record dedupe (#93) never
    sees it — before this fix two identical calls produced 'seed\\nBLOCK\\nBLOCK'.
    """
    persona, notes = _notes_persona(tmp_path, monkeypatch)
    target = notes / "note.md"
    target.write_text("seed\n", encoding="utf-8")

    first = propose_write(path=str(target), content="- a line", op="append", persona_dir=persona)
    after_first = target.read_text(encoding="utf-8")
    second = propose_write(path=str(target), content="- a line", op="append", persona_dir=persona)

    assert first.get("status") == "written", first
    assert second.get("status") == "already_present", second
    assert second.get("deduped") is True
    assert target.read_text(encoding="utf-8") == after_first, "the second call must write nothing"
    assert after_first.count("- a line") == 1


def test_notes_append_writes_short_content_that_is_only_an_incidental_substring(
    tmp_path, monkeypatch
):
    """The containment check is line-anchored, not a substring test (#105).

    "done" sitting inside "well done, truly" must NOT count as already present —
    silently dropping a legitimate short note is the same class of harm as the
    doubling this guards against.
    """
    persona, notes = _notes_persona(tmp_path, monkeypatch)
    target = notes / "note.md"
    target.write_text("well done, truly\n", encoding="utf-8")

    out = propose_write(path=str(target), content="done", op="append", persona_dir=persona)

    assert out.get("status") == "written", out
    assert target.read_text(encoding="utf-8") == "well done, truly\ndone"


def test_consent_path_still_commits_content_already_in_the_file(tmp_path, monkeypatch):
    """CANARY for #105's placement decision — do NOT move the check into commit_write.

    commit_write has two callers: the autonomous notes branch, and the
    /persona/writes/{rid}/approve consent gate. A containment check inside
    commit_write would silently refuse a write the USER HAD JUST APPROVED.
    This asserts the consented path is untouched; it fails if the check ever
    slides down a layer.
    """
    from datetime import UTC, datetime

    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.store import MemoryStore

    h = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: h)
    (h / "Documents").mkdir(parents=True)
    persona = _persona(tmp_path)
    target = h / "Documents" / "doc.md"
    target.write_text("- a line\n", encoding="utf-8")  # content ALREADY present

    rid = pending.create(persona, op="append", resolved_path=str(target.resolve()),
                         content="- a line", now=datetime.now(UTC))
    store = MemoryStore(persona / "memories.db")
    try:
        res = commit_write(persona, rid, store=store)
    finally:
        store.close()

    assert res.get("ok") is True, res
    assert target.read_text(encoding="utf-8").count("- a line") == 2, (
        "an approved write must commit even when the content is already present"
    )
