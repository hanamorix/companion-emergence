# tests/unit/brain/files/test_commit.py
from datetime import UTC, datetime
from pathlib import Path

from brain.files import pending
from brain.files.commit import commit_write, decline_write
from brain.memory.store import MemoryStore


# NOTE (deviation from plan): mirrors the Task 3 test fix. The guard's
# persona-substrate deny rule (Task 1) refuses any path inside persona_dir.
# The plan's Task 4 tests passed persona_dir=tmp_path while home=tmp_path/"home"
# — nesting the user's home *inside* her substrate, so every target is denied
# and the tests can't pass. In production persona_dir is
# $KINDLED_HOME/personas/<name>/, never an ancestor of ~. We make persona_dir a
# sibling of home via a helper; all assertions are otherwise verbatim.
def _persona(tmp_path) -> Path:
    return tmp_path / "persona"


def _store(persona_dir):
    return MemoryStore(persona_dir / "memories.db")


def _promote_pending(persona_dir):
    """Drain the consolidation gate with a promote-all classifier.

    file_write is a GATED type (Root-2 stopgap): its wire-back memory enqueues to the
    pending queue and only reaches memories.db after a consolidation drain. Tests that
    assert the end-state DB row run this to promote the candidate.
    """
    from brain.engines.consolidation import Decision, run_consolidation

    s = MemoryStore(persona_dir / "memories.db")
    try:
        run_consolidation(
            s, persona_dir=persona_dir, classifier=lambda _c, _ctx: Decision("new")
        )
    finally:
        s.close()


def test_commit_create_writes_file_and_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    persona = _persona(tmp_path)
    target = tmp_path / "home" / "out" / "n.md"
    rid = pending.create(
        persona,
        op="create",
        resolved_path=str(target.resolve()),
        content="body",
        now=datetime.now(UTC),
    )
    store = _store(persona)
    res = commit_write(persona, rid, store=store)
    assert res["ok"]
    assert target.read_text() == "body"
    assert pending.get(persona, rid)["status"] == "committed"
    # file_write is GATED: the wire-back memory enqueues, not written straight to db.
    assert not store.list_by_type("file_write", active_only=True, limit=5)
    store.close()
    _promote_pending(persona)  # drain → promote the candidate
    store2 = _store(persona)
    assert store2.list_by_type("file_write", active_only=True, limit=5)  # wire-back memory
    store2.close()


def test_commit_append_preserves_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    persona = _persona(tmp_path)
    target = tmp_path / "home" / "log.md"
    target.parent.mkdir(parents=True)
    target.write_text("old\n")
    rid = pending.create(
        persona,
        op="append",
        resolved_path=str(target.resolve()),
        content="new\n",
        now=datetime.now(UTC),
    )
    store = _store(persona)
    commit_write(persona, rid, store=store)
    assert target.read_text() == "old\nnew\n"
    store.close()


def test_commit_revalidates_guard(tmp_path, monkeypatch):
    """A pending create whose path now EXISTS is refused at commit (TOCTOU)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    persona = _persona(tmp_path)
    target = tmp_path / "home" / "x.md"
    target.parent.mkdir(parents=True)
    rid = pending.create(
        persona,
        op="create",
        resolved_path=str(target.resolve()),
        content="c",
        now=datetime.now(UTC),
    )
    target.write_text("appeared")  # the file showed up between propose and approve
    store = _store(persona)
    res = commit_write(persona, rid, store=store)
    assert not res["ok"]
    assert target.read_text() == "appeared"  # not clobbered
    store.close()


def test_decline_writes_nothing_records_memory(tmp_path):
    persona = _persona(tmp_path)
    rid = pending.create(
        persona,
        op="create",
        resolved_path="/tmp/none",
        content="secret",
        now=datetime.now(UTC),
    )
    store = MemoryStore(persona / "memories.db")
    decline_write(persona, rid, store=store)
    assert pending.get(persona, rid)["status"] == "declined"
    # file_write is GATED: enqueues, promoted on drain.
    assert not store.list_by_type("file_write", active_only=True, limit=5)
    store.close()
    _promote_pending(persona)
    store2 = MemoryStore(persona / "memories.db")
    mems = store2.list_by_type("file_write", active_only=True, limit=5)
    assert mems and "secret" not in mems[0].content  # declined content NOT stored
    store2.close()


def test_committed_write_surfaces_in_feed(tmp_path, monkeypatch):
    """A committed write wires a file_write memory that the feed source reads."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    persona = _persona(tmp_path)
    target = tmp_path / "home" / "feedme.md"
    target.parent.mkdir(parents=True)
    rid = pending.create(
        persona,
        op="create",
        resolved_path=str(target.resolve()),
        content="diary",
        now=datetime.now(UTC),
    )
    store = _store(persona)
    commit_write(persona, rid, store=store)
    store.close()
    # BLAST RADIUS (accepted, see changes/gate-leak-coverage/decisions.md): the Feed source
    # build_file_write_entries reads memories.db, so a committed file-write is NOT visible in
    # the Feed until the consolidation gate promotes the (now gated) file_write candidate —
    # a user-facing latency that mirrors the self-model latency (criteria A3). Drain to promote.
    _promote_pending(persona)

    from brain.bridge.feed import build_file_write_entries

    entries = build_file_write_entries(persona, limit=5)
    assert entries
    assert entries[0].type == "file_write"
    assert entries[0].opener == "I wrote to a file —"
    assert str(target.resolve()) in entries[0].body
