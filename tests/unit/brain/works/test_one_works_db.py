"""#606: one works database per persona.

The maker, the Feed and propose_write opened ``<persona>/works.db``; save_work, list_works, read_work, search_works and
the integrity walker opened ``<persona>/data/works.db``. So the works tools never saw a single making, and the walker
checked a stale file. ``brain.works.location.works_db_path`` is now the one resolver, and a persona that still has the
old ``data/works.db`` has its rows merged in, by id, the first time the path is resolved.

These tests build the two paths by hand on purpose: they stand in for a persona written by an older brain.
"""

from __future__ import annotations

import gc
import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

import brain.works.store as store_module
from brain.maker.charge import MakerCharge, save_charge
from brain.maker.making_runner import make_and_wire
from brain.memory.store import MemoryStore
from brain.tools.impls.list_works import list_works
from brain.tools.impls.read_work import read_work
from brain.tools.impls.search_works import search_works
from brain.works import Work
from brain.works.storage import write_markdown
from brain.works.store import WorksStore


def _seed(db: Path, persona_dir: Path, wid: str, title: str, content: str, *, origin: str = "tool") -> None:
    work = Work(
        id=wid,
        title=title,
        type="story",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
        session_id=None,
        word_count=len(content.split()),
        summary=None,
        origin=origin,
    )
    assert _store(db).insert(work, content=content)
    write_markdown(persona_dir, work, content=content)


def _store(db: Path) -> WorksStore:
    """A WorksStore on ``db`` that holds no connection once a call returns.

    WorksStore's first open of a path is never closed: a dropped sqlite3 connection lives until the cycle collector runs,
    and Windows refuses to rename a file a live handle holds. So the schema is created here through a connection that is
    closed, and the path is marked initialised, so the store skips that open. Its own insert closes what it opens.
    """
    key = str(db.resolve())
    if key not in WorksStore._INITIALISED_PATHS:
        db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(db))) as conn:
            conn.executescript(store_module._SCHEMA_SQL)
        WorksStore._INITIALISED_PATHS.add(key)
    return WorksStore(db)


def _release_handles() -> None:
    """The maker's own path opens works.db through stores this test can't reach, so collect them before the file is
    damaged on purpose. One full collection; it costs CPU, so only the test that needs it calls this."""
    gc.collect()


def test_list_works_sees_both_files_once_after_start_up(tmp_path):
    _seed(tmp_path / "works.db", tmp_path, "aaaaaaaaaaaa", "Made by the maker", "harbour light", origin="maker")
    _seed(tmp_path / "data" / "works.db", tmp_path, "bbbbbbbbbbbb", "Imported", "an old letter")
    # The same work in both files must come out once.
    _seed(tmp_path / "works.db", tmp_path, "cccccccccccc", "In both", "twice told")
    _seed(tmp_path / "data" / "works.db", tmp_path, "cccccccccccc", "In both", "twice told")

    ids = [w["id"] for w in list_works(persona_dir=tmp_path, limit=50)]

    assert sorted(ids) == ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"]
    assert not (tmp_path / "data" / "works.db").exists()
    assert (tmp_path / "data" / "works.db.migrated").is_file()
    # The old file was closed when it was renamed: its log was folded in, so no -wal travelled with it. (Windows refuses
    # to rename a file a handle still holds, which is how a leaked handle showed up there.)
    assert not (tmp_path / "data" / "works.db.migrated-wal").exists()


def test_merged_works_stay_searchable_and_readable(tmp_path):
    _seed(tmp_path / "data" / "works.db", tmp_path, "bbbbbbbbbbbb", "Imported", "an old letter about lanterns")

    hits = [w["id"] for w in search_works("lanterns", persona_dir=tmp_path)]
    read = read_work("bbbbbbbbbbbb", persona_dir=tmp_path)

    assert hits == ["bbbbbbbbbbbb"]
    assert read["content"] == "an old letter about lanterns"
    assert not (tmp_path / "data" / "works.db").exists()


class _FakeProvider:
    def complete(self, prompt):
        return json.dumps(
            {"type": "vignette", "title": "Dusk", "content": "Light fell slow over the harbour.",
             "disposition": "eventual_share"}
        )


def _make_one(persona_dir: Path) -> None:
    """One making, through the maker's own discharge path."""
    save_charge(persona_dir, MakerCharge(99.0, "2026-06-14T00:00:00+00:00", None, 0))
    store = MemoryStore(persona_dir / "memories.db")
    try:
        make_and_wire(persona_dir=persona_dir, store=store, provider=_FakeProvider(), now=None)
    finally:
        store.close()


def test_a_making_is_visible_to_the_works_tools(tmp_path):
    _make_one(tmp_path)

    listed = list_works(persona_dir=tmp_path)
    assert [w["title"] for w in listed] == ["Dusk"]
    wid = listed[0]["id"]
    assert read_work(wid, persona_dir=tmp_path)["content"] == "Light fell slow over the harbour."
    assert [w["id"] for w in search_works("harbour", persona_dir=tmp_path)] == [wid]


def test_a_saved_work_lands_in_the_file_the_maker_uses(tmp_path):
    from brain.tools.impls.save_work import save_work

    out = save_work(title="Note", type="idea", content="a small idea", persona_dir=tmp_path)

    assert WorksStore(tmp_path / "works.db").get(out["id"]) is not None
    assert not (tmp_path / "data" / "works.db").exists()


def test_the_walker_checks_the_file_the_store_uses(tmp_path):
    from brain.health.walker import walk_persona

    _make_one(tmp_path)
    _release_handles()
    for sidecar in ("works.db-wal", "works.db-shm"):  # else SQLite can serve the pages from the log
        (tmp_path / sidecar).unlink(missing_ok=True)
    (tmp_path / "works.db").write_bytes(b"this is not a sqlite database" * 64)

    files = {a.file for a in walk_persona(tmp_path) if a.kind == "sqlite_integrity_fail"}

    assert "works.db" in files


def test_an_unreadable_old_file_is_left_in_place(tmp_path):
    _seed(tmp_path / "works.db", tmp_path, "aaaaaaaaaaaa", "Live", "harbour light", origin="maker")
    legacy = tmp_path / "data" / "works.db"
    legacy.write_bytes(b"not a database" * 64)  # data/ exists: _seed wrote the markdown there

    with pytest.warns(RuntimeWarning, match="could not merge"):
        ids = [w["id"] for w in list_works(persona_dir=tmp_path)]
    ids_again = [w["id"] for w in list_works(persona_dir=tmp_path)]  # warned once per process, not per call

    assert ids == ids_again == ["aaaaaaaaaaaa"]
    assert legacy.read_bytes() == b"not a database" * 64
    assert not (tmp_path / "data" / "works.db.migrated").exists()


def test_a_lock_that_cannot_be_taken_leaves_the_old_file_and_returns_the_live_path(tmp_path, monkeypatch):
    """A read-only persona folder can't hold the lock sidecar: the merge warns and is skipped, nothing raises."""
    import contextlib

    import brain.works.location as location

    @contextlib.contextmanager
    def refusing_lock(path):
        raise PermissionError(f"cannot create {path}.lock")
        yield  # pragma: no cover

    monkeypatch.setattr(location, "file_lock", refusing_lock)
    _seed(tmp_path / "data" / "works.db", tmp_path, "bbbbbbbbbbbb", "Old", "first import")
    old = (tmp_path / "data" / "works.db").read_bytes()

    with pytest.warns(RuntimeWarning, match="could not merge"):
        path = location.works_db_path(tmp_path)

    assert path == tmp_path / "works.db"
    assert (tmp_path / "data" / "works.db").read_bytes() == old
    assert not (tmp_path / "data" / "works.db.migrated").exists()


def test_a_second_old_file_keeps_the_first_backup(tmp_path):
    from brain.works.location import works_db_path

    _seed(tmp_path / "data" / "works.db", tmp_path, "bbbbbbbbbbbb", "First", "first import")
    works_db_path(tmp_path)
    first_backup = (tmp_path / "data" / "works.db.migrated").read_bytes()
    # An older brain, after a rollback, recreates the old file and saves into it.
    WorksStore._INITIALISED_PATHS.discard(str((tmp_path / "data" / "works.db").resolve()))
    _seed(tmp_path / "data" / "works.db", tmp_path, "dddddddddddd", "Second", "made while rolled back")

    works_db_path(tmp_path)

    assert (tmp_path / "data" / "works.db.migrated").read_bytes() == first_backup
    assert (tmp_path / "data" / "works.db.migrated.1").is_file()
    assert sorted(w["id"] for w in list_works(persona_dir=tmp_path)) == ["bbbbbbbbbbbb", "dddddddddddd"]


def test_no_code_builds_a_works_db_path_by_hand():
    """Every opener goes through ``works_db_path``; a hand-built path is how the two files came about."""
    repo = Path(__file__).resolve().parents[4]
    allowed = {"brain/works/location.py"}
    literal = re.compile(r"""["']works\.db["']""")
    offenders = sorted(
        rel
        for path in (repo / "brain").rglob("*.py")
        if (rel := path.relative_to(repo).as_posix()) not in allowed
        and literal.search(path.read_text(encoding="utf-8"))
    )
    assert offenders == []
