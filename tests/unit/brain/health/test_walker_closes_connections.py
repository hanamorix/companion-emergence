"""#779: `walk_persona` closes every SQLite connection it opens.

`with sqlite3.connect(...) as conn:` only commits or rolls back; it never closes. A connection left open holds its file
until the cycle collector runs, and Windows refuses to rename, replace or delete a held file (`WinError 32`), so a walk
that has just reported the works database as corrupt kept that very file open.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.works import Work
from brain.works.store import WorksStore

# Main's walker checks data/works.db and #768's checks works.db; seeding both keeps this test true either side of it.
_WORKS_PATHS = ("works.db", "data/works.db")


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:  # "Cannot operate on a closed database"
        return True
    except sqlite3.DatabaseError:  # open, on a file that is not a database
        return False
    return False


def _persona(tmp_path: Path, *, corrupt_works: bool) -> Path:
    persona = tmp_path / "persona"
    (persona / "data").mkdir(parents=True)
    MemoryStore(db_path=persona / "memories.db").close()
    HebbianMatrix(db_path=persona / "hebbian.db").close()
    for name in _WORKS_PATHS:
        db = persona / name
        if corrupt_works:
            db.write_bytes(b"this is not a sqlite database" * 64)
        else:
            work = Work(id="aaaaaaaaaaaa", title="Dusk", type="story", created_at=datetime(2026, 5, 5, tzinfo=UTC),
                        session_id=None, word_count=2, summary=None, origin="tool")
            assert WorksStore(db).insert(work, content="harbour light")
    return persona


@pytest.mark.parametrize("corrupt_works", [False, True], ids=["valid works db", "corrupt works db"])
def test_every_connection_the_walk_opens_is_closed_when_it_returns(tmp_path, monkeypatch, corrupt_works):
    persona = _persona(tmp_path, corrupt_works=corrupt_works)
    opened: list[tuple[str, sqlite3.Connection]] = []
    real = sqlite3.connect

    def recording(database, *args, **kwargs):
        conn = real(database, *args, **kwargs)
        opened.append((Path(database).relative_to(persona).as_posix(), conn))
        return conn

    monkeypatch.setattr(sqlite3, "connect", recording)
    anomalies = walk_persona(persona)

    works_checked = {name for name, _ in opened} & set(_WORKS_PATHS)
    assert works_checked, "the walk opened no works database, so this test checked nothing"
    assert [name for name, conn in opened if not _is_closed(conn)] == []
    flagged = {a.file for a in anomalies if a.kind == "sqlite_integrity_fail"}
    assert (flagged >= works_checked) if corrupt_works else not (flagged & works_checked)
