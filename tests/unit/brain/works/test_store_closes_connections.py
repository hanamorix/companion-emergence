"""WorksStore closes every connection it opens.

A dropped ``sqlite3.Connection`` is not freed when its last reference goes: CPython holds it in a reference cycle (its
statement cache), so only the cycle collector closes it. Until then it holds the database file, and Windows refuses to
rename or delete a file a handle holds. So the store closes each connection itself rather than leaving it to garbage.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

import brain.works.store as store_module
from brain.works import Work
from brain.works.store import WorksStore


def _work(wid: str = "aaaaaaaaaaaa") -> Work:
    return Work(
        id=wid,
        title="Lanterns",
        type="story",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
        session_id=None,
        word_count=3,
        summary=None,
    )


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


@pytest.fixture
def opened(monkeypatch):
    """Every connection the store opens, in order."""
    seen: list[sqlite3.Connection] = []
    real = store_module.open_store_connection

    def recording(*args, **kwargs):
        conn = real(*args, **kwargs)
        seen.append(conn)
        return conn

    monkeypatch.setattr(store_module, "open_store_connection", recording)
    return seen


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: None, id="init"),
        pytest.param(lambda s: s.schema_version(), id="schema_version"),
        pytest.param(lambda s: s.insert(_work(), content="old letter lanterns"), id="insert"),
        pytest.param(lambda s: s.get("aaaaaaaaaaaa"), id="get"),
        pytest.param(lambda s: s.list_recent(limit=5), id="list_recent"),
        pytest.param(lambda s: s.search("lanterns", limit=5), id="search"),
        pytest.param(lambda s: s.delete("aaaaaaaaaaaa"), id="delete"),
    ],
)
def test_each_method_closes_the_connections_it_opens(tmp_path, opened, call):
    WorksStore._INITIALISED_PATHS.discard(str((tmp_path / "works.db").resolve()))
    store = WorksStore(tmp_path / "works.db")

    call(store)

    assert opened, "the store opened no connection"
    assert [_is_closed(c) for c in opened] == [True] * len(opened)


def test_close_closes_the_raw_connections_it_handed_out(tmp_path):
    """The maker's ambient, disclosure and share readers query ``store._conn`` and then call ``store.close()``."""
    store = WorksStore(tmp_path / "works.db")
    first, second = store._conn, store._conn
    first.execute("SELECT COUNT(*) FROM works").fetchone()

    store.close()
    store.close()  # a second close is harmless

    assert (_is_closed(first), _is_closed(second)) == (True, True)
