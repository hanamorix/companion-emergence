import sqlite3

from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def _work(content="poem body"):
    from datetime import UTC, datetime

    return Work(
        id=make_work_id(content),
        title="t",
        type="poem",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=2,
        summary=None,
        disposition="private",
        private_reason="too close",
        origin="maker",
        charge_sources='["grief"]',
        shared_at=None,
    )


def test_new_columns_round_trip(tmp_path):
    store = WorksStore(tmp_path / "works.db")
    assert store.insert(_work(), content="poem body") is True
    got = store.get(make_work_id("poem body"))
    assert got.disposition == "private"
    assert got.private_reason == "too close"
    assert got.origin == "maker"
    store.close()


def test_migration_from_v1_adds_columns(tmp_path):
    # Build a v1 schema db by hand, then open with the v2 store → ALTER runs.
    db = tmp_path / "works.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE works (id TEXT PRIMARY KEY, title TEXT NOT NULL, type TEXT NOT NULL,"
        " created_at TEXT NOT NULL, session_id TEXT, content_path TEXT NOT NULL,"
        " word_count INTEGER NOT NULL, summary TEXT); PRAGMA user_version=1;"
    )
    conn.commit()
    conn.close()
    store = WorksStore(db)  # opening must migrate v1 → v2 without error
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(works)")}
    assert {"disposition", "private_reason", "origin", "charge_sources", "shared_at"} <= cols
    store.close()
