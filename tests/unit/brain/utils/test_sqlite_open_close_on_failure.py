"""C9 / spec B3 (#496): a store that fails to open closes the connection it made.

At base no constructor closed its connection after a failure past `sqlite3.connect`
(the connection stayed open until the traceback was released). Each case injects
one failure into one store's open and asserts two things: the very same exception
object propagates (nothing wrapped, replaced or swallowed), and every connection
that open made has a recorded `<close>` marker.

The recording connection is the one from the equivalence file. This file imports the
helper module only inside the helper-direct tests, so that before the helper exists
the per-store cases fail on "no close marker" and not on an ImportError.
"""

from __future__ import annotations

import sqlite3

import pytest

from tests.unit.brain.utils.test_sqlite_open_equivalence import CLOSE, Recorder, install

SCHEMA_STEP = "CREATE TABLE IF NOT EXISTS"
LOCKED = "database is locked"

LEGACY_MEMORIES = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    emotions_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_accessed_at TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    protected INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
"""
LEGACY_PEERS = """
CREATE TABLE peers (
    peer_id TEXT PRIMARY KEY, identity_pub TEXT NOT NULL,
    fingerprint TEXT NOT NULL, consent_state TEXT NOT NULL,
    relay_url TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
INSERT INTO peers VALUES ('kid_old','cc','kid_old','paired',NULL,'t','t');
"""


def _write_legacy(path: str, script: str) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(script)
    conn.commit()
    conn.close()


def _open(kind: str, path: str):
    if kind == "memory":
        from brain.memory.store import MemoryStore

        return MemoryStore(path)
    if kind == "soul":
        from brain.soul.store import SoulStore

        return SoulStore(path)
    if kind == "hebbian":
        from brain.memory.hebbian import HebbianMatrix

        return HebbianMatrix(path)
    if kind == "kindled":
        from brain.kindled_link.store import KindledLinkStore

        return KindledLinkStore(path)
    raise AssertionError(kind)


def _assert_failed_open_closed(rec: Recorder, exc: BaseException, raised) -> None:
    assert raised.value is exc, "the injected exception was wrapped or replaced"
    assert rec.conns, "the open never reached sqlite3.connect"
    for n, conn in enumerate(rec.conns):
        assert CLOSE in conn.trace, f"no close marker on connection {n}: {conn.trace[-3:]}"


STORE_CASES = [
    pytest.param(PRAGMA, exc, id=name)
    for name, PRAGMA, exc in [
        ("executescript-locked", SCHEMA_STEP, sqlite3.OperationalError(LOCKED)),
        ("busy-timeout-pragma", "PRAGMA busy_timeout", sqlite3.OperationalError(LOCKED)),
        ("integrity-probe-runtime-error", "PRAGMA integrity_check", RuntimeError("probe broke")),
        ("keyboard-interrupt-at-schema", SCHEMA_STEP, KeyboardInterrupt()),
    ]
]


@pytest.mark.parametrize("needle,exc", STORE_CASES)
@pytest.mark.parametrize("kind", ["memory", "soul", "hebbian", "kindled"])
def test_failed_store_open_closes_connection(kind, needle, exc, tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    rec.fail_on(needle, exc)
    with pytest.raises(type(exc)) as raised:
        _open(kind, str(tmp_path / f"{kind}.db"))
    _assert_failed_open_closed(rec, exc, raised)


@pytest.mark.parametrize(
    "kind,script,alter",
    [
        ("memory", LEGACY_MEMORIES, "ALTER TABLE memories ADD COLUMN"),
        ("kindled", LEGACY_PEERS, "ALTER TABLE peers ADD COLUMN relay_mailbox"),
    ],
    ids=["memory-legacy-schema", "kindled-legacy-schema"],
)
def test_failed_migration_step_closes_connection(
    kind, script, alter, tmp_path, monkeypatch
) -> None:
    path = str(tmp_path / f"{kind}.db")
    _write_legacy(path, script)
    exc = sqlite3.OperationalError(LOCKED)
    rec = install(monkeypatch, tmp_path)
    rec.fail_on(alter, exc)
    with pytest.raises(sqlite3.OperationalError) as raised:
        _open(kind, path)
    assert any(alter in s for s in rec.conns[0].trace), "the ALTER step was never reached"
    _assert_failed_open_closed(rec, exc, raised)


WORKS_CONNECT_CASES = [
    pytest.param("PRAGMA journal_mode = WAL", id="strict-wal"),
    pytest.param("PRAGMA busy_timeout", id="busy-timeout-pragma"),
    pytest.param("PRAGMA foreign_keys", id="foreign-keys-pragma"),
]


@pytest.mark.parametrize("needle", WORKS_CONNECT_CASES)
def test_failed_works_connect_closes_connection(needle, tmp_path, monkeypatch) -> None:
    from brain.works.store import WorksStore

    store = WorksStore(tmp_path / "works.db")  # first-init runs before the recorder exists
    exc = sqlite3.OperationalError("injected pragma failure")
    rec = install(monkeypatch, tmp_path)
    rec.fail_on(needle, exc)
    with pytest.raises(sqlite3.OperationalError) as raised:
        store._connect()
    _assert_failed_open_closed(rec, exc, raised)


@pytest.mark.parametrize(
    "needle,via,reached",
    [
        ("CREATE TABLE", "executescript", None),
        ("PRAGMA user_version = ", "execute", "PRAGMA user_version = "),
    ],
    ids=["first-init-executescript", "first-init-migrate-user-version"],
)
def test_failed_works_first_init_closes_connection(
    needle, via, reached, tmp_path, monkeypatch
) -> None:
    from brain.works.store import WorksStore

    path = tmp_path / "fresh" / "works.db"
    exc = sqlite3.OperationalError(LOCKED)
    rec = install(monkeypatch, tmp_path)
    rec.fail_on(needle, exc, via=via)
    with pytest.raises(sqlite3.OperationalError) as raised:
        WorksStore(path)
    assert len(rec.conns) == 1, "first-init must open exactly one connection"
    if reached is not None:
        assert any(reached in s for s in rec.conns[0].trace), "the migration step was not reached"
    _assert_failed_open_closed(rec, exc, raised)
    assert str(path.resolve()) not in WorksStore._INITIALISED_PATHS


# ---- the guard itself (helper-direct; the import is inside each test) ----


class _FakeConn:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.closed = 0
        self._close_error = close_error

    def close(self) -> None:
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


def test_close_on_error_passes_the_exception_through_and_closes_once() -> None:
    from brain.utils.sqlite_open import close_on_error

    conn = _FakeConn()
    exc = RuntimeError("boom")
    with pytest.raises(RuntimeError) as raised, close_on_error(conn):
        raise exc
    assert raised.value is exc
    assert conn.closed == 1


def test_close_on_error_does_not_close_on_normal_exit() -> None:
    from brain.utils.sqlite_open import close_on_error

    conn = _FakeConn()
    with close_on_error(conn):
        pass
    assert conn.closed == 0


def test_close_on_error_second_close_is_harmless(tmp_path) -> None:
    from brain.utils.sqlite_open import close_on_error

    conn = sqlite3.connect(str(tmp_path / "x.db"))
    conn.close()
    exc = ValueError("after an earlier close")
    with pytest.raises(ValueError) as raised, close_on_error(conn):
        raise exc
    assert raised.value is exc


def test_close_on_error_a_failing_close_does_not_mask_the_original(caplog) -> None:
    from brain.utils.sqlite_open import close_on_error

    conn = _FakeConn(close_error=sqlite3.OperationalError("close failed"))
    exc = RuntimeError("the original")
    with caplog.at_level("DEBUG", logger="brain.utils.sqlite_open"):
        with pytest.raises(RuntimeError) as raised, close_on_error(conn):
            raise exc
    assert raised.value is exc
    assert conn.closed == 1
    assert any("also failed" in r.getMessage() for r in caplog.records)


def test_close_on_error_catches_base_exception() -> None:
    from brain.utils.sqlite_open import close_on_error

    conn = _FakeConn()
    with pytest.raises(KeyboardInterrupt), close_on_error(conn):
        raise KeyboardInterrupt
    assert conn.closed == 1
