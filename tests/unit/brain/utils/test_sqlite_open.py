"""Unit tests of `brain.utils.sqlite_open` on real SQLite files and `:memory:`."""

from __future__ import annotations

import dataclasses
import inspect
import sqlite3

import pytest

from brain import dev_constants
from brain.health.anomaly import BrainIntegrityError
from brain.utils import sqlite_open
from brain.utils.sqlite_open import (
    DEFAULT_PROFILE,
    STRICT_WAL_PROFILE,
    OpenProfile,
    check_integrity,
    close_on_error,
    open_store_connection,
)
from tests.unit.brain.utils.test_sqlite_open_equivalence import (
    CLOSE,
    NOT_OK_ROWS,
    install,
)


def test_statements_run_in_the_documented_order(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    seen = []
    conn = open_store_connection(
        tmp_path / "a.db",
        caller="TestStore",
        integrity_check=True,
        setup=lambda c: seen.append(list(c.trace)),
    )
    assert rec.conns == [conn]
    assert seen == [
        ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
    ]
    assert rec.connects == [
        {
            # the recorder gives paths a tmp token and forward slashes, on every OS
            "args": ["<TMP>/a.db"],
            "kwargs": {"timeout": 5.0},
        }
    ]
    conn.close()


def test_integrity_false_issues_no_integrity_statement(tmp_path, monkeypatch) -> None:
    install(monkeypatch, tmp_path)
    conn = open_store_connection(tmp_path / "a.db", caller="TestStore", integrity_check=False)
    assert conn.trace == ["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
    conn.close()


def test_integrity_check_is_a_required_keyword() -> None:
    p = inspect.signature(open_store_connection).parameters["integrity_check"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY
    assert p.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        open_store_connection(":memory:", caller="TestStore")  # type: ignore[call-arg]


def test_the_profile_is_two_fields_frozen_and_sourced_from_the_constant() -> None:
    assert [f.name for f in dataclasses.fields(OpenProfile)] == ["busy_timeout_s", "wal_fallback"]
    assert DEFAULT_PROFILE == OpenProfile(dev_constants.SQLITE_STORE_BUSY_TIMEOUT_S, True)
    assert STRICT_WAL_PROFILE == OpenProfile(dev_constants.SQLITE_STORE_BUSY_TIMEOUT_S, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_PROFILE.wal_fallback = False  # type: ignore[misc]


def test_a_profile_timeout_reaches_connect_and_the_pragma(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    conn = open_store_connection(
        tmp_path / "a.db",
        caller="TestStore",
        integrity_check=False,
        profile=OpenProfile(busy_timeout_s=30.0),
    )
    assert rec.connects[0]["kwargs"]["timeout"] == 30.0
    assert "PRAGMA busy_timeout = 30000" in conn.trace
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    conn.close()


def test_wal_failure_is_swallowed_by_the_default_profile(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    rec.fail_on("PRAGMA journal_mode = WAL", sqlite3.OperationalError("no WAL"))
    conn = open_store_connection(tmp_path / "a.db", caller="TestStore", integrity_check=False)
    assert "PRAGMA busy_timeout = 5000" in conn.trace
    conn.close()


def test_only_operational_error_is_swallowed_on_the_wal_pragma(tmp_path, monkeypatch) -> None:
    """Base caught `sqlite3.OperationalError` only: any other DatabaseError from the
    WAL pragma (a non-database file opened without the integrity check) propagates."""
    rec = install(monkeypatch, tmp_path)
    exc = sqlite3.DatabaseError("file is not a database")
    rec.fail_on("PRAGMA journal_mode = WAL", exc)
    with pytest.raises(sqlite3.DatabaseError) as raised:
        open_store_connection(
            tmp_path / "a.db", caller="TestStore", integrity_check=False, profile=DEFAULT_PROFILE
        )
    assert raised.value is exc
    assert CLOSE in rec.conns[0].trace
    assert not any("busy_timeout" in s for s in rec.conns[0].trace)


def test_a_fractional_timeout_reaches_connect_and_the_pragma_unrounded(
    tmp_path, monkeypatch
) -> None:
    rec = install(monkeypatch, tmp_path)
    conn = open_store_connection(
        tmp_path / "a.db",
        caller="TestStore",
        integrity_check=False,
        profile=OpenProfile(busy_timeout_s=0.5),
    )
    assert rec.connects[0]["kwargs"]["timeout"] == 0.5
    assert "PRAGMA busy_timeout = 500" in conn.trace
    conn.close()


def test_wal_failure_propagates_and_closes_under_the_strict_profile(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    exc = sqlite3.OperationalError("no WAL")
    rec.fail_on("PRAGMA journal_mode = WAL", exc)
    with pytest.raises(sqlite3.OperationalError) as raised:
        open_store_connection(
            tmp_path / "a.db",
            caller="TestStore",
            integrity_check=False,
            profile=STRICT_WAL_PROFILE,
        )
    assert raised.value is exc
    assert CLOSE in rec.conns[0].trace
    assert not any("busy_timeout" in s for s in rec.conns[0].trace)


def test_not_ok_result_raises_with_the_detail_and_closes(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    rec.integrity_result = NOT_OK_ROWS
    with pytest.raises(BrainIntegrityError) as raised:
        open_store_connection(tmp_path / "a.db", caller="TestStore", integrity_check=True)
    assert str(raised.value) == (
        f"integrity check failed for {tmp_path / 'a.db'}: *** in database main ***; row 1 missing"
    )
    assert CLOSE in rec.conns[0].trace
    with pytest.raises(sqlite3.ProgrammingError):
        rec.conns[0].execute("SELECT 1")
    assert not any("journal_mode" in s or "busy_timeout" in s for s in rec.conns[0].trace)


def test_check_integrity_does_not_close_the_connection(tmp_path, monkeypatch) -> None:
    rec = install(monkeypatch, tmp_path)
    conn = sqlite3.connect(str(tmp_path / "a.db"))
    check_integrity(conn, tmp_path / "a.db", caller="TestStore")  # ok result: no raise
    rec.integrity_result = NOT_OK_ROWS
    with pytest.raises(BrainIntegrityError):
        check_integrity(conn, tmp_path / "a.db", caller="TestStore")
    assert CLOSE not in conn.trace
    conn.close()


def test_the_caller_label_reaches_the_retry_log(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("brain.health.integrity_retry.time.sleep", lambda s: None)
    rec = install(monkeypatch, tmp_path)
    rec.fail_on("PRAGMA integrity_check", sqlite3.OperationalError("disk I/O error"), times=1)
    with caplog.at_level("WARNING", logger="brain.health.integrity_retry"):
        open_store_connection(
            tmp_path / "a.db", caller="LabelledStore", integrity_check=True
        ).close()
    assert any("LabelledStore" in r.getMessage() for r in caplog.records)


def test_row_factory_is_left_alone(tmp_path) -> None:
    conn = open_store_connection(tmp_path / "a.db", caller="TestStore", integrity_check=True)
    assert conn.row_factory is None
    assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    conn.close()


def test_setup_runs_last_gets_the_connection_and_can_set_a_row_factory(tmp_path) -> None:
    def setup(conn: sqlite3.Connection) -> None:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE t (x)")

    conn = open_store_connection(
        tmp_path / "a.db", caller="TestStore", integrity_check=True, setup=setup
    )
    assert conn.row_factory is sqlite3.Row
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    conn.close()


def test_a_failing_setup_closes_the_connection_and_propagates_the_same_exception(
    tmp_path, monkeypatch
) -> None:
    rec = install(monkeypatch, tmp_path)
    exc = RuntimeError("setup broke")

    def setup(conn: sqlite3.Connection) -> None:
        raise exc

    with pytest.raises(RuntimeError) as raised:
        open_store_connection(
            tmp_path / "a.db", caller="TestStore", integrity_check=True, setup=setup
        )
    assert raised.value is exc
    assert CLOSE in rec.conns[0].trace


def test_the_in_memory_database_opens_with_the_default_profile() -> None:
    conn = open_store_connection(":memory:", caller="TestStore", integrity_check=True)
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    conn.close()


def test_close_on_error_is_exported() -> None:
    assert sqlite_open.close_on_error is close_on_error
