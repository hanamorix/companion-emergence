"""Tests for the shared PRAGMA integrity_check retry helper
(`brain.health.integrity_retry.run_integrity_check_with_retry`) — the single
source of truth backing all 4 sqlite stores' integrity-check-on-open
behavior (C16 Windows CI flake fix, ram-spike-fix INC-10 follow-up,
generalized from MemoryStore-only to MemoryStore/HebbianMatrix/
KindledLinkStore/SoulStore per orchestrator directive 2026-09-28).

Tests the mechanism ONCE, directly against a minimal fake connection (not
through any of the 4 real stores) — each store's own test file only needs a
thin integration test proving IT calls this helper, not re-proving the
retry/classify logic itself.
"""

from __future__ import annotations

import sqlite3

import pytest

from brain import dev_constants
from brain.health import integrity_retry
from brain.health.anomaly import BrainIntegrityError


class _FakeConn:
    """Minimal stand-in for `sqlite3.Connection` exposing only what
    `run_integrity_check_with_retry` touches: `execute(...).fetchall()` and
    `close()`. `execute` itself doubles as the returned "cursor" (its own
    `fetchall` returns the canned healthy result) since nothing else is
    needed for this helper's own tests."""

    def __init__(self, side_effect) -> None:
        self.calls = 0
        self.closed = False
        self._side_effect = side_effect

    def execute(self, sql):
        assert sql.strip() == "PRAGMA integrity_check"
        self.calls += 1
        self._side_effect(self.calls)
        return self

    def fetchall(self):
        return [("ok",)]

    def close(self) -> None:
        self.closed = True


def _no_error(_call_n: int) -> None:
    return None


def test_clean_db_passes_on_first_attempt_zero_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integrity_retry.time,
        "sleep",
        lambda *_a, **_kw: pytest.fail("a clean check must never sleep/retry"),
    )
    conn = _FakeConn(_no_error)
    result = integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert result == [("ok",)]
    assert conn.calls == 1
    assert not conn.closed


def test_transient_disk_io_error_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(integrity_retry.time, "sleep", lambda *_a, **_kw: None)

    def _fails_then_clears(call_n: int) -> None:
        if call_n < dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS:
            raise sqlite3.OperationalError("disk I/O error")

    conn = _FakeConn(_fails_then_clears)
    result = integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert result == [("ok",)]
    assert conn.calls == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS
    assert not conn.closed


@pytest.mark.parametrize("message", list(integrity_retry.TRANSIENT_INTEGRITY_ERROR_SUBSTRINGS))
def test_every_allowlist_entry_retries(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    monkeypatch.setattr(integrity_retry.time, "sleep", lambda *_a, **_kw: None)

    def _fails_once_then_clears(call_n: int) -> None:
        if call_n == 1:
            raise sqlite3.OperationalError(message)

    conn = _FakeConn(_fails_once_then_clears)
    result = integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert result == [("ok",)]
    assert conn.calls == 2


def test_transient_error_exhausts_retry_budget_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integrity_retry.time, "sleep", lambda *_a, **_kw: None)

    def _always_disk_io_error(_call_n: int) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    conn = _FakeConn(_always_disk_io_error)
    with pytest.raises(BrainIntegrityError):
        integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert conn.calls == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS
    assert conn.closed


def test_malformed_disk_image_raises_immediately_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integrity_retry.time,
        "sleep",
        lambda *_a, **_kw: pytest.fail("must not retry a genuine-corruption message"),
    )

    def _malformed(_call_n: int) -> None:
        raise sqlite3.OperationalError("database disk image is malformed")

    conn = _FakeConn(_malformed)
    with pytest.raises(BrainIntegrityError):
        integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert conn.calls == 1
    assert conn.closed


def test_unrecognized_message_raises_immediately_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The concrete regression case (round-1 red-team F3/G3 on the original
    MemoryStore-only fix): a garbled file's real SQLite message, "file is not
    a database", matches neither "malformed" nor "corrupt" and is NOT on the
    transient allowlist — it must raise on the first attempt, not be
    mistaken for a clearing condition."""
    monkeypatch.setattr(
        integrity_retry.time,
        "sleep",
        lambda *_a, **_kw: pytest.fail("must not retry an unrecognized message"),
    )

    def _unrecognized(_call_n: int) -> None:
        raise sqlite3.OperationalError("file is not a database")

    conn = _FakeConn(_unrecognized)
    with pytest.raises(BrainIntegrityError):
        integrity_retry.run_integrity_check_with_retry(conn, "db.sqlite", caller="Test")
    assert conn.calls == 1
