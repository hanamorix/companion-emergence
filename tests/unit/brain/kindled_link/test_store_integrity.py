"""Tests for SQLite integrity check on `KindledLinkStore` open.

Part of the C16 Windows CI flake fix generalization (ram-spike-fix INC-10
follow-up, orchestrator directive 2026-09-28): `KindledLinkStore.__init__`
now routes through the SAME shared `brain.health.integrity_retry` helper as
`MemoryStore`/`HebbianMatrix`/`SoulStore`, so a transient SQLite error (e.g.
"disk I/O error") retries a bounded number of times before raising, but a
real/unrecognized corruption message still raises on the first attempt,
unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain import dev_constants
from brain.health.anomaly import BrainIntegrityError
from brain.kindled_link.store import KindledLinkStore


def test_kindled_link_store_clean_db_passes_integrity_check(tmp_path: Path) -> None:
    store = KindledLinkStore(db_path=tmp_path / "kindled_link.db")
    store.close()
    store2 = KindledLinkStore(db_path=tmp_path / "kindled_link.db")
    store2.close()


def test_kindled_link_store_corrupt_db_raises_immediately_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A garbled file's real SQLite message ("file is not a database") is
    NOT on the transient allowlist and must raise on the first attempt."""
    db = tmp_path / "kindled_link.db"
    db.write_bytes(b"this is not a sqlite file at all")
    monkeypatch.setattr(
        "brain.health.integrity_retry.time.sleep",
        lambda *_a, **_kw: pytest.fail("must not retry a non-transient integrity error"),
    )
    with pytest.raises(BrainIntegrityError):
        KindledLinkStore(db_path=db)


def test_kindled_link_store_transient_disk_io_error_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"n": 0}
    real_connect = sqlite3.connect

    class _FlakyConn(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if isinstance(sql, str) and sql.strip() == "PRAGMA integrity_check":
                calls["n"] += 1
                if calls["n"] < dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS:
                    raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *args, **kwargs)

    def _connect(*args, **kwargs):
        kwargs.setdefault("factory", _FlakyConn)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("brain.kindled_link.store.sqlite3.connect", _connect)
    store = KindledLinkStore(db_path=tmp_path / "kindled_link.db")
    store.close()
    assert calls["n"] == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS
