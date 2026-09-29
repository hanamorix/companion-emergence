"""Tests for SQLite integrity check on store + hebbian open.

Also covers the C16 Windows CI flake fix (ram-spike-fix INC-10 follow-up,
changes/c16-windows-integrity-flake/): `MemoryStore.__init__`'s integrity
check now retries a bounded number of times on a KNOWN-transient SQLite
error (e.g. "disk I/O error" — observed cause: a Windows process hard-kill
can leave a killed process's WAL/-shm memory-mapped section released
slightly after `proc.wait()` returns, so the very next `MemoryStore` open
sees a transient I/O error) before raising `BrainIntegrityError`, but any
message NOT on that small allowlist — including real corruption and any
unrecognized message — still raises immediately, exactly as before.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain import dev_constants
from brain.health.anomaly import BrainIntegrityError
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


def _patch_integrity_check_execute(
    monkeypatch: pytest.MonkeyPatch,
    side_effect,
    target: str = "brain.memory.store.sqlite3.connect",
):
    """Make a store module's `sqlite3.connect(...)` return a connection whose
    `PRAGMA integrity_check` calls go through `side_effect(call_count)`
    (1-indexed) instead of the real pragma; every other statement (schema
    creation, WAL pragma, etc.) hits the real, underlying connection
    unchanged. `side_effect` returning normally means "run the real pragma
    this attempt" (used to let a flaky call eventually succeed); raising
    inside `side_effect` propagates as the simulated `PRAGMA integrity_check`
    failure.

    `sqlite3.Connection` is a C-extension type and cannot be monkeypatched in
    place ("cannot set 'execute' attribute of immutable type
    'sqlite3.Connection'") — so this subclasses it via `sqlite3.connect`'s own
    `factory=` parameter instead, and patches only the ONE store module's own
    `sqlite3.connect` reference named by `target` (default `MemoryStore`'s;
    pass e.g. `"brain.memory.hebbian.sqlite3.connect"` for `HebbianMatrix`)
    — not the global `sqlite3.connect` — scoped to this test by
    `monkeypatch`'s auto-restore.
    """
    calls = {"n": 0}
    real_connect = sqlite3.connect

    class _FlakyIntegrityCheckConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if isinstance(sql, str) and sql.strip() == "PRAGMA integrity_check":
                calls["n"] += 1
                side_effect(calls["n"])
            return super().execute(sql, *args, **kwargs)

    def _connect(*args, **kwargs):
        kwargs.setdefault("factory", _FlakyIntegrityCheckConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(target, _connect)
    return calls


def test_memory_store_clean_db_passes_integrity_check(tmp_path: Path) -> None:
    """A clean store opens without raising."""
    store = MemoryStore(db_path=tmp_path / "memories.db")
    store.close()
    # Re-open — integrity check runs again on fresh open
    store2 = MemoryStore(db_path=tmp_path / "memories.db")
    store2.close()


def test_memory_store_corrupt_db_raises_integrity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file with bad SQLite header → BrainIntegrityError on open,
    IMMEDIATELY, with no retry. Real SQLite output for this exact scenario is
    `sqlite3.DatabaseError("file is not a database")` — it contains neither
    "malformed" nor "corrupt", so this is the regression test for the C16
    fix's central defect (round-1 red-team F3/G3): a naive "retry anything
    not explicitly recognized as corruption" classifier would have
    misclassified this real-corruption message as transient and retried it.
    Asserting `time.sleep` is never called proves no retry happened (the
    retry path is the only thing in this constructor that sleeps).
    """
    db = tmp_path / "memories.db"
    db.write_bytes(b"this is not a SQLite database")
    monkeypatch.setattr(
        "brain.health.integrity_retry.time.sleep",
        lambda *_a, **_kw: pytest.fail(
            "a real-corruption / unrecognized integrity error must not retry"
        ),
    )
    with pytest.raises(BrainIntegrityError):
        MemoryStore(db_path=db)


def test_memory_store_malformed_disk_image_raises_immediately_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other real-SQLite corruption message family ("database disk image
    is malformed") is deliberately NOT on the transient allowlist and must
    also raise on the first attempt."""

    def _always_malformed(_call_n: int) -> None:
        raise sqlite3.OperationalError("database disk image is malformed")

    calls = _patch_integrity_check_execute(monkeypatch, _always_malformed)
    with pytest.raises(BrainIntegrityError):
        MemoryStore(db_path=tmp_path / "memories.db")
    assert calls["n"] == 1


def test_memory_store_transient_disk_io_error_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual C16 CI failure signature: `PRAGMA integrity_check` raises
    "disk I/O error" (a KNOWN-transient message) on the first two attempts,
    then clears. `MemoryStore(...)` must succeed without raising."""

    def _fails_then_clears(call_n: int) -> None:
        if call_n < dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS:
            raise sqlite3.OperationalError("disk I/O error")
        # Final attempt: let the real pragma run (returns [("ok",)]).

    calls = _patch_integrity_check_execute(monkeypatch, _fails_then_clears)
    store = MemoryStore(db_path=tmp_path / "memories.db")
    store.close()
    assert calls["n"] == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS


@pytest.mark.parametrize(
    "message",
    [
        "database is locked",
        "unable to open database file",
    ],
)
def test_memory_store_other_allowlisted_messages_also_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    """Every entry in
    `brain.health.integrity_retry.TRANSIENT_INTEGRITY_ERROR_SUBSTRINGS`
    retries, not just the "disk I/O error" signature the C16 CI failure
    actually produced —
    stage-6 red-team coverage-challenge gap (the other two allowlist entries
    were previously exercised only by the shared generic matching logic, not
    a dedicated test each)."""

    def _fails_once_then_clears(call_n: int) -> None:
        if call_n == 1:
            raise sqlite3.OperationalError(message)

    calls = _patch_integrity_check_execute(monkeypatch, _fails_once_then_clears)
    store = MemoryStore(db_path=tmp_path / "memories.db")
    store.close()
    assert calls["n"] == 2


def test_memory_store_transient_disk_io_error_exhausts_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient-classified error that NEVER clears still raises
    `BrainIntegrityError`, after exactly the budgeted number of attempts —
    proves the retry is bounded, not a hang risk."""

    def _always_disk_io_error(_call_n: int) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    calls = _patch_integrity_check_execute(monkeypatch, _always_disk_io_error)
    with pytest.raises(BrainIntegrityError):
        MemoryStore(db_path=tmp_path / "memories.db")
    assert calls["n"] == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS


def test_hebbian_matrix_clean_db_passes(tmp_path: Path) -> None:
    h = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h.close()
    h2 = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h2.close()


def test_hebbian_matrix_corrupt_db_raises(tmp_path: Path) -> None:
    db = tmp_path / "hebbian.db"
    db.write_bytes(b"not sqlite")
    with pytest.raises(BrainIntegrityError):
        HebbianMatrix(db_path=db)


def test_hebbian_matrix_transient_disk_io_error_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HebbianMatrix routes through the SAME shared
    `brain.health.integrity_retry` helper as MemoryStore (orchestrator
    directive 2026-09-28: extract into one shared helper so the allowlist
    can't drift between stores)."""

    def _fails_then_clears(call_n: int) -> None:
        if call_n < dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS:
            raise sqlite3.OperationalError("disk I/O error")

    calls = _patch_integrity_check_execute(
        monkeypatch, _fails_then_clears, target="brain.memory.hebbian.sqlite3.connect"
    )
    h = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    h.close()
    assert calls["n"] == dev_constants.SQLITE_INTEGRITY_CHECK_RETRY_ATTEMPTS


def test_daemon_dirty_restart_shape_memorystore_then_hebbianmatrix_no_false_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the concrete gap named when the MemoryStore-only
    fix shipped (round-1/round-2 plan red-team M1/G1): `brain/bridge/
    daemon.py`'s dirty-restart recovery opens `MemoryStore` then, one line
    later, `HebbianMatrix` against the same persona directory. Before this
    change generalized the fix to all 4 stores, a transient Windows I/O
    error on the SECOND open (`HebbianMatrix`) would still have false-
    alarmed as `BrainIntegrityError` even though `MemoryStore` was already
    protected. Reproduces that exact shape: MemoryStore opens clean, then
    HebbianMatrix hits a transient error that clears on retry — neither
    raises.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    # MemoryStore open: clean, no injected error (mirrors daemon.py's first
    # open succeeding normally).
    store = MemoryStore(db_path=persona_dir / "memories.db")
    store.close()

    # HebbianMatrix open, immediately after (mirrors daemon.py:86-87): inject
    # a transient "disk I/O error" on its first attempt only.
    def _hebbian_fails_once_then_clears(call_n: int) -> None:
        if call_n == 1:
            raise sqlite3.OperationalError("disk I/O error")

    calls = _patch_integrity_check_execute(
        monkeypatch,
        _hebbian_fails_once_then_clears,
        target="brain.memory.hebbian.sqlite3.connect",
    )
    hebbian = HebbianMatrix(db_path=persona_dir / "hebbian.db")
    hebbian.close()
    assert calls["n"] == 2, "must have retried once, not false-alarmed on the first attempt"
