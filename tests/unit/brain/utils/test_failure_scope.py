"""failure_scope / settle_scope: what a failure depends on decides what happens to the item it hit (#459, #460).

"environment": the failure depends on the store or the filesystem, not on the item: it affects any item and clears when the
environment is fixed (keep the item, retry later). "item": it depends on the item's own data and will fail the same way
again (set it aside with evidence). Classification is by exception TYPE, never by message text. A store that is closed or used
from the wrong thread raises `sqlite3.ProgrammingError` for EVERY item, so an item-scope verdict is confirmed against the store
itself before anyone acts on it (`settle_scope`).
"""

from __future__ import annotations

import errno
import sqlite3
import threading

import pytest

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.health.json_io import JsonFileError
from brain.memory.store import MemoryStore
from brain.utils.failure_scope import failure_scope, settle_scope

_ENVIRONMENT = [
    OSError("boom"),
    PermissionError("denied"),
    FileNotFoundError("gone"),
    TimeoutError("slow"),
    OSError(errno.ENOSPC, "No space left on device"),
    OSError(errno.EIO, "Input/output error"),
    JsonFileError("/x", "io", "denied"),
    JsonFileError("/x", "malformed", "bad"),
    sqlite3.OperationalError("database is locked"),
    sqlite3.OperationalError("no such table: memories"),
    sqlite3.OperationalError("disk I/O error"),
    sqlite3.DatabaseError("database disk image is malformed"),
    sqlite3.InternalError("internal"),
    sqlite3.NotSupportedError("nope"),
]
_ITEM = [
    sqlite3.IntegrityError("UNIQUE constraint failed: memories.id"),
    sqlite3.DataError("string or blob too big"),
    sqlite3.ProgrammingError("Error binding parameter 1"),
    sqlite3.InterfaceError("bad"),
    TypeError("not JSON serialisable"),
    ValueError("bad value"),
    KeyError("k"),
    RuntimeError("boom"),
    type("Custom", (Exception,), {})("custom"),
]


@pytest.mark.parametrize("exc", _ENVIRONMENT, ids=lambda e: f"{type(e).__name__}:{str(e)[:20]}")
def test_environment_types(exc):
    # C-FS-a
    assert failure_scope(exc) == "environment"


@pytest.mark.parametrize("exc", _ITEM, ids=lambda e: f"{type(e).__name__}:{str(e)[:20]}")
def test_item_types(exc):
    # C-FS-b
    assert failure_scope(exc) == "item"


def test_classification_ignores_the_message_text():
    # C-FS-c: the same class with different words gives the same answer; JsonFileError is an OSError AND a ValueError.
    assert failure_scope(sqlite3.OperationalError("UNIQUE constraint failed")) == "environment"
    assert failure_scope(sqlite3.IntegrityError("database is locked")) == "item"
    assert failure_scope(JsonFileError("/x", "shape", "expected dict")) == "environment"


@pytest.mark.parametrize("exc", [ThrottleDeferred("not now"), ProviderAuthDeferred("provider auth expired - deferred")])
def test_a_deferral_is_environment_although_it_is_a_runtime_error(exc):
    # C-FS-e (#473): a deferral says "not now".
    assert isinstance(exc, RuntimeError)
    assert failure_scope(exc) == "environment"


class _NoProbe:
    def get(self, *_a, **_k):
        raise AssertionError("an environment failure must not probe the store")


def test_settle_scope_does_not_probe_for_an_environment_failure():
    assert settle_scope(sqlite3.OperationalError("database is locked"), _NoProbe()) == "environment"
    assert settle_scope(ProviderAuthDeferred("deferred"), _NoProbe()) == "environment"


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


def test_settle_scope_keeps_item_when_the_store_is_healthy(store):
    # C-FS-d(3): a healthy store with a per-row failure.
    assert settle_scope(sqlite3.ProgrammingError("Error binding parameter 1"), store) == "item"
    assert settle_scope(ValueError("bad"), store) == "item"


def test_settle_scope_reclassifies_a_closed_store_as_environment(tmp_path):
    # C-FS-d(1): REAL fault: "Cannot operate on a closed database" hits every item.
    closed = MemoryStore(tmp_path / "closed.db")
    closed.close()
    with pytest.raises(sqlite3.ProgrammingError) as excinfo:
        closed.get("", bump=False)

    assert failure_scope(excinfo.value) == "item"  # by type alone it looks per-row
    assert settle_scope(excinfo.value, closed) == "environment"


def test_settle_scope_reclassifies_a_wrong_thread_store_as_environment(store):
    # C-FS-d(2): REAL fault: a store used from another thread raises ProgrammingError on every call.
    seen: dict = {}

    def other_thread():
        try:
            store.get("", bump=False)
        except sqlite3.ProgrammingError as exc:
            seen["exc"] = exc
            seen["scope"] = settle_scope(exc, store)

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()

    assert seen["scope"] == "environment"
