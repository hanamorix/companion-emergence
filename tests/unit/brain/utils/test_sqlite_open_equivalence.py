"""C9: the open sequence of the five SQLite stores, pinned pragma for pragma.

Probe: `sqlite3.connect` is wrapped so every connection is a `sqlite3.Connection`
subclass that records, in order, every statement it runs (through
`set_trace_callback`, which sees `execute`, `executescript` and cursors alike)
plus `<commit>` and `<close>` markers. `Recorder.trace()` keeps from that only the policy
pragmas (`integrity_check`, `journal_mode`, `busy_timeout`, `foreign_keys`, ...) in order
and `<close>`: schema statements (DDL, introspection pragmas, migration steps and the
commits after them) follow the schema, which changes with every column or table, and are
not the open sequence (#527). The connect arguments and the number of connect calls are
recorded as well. Each store is opened on a fresh file (and on `:memory:` where the
store allows it) and the recording is compared with GOLDEN, a set of literals captured
from the unmodified stores (base f9f18bd3, before the helper existed; main merged at
a17ab2e6 changed no store open sequence), schema statements removed. The same file runs
unchanged against the refactored stores.

A schema change needs no refresh of GOLDEN. A legitimate change to GOLDEN is a change to a
connect argument, a policy pragma or their order.

Failure paths: a failed open now closes its connection (spec B3, #496), which
base did not. So on a failure path the golden holds the statement list with the
`<close>` markers removed, and the closure itself is asserted separately.

Paths are recorded with a token for the tmp dir and forward slashes, so Windows runs match.
"""

from __future__ import annotations

import inspect
import logging
import re
import sqlite3
from pathlib import Path

import pytest

from brain import dev_constants

# Pragmas that read or version the schema; a migration issues them once per table or column.
SCHEMA_PRAGMAS = frozenset(
    {
        "table_info",
        "table_xinfo",
        "table_list",
        "index_list",
        "index_info",
        "index_xinfo",
        "foreign_key_list",
        "schema_version",
        "user_version",
    }
)
CLOSE = "<close>"
COMMIT = "<commit>"
NOT_OK_ROWS = [("*** in database main ***",), ("row 1 missing",)]

OPEN_CASES = [
    ("memory", "file", True),
    ("memory", "file", False),
    ("memory", "memory", True),
    ("memory", "memory", False),
    ("soul", "file", True),
    ("soul", "memory", True),
    ("hebbian", "file", True),
    ("hebbian", "file", False),
    ("hebbian", "memory", True),
    ("hebbian", "memory", False),
    ("kindled", "file", True),
    ("kindled", "file", False),
    ("kindled", "memory", True),
    ("kindled", "memory", False),
    ("works", "file", None),
]
CHECKED_KINDS = ["memory", "soul", "hebbian", "kindled"]
ALL_KINDS = [*CHECKED_KINDS, "works"]
CALLER_LABEL = {
    "memory": "MemoryStore",
    "soul": "SoulStore",
    "hebbian": "HebbianMatrix",
    "kindled": "KindledLinkStore",
}


# --------------------------------------------------------------------------
# The recording connection
# --------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class Recorder:
    """What the patched `sqlite3.connect` saw, plus the failures to inject."""

    def __init__(self, tmp_token: str):
        self.tmp_token = tmp_token
        self.connects: list[dict] = []
        self.conns: list[sqlite3.Connection] = []
        self.failures: list[dict] = []
        self.integrity_result = None

    def fail_on(
        self,
        needle: str,
        exc: BaseException,
        times: int | None = None,
        via: str | None = None,
    ) -> None:
        """Raise `exc` when a statement containing `needle` is about to run.

        `via` restricts it to "execute" or "executescript" (a schema script can
        contain the same text as a single statement).
        """
        self.failures.append({"needle": needle, "exc": exc, "remaining": times, "via": via})

    def norm(self, value):
        """Replace the tmp dir by a token and give the rest of the path forward slashes
        (on Windows the same path would otherwise differ from the golden)."""
        text = str(value).replace(self.tmp_token, "<TMP>")
        return re.sub(r"<TMP>[^\s:'\"]*", lambda m: m.group(0).replace("\\", "/"), text)

    def trace(self, conn, *, keep_close: bool = True) -> list[str]:
        """The policy statements one connection ran: its pragmas (in order) and `<close>`.

        Schema work is not part of the open sequence, so a PR that adds a column or a table
        must not change it (#527): after dropping leading whitespace and `--` comment lines,
        only `PRAGMA` items are kept (case-insensitively), minus the introspection and
        versioning pragmas a migration issues once per table or column. That also drops
        SQLite's own one-line `-- ...` bookkeeping statements, which differ between builds."""
        out = []
        for item in conn.trace:
            if item == CLOSE:
                if keep_close:
                    out.append(item)
                continue
            if item == COMMIT:
                continue  # a migration commits after each step: placement follows the schema, not the policy
            kept = _policy_pragma(item)
            if kept is not None:
                out.append(self.norm(kept))
        return out


def _policy_pragma(item: str) -> str | None:
    """`item` with leading whitespace and comment lines removed and whitespace collapsed, if it is a policy
    pragma; None for anything else (schema statements, DML, introspection pragmas, SQLite's own lines)."""
    lines = item.strip().splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("--")):
        lines.pop(0)
    text = " ".join(" ".join(lines).split())
    if not text.upper().startswith("PRAGMA"):
        return None
    name = re.match(r"PRAGMA\s+(?:[\"\']?\w+[\"\']?\.)?[\"\']?(\w+)", text, re.IGNORECASE)
    if name is None or name.group(1).lower() in SCHEMA_PRAGMAS:
        return None
    return text


def _connection_class(rec: Recorder):
    class _Traced(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trace: list[str] = []
            self.set_trace_callback(self.trace.append)

        def _intercept(self, sql: str, kind: str) -> None:
            for entry in rec.failures:
                if (
                    entry["via"] in (None, kind)
                    and entry["needle"] in sql
                    and entry["remaining"] != 0
                ):
                    if entry["remaining"] is not None:
                        entry["remaining"] -= 1
                    self.trace.append(sql if kind == "execute" else f"<{kind} raised>")
                    raise entry["exc"]

        def execute(self, sql, *args):
            self._intercept(sql, "execute")
            if rec.integrity_result is not None and sql == "PRAGMA integrity_check":
                self.trace.append(sql)
                return _FakeCursor(rec.integrity_result)
            return super().execute(sql, *args)

        def executescript(self, script):
            self._intercept(script, "executescript")
            return super().executescript(script)

        def commit(self):
            self.trace.append(COMMIT)
            super().commit()

        def close(self):
            self.trace.append(CLOSE)
            super().close()

    return _Traced


def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Recorder:
    """Wrap `sqlite3.connect` so every connection it makes is recorded."""
    rec = Recorder(str(tmp_path))
    cls = _connection_class(rec)
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs.pop("factory", None)
        kw = {k: rec.norm(v) if isinstance(v, (str, Path)) else v for k, v in kwargs.items()}
        if "timeout" not in kw and len(args) < 2:
            kw["timeout"] = 5.0
        rec.connects.append({"args": [rec.norm(a) for a in args], "kwargs": kw})
        conn = real_connect(*args, factory=cls, **kwargs)
        rec.conns.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return rec


# --------------------------------------------------------------------------
# Opening each store
# --------------------------------------------------------------------------


def _target(kind: str, target: str, tmp_path: Path) -> str:
    return ":memory:" if target == "memory" else str(tmp_path / f"{kind}.db")


def _build(kind: str, path: str, integrity: bool | None):
    if kind == "memory":
        from brain.memory.store import MemoryStore

        return MemoryStore(path, integrity_check=integrity)
    if kind == "soul":
        from brain.soul.store import SoulStore

        return SoulStore(path)
    if kind == "hebbian":
        from brain.memory.hebbian import HebbianMatrix

        return HebbianMatrix(path, integrity_check=integrity)
    if kind == "kindled":
        from brain.kindled_link.store import KindledLinkStore

        return KindledLinkStore(path, integrity_check=integrity)
    raise AssertionError(kind)


def open_store(kind, path, integrity, rec_getter):
    """Open one store; return (rec-visible connection, store object or None, error or None).

    For works the thing opened is `WorksStore._connect()` on a store whose first-init
    already ran before the recorder was installed (so only `_connect` is recorded).
    """
    if kind == "works":
        from brain.works.store import WorksStore

        store = WorksStore(Path(path))
        rec = rec_getter()
        try:
            return rec, store._connect(), None
        except BaseException as exc:  # noqa: BLE001 - the probe reports any failure
            return rec, None, exc
    rec = rec_getter()
    try:
        store = _build(kind, path, integrity)
    except BaseException as exc:  # noqa: BLE001
        return rec, None, exc
    return rec, store._conn, None


def snapshot(rec: Recorder, conn, error, *, failure: bool, unordered: bool = False) -> dict:
    """What the probe saw. `unordered` (works only, spec B1): its `_connect` pragmas are compared as a set."""
    traces = []
    for c in rec.conns:
        t = rec.trace(c, keep_close=not failure)
        traces.append(sorted(t) if unordered and all(s.startswith("PRAGMA") for s in t) else t)
    snap = {
        "connects": rec.connects,
        "traces": traces,
        "error": None if error is None else [type(error).__name__, rec.norm(error)],
    }
    if conn is not None:
        snap["end_state"] = {
            "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
            "busy_timeout": conn.execute("PRAGMA busy_timeout").fetchone()[0],
            "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
            "row_factory": "Row" if conn.row_factory is sqlite3.Row else repr(conn.row_factory),
            "isolation_level": conn.isolation_level,
        }
    return snap


def run_case(kind, target, integrity, tmp_path, monkeypatch, *, configure=None, prepare=None):
    path = _target(kind, target, tmp_path)
    if prepare is not None:
        prepare(path)

    def getter():
        rec = install(monkeypatch, tmp_path)
        if configure is not None:
            configure(rec)
        return rec

    rec, conn, error = open_store(kind, path, integrity, getter)
    return rec, conn, error


def success_snapshot(kind, target, integrity, tmp_path, monkeypatch) -> dict:
    rec, conn, error = run_case(kind, target, integrity, tmp_path, monkeypatch)
    assert error is None, error
    return snapshot(rec, conn, error, failure=False, unordered=kind == "works")


def failure_snapshot(kind, tmp_path, monkeypatch, configure, prepare=None):
    rec, conn, error = run_case(
        kind, "file", True, tmp_path, monkeypatch, configure=configure, prepare=prepare
    )
    return rec, conn, error, snapshot(rec, conn, error, failure=True, unordered=kind == "works")


def _not_sqlite(path: str) -> None:
    Path(path).write_bytes(b"this is not a sqlite database " * 200)


def _not_ok(rec: Recorder) -> None:
    rec.integrity_result = NOT_OK_ROWS


def _wal_fails(rec: Recorder) -> None:
    rec.fail_on("PRAGMA journal_mode = WAL", sqlite3.OperationalError("WAL not available"))


def _case_id(case) -> str:
    return ":".join(str(x) for x in case)


# --------------------------------------------------------------------------
# GOLDEN: what each store's open ran, as captured from base f9f18bd3 (the unmodified
# stores) with the schema statements left out (#527), pasted as literals. A schema
# change does not touch it; a change to the connect arguments, the policy pragmas or
# their order does, and then the new value is a deliberate decision.
# --------------------------------------------------------------------------

GOLDEN: dict[str, dict] = {
    "open:memory:file:True": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 30000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 30000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:memory:file:False": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 30000"]],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 30000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:memory:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 30.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 30000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 30000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:memory:memory:False": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 30.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 30000"]],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 30000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:soul:file:True": {
        "connects": [{"args": ["<TMP>/soul.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:soul:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:hebbian:file:True": {
        "connects": [{"args": ["<TMP>/hebbian.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "None",
            "isolation_level": "",
        },
    },
    "open:hebbian:file:False": {
        "connects": [{"args": ["<TMP>/hebbian.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "None",
            "isolation_level": "",
        },
    },
    "open:hebbian:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "None",
            "isolation_level": "",
        },
    },
    "open:hebbian:memory:False": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "None",
            "isolation_level": "",
        },
    },
    "open:kindled:file:True": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:kindled:file:False": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:kindled:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:kindled:memory:False": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]],
        "error": None,
        "end_state": {
            "journal_mode": "memory",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "open:works:file:None": {
        "connects": [{"args": ["<TMP>/works.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA busy_timeout = 5000", "PRAGMA foreign_keys = ON", "PRAGMA journal_mode = WAL"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "wal",
            "busy_timeout": 5000,
            "foreign_keys": 1,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "notsqlite:memory": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [[]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/memory.db: file is not a database",
        ],
    },
    "notok:memory": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [["PRAGMA integrity_check"]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/memory.db: *** in database main ***; row 1 missing",
        ],
    },
    "notsqlite:soul": {
        "connects": [{"args": ["<TMP>/soul.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [[]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/soul.db: file is not a database",
        ],
    },
    "notok:soul": {
        "connects": [{"args": ["<TMP>/soul.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA integrity_check"]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/soul.db: *** in database main ***; row 1 missing",
        ],
    },
    "notsqlite:hebbian": {
        "connects": [{"args": ["<TMP>/hebbian.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [[]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/hebbian.db: file is not a database",
        ],
    },
    "notok:hebbian": {
        "connects": [{"args": ["<TMP>/hebbian.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA integrity_check"]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/hebbian.db: *** in database main ***; row 1 missing",
        ],
    },
    "notsqlite:kindled": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [[]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/kindled.db: file is not a database",
        ],
    },
    "notok:kindled": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA integrity_check"]],
        "error": [
            "BrainIntegrityError",
            "integrity check failed for <TMP>/kindled.db: *** in database main ***; row 1 missing",
        ],
    },
    "walfail:memory": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 30000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "delete",
            "busy_timeout": 30000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "walfail:soul": {
        "connects": [{"args": ["<TMP>/soul.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "delete",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "walfail:hebbian": {
        "connects": [{"args": ["<TMP>/hebbian.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "delete",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "None",
            "isolation_level": "",
        },
    },
    "walfail:kindled": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            ["PRAGMA integrity_check", "PRAGMA journal_mode = WAL", "PRAGMA busy_timeout = 5000"]
        ],
        "error": None,
        "end_state": {
            "journal_mode": "delete",
            "busy_timeout": 5000,
            "foreign_keys": 0,
            "row_factory": "Row",
            "isolation_level": "",
        },
    },
    "walfail:works": {
        "connects": [{"args": ["<TMP>/works.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [["PRAGMA journal_mode = WAL"]],
        "error": ["OperationalError", "WAL not available"],
    },
}

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", OPEN_CASES, ids=_case_id)
def test_c1_c2_open_statements_connect_args_and_end_state(case, tmp_path, monkeypatch) -> None:
    kind, target, integrity = case
    snap = success_snapshot(kind, target, integrity, tmp_path, monkeypatch)
    assert snap == GOLDEN[f"open:{_case_id(case)}"]
    assert CLOSE not in [s for t in snap["traces"] for s in t]
    if integrity is False or kind == "works":
        assert "PRAGMA integrity_check" not in [s for t in snap["traces"] for s in t]


def test_c1b_memories_timeout_is_read_at_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dev_constants, "MEMORIES_DB_BUSY_TIMEOUT_S", 7.0)
    snap = success_snapshot("memory", "file", True, tmp_path, monkeypatch)
    assert snap["connects"][0]["kwargs"]["timeout"] == 7.0
    assert "PRAGMA busy_timeout = 7000" in snap["traces"][0]


@pytest.mark.parametrize("kind", CHECKED_KINDS)
def test_c3a_non_sqlite_file_raises_integrity_error(kind, tmp_path, monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("brain.health.integrity_retry.time.sleep", sleeps.append)
    rec, _, error, snap = failure_snapshot(kind, tmp_path, monkeypatch, None, _not_sqlite)
    assert snap == GOLDEN[f"notsqlite:{kind}"]
    assert type(error).__name__ == "BrainIntegrityError"
    assert sleeps == []
    assert CLOSE in rec.conns[0].trace


@pytest.mark.parametrize("kind", CHECKED_KINDS)
def test_c3b_not_ok_result_raises_and_closes(kind, tmp_path, monkeypatch) -> None:
    rec, _, error, snap = failure_snapshot(kind, tmp_path, monkeypatch, _not_ok)
    assert snap == GOLDEN[f"notok:{kind}"]
    assert type(error).__name__ == "BrainIntegrityError"
    assert str(error).endswith("*** in database main ***; row 1 missing")
    assert CLOSE in rec.conns[0].trace
    with pytest.raises(sqlite3.ProgrammingError):
        rec.conns[0].execute("SELECT 1")
    trace = rec.conns[0].trace
    assert not any("journal_mode" in s or "busy_timeout" in s for s in trace)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_c4_wal_policy(kind, tmp_path, monkeypatch) -> None:
    rec, conn, error, snap = failure_snapshot(kind, tmp_path, monkeypatch, _wal_fails)
    assert snap == GOLDEN[f"walfail:{kind}"]
    if kind == "works":
        assert isinstance(error, sqlite3.OperationalError)
    else:
        assert error is None
        assert conn is not None


@pytest.mark.parametrize("kind", CHECKED_KINDS)
def test_c3e_transient_retry_log_names_the_store(kind, tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setattr("brain.health.integrity_retry.time.sleep", lambda s: None)

    def configure(rec: Recorder) -> None:
        rec.fail_on("PRAGMA integrity_check", sqlite3.OperationalError("disk I/O error"), times=1)

    with caplog.at_level(logging.WARNING, logger="brain.health.integrity_retry"):
        _, conn, error, _ = failure_snapshot(kind, tmp_path, monkeypatch, configure)
    assert error is None and conn is not None
    messages = [r.getMessage() for r in caplog.records if "succeeded after retry" in r.getMessage()]
    assert len(messages) == 1
    assert CALLER_LABEL[kind] in messages[0]


def test_c10_integrity_cadence_signatures_are_unchanged() -> None:
    from brain.kindled_link.store import KindledLinkStore
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.soul.store import SoulStore

    for cls in (MemoryStore, HebbianMatrix, KindledLinkStore):
        p = inspect.signature(cls.__init__).parameters["integrity_check"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is True
    assert "integrity_check" not in inspect.signature(SoulStore.__init__).parameters


def test_the_path_token_normalisation_handles_windows_paths(tmp_path) -> None:
    rec = Recorder(str(tmp_path))
    assert rec.norm(f"oops at {tmp_path}/x.db") == "oops at <TMP>/x.db"
    windows = Recorder("C:\\Users\\ci\\tmp")
    assert windows.norm("C:\\Users\\ci\\tmp\\memory.db") == "<TMP>/memory.db"
    assert (
        windows.norm("integrity check failed for C:\\Users\\ci\\tmp\\a.db: not a database")
        == "integrity check failed for <TMP>/a.db: not a database"
    )


def test_the_trace_keeps_policy_pragmas_and_drops_schema_work() -> None:
    from types import SimpleNamespace

    conn = SimpleNamespace(
        trace=[
            "PRAGMA integrity_check",
            "\nCREATE TABLE a(x);",
            "create index idx on a(x)",
            "-- REPLACE INTO 'main'.'memories_fts_data'(id, block) VALUES(?,?)",
            "-- PRAGMA 'main'.data_version",
            "\n-- why\npragma foreign_keys = ON;",
            "PRAGMA table_info(a)",
            "PRAGMA main.index_list(a)",
            'PRAGMA "main".table_xinfo(a)',
            "PRAGMA user_version = 3",
            "PRAGMA synchronous = FULL",
            "ALTER TABLE a ADD COLUMN y",
            "UPDATE a SET x = 1",
            COMMIT,
            "PRAGMA journal_mode = WAL",
            CLOSE,
        ]
    )
    rec = Recorder("/tmp/x")
    assert rec.trace(conn) == [
        "PRAGMA integrity_check",
        "pragma foreign_keys = ON;",
        "PRAGMA synchronous = FULL",
        "PRAGMA journal_mode = WAL",
        CLOSE,
    ]
    assert rec.trace(conn, keep_close=False) == [
        "PRAGMA integrity_check",
        "pragma foreign_keys = ON;",
        "PRAGMA synchronous = FULL",
        "PRAGMA journal_mode = WAL",
    ]
