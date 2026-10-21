"""C9: the open sequence of the five SQLite stores, pinned statement for statement.

Probe: `sqlite3.connect` is wrapped so every connection is a `sqlite3.Connection`
subclass that records, in order, every statement it runs (through
`set_trace_callback`, which sees `execute`, `executescript` and cursors alike)
plus `<commit>` and `<close>` markers. The connect arguments and the number of
connect calls are recorded as well. Each store is opened on a fresh file (and on
`:memory:` where the store allows it) and the recording is compared with
GOLDEN, a set of literals captured from the unmodified stores (base f9f18bd3,
before the helper existed; main merged at a17ab2e6 changed no store open
sequence). The same file runs unchanged against the refactored stores.

Failure paths: a failed open now closes its connection (spec B3, #496), which
base did not. So on a failure path the golden holds the statement list with the
`<close>` markers removed, and the closure itself is asserted separately.

Left out of the recording, because they come from SQLite and not from the store: the
one-line "-- <SQL>" statements SQLite reports for its own FTS5 bookkeeping (see
SQLITE_OWN_STATEMENT; the set differs between SQLite builds). Paths are recorded with a
token for the tmp dir and forward slashes, so Windows runs match.

Statements longer than ABBREV characters (the schema DDL) are kept as their
first 40 characters, the length and a sha1 prefix of the whitespace-collapsed
text, so the golden stays readable and a one-character change still shows.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import re
import sqlite3
from pathlib import Path

import pytest

from brain import dev_constants

ABBREV = 80
PREFIX = 40
# SQLite itself reports some statements it runs for its own purposes (the shadow-table
# writes behind an FTS5 table, `PRAGMA data_version`, a `SELECT` of the FTS5 config) to
# the trace callback as ONE line that starts with "-- " and a SQL keyword. Which of them
# appear depends on the SQLite build (observed: this machine's 3.46.1 reports a
# `-- PRAGMA 'main'.data_version` that the SQLite on the CI runners does not). They are
# not statements the store issued, so they are left out. A statement of the store's own
# that starts with a comment (the schema DDL) always has the comment on its own line,
# so it is several lines and never matches.
SQLITE_OWN_STATEMENT = re.compile(
    r"-- (?:CREATE|DROP|ALTER|INSERT|REPLACE|DELETE|UPDATE|SELECT|PRAGMA|BEGIN|COMMIT|"
    r"ROLLBACK|SAVEPOINT|RELEASE) [^\n]*"
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
        out = []
        for item in conn.trace:
            if item == CLOSE and not keep_close:
                continue
            if SQLITE_OWN_STATEMENT.fullmatch(item):
                continue
            out.append(self.abbrev(item))
        return out

    def abbrev(self, stmt: str) -> str:
        text = " ".join(self.norm(stmt).split())
        if len(text) <= ABBREV:
            return text
        digest = hashlib.sha1(text.encode()).hexdigest()[:12]
        return f"{text[:PREFIX]}...[{len(text)}:{digest}]"


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


def snapshot(rec: Recorder, conn, error, *, failure: bool) -> dict:
    traces = []
    for c in rec.conns:
        t = rec.trace(c, keep_close=not failure)
        traces.append(sorted(t) if _is_pragma_set(t) else t)
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


def _is_pragma_set(trace: list[str]) -> bool:
    """A works `_connect` trace: only pragmas, compared as a set (spec B1)."""
    return bool(trace) and all(
        s.startswith("PRAGMA") and not s.startswith("PRAGMA integrity_check") for s in trace
    )


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
    return snapshot(rec, conn, error, failure=False)


def failure_snapshot(kind, tmp_path, monkeypatch, configure, prepare=None):
    rec, conn, error = run_case(
        kind, "file", True, tmp_path, monkeypatch, configure=configure, prepare=prepare
    )
    return rec, conn, error, snapshot(rec, conn, error, failure=True)


def _not_sqlite(path: str) -> None:
    Path(path).write_bytes(b"this is not a sqlite database " * 200)


def _not_ok(rec: Recorder) -> None:
    rec.integrity_result = NOT_OK_ROWS


def _wal_fails(rec: Recorder) -> None:
    rec.fail_on("PRAGMA journal_mode = WAL", sqlite3.OperationalError("WAL not available"))


def _case_id(case) -> str:
    return ":".join(str(x) for x in case)


# --------------------------------------------------------------------------
# GOLDEN: golden captured from base f9f18bd3 before the refactor (the unmodified
# stores), pasted as literals
# --------------------------------------------------------------------------

GOLDEN: dict[str, dict] = {
    "open:memory:file:True": {
        "connects": [{"args": ["<TMP>/memory.db"], "kwargs": {"timeout": 30.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 30000",
                "CREATE TABLE IF NOT EXISTS memories ( id...[654:214cc88e1235]",
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);",
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
                "-- F1 (#259): per-cluster centroid vecto...[774:bff0f089e1be]",
                "-- F2a (#250 inc4): real-query calibrati...[6590:5dc5a9416ee5]",
                "CREATE INDEX IF NOT EXISTS idx_calibrati...[89:a7d78ad56dc4]",
                '-- F2c inc8 (spec §3 "Retention vs the w...[1021:7f26758bffe7]',
                "-- F2a (#250 inc7): the DB-adaptive abst...[3015:758e9b06ee34]",
                "-- Name-recall fix R2 (spec §2, S18/S25/...[1070:64d11328530c]",
                "-- Cost series (name-recall fix S146, S1...[1096:a776bf830e8c]",
                "CREATE INDEX IF NOT EXISTS idx_cost_seri...[123:bcf7da8b11ed]",
                "CREATE TABLE IF NOT EXISTS cost_series_c...[277:953edfb5b3d6]",
                "-- `cost_series_verdicts` (S179): the S1...[742:17ce257dfdfa]",
                "-- F2c (inc2, red-team fix F-1, spec §2)...[1709:9d93cce1c961]",
                "-- F2c (inc3, spec §5): per-persona JUDG...[2257:d0a743b01608]",
                "-- External-content FTS5 shadow index (P...[346:8186965c720b]",
                "-- Tier-1 passive-recall salience: zero-...[421:6025bc9f6f82]",
                "-- Schema-level sync triggers: they live...[371:0d99e5f474c0]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[178:486cd7592085]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[263:f018e7f9b61c]",
                "PRAGMA table_info(memories)",
                "PRAGMA table_info(calibration_log)",
                "PRAGMA table_info(reranker_floor_calibration)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 30000",
                "CREATE TABLE IF NOT EXISTS memories ( id...[654:214cc88e1235]",
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);",
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
                "-- F1 (#259): per-cluster centroid vecto...[774:bff0f089e1be]",
                "-- F2a (#250 inc4): real-query calibrati...[6590:5dc5a9416ee5]",
                "CREATE INDEX IF NOT EXISTS idx_calibrati...[89:a7d78ad56dc4]",
                '-- F2c inc8 (spec §3 "Retention vs the w...[1021:7f26758bffe7]',
                "-- F2a (#250 inc7): the DB-adaptive abst...[3015:758e9b06ee34]",
                "-- Name-recall fix R2 (spec §2, S18/S25/...[1070:64d11328530c]",
                "-- Cost series (name-recall fix S146, S1...[1096:a776bf830e8c]",
                "CREATE INDEX IF NOT EXISTS idx_cost_seri...[123:bcf7da8b11ed]",
                "CREATE TABLE IF NOT EXISTS cost_series_c...[277:953edfb5b3d6]",
                "-- `cost_series_verdicts` (S179): the S1...[742:17ce257dfdfa]",
                "-- F2c (inc2, red-team fix F-1, spec §2)...[1709:9d93cce1c961]",
                "-- F2c (inc3, spec §5): per-persona JUDG...[2257:d0a743b01608]",
                "-- External-content FTS5 shadow index (P...[346:8186965c720b]",
                "-- Tier-1 passive-recall salience: zero-...[421:6025bc9f6f82]",
                "-- Schema-level sync triggers: they live...[371:0d99e5f474c0]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[178:486cd7592085]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[263:f018e7f9b61c]",
                "PRAGMA table_info(memories)",
                "PRAGMA table_info(calibration_log)",
                "PRAGMA table_info(reranker_floor_calibration)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "<commit>",
            ]
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
    "open:memory:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 30.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 30000",
                "CREATE TABLE IF NOT EXISTS memories ( id...[654:214cc88e1235]",
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);",
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
                "-- F1 (#259): per-cluster centroid vecto...[774:bff0f089e1be]",
                "-- F2a (#250 inc4): real-query calibrati...[6590:5dc5a9416ee5]",
                "CREATE INDEX IF NOT EXISTS idx_calibrati...[89:a7d78ad56dc4]",
                '-- F2c inc8 (spec §3 "Retention vs the w...[1021:7f26758bffe7]',
                "-- F2a (#250 inc7): the DB-adaptive abst...[3015:758e9b06ee34]",
                "-- Name-recall fix R2 (spec §2, S18/S25/...[1070:64d11328530c]",
                "-- Cost series (name-recall fix S146, S1...[1096:a776bf830e8c]",
                "CREATE INDEX IF NOT EXISTS idx_cost_seri...[123:bcf7da8b11ed]",
                "CREATE TABLE IF NOT EXISTS cost_series_c...[277:953edfb5b3d6]",
                "-- `cost_series_verdicts` (S179): the S1...[742:17ce257dfdfa]",
                "-- F2c (inc2, red-team fix F-1, spec §2)...[1709:9d93cce1c961]",
                "-- F2c (inc3, spec §5): per-persona JUDG...[2257:d0a743b01608]",
                "-- External-content FTS5 shadow index (P...[346:8186965c720b]",
                "-- Tier-1 passive-recall salience: zero-...[421:6025bc9f6f82]",
                "-- Schema-level sync triggers: they live...[371:0d99e5f474c0]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[178:486cd7592085]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[263:f018e7f9b61c]",
                "PRAGMA table_info(memories)",
                "PRAGMA table_info(calibration_log)",
                "PRAGMA table_info(reranker_floor_calibration)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 30000",
                "CREATE TABLE IF NOT EXISTS memories ( id...[654:214cc88e1235]",
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);",
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
                "-- F1 (#259): per-cluster centroid vecto...[774:bff0f089e1be]",
                "-- F2a (#250 inc4): real-query calibrati...[6590:5dc5a9416ee5]",
                "CREATE INDEX IF NOT EXISTS idx_calibrati...[89:a7d78ad56dc4]",
                '-- F2c inc8 (spec §3 "Retention vs the w...[1021:7f26758bffe7]',
                "-- F2a (#250 inc7): the DB-adaptive abst...[3015:758e9b06ee34]",
                "-- Name-recall fix R2 (spec §2, S18/S25/...[1070:64d11328530c]",
                "-- Cost series (name-recall fix S146, S1...[1096:a776bf830e8c]",
                "CREATE INDEX IF NOT EXISTS idx_cost_seri...[123:bcf7da8b11ed]",
                "CREATE TABLE IF NOT EXISTS cost_series_c...[277:953edfb5b3d6]",
                "-- `cost_series_verdicts` (S179): the S1...[742:17ce257dfdfa]",
                "-- F2c (inc2, red-team fix F-1, spec §2)...[1709:9d93cce1c961]",
                "-- F2c (inc3, spec §5): per-persona JUDG...[2257:d0a743b01608]",
                "-- External-content FTS5 shadow index (P...[346:8186965c720b]",
                "-- Tier-1 passive-recall salience: zero-...[421:6025bc9f6f82]",
                "-- Schema-level sync triggers: they live...[371:0d99e5f474c0]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[178:486cd7592085]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[263:f018e7f9b61c]",
                "PRAGMA table_info(memories)",
                "PRAGMA table_info(calibration_log)",
                "PRAGMA table_info(reranker_floor_calibration)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "<commit>",
            ]
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
    "open:soul:file:True": {
        "connects": [{"args": ["<TMP>/soul.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS crystallizati...[348:b11f8f297aff]",
                "CREATE INDEX IF NOT EXISTS idx_crystal_revoked ON crystallizations(revoked_at);",
                "CREATE INDEX IF NOT EXISTS idx_crystal_c...[84:a93c914e8614]",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS crystallizati...[348:b11f8f297aff]",
                "CREATE INDEX IF NOT EXISTS idx_crystal_revoked ON crystallizations(revoked_at);",
                "CREATE INDEX IF NOT EXISTS idx_crystal_c...[84:a93c914e8614]",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS hebbian_edges...[222:96a0de2b7939]",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS hebbian_edges...[222:96a0de2b7939]",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);",
                "<commit>",
            ]
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
    "open:hebbian:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS hebbian_edges...[222:96a0de2b7939]",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS hebbian_edges...[222:96a0de2b7939]",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);",
                "<commit>",
            ]
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
    "open:kindled:file:True": {
        "connects": [{"args": ["<TMP>/kindled.db"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS peers ( peer_...[234:151307d17705]",
                "CREATE TABLE IF NOT EXISTS consumed_invi...[102:268a03589040]",
                "CREATE TABLE IF NOT EXISTS seq_high_wate...[158:f4797bf2ab58]",
                "CREATE TABLE IF NOT EXISTS sessions ( pe...[257:b9ddcfee76ec]",
                "CREATE TABLE IF NOT EXISTS peer_counters...[167:a5b0207ce990]",
                "CREATE TABLE IF NOT EXISTS outbound_draf...[209:5a08b6aaf09d]",
                "CREATE TABLE IF NOT EXISTS transcript ( ...[241:9ea8e7bdf663]",
                "CREATE TABLE IF NOT EXISTS disclosure_bu...[122:8aeb0e5c44bb]",
                "CREATE TABLE IF NOT EXISTS relationship_...[276:ca5d525e8c81]",
                "CREATE TABLE IF NOT EXISTS peer_emotion_...[129:bb0e63259374]",
                "CREATE TABLE IF NOT EXISTS local_identit...[88:c944b0a7d24e]",
                "CREATE TABLE IF NOT EXISTS outbound_seq ...[149:63da61eeb1ea]",
                "CREATE TABLE IF NOT EXISTS pending_hands...[239:4eb923f0b9e8]",
                "CREATE TABLE IF NOT EXISTS session_keys ...[230:82d1f598c53c]",
                "CREATE TABLE IF NOT EXISTS pending_rotat...[135:2f740091a6b9]",
                "PRAGMA table_info(peers)",
                "PRAGMA table_info(peers)",
                "ALTER TABLE peers ADD COLUMN previous_identity_pub TEXT",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS peers ( peer_...[234:151307d17705]",
                "CREATE TABLE IF NOT EXISTS consumed_invi...[102:268a03589040]",
                "CREATE TABLE IF NOT EXISTS seq_high_wate...[158:f4797bf2ab58]",
                "CREATE TABLE IF NOT EXISTS sessions ( pe...[257:b9ddcfee76ec]",
                "CREATE TABLE IF NOT EXISTS peer_counters...[167:a5b0207ce990]",
                "CREATE TABLE IF NOT EXISTS outbound_draf...[209:5a08b6aaf09d]",
                "CREATE TABLE IF NOT EXISTS transcript ( ...[241:9ea8e7bdf663]",
                "CREATE TABLE IF NOT EXISTS disclosure_bu...[122:8aeb0e5c44bb]",
                "CREATE TABLE IF NOT EXISTS relationship_...[276:ca5d525e8c81]",
                "CREATE TABLE IF NOT EXISTS peer_emotion_...[129:bb0e63259374]",
                "CREATE TABLE IF NOT EXISTS local_identit...[88:c944b0a7d24e]",
                "CREATE TABLE IF NOT EXISTS outbound_seq ...[149:63da61eeb1ea]",
                "CREATE TABLE IF NOT EXISTS pending_hands...[239:4eb923f0b9e8]",
                "CREATE TABLE IF NOT EXISTS session_keys ...[230:82d1f598c53c]",
                "CREATE TABLE IF NOT EXISTS pending_rotat...[135:2f740091a6b9]",
                "PRAGMA table_info(peers)",
                "PRAGMA table_info(peers)",
                "ALTER TABLE peers ADD COLUMN previous_identity_pub TEXT",
                "<commit>",
            ]
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
    "open:kindled:memory:True": {
        "connects": [{"args": [":memory:"], "kwargs": {"timeout": 5.0}}],
        "traces": [
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS peers ( peer_...[234:151307d17705]",
                "CREATE TABLE IF NOT EXISTS consumed_invi...[102:268a03589040]",
                "CREATE TABLE IF NOT EXISTS seq_high_wate...[158:f4797bf2ab58]",
                "CREATE TABLE IF NOT EXISTS sessions ( pe...[257:b9ddcfee76ec]",
                "CREATE TABLE IF NOT EXISTS peer_counters...[167:a5b0207ce990]",
                "CREATE TABLE IF NOT EXISTS outbound_draf...[209:5a08b6aaf09d]",
                "CREATE TABLE IF NOT EXISTS transcript ( ...[241:9ea8e7bdf663]",
                "CREATE TABLE IF NOT EXISTS disclosure_bu...[122:8aeb0e5c44bb]",
                "CREATE TABLE IF NOT EXISTS relationship_...[276:ca5d525e8c81]",
                "CREATE TABLE IF NOT EXISTS peer_emotion_...[129:bb0e63259374]",
                "CREATE TABLE IF NOT EXISTS local_identit...[88:c944b0a7d24e]",
                "CREATE TABLE IF NOT EXISTS outbound_seq ...[149:63da61eeb1ea]",
                "CREATE TABLE IF NOT EXISTS pending_hands...[239:4eb923f0b9e8]",
                "CREATE TABLE IF NOT EXISTS session_keys ...[230:82d1f598c53c]",
                "CREATE TABLE IF NOT EXISTS pending_rotat...[135:2f740091a6b9]",
                "PRAGMA table_info(peers)",
                "PRAGMA table_info(peers)",
                "ALTER TABLE peers ADD COLUMN previous_identity_pub TEXT",
                "<commit>",
            ]
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
        "traces": [
            [
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS peers ( peer_...[234:151307d17705]",
                "CREATE TABLE IF NOT EXISTS consumed_invi...[102:268a03589040]",
                "CREATE TABLE IF NOT EXISTS seq_high_wate...[158:f4797bf2ab58]",
                "CREATE TABLE IF NOT EXISTS sessions ( pe...[257:b9ddcfee76ec]",
                "CREATE TABLE IF NOT EXISTS peer_counters...[167:a5b0207ce990]",
                "CREATE TABLE IF NOT EXISTS outbound_draf...[209:5a08b6aaf09d]",
                "CREATE TABLE IF NOT EXISTS transcript ( ...[241:9ea8e7bdf663]",
                "CREATE TABLE IF NOT EXISTS disclosure_bu...[122:8aeb0e5c44bb]",
                "CREATE TABLE IF NOT EXISTS relationship_...[276:ca5d525e8c81]",
                "CREATE TABLE IF NOT EXISTS peer_emotion_...[129:bb0e63259374]",
                "CREATE TABLE IF NOT EXISTS local_identit...[88:c944b0a7d24e]",
                "CREATE TABLE IF NOT EXISTS outbound_seq ...[149:63da61eeb1ea]",
                "CREATE TABLE IF NOT EXISTS pending_hands...[239:4eb923f0b9e8]",
                "CREATE TABLE IF NOT EXISTS session_keys ...[230:82d1f598c53c]",
                "CREATE TABLE IF NOT EXISTS pending_rotat...[135:2f740091a6b9]",
                "PRAGMA table_info(peers)",
                "PRAGMA table_info(peers)",
                "ALTER TABLE peers ADD COLUMN previous_identity_pub TEXT",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 30000",
                "CREATE TABLE IF NOT EXISTS memories ( id...[654:214cc88e1235]",
                "CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);",
                "CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(active);",
                "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);",
                "-- F1 (#259): per-cluster centroid vecto...[774:bff0f089e1be]",
                "-- F2a (#250 inc4): real-query calibrati...[6590:5dc5a9416ee5]",
                "CREATE INDEX IF NOT EXISTS idx_calibrati...[89:a7d78ad56dc4]",
                '-- F2c inc8 (spec §3 "Retention vs the w...[1021:7f26758bffe7]',
                "-- F2a (#250 inc7): the DB-adaptive abst...[3015:758e9b06ee34]",
                "-- Name-recall fix R2 (spec §2, S18/S25/...[1070:64d11328530c]",
                "-- Cost series (name-recall fix S146, S1...[1096:a776bf830e8c]",
                "CREATE INDEX IF NOT EXISTS idx_cost_seri...[123:bcf7da8b11ed]",
                "CREATE TABLE IF NOT EXISTS cost_series_c...[277:953edfb5b3d6]",
                "-- `cost_series_verdicts` (S179): the S1...[742:17ce257dfdfa]",
                "-- F2c (inc2, red-team fix F-1, spec §2)...[1709:9d93cce1c961]",
                "-- F2c (inc3, spec §5): per-persona JUDG...[2257:d0a743b01608]",
                "-- External-content FTS5 shadow index (P...[346:8186965c720b]",
                "-- Tier-1 passive-recall salience: zero-...[421:6025bc9f6f82]",
                "-- Schema-level sync triggers: they live...[371:0d99e5f474c0]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[178:486cd7592085]",
                "CREATE TRIGGER IF NOT EXISTS memories_ft...[263:f018e7f9b61c]",
                "PRAGMA table_info(memories)",
                "PRAGMA table_info(calibration_log)",
                "PRAGMA table_info(reranker_floor_calibration)",
                "CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS crystallizati...[348:b11f8f297aff]",
                "CREATE INDEX IF NOT EXISTS idx_crystal_revoked ON crystallizations(revoked_at);",
                "CREATE INDEX IF NOT EXISTS idx_crystal_c...[84:a93c914e8614]",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS hebbian_edges...[222:96a0de2b7939]",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);",
                "CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);",
                "<commit>",
            ]
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
            [
                "PRAGMA integrity_check",
                "PRAGMA journal_mode = WAL",
                "PRAGMA busy_timeout = 5000",
                "CREATE TABLE IF NOT EXISTS peers ( peer_...[234:151307d17705]",
                "CREATE TABLE IF NOT EXISTS consumed_invi...[102:268a03589040]",
                "CREATE TABLE IF NOT EXISTS seq_high_wate...[158:f4797bf2ab58]",
                "CREATE TABLE IF NOT EXISTS sessions ( pe...[257:b9ddcfee76ec]",
                "CREATE TABLE IF NOT EXISTS peer_counters...[167:a5b0207ce990]",
                "CREATE TABLE IF NOT EXISTS outbound_draf...[209:5a08b6aaf09d]",
                "CREATE TABLE IF NOT EXISTS transcript ( ...[241:9ea8e7bdf663]",
                "CREATE TABLE IF NOT EXISTS disclosure_bu...[122:8aeb0e5c44bb]",
                "CREATE TABLE IF NOT EXISTS relationship_...[276:ca5d525e8c81]",
                "CREATE TABLE IF NOT EXISTS peer_emotion_...[129:bb0e63259374]",
                "CREATE TABLE IF NOT EXISTS local_identit...[88:c944b0a7d24e]",
                "CREATE TABLE IF NOT EXISTS outbound_seq ...[149:63da61eeb1ea]",
                "CREATE TABLE IF NOT EXISTS pending_hands...[239:4eb923f0b9e8]",
                "CREATE TABLE IF NOT EXISTS session_keys ...[230:82d1f598c53c]",
                "CREATE TABLE IF NOT EXISTS pending_rotat...[135:2f740091a6b9]",
                "PRAGMA table_info(peers)",
                "PRAGMA table_info(peers)",
                "ALTER TABLE peers ADD COLUMN previous_identity_pub TEXT",
                "<commit>",
            ]
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


def test_the_path_token_normalisation_is_not_vacuous(tmp_path) -> None:
    rec = Recorder(str(tmp_path))
    assert rec.norm(f"oops at {tmp_path}/x.db") == "oops at <TMP>/x.db"
    windows = Recorder("C:\\Users\\ci\\tmp")
    assert windows.norm("C:\\Users\\ci\\tmp\\memory.db") == "<TMP>/memory.db"
    assert (
        windows.norm("integrity check failed for C:\\Users\\ci\\tmp\\a.db: not a database")
        == "integrity check failed for <TMP>/a.db: not a database"
    )
    assert re.fullmatch(r".{40}\.\.\.\[\d+:[0-9a-f]{12}\]", rec.abbrev("x" * 300))


def test_sqlite_own_trace_lines_are_dropped_and_the_stores_own_are_not() -> None:
    own = [
        "-- REPLACE INTO 'main'.'memories_fts_data'(id, block) VALUES(?,?)",
        "-- PRAGMA 'main'.data_version",
        "-- SELECT k, v FROM 'main'.'memories_fts_config'",
        "-- CREATE TABLE 'main'.'memories_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
    ]
    theirs = [
        "-- F1 (#259): per-cluster centroids\nCREATE TABLE IF NOT EXISTS t (x)",
        "PRAGMA journal_mode = WAL",
        "CREATE INDEX IF NOT EXISTS idx ON t(x);",
    ]
    assert all(SQLITE_OWN_STATEMENT.fullmatch(s) for s in own)
    assert not any(SQLITE_OWN_STATEMENT.fullmatch(s) for s in theirs)
