"""#460 (+ the same defect on a second read): the soul-candidate repair must not record itself complete after a read it could not do.

`_repair_locked` marked the repair done when its read of the candidates file failed, so a transient I/O fault retired the repair
for good with the candidates unrepaired; and `_fetch_content` swallowed every `store.get` failure, so a transient database fault
made the repair EXPIRE the candidate and record complete. Decided by cause, as in the consolidation gate (`failure_scope`): a fault of
the environment (I/O, a locked or closed store) aborts the run with nothing saved and no state file, to be retried at the next start;
a per-record fault leaves THAT record exactly as it was and the rest are processed; if every target failed that way the run is not
recorded as complete (all rows failing is systemic).

Synthetic names only.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

import brain.health.soul_candidate_repair as repair
from brain.health.json_io import JsonFileError, Recovery
from brain.health.soul_candidate_repair import (
    run_soul_candidate_repair,
    should_run_soul_candidate_repair,
)
from brain.memory.store import Memory, MemoryStore

_CANDIDATES = "soul_candidates.jsonl"
_STATE = "soul_candidate_repair_state.json"


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)


def _memory(store: MemoryStore, content: str) -> Memory:
    mem = Memory.create_new(content=content, memory_type="monologue_soul_candidate", domain="interior", emotions={})
    store.create(mem)
    return mem


def _target(memory_id: str, text: str = "Ordinary trust") -> dict:
    return {
        "memory_id": memory_id,
        "text": text,
        "label": "monologue_soul_candidate",
        "importance": 8,
        "session_id": "monologue",
        "queued_at": "2026-06-01T10:00:00+00:00",
        "status": "auto_pending",
        "defer_count": 5,
        "last_deferred_at": "2026-06-10T08:00:00+00:00",
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / _CANDIDATES
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _read(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / _CANDIDATES).read_text(encoding="utf-8").splitlines() if line.strip()]


def _raise_on_the_raise_read(monkeypatch, exc_factory):
    """Fail ONLY the RAISE read inside `_repair_locked`; `should_run`'s default-policy read is untouched."""
    real = repair.read_jsonl

    def read_jsonl(path, *args, **kwargs):
        if kwargs.get("recover") is Recovery.RAISE:
            raise exc_factory(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(repair, "read_jsonl", read_jsonl)


def _fail_get(monkeypatch, store, match, exc):
    real = store.get

    def get(memory_id, *a, **k):
        if match(memory_id):
            raise exc
        return real(memory_id, *a, **k)

    monkeypatch.setattr(store, "get", get)


# ---------------------------------------------------------------------------
# the candidates read
# ---------------------------------------------------------------------------


def test_a_failed_read_leaves_the_repair_pending(tmp_path, monkeypatch, caplog):
    # C460-a
    store = _store(tmp_path)
    try:
        mem = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
        path = _write(tmp_path, [_target(mem.id)])
        before = path.read_bytes()
        assert should_run_soul_candidate_repair(tmp_path) is True
        _raise_on_the_raise_read(monkeypatch, lambda p: JsonFileError(p, "io", "denied"))

        with caplog.at_level(logging.WARNING):
            report = run_soul_candidate_repair(tmp_path, store=store)

        assert report.status == "skipped" and report.repaired == 0 and report.expired == 0
        assert not (tmp_path / _STATE).exists()
        assert should_run_soul_candidate_repair(tmp_path) is True
        assert path.read_bytes() == before
        assert any(_CANDIDATES in r.getMessage() and "denied" in r.getMessage() for r in caplog.records)

        monkeypatch.undo()  # C460-b: once readable, the next run repairs the target and completes
        report = run_soul_candidate_repair(tmp_path, store=store)

        assert report.status == "complete" and report.repaired == 1
        assert json.loads((tmp_path / _STATE).read_text())["status"] == "complete"
    finally:
        store.close()


def test_a_bug_in_the_read_is_not_swallowed(tmp_path, monkeypatch):
    # C460-a: only JsonFileError (the one thing the RAISE read raises) is handled.
    store = _store(tmp_path)
    try:
        mem = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
        _write(tmp_path, [_target(mem.id)])
        _raise_on_the_raise_read(monkeypatch, lambda p: RuntimeError("a bug"))

        with pytest.raises(RuntimeError, match="a bug"):
            run_soul_candidate_repair(tmp_path, store=store)
        assert not (tmp_path / _STATE).exists()
    finally:
        store.close()


# ---------------------------------------------------------------------------
# the memory read (store.get)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exc", [sqlite3.OperationalError("database is locked"), OSError("disk I/O error")],
                         ids=["sqlite-locked", "os-error"])
def test_an_environment_fault_on_the_memory_read_aborts_the_run_with_nothing_saved(tmp_path, monkeypatch, exc):
    # C460-d
    store = _store(tmp_path)
    try:
        mem = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
        path = _write(tmp_path, [_target(mem.id)])
        before = path.read_bytes()
        _fail_get(monkeypatch, store, lambda mid: mid == mem.id, exc)

        with pytest.raises(type(exc)):
            run_soul_candidate_repair(tmp_path, store=store)

        assert path.read_bytes() == before and not (tmp_path / _STATE).exists()
        monkeypatch.undo()
        report = run_soul_candidate_repair(tmp_path, store=store)
        assert report.status == "complete" and report.repaired == 1
    finally:
        store.close()


def test_a_per_record_fault_leaves_that_record_alone_and_the_rest_are_processed(tmp_path, monkeypatch, caplog):
    # C460-e
    store = _store(tmp_path)
    try:
        bad = _memory(store, "Bad row content")
        good = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
        _write(tmp_path, [_target(bad.id, "First fragment"), _target(good.id)])
        _fail_get(monkeypatch, store, lambda mid: mid == bad.id, ValueError("a row that cannot be read"))

        with caplog.at_level(logging.INFO):
            report = run_soul_candidate_repair(tmp_path, store=store)

        first, second = _read(tmp_path)
        assert first["status"] == "auto_pending" and first["text"] == "First fragment" and first["defer_count"] == 5
        assert second["defer_count"] == 0 and "goodnight" in second["text"]
        assert report.status == "complete" and report.repaired == 1 and report.expired == 0
        assert json.loads((tmp_path / _STATE).read_text())["status"] == "complete"
        assert sum(bad.id in r.getMessage() for r in caplog.records) == 1  # the record id, once
    finally:
        store.close()


def test_a_genuinely_missing_memory_row_still_expires_the_record(tmp_path):
    # C460-e PIN: `store.get` returning None is "no recoverable context", as before.
    store = _store(tmp_path)
    try:
        _write(tmp_path, [_target("no-such-memory-id")])

        report = run_soul_candidate_repair(tmp_path, store=store)

        assert report.expired == 1 and _read(tmp_path)[0]["status"] == "expired"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# a store that is closed or on the wrong thread hits EVERY record
# ---------------------------------------------------------------------------


def test_a_closed_store_aborts_the_run_and_records_nothing(tmp_path):
    # C460-f(1): REAL fault: `ProgrammingError("Cannot operate on a closed database")` for every row.
    store = _store(tmp_path)
    mem = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
    path = _write(tmp_path, [_target(mem.id)])
    before = path.read_bytes()
    store.close()

    with pytest.raises(sqlite3.ProgrammingError):
        run_soul_candidate_repair(tmp_path, store=store)

    assert path.read_bytes() == before and not (tmp_path / _STATE).exists()


def test_a_wrong_thread_store_aborts_the_run_and_records_nothing(tmp_path):
    # C460-f(1): REAL fault: a store used from another thread.
    import threading

    store = _store(tmp_path)
    try:
        mem = _memory(store, "Ordinary trust, she let the goodnight be ordinary")
        path = _write(tmp_path, [_target(mem.id)])
        before = path.read_bytes()
        outcome: dict = {}

        def run():
            try:
                run_soul_candidate_repair(tmp_path, store=store)
            except sqlite3.ProgrammingError as exc:
                outcome["exc"] = exc

        t = threading.Thread(target=run)
        t.start()
        t.join()

        assert "exc" in outcome
        assert path.read_bytes() == before and not (tmp_path / _STATE).exists()
    finally:
        store.close()


def test_every_target_failing_per_record_does_not_record_completion(tmp_path, monkeypatch, caplog):
    # C460-f(2): item faults are per row; all rows failing on a healthy store is systemic.
    store = _store(tmp_path)
    try:
        one, two = _memory(store, "one"), _memory(store, "two")
        path = _write(tmp_path, [_target(one.id, "A"), _target(two.id, "B")])
        before = path.read_bytes()
        _fail_get(monkeypatch, store, lambda mid: mid in (one.id, two.id), ValueError("unreadable row"))

        with caplog.at_level(logging.WARNING):
            report = run_soul_candidate_repair(tmp_path, store=store)

        assert path.read_bytes() == before
        assert not (tmp_path / _STATE).exists() and report.status == "skipped"
        assert any("2" in r.getMessage() and "target" in r.getMessage() for r in caplog.records)
        assert should_run_soul_candidate_repair(tmp_path) is True
    finally:
        store.close()
