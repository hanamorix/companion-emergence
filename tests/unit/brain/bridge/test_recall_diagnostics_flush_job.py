"""Recall diagnostics (name-recall fix S189): the `recall_diagnostics_flush` gated
job, and the two clean-stop flushes (the bridge's stop, `nell chat --no-bridge`'s
exit). The job table is real; the persona is a tmp directory; synthetic data only."""

from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime

from brain import cli
from brain.bridge import server
from brain.bridge.central_cadence import GATED_JOB_ORDER, JobOutcome
from brain.memory import recall_diagnostics as rd
from brain.memory.store import MemoryStore
from tests.unit.brain.bridge.test_central_cadence import _persona, _real_jobs


def _job(persona_dir):
    return {j.name: j for j in _real_jobs(persona_dir)}["recall_diagnostics_flush"]


def _rec(n: int) -> dict:
    return {"ts": datetime.now(UTC).isoformat(), "source": "passive", "n": n}


def _lines(persona_dir) -> list[int]:
    path = rd.diagnostics_path(persona_dir)
    if not path.exists():
        return []
    return [json.loads(x)["n"] for x in path.read_text().splitlines() if x.strip()]


def _buffer_one(persona_dir, n: int = 1) -> None:
    """Make one append meet a held lock (in a thread so a blocking append fails, not hangs)."""
    path = rd.diagnostics_path(persona_dir)
    with rd.file_lock(path):
        t = threading.Thread(target=rd.append_record, args=(path, _rec(n)), daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
    # the lock was released after the append buffered; make sure the retry did not already write it
    assert rd.has_pending(path) or _lines(persona_dir) == [n]


def test_the_flush_job_follows_the_cost_series_flush_in_the_order() -> None:
    i = GATED_JOB_ORDER.index("cost_series_flush")
    assert GATED_JOB_ORDER[i + 1] == "recall_diagnostics_flush"


def test_due_iff_records_are_buffered_for_this_personas_log(tmp_path) -> None:
    persona = _persona(tmp_path)
    job = _job(persona)
    assert job.has_work() is False
    path = rd.diagnostics_path(persona)
    with rd.file_lock(path):
        t = threading.Thread(target=rd.append_record, args=(path, _rec(1)), daemon=True)
        t.start()
        t.join(timeout=5)
        assert job.has_work() is True
        other = tmp_path / "another-persona"
        other.mkdir()
        assert _job(other).has_work() is False  # keyed by the log path
    assert job.run() is JobOutcome.COMPLETED
    assert _lines(persona) == [1]
    assert job.has_work() is False


def test_the_run_leaves_the_records_buffered_while_the_lock_is_still_held(tmp_path) -> None:
    persona = _persona(tmp_path)
    path = rd.diagnostics_path(persona)
    with rd.file_lock(path):
        t = threading.Thread(target=rd.append_record, args=(path, _rec(1)), daemon=True)
        t.start()
        t.join(timeout=5)
        done = []
        r = threading.Thread(target=lambda: done.append(_job(persona).run()), daemon=True)
        r.start()
        r.join(timeout=5)
        assert not r.is_alive(), "the lull job waited on the held lock"
        assert rd.has_pending(path)  # still buffered: the next lull retries
        assert _lines(persona) == []
    assert rd.flush_pending(path) == 1


def test_the_bridge_clean_stop_writes_the_buffered_records(tmp_path) -> None:
    persona = _persona(tmp_path)
    path = rd.diagnostics_path(persona)
    with rd.file_lock(path):
        t = threading.Thread(target=rd.append_record, args=(path, _rec(7)), daemon=True)
        t.start()
        t.join(timeout=5)
    assert rd.has_pending(path) or _lines(persona) == [7]
    asyncio.run(server._flush_cost_series_at_stop(persona))
    assert _lines(persona) == [7]
    assert not rd.has_pending(path)


def test_the_no_bridge_exit_writes_the_buffered_records_and_never_raises(tmp_path, monkeypatch) -> None:
    persona = _persona(tmp_path)
    path = rd.diagnostics_path(persona)
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        with rd.file_lock(path):
            t = threading.Thread(target=rd.append_record, args=(path, _rec(9)), daemon=True)
            t.start()
            t.join(timeout=5)
        cli._flush_cost_series_at_exit(store)
        assert _lines(persona) == [9]
        monkeypatch.setattr(rd, "flush_pending", lambda _p: (_ for _ in ()).throw(RuntimeError("x")))
        cli._flush_cost_series_at_exit(store)  # a failing diagnostics flush does not break the exit
    finally:
        store.close()
