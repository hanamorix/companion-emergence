"""Cost-series increment (name-recall fix S146, S164): the `cost_series_flush`
gated job (criterion CS17b) and the bridge's clean-stop flush (CS17e).

The job table, the central function and the real `cost_series` buffer are
real; the persona is a tmp directory with a real memories.db. Synthetic data
only.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime

import pytest

from brain.bridge import central_cadence, supervisor
from brain.bridge.central_cadence import GATED_JOB_ORDER, JobOutcome
from brain.memory import cost_series
from brain.memory.store import MemoryStore
from tests.unit.brain.bridge.test_central_cadence import _persona, _real_jobs


def _job(persona_dir, name="cost_series_flush"):
    return {j.name: j for j in _real_jobs(persona_dir)}[name]


def _job_with(persona_dir, **overrides):
    return {j.name: j for j in _real_jobs(persona_dir, **overrides)}["cost_series_flush"]


def _rows(persona_dir) -> int:
    con = sqlite3.connect(persona_dir / "memories.db")
    try:
        return con.execute("SELECT COUNT(*) FROM cost_series_samples").fetchone()[0]
    finally:
        con.close()


def _buffer(persona_dir, n: int = 3) -> MemoryStore:
    store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
    series = cost_series.CostSeries("rerank")
    for i in range(1, n + 1):
        series.record(store, "m", 1000 * i, 0.1 * i)
    return store


def test_cs17b_the_flush_job_follows_the_remeasure_job_in_the_order() -> None:
    i = GATED_JOB_ORDER.index("rerank_width_remeasure")
    assert GATED_JOB_ORDER[i + 1] == "cost_series_flush"


def test_cs17b_due_iff_the_buffer_holds_samples_for_this_personas_database(tmp_path) -> None:
    persona = _persona(tmp_path)
    job = _job(persona)
    assert job.has_work() is False
    store = _buffer(persona)
    try:
        assert job.has_work() is True
        other = tmp_path / "another-persona"
        other.mkdir()
        assert _job(other).has_work() is False, "samples are keyed by the database path"
    finally:
        store.close()


def test_cs17b_the_run_writes_the_samples_with_the_tick_store_and_empties_the_buffer(
    tmp_path, monkeypatch
) -> None:
    persona = _persona(tmp_path)
    store = _buffer(persona, 3)
    try:
        assert _rows(persona) == 0
        seen: list[object] = []
        real_flush = cost_series.flush
        monkeypatch.setattr(cost_series, "flush", lambda s: seen.append(s) or real_flush(s))
        assert _job(persona).run() is JobOutcome.COMPLETED
        assert len(seen) == 1 and isinstance(seen[0], MemoryStore)
        assert _rows(persona) == 3
        assert not cost_series.has_pending(persona / "memories.db")
    finally:
        store.close()


def test_cs17b_the_run_uses_the_ticks_shared_store_when_there_is_one(tmp_path, monkeypatch) -> None:
    persona = _persona(tmp_path)
    shared = _buffer(persona, 2)
    try:
        seen: list[object] = []
        real_flush = cost_series.flush
        monkeypatch.setattr(cost_series, "flush", lambda s: seen.append(s) or real_flush(s))
        job = _job_with(persona, tick_ctx={"store": shared})
        assert job.run() is JobOutcome.COMPLETED
        assert seen == [shared]
        assert _rows(persona) == 2
    finally:
        shared.close()


def test_cs17b_a_failing_write_keeps_the_samples_for_the_next_lull(tmp_path, monkeypatch) -> None:
    persona = _persona(tmp_path)
    store = _buffer(persona, 2)
    try:
        monkeypatch.setattr(
            MemoryStore,
            "record_cost_samples",
            lambda self, *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
        )
        assert _job(persona).run() is JobOutcome.COMPLETED
        assert cost_series.has_pending(persona / "memories.db")
        monkeypatch.undo()
        assert _job(persona).run() is JobOutcome.COMPLETED
        assert _rows(persona) == 2
    finally:
        store.close()


def test_cs17b_a_buffered_sample_older_than_the_window_is_dropped_not_written(tmp_path, monkeypatch) -> None:
    persona = _persona(tmp_path)
    store = _buffer(persona, 2)
    try:
        real = cost_series._wall_clock
        monkeypatch.setattr(cost_series, "_wall_clock", lambda: real() + cost_series.window_seconds() + 10.0)
        assert _job(persona).run() is JobOutcome.COMPLETED
        assert _rows(persona) == 0
        assert not cost_series.has_pending(persona / "memories.db")
    finally:
        store.close()


def test_cs17b_the_whole_pass_saves_the_remeasure_jobs_sample_in_the_same_lull(tmp_path, monkeypatch) -> None:
    persona = _persona(tmp_path)
    store = _buffer(persona, 1)
    try:
        ran: list[str] = []
        real = central_cadence.run_central_pass
        jobs = [
            j
            for j in _real_jobs(persona)
            if j.name in {"rerank_width_remeasure", "cost_series_flush"}
        ]
        monkeypatch.setattr(supervisor.reranker, "stuck_width_due", lambda _m, _f: True)
        monkeypatch.setattr(
            supervisor.reranker,
            "remeasure_stuck_width",
            lambda _m, s: ran.append("remeasure")
            or cost_series.CostSeries("rerank").record(s, "m", 5000, 0.5)
            or True,
        )
        real(
            persona,
            jobs,
            is_idle=lambda: True,
            slot_available=lambda: True,
            now_wall=lambda: datetime.now(UTC),
        )
        assert ran == ["remeasure"]
        assert _rows(persona) == 2, "the re-measure's sample and the earlier one are both saved"
    finally:
        store.close()


@pytest.mark.parametrize("fail", [False, True])
def test_cs17e_the_bridge_teardown_flushes_after_the_supervisor_stops(tmp_path, monkeypatch, fail) -> None:
    """The lifespan teardown (server.py) saves the buffered samples once the
    supervisor thread has stopped, and a failing flush does not stop the
    teardown."""
    from brain.bridge import server

    persona = _persona(tmp_path)
    store = _buffer(persona, 2)
    try:
        order: list[str] = []
        real_flush = cost_series.flush

        def flush(s):
            order.append("flush")
            if fail:
                raise RuntimeError("flush failed")
            return real_flush(s)

        monkeypatch.setattr(cost_series, "flush", flush)
        asyncio.run(server._flush_cost_series_at_stop(persona))
        assert order == ["flush"]
        assert _rows(persona) == (0 if fail else 2)
    finally:
        store.close()
