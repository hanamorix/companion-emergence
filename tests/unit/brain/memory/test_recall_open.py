"""Unit tests for the one `open_memory` door (#231 recall-bump consolidation).

Covers the semantics of a single open event: the +1.0 recall bump together with the `last_accessed_at` touch (every full
open is an access), the persona_dir-gated reappraisal enqueue (reconciliation C), the `seen`-based per-pass dedup
(reconciliation D) and the persona_dir-gated report of the showing to the emotion engine. The older tests fail against
4c916b24 because `open_memory` (and its module) does not exist there.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory.pending import PendingQueue
from brain.memory.recall_open import open_memory
from brain.memory.store import Memory, MemoryStore


def _rc(store: MemoryStore, mid: str) -> float:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _la(store: MemoryStore, mid: str):
    return store._conn.execute(  # noqa: SLF001
        "SELECT last_accessed_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _reappraisal_ids(rows: list[dict]) -> list[str]:
    return [r["memory_id"] for r in rows if r.get("_route") == "reappraise_importance"]


def _seed(store: MemoryStore) -> Memory:
    m = Memory.create_new(content="a memory to open", memory_type="event", domain="d")
    store.create(m)
    return m


def test_open_bumps_recall_and_sets_last_accessed(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)
    assert _la(store, m.id) is None  # freshly created, never accessed

    open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _la(store, m.id) is not None  # any full open is a genuine access


def test_enqueue_fires_when_persona_dir_set(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)

    open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(m.id) == 1


def test_enqueue_skipped_when_persona_dir_none(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)

    # No persona_dir -> no PendingQueue to build; must bump recall_count but not
    # raise and not write any queue.
    open_memory(m, store=store, persona_dir=None, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _la(store, m.id) is not None  # the legacy path sets last_accessed_at too
    assert not (tmp_path / "pending_candidates.jsonl").exists()


def test_seen_dedups_within_one_pass(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)
    seen: set[str] = set()

    open_memory(m, store=store, persona_dir=tmp_path, seen=seen)
    open_memory(m, store=store, persona_dir=tmp_path, seen=seen)

    # Second call is a no-op: bumped once, enqueued once.
    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _reappraisal_ids(PendingQueue(tmp_path).drain()).count(m.id) == 1


def test_no_seen_opens_each_call(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)

    # seen=None -> each call is its own open event (per-call behaviour).
    open_memory(m, store=store, persona_dir=tmp_path, seen=None)
    open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    # Without a seen-set each call is its own open event: recall_count is bumped for EVERY call (the property this test protects).
    assert _rc(store, m.id) == pytest.approx(2.0)
    # The memory is queued for re-appraisal, and the gate takes it ONCE: each open appends a line (the hot path reads nothing, #470
    # review item 4), and `claim()` collapses the repeats per memory id so an outage cannot pile them up. One item is all the gate needs.
    assert _reappraisal_ids(PendingQueue(tmp_path).claim().entries).count(m.id) == 1


# ---------------------------------------------------------------------------
# `pending_ids` collector (restored batching, #231 follow-up): a passive
# render pass can defer the enqueue to a caller-owned list instead of writing
# the queue file immediately, so N full-opens in one turn cost one flush
# instead of N file-lock/open/append syscalls.
# ---------------------------------------------------------------------------


def test_pending_ids_collector_defers_enqueue_instead_of_writing_immediately(
    tmp_path: Path,
) -> None:
    """With a `pending_ids` collector supplied, `open_memory` still bumps
    recall_count immediately, but APPENDS the id to the collector instead of
    writing the pending-candidates queue file — nothing is enqueued until the
    caller flushes the collector itself."""
    store = MemoryStore(":memory:")
    m = _seed(store)
    pending_ids: list[str] = []

    open_memory(m, store=store, persona_dir=tmp_path, seen=None, pending_ids=pending_ids)

    assert _rc(store, m.id) == pytest.approx(1.0)  # bump still happens immediately
    assert pending_ids == [m.id]  # id landed in the collector...
    assert not (tmp_path / "pending_candidates.jsonl").exists()  # ...not the queue file

    # The caller's own flush is what actually enqueues it.
    written = PendingQueue(tmp_path).enqueue_reappraisals(pending_ids, source="recall")
    assert written == 1
    assert _reappraisal_ids(PendingQueue(tmp_path).drain()) == [m.id]


def test_pending_ids_collector_batches_multiple_opens_into_one_list(tmp_path: Path) -> None:
    """Several full-opens sharing one `pending_ids` collector across one
    passive pass all land in the SAME list, ready for one combined
    `enqueue_reappraisals` flush — the shape the restored batching depends on."""
    store = MemoryStore(":memory:")
    m1 = _seed(store)
    m2 = _seed(store)
    pending_ids: list[str] = []
    seen: set[str] = set()

    open_memory(m1, store=store, persona_dir=tmp_path, seen=seen, pending_ids=pending_ids)
    open_memory(m2, store=store, persona_dir=tmp_path, seen=seen, pending_ids=pending_ids)

    assert pending_ids == [m1.id, m2.id]
    assert not (tmp_path / "pending_candidates.jsonl").exists()


def test_open_without_collector_still_enqueues_immediately(tmp_path: Path) -> None:
    """The `read_full_memory` path never passes `pending_ids` (it
    opens exactly one memory per call), so it must keep enqueuing immediately,
    unaffected by the new collector parameter."""
    store = MemoryStore(":memory:")
    m = _seed(store)

    open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    assert _reappraisal_ids(PendingQueue(tmp_path).drain()) == [m.id]


# ---------------------------------------------------------------------------
# The report of the showing to the emotion engine.
# ---------------------------------------------------------------------------


def test_open_reports_each_full_open_to_the_engine_once(
    tmp_path: Path, emotion_engine_stub
) -> None:
    store = MemoryStore(":memory:")

    def shown_ids() -> list[str]:
        return [c[1].id for c in emotion_engine_stub.recorded_calls if c[0] == "on_memory_shown"]

    # (1) no collector (read_full_memory's shape): one report, with that memory and a UTC time inside the call
    m1 = _seed(store)
    before = datetime.now(UTC)
    open_memory(m1, store=store, persona_dir=tmp_path, seen=None)
    after = datetime.now(UTC)
    assert shown_ids() == [m1.id]
    shown = [c for c in emotion_engine_stub.recorded_calls if c[0] == "on_memory_shown"]
    at = shown[0][2]
    assert at.tzinfo is not None and at.utcoffset() == timedelta(0)  # a naive or None time fails
    assert before <= at <= after

    # (2) the recall block's shape: a collector is passed, the id lands in it, and the open is still reported
    m2 = _seed(store)
    collector: list[str] = []
    open_memory(m2, store=store, persona_dir=tmp_path, seen=None, pending_ids=collector)
    assert collector == [m2.id]
    assert shown_ids() == [m1.id, m2.id]

    # (3) the same id with the same `seen` is not reported twice
    m3 = _seed(store)
    seen: set[str] = set()
    open_memory(m3, store=store, persona_dir=tmp_path, seen=seen)
    open_memory(m3, store=store, persona_dir=tmp_path, seen=seen)
    assert shown_ids().count(m3.id) == 1

    # (4) no persona dir (the legacy path): nothing is reported
    m4 = _seed(store)
    open_memory(m4, store=store, persona_dir=None, seen=None)
    assert m4.id not in shown_ids()


def test_open_queues_a_showing_in_the_real_engine(tmp_path: Path) -> None:
    """No stub: the door's report reaches the real engine and one `showing` event for the memory is queued."""
    from brain.emotion import engine_files

    store = MemoryStore(":memory:")  # a file-backed store would also queue a `commit` event
    m = Memory.create_new(
        content="a memory the add_memory tool wrote",
        memory_type="event",
        domain="d",
        emotions={"love": 6.0},
        metadata={"tags": ["t"], "importance": 6.0},
    )
    store.create(m)

    open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    # No state file is ever saved for this persona, so the stored position is the default `queue_line`, 0.
    queued = engine_files.read_queue(tmp_path, from_line=0)
    showings = [e for e in queued.events if e.kind == "showing"]
    assert [e.memory_id for e in showings] == [m.id]
    assert queued.end_line == 1  # one complete line was written


def test_open_survives_an_engine_failure_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raise from the engine report never escapes the door: the bump and the enqueue are already done, and it is logged."""
    from brain.emotion import engine

    def boom(persona_dir: Path):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(engine, "engine_for", boom)
    store = MemoryStore(":memory:")
    m = _seed(store)

    with caplog.at_level(logging.ERROR, logger="brain.memory.recall_open"):
        open_memory(m, store=store, persona_dir=tmp_path, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _la(store, m.id) is not None
    assert _reappraisal_ids(PendingQueue(tmp_path).drain()) == [m.id]
    errors = [
        r
        for r in caplog.records
        if r.name == "brain.memory.recall_open" and r.levelno >= logging.ERROR
    ]
    assert len(errors) == 1
