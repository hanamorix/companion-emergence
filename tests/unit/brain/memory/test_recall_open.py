"""Unit tests for the one `open_memory` door (#231 recall-bump consolidation).

Covers the four semantics of a single open event: the always-+1.0 recall bump,
the deliberate-only `last_accessed_at` touch (reconciliation B), the
persona_dir-gated reappraisal enqueue (reconciliation C), and the `seen`-based
per-pass dedup (reconciliation D). These fail against 4c916b24 because
`open_memory` (and its module) does not exist there.
"""

from __future__ import annotations

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


def test_passive_open_bumps_recall_but_not_last_accessed(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)
    assert _la(store, m.id) is None  # freshly created, never accessed

    open_memory(m, store=store, persona_dir=tmp_path, deliberate=False, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _la(store, m.id) is None  # passive surface is NOT a genuine access


def test_deliberate_open_bumps_recall_and_touches_last_accessed(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)
    assert _la(store, m.id) is None

    open_memory(m, store=store, persona_dir=tmp_path, deliberate=True, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _la(store, m.id) is not None  # deliberate read IS a genuine access


def test_enqueue_fires_on_both_modes_when_persona_dir_set(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    passive_mem = _seed(store)
    deliberate_mem = _seed(store)

    open_memory(passive_mem, store=store, persona_dir=tmp_path, deliberate=False, seen=None)
    open_memory(deliberate_mem, store=store, persona_dir=tmp_path, deliberate=True, seen=None)

    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(passive_mem.id) == 1
    assert ids.count(deliberate_mem.id) == 1


def test_enqueue_skipped_when_persona_dir_none(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)

    # No persona_dir -> no PendingQueue to build; must bump recall_count but not
    # raise and not write any queue.
    open_memory(m, store=store, persona_dir=None, deliberate=False, seen=None)

    assert _rc(store, m.id) == pytest.approx(1.0)
    assert not (tmp_path / "pending_candidates.jsonl").exists()


def test_seen_dedups_within_one_pass(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)
    seen: set[str] = set()

    open_memory(m, store=store, persona_dir=tmp_path, deliberate=False, seen=seen)
    open_memory(m, store=store, persona_dir=tmp_path, deliberate=False, seen=seen)

    # Second call is a no-op: bumped once, enqueued once.
    assert _rc(store, m.id) == pytest.approx(1.0)
    assert _reappraisal_ids(PendingQueue(tmp_path).drain()).count(m.id) == 1


def test_no_seen_opens_each_call(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    m = _seed(store)

    # seen=None -> each call is its own open event (per-call behaviour).
    open_memory(m, store=store, persona_dir=tmp_path, deliberate=False, seen=None)
    open_memory(m, store=store, persona_dir=tmp_path, deliberate=False, seen=None)

    assert _rc(store, m.id) == pytest.approx(2.0)
    assert _reappraisal_ids(PendingQueue(tmp_path).drain()).count(m.id) == 2


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

    open_memory(
        m, store=store, persona_dir=tmp_path, deliberate=False, seen=None, pending_ids=pending_ids
    )

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

    open_memory(
        m1, store=store, persona_dir=tmp_path, deliberate=False, seen=seen, pending_ids=pending_ids
    )
    open_memory(
        m2, store=store, persona_dir=tmp_path, deliberate=False, seen=seen, pending_ids=pending_ids
    )

    assert pending_ids == [m1.id, m2.id]
    assert not (tmp_path / "pending_candidates.jsonl").exists()


def test_deliberate_open_without_collector_still_enqueues_immediately(tmp_path: Path) -> None:
    """The deliberate `read_full_memory` path never passes `pending_ids` (it
    opens exactly one memory per call), so it must keep enqueuing immediately,
    unaffected by the new collector parameter."""
    store = MemoryStore(":memory:")
    m = _seed(store)

    open_memory(m, store=store, persona_dir=tmp_path, deliberate=True, seen=None)

    assert _reappraisal_ids(PendingQueue(tmp_path).drain()) == [m.id]
