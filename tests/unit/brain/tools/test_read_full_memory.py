"""read_full_memory tool — full body + deliberate-read bump + wiring (P2 — C7/C10)."""

from __future__ import annotations

from pathlib import Path

from brain.chat.tool_inventory import build_tool_inventory
from brain.chat.tool_recruit import REFLEXIVE_CORE
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.tools import NELL_TOOL_NAMES, SCHEMAS
from brain.tools.dispatch import dispatch
from brain.tools.impls.read_full_memory import read_full_memory


def _recall_count(store: MemoryStore, mid: str) -> int:
    return store._conn.execute(
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _last_accessed(store: MemoryStore, mid: str):
    return store._conn.execute(
        "SELECT last_accessed_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def test_c7_read_full_memory_returns_full_body_and_bumps_recall_count(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    long_body = "Jordan " + ("mattered very much and here is a lot more detail. " * 20)
    m = Memory.create_new(content=long_body, memory_type="event", domain="d")
    store.create(m)

    before = _recall_count(store, m.id)
    res = read_full_memory(m.id, store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert res["content"] == long_body  # full, untruncated
    assert _recall_count(store, m.id) - before == 1  # deliberate-read bump


def test_read_full_memory_touches_last_accessed_and_enqueues_reappraisal(tmp_path: Path) -> None:
    """#231 reconciliation C: a deliberate read now routes through the one
    `open_memory` door — it still touches `last_accessed_at` (a genuine access)
    AND now enqueues the memory for reappraisal ("if a memory gets opened, it
    goes back into the reappraisal queue, doesn't matter how"). Pre-fix,
    `read_full_memory` bumped recall_count via `store.get()` but NEVER
    enqueued."""
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    m = Memory.create_new(content="Jordan mattered", memory_type="event", domain="d")
    store.create(m)
    assert _last_accessed(store, m.id) is None
    PendingQueue(tmp_path).drain()

    read_full_memory(m.id, store=store, hebbian=hebbian, persona_dir=tmp_path)

    # Deliberate access still touches the freshness anchor.
    assert _last_accessed(store, m.id) is not None

    # Reconciliation C: the deliberate read is now enqueued for reappraisal.
    rows = PendingQueue(tmp_path).drain()
    ids = [r["memory_id"] for r in rows if r.get("_route") == "reappraise_importance"]
    assert ids.count(m.id) == 1


def test_c7_read_full_memory_not_found(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    res = read_full_memory("no-such-id", store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert res == {"error": "not found", "id": "no-such-id"}


def test_c10_read_full_memory_registered_and_reachable(tmp_path: Path) -> None:
    assert "read_full_memory" in NELL_TOOL_NAMES
    assert "read_full_memory" in SCHEMAS
    assert set(SCHEMAS) == set(NELL_TOOL_NAMES)  # schema set-equality holds
    assert "read_full_memory" in REFLEXIVE_CORE  # always-available tier
    assert "`read_full_memory`" in build_tool_inventory("Nell")  # appears in inventory

    # Dispatches without crashing (unknown id → soft error dict).
    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    out = dispatch(
        "read_full_memory",
        {"memory_id": "nope"},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    assert out["error"] == "not found"
