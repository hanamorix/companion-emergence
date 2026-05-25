"""Recovery engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.forgetting import graveyard
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.recovery.source_reader import read_source_memories


@dataclass
class RestorePlan:
    mode: str = "source"
    missing: dict[str, Memory] = field(default_factory=dict)
    unfade: dict[str, str] = field(default_factory=dict)
    missing_summaries: dict[str, Memory] = field(default_factory=dict)
    source_edges: list = field(default_factory=list)
    graveyard_neighbors: dict = field(default_factory=dict)


def _build_restore_plan(persona_dir: Path, *, source_dir: Path | None) -> RestorePlan:
    store = MemoryStore(persona_dir / "memories.db")
    try:
        rows = store._conn.execute("SELECT id, state FROM memories").fetchall()
    finally:
        store.close()
    current_ids = {r["id"] for r in rows}
    faded_ids = {r["id"] for r in rows if r["state"] == "fading"}
    if source_dir is None:
        plan = RestorePlan(mode="graveyard")
        store = MemoryStore(persona_dir / "memories.db")
        try:
            rows = store._conn.execute("SELECT id FROM memories").fetchall()
        finally:
            store.close()
        current_ids = {r["id"] for r in rows}
        for entry in graveyard.read_all(persona_dir):
            mid = entry.get("memory_id")
            if not mid or mid in current_ids:
                continue
            created_raw = entry.get("created_at_iso")
            try:
                created = datetime.fromisoformat(created_raw.replace("Z", "+00:00")) if created_raw else datetime.now(UTC)
            except (ValueError, AttributeError):
                created = datetime.now(UTC)
            plan.missing_summaries[mid] = Memory(
                id=mid,
                content=entry.get("summary") or "",
                memory_type=entry.get("memory_type") or "conversation",
                domain=entry.get("domain") or "us",
                created_at=created,
                emotions=dict(entry.get("emotion_at_ingest") or {}),
            )
            plan.graveyard_neighbors[mid] = [
                (nid, float(w)) for nid, w in (entry.get("hebbian_neighbors") or [])
            ]
        return plan
    plan = RestorePlan(mode="source")
    for mid, mem in read_source_memories(source_dir).items():
        if mid not in current_ids:
            plan.missing[mid] = mem
        elif mid in faded_ids:
            plan.unfade[mid] = mem.content
    return plan


def _repair_edges(persona_dir: Path, plan: RestorePlan) -> tuple[int, int]:
    """Restore recoverable edges, then prune any edge still dangling.

    Returns (edges_repaired, edges_pruned_unrecoverable). Runs AFTER memory
    restores so endpoints exist; the final prune is the only place we delete
    dangling edges, and only once restoration can no longer save them.
    """
    store = MemoryStore(persona_dir / "memories.db")
    try:
        valid = {r["id"] for r in store._conn.execute("SELECT id FROM memories")}
    finally:
        store.close()

    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    repaired = 0
    pruned = 0
    try:
        if plan.mode == "source":
            for a, b, w in plan.source_edges:
                if a in valid and b in valid and hebbian.ensure_edge(a, b, w):
                    repaired += 1
        else:
            for mid, neighbours in plan.graveyard_neighbors.items():
                if mid not in valid:
                    continue
                for nid, w in neighbours:
                    if nid in valid and hebbian.ensure_edge(mid, nid, w):
                        repaired += 1

        rows = hebbian._conn.execute(
            "SELECT DISTINCT memory_a FROM hebbian_edges "
            "UNION SELECT DISTINCT memory_b FROM hebbian_edges"
        ).fetchall()
        referenced = {r[0] for r in rows}
        for mid in referenced - valid:
            pruned += hebbian.remove_memory(mid)
    finally:
        hebbian.close()
    return repaired, pruned


def _apply_memory_restores(persona_dir: Path, plan: RestorePlan) -> dict[str, int]:
    counts = {"restored_full": 0, "restored_summary": 0, "unfaded": 0}
    store = MemoryStore(persona_dir / "memories.db")
    try:
        now = datetime.now(UTC)
        for mem in plan.missing.values():
            mem.last_accessed_at = now
            store.create(mem)
            counts["restored_full"] += 1
        for mem in plan.missing_summaries.values():
            mem.last_accessed_at = now
            store.create(mem)
            counts["restored_summary"] += 1
        for mid, original in plan.unfade.items():
            store.update(mid, content=original)
            store._conn.execute(
                "UPDATE memories SET state='active', content_snapshot=NULL WHERE id=?",
                (mid,),
            )
            store._conn.commit()
            counts["unfaded"] += 1
    finally:
        store.close()
    return counts
