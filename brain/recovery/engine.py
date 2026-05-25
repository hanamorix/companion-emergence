"""Recovery engine."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from brain.forgetting import graveyard
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.recovery.report import RecoveryReport
from brain.recovery.source_reader import read_source_edges, read_source_memories


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
    plan.source_edges = read_source_edges(source_dir)
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


def run_recovery(
    persona_dir: Path,
    *,
    source_dir: Path | None,
    dry_run: bool = False,
) -> RecoveryReport:
    """Backup → restore → repair → grace-refresh → report."""
    started = time.monotonic()
    if source_dir is not None and source_dir.resolve() == persona_dir.resolve():
        raise ValueError("--from must not be the persona being recovered")
    plan = _build_restore_plan(persona_dir, source_dir=source_dir)
    if dry_run:
        return RecoveryReport(
            persona=persona_dir.name, mode=plan.mode,
            source_dir=str(source_dir) if source_dir else None,
            memories_restored_full=len(plan.missing),
            memories_restored_summary=len(plan.missing_summaries),
            memories_unfaded=len(plan.unfade),
            edges_repaired=0, edges_pruned_unrecoverable=0,
            backup_path=None, elapsed_seconds=time.monotonic() - started, dry_run=True,
        )
    # backup
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    backup_dir = persona_dir / f"recover-backup-{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ("memories.db", "hebbian.db"):
        src_file = persona_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, backup_dir / name)
    backup = str(backup_dir)

    counts = _apply_memory_restores(persona_dir, plan)
    repaired, pruned = _repair_edges(persona_dir, plan)

    # grace-refresh
    from brain.felt_time.state import load_or_recover
    p_manifest = persona_dir / "source-manifest.json"
    mdata: dict = {}
    if p_manifest.exists():
        try:
            mdata = json.loads(p_manifest.read_text())
        except (OSError, json.JSONDecodeError):
            mdata = {}
    state, _ = load_or_recover(persona_dir)
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    mdata["migrated_at_utc"] = now_iso
    mdata["lived_age_hours_at_migration"] = float(state.lived_age_hours)
    mdata["recovered_at"] = now_iso
    p_manifest.write_text(json.dumps(mdata, indent=2) + "\n", encoding="utf-8")

    return RecoveryReport(
        persona=persona_dir.name, mode=plan.mode,
        source_dir=str(source_dir) if source_dir else None,
        memories_restored_full=counts["restored_full"],
        memories_restored_summary=counts["restored_summary"],
        memories_unfaded=counts["unfaded"],
        edges_repaired=repaired, edges_pruned_unrecoverable=pruned,
        backup_path=backup, elapsed_seconds=time.monotonic() - started, dry_run=False,
    )


def preflight_recovery(persona_dir: Path, *, source_dir: Path | None) -> dict:
    """Pure read — what recovery WOULD do. Used by the wizard preflight."""
    plan = _build_restore_plan(persona_dir, source_dir=source_dir)
    return {
        "mode": plan.mode,
        "missing": len(plan.missing) + len(plan.missing_summaries),
        "unfade": len(plan.unfade),
    }
