"""recall_resonance event source for v0.0.11.

Per-heartbeat-tick batch evaluator: walks the top-N most-active memories
(by HebbianMatrix neighbor-weight sum), computes their current activation,
compares against per-memory EMA baseline, emits initiate candidates for
memories whose activation has spiked vs. their own history.

Spec: docs/superpowers/specs/2026-05-13-v0.0.11-design.md (Section D)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.initiate.emit import emit_initiate_candidate, read_candidates
from brain.initiate.new_sources import (
    GateThresholds,
    check_shared_meta_gates,
    load_gate_thresholds,
    write_gate_rejection,
)
from brain.initiate.schemas import SemanticContext
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineRow:
    memory_id: str
    ema_mean: float
    ema_var: float
    last_updated_ts: str
    update_count: int


class MemoryActivationBaseline:
    """SQLite-backed per-memory EMA mean + variance.

    EMA update math (West 1979 hybrid):
        delta = current - ema_mean_old
        ema_mean_new = ema_mean_old + α * delta
        ema_var_new = (1 - α) * (ema_var_old + α * delta**2)

    O(1) update, O(1) query, one row per memory ever observed.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_activation_baseline (
                memory_id TEXT PRIMARY KEY,
                ema_mean REAL NOT NULL,
                ema_var REAL NOT NULL,
                last_updated_ts TEXT NOT NULL,
                update_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MemoryActivationBaseline:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get(self, memory_id: str) -> BaselineRow | None:
        row = self._conn.execute(
            "SELECT memory_id, ema_mean, ema_var, last_updated_ts, update_count "
            "FROM memory_activation_baseline WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return BaselineRow(
            memory_id=row["memory_id"],
            ema_mean=row["ema_mean"],
            ema_var=row["ema_var"],
            last_updated_ts=row["last_updated_ts"],
            update_count=row["update_count"],
        )

    def update(self, memory_id: str, current: float, *, alpha: float) -> None:
        """Apply one EMA update for the given memory.

        Seeds with (mean=current, var=0) if the memory has no row yet.
        """
        existing = self.get(memory_id)
        now_iso = datetime.now().isoformat()
        if existing is None:
            new_mean = current
            new_var = 0.0
            new_count = 1
        else:
            delta = current - existing.ema_mean
            new_mean = existing.ema_mean + alpha * delta
            new_var = (1 - alpha) * (existing.ema_var + alpha * delta * delta)
            new_count = existing.update_count + 1
        self._conn.execute(
            """
            INSERT INTO memory_activation_baseline
                (memory_id, ema_mean, ema_var, last_updated_ts, update_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                ema_mean = excluded.ema_mean,
                ema_var = excluded.ema_var,
                last_updated_ts = excluded.last_updated_ts,
                update_count = excluded.update_count
            """,
            (memory_id, new_mean, new_var, now_iso, new_count),
        )
        self._conn.commit()


def compute_current_activation(hebbian: HebbianMatrix, memory_id: str) -> float:
    """Sum of weights of all Hebbian neighbors of memory_id.

    A memory with no edges has activation 0.0. Higher activation = the
    memory is more connected to the rest of the brain's recall graph.
    """
    return sum(weight for _, weight in hebbian.neighbors(memory_id))


def top_n_most_active_memories(
    hebbian: HebbianMatrix,
    *,
    n: int,
) -> list[tuple[str, float]]:
    """Return the n memory IDs with the highest activation, sorted desc.

    Iterates the underlying edges table once (O(E)) and accumulates
    activation per memory. For very large E this is suboptimal but
    bounded by available RAM (50k edges fits comfortably).
    """
    activation: dict[str, float] = {}
    # Reach into HebbianMatrix's internal connection to iterate all edges
    # in one query. The alternative (list memories + neighbors per memory)
    # is O(M * deg), which is worse for sparse graphs. If/when HebbianMatrix
    # exposes a public all_node_activations() helper, switch to that.
    cur = hebbian._conn.execute(
        "SELECT memory_a, memory_b, weight FROM hebbian_edges WHERE weight > 0"
    )
    for row in cur.fetchall():
        a, b, w = row[0], row[1], row[2]
        activation[a] = activation.get(a, 0.0) + w
        activation[b] = activation.get(b, 0.0) + w
    sorted_pairs = sorted(activation.items(), key=lambda p: p[1], reverse=True)
    return sorted_pairs[:n]


def gate_recall_resonance(
    persona_dir: Path,
    *,
    memory_id: str,
    z_score: float,
    staleness_days: float,
    thresholds: GateThresholds,
    now: datetime | None = None,
) -> tuple[bool, str | None]:
    """Per-source gate for recall_resonance.

    Checks, in order: z-threshold, memory staleness, then anti-flood for
    the same memory ID.
    """
    if z_score < thresholds.recall_resonance_z_threshold:
        return False, "z_threshold"
    if staleness_days < thresholds.recall_resonance_staleness_min_days:
        return False, "staleness_min"

    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=thresholds.recall_resonance_anti_flood_hours)
    for candidate in read_candidates(persona_dir):
        if candidate.source != "recall_resonance" or candidate.source_id != memory_id:
            continue
        try:
            candidate_ts = datetime.fromisoformat(candidate.ts)
        except ValueError:
            continue
        if candidate_ts.tzinfo is None and cutoff.tzinfo is not None:
            candidate_ts = candidate_ts.replace(tzinfo=UTC)
        if candidate_ts >= cutoff:
            return False, "anti_flood"

    return True, None


def emit_recall_resonance_candidate(
    persona_dir: Path,
    *,
    memory: Memory,
    z_score: float,
    ema_mean: float,
    current_activation: float,
    staleness_days: float,
    top_neighbors: list[tuple[str, float]],
    now: datetime,
) -> None:
    """Write a recall_resonance candidate into the initiate queue."""
    neighbor_ids = [neighbor_id for neighbor_id, _ in top_neighbors]
    semantic_context = SemanticContext(
        linked_memory_ids=[memory.id, *neighbor_ids],
        topic_tags=list(memory.tags),
        source_meta={
            "memory_id": memory.id,
            "z_score": round(z_score, 3),
            "ema_mean_at_evaluation": round(ema_mean, 3),
            "current_activation": round(current_activation, 3),
            "staleness_days": int(staleness_days),
            "top_neighbors": [
                {"id": neighbor_id, "weight": round(weight, 3)}
                for neighbor_id, weight in top_neighbors
            ],
        },
    )
    emit_initiate_candidate(
        persona_dir,
        kind="message",
        source="recall_resonance",
        source_id=memory.id,
        semantic_context=semantic_context,
        now=now,
    )


def run_resonance_tick(
    persona_dir: Path,
    *,
    now: datetime | None = None,
    is_rest_state: bool = False,
) -> None:
    """Per-heartbeat batch evaluator for recall_resonance.

    Evaluates top-N active Hebbian memories, updates their EMA baselines,
    and emits candidates when current activation spikes above history.
    """
    now = now or datetime.now(UTC)
    thresholds = load_gate_thresholds(persona_dir)

    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        top = top_n_most_active_memories(
            hebbian,
            n=thresholds.recall_resonance_top_n,
        )
    finally:
        hebbian.close()

    if not top:
        return

    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    store_path = persona_dir / "memories.db"
    store = MemoryStore(store_path) if store_path.exists() else None
    baseline = MemoryActivationBaseline(persona_dir / "memory_activation_baseline.db")

    try:
        for memory_id, current_activation in top:
            row = baseline.get(memory_id)
            baseline.update(
                memory_id,
                current_activation,
                alpha=thresholds.recall_resonance_ema_alpha,
            )

            if row is None or row.update_count < thresholds.recall_resonance_bootstrap_min_count:
                continue

            memory = store.get(memory_id) if store is not None else None
            if memory is None:
                continue

            last_accessed = memory.last_accessed_at or memory.created_at
            if last_accessed.tzinfo is None and now.tzinfo is not None:
                last_accessed = last_accessed.replace(tzinfo=UTC)
            staleness_days = (now - last_accessed).total_seconds() / 86_400.0

            stdev = math.sqrt(max(row.ema_var, 1e-6))
            z_score = (current_activation - row.ema_mean) / stdev

            allowed, reason = gate_recall_resonance(
                persona_dir,
                memory_id=memory_id,
                z_score=z_score,
                staleness_days=staleness_days,
                thresholds=thresholds,
                now=now,
            )
            if not allowed:
                write_gate_rejection(
                    persona_dir,
                    ts=now,
                    source="recall_resonance",
                    source_id=memory_id,
                    gate_name=reason or "unknown",
                    threshold_value=thresholds.recall_resonance_z_threshold,
                    observed_value=z_score,
                )
                continue

            meta_ok, meta_reason = check_shared_meta_gates(
                persona_dir,
                source="recall_resonance",
                now=now,
                is_rest_state=is_rest_state,
                thresholds=thresholds,
            )
            if not meta_ok:
                write_gate_rejection(
                    persona_dir,
                    ts=now,
                    source="recall_resonance",
                    source_id=memory_id,
                    gate_name=meta_reason or "meta",
                    threshold_value=0.0,
                    observed_value=0.0,
                )
                continue

            top_neighbors = sorted(
                hebbian.neighbors(memory_id),
                key=lambda pair: pair[1],
                reverse=True,
            )[:3]
            emit_recall_resonance_candidate(
                persona_dir,
                memory=memory,
                z_score=z_score,
                ema_mean=row.ema_mean,
                current_activation=current_activation,
                staleness_days=staleness_days,
                top_neighbors=top_neighbors,
                now=now,
            )
    finally:
        baseline.close()
        if store is not None:
            store.close()
        hebbian.close()
