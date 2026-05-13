"""recall_resonance event source for v0.0.11.

Per-heartbeat-tick batch evaluator: walks the top-N most-active memories
(by HebbianMatrix neighbor-weight sum), computes their current activation,
compares against per-memory EMA baseline, emits initiate candidates for
memories whose activation has spiked vs. their own history.

Spec: docs/superpowers/specs/2026-05-13-v0.0.11-design.md (Section D)
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


from brain.memory.hebbian import HebbianMatrix  # noqa: E402


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
