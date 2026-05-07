"""Hebbian connection matrix with spreading activation.

Edges are undirected: edge (a, b) is stored canonically with the lower id
first to avoid duplicate rows. Weight accumulates over repeated strengthen()
calls; decay_all() reduces all weights (floored at 0); garbage_collect()
removes weak edges to keep the graph compact.

Spreading activation is a bounded BFS that propagates seed activation
through the graph, attenuating by (weight * decay_per_hop) at each hop.
Multi-path arrivals take the max (not sum) — prevents an activation
runaway on densely connected graphs.

Design per spec Section 4.1 (brain/memory/hebbian.py) and OG's F32/F33
Hebbian work.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path


class HebbianMatrix:
    """SQLite-backed sparse weighted graph between memory ids."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS hebbian_edges (
        memory_a TEXT NOT NULL,
        memory_b TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 0.0,
        last_strengthened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (memory_a, memory_b)
    );

    CREATE INDEX IF NOT EXISTS idx_hebbian_a ON hebbian_edges(memory_a);
    CREATE INDEX IF NOT EXISTS idx_hebbian_b ON hebbian_edges(memory_b);
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        try:
            result = self._conn.execute("PRAGMA integrity_check").fetchall()
        except sqlite3.DatabaseError as exc:
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(str(db_path), str(exc)) from exc
        if result != [("ok",)]:
            detail = "; ".join(str(row[0]) for row in result)
            self._conn.close()
            from brain.health.anomaly import BrainIntegrityError

            raise BrainIntegrityError(str(db_path), detail)
        # WAL + 5s busy_timeout — set AFTER the integrity check so a
        # corrupt-file probe surfaces BrainIntegrityError, not a pragma
        # crash. In-memory dbs reject WAL; fallback keeps `:memory:` ok.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def strengthen(self, a: str, b: str, delta: float = 0.1) -> None:
        """Add `delta` to the weight of edge (a, b). Creates the edge if new.

        `delta` must be positive — the module contract is that weights are
        non-negative. Callers that want to weaken an edge use `decay_all`
        or `garbage_collect`. Negative delta raises ValueError.
        """
        if a == b:
            return  # self-edges not tracked
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta!r}")
        lo, hi = _canonical(a, b)
        self._conn.execute(
            """
            INSERT INTO hebbian_edges (memory_a, memory_b, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(memory_a, memory_b)
                DO UPDATE SET weight = weight + excluded.weight,
                              last_strengthened_at = CURRENT_TIMESTAMP
            """,
            (lo, hi, delta),
        )
        self._conn.commit()

    def weight(self, a: str, b: str) -> float:
        """Return the weight of edge (a, b). Zero if no edge."""
        if a == b:
            return 0.0
        lo, hi = _canonical(a, b)
        row = self._conn.execute(
            "SELECT weight FROM hebbian_edges WHERE memory_a = ? AND memory_b = ?",
            (lo, hi),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def neighbors(self, memory_id: str) -> list[tuple[str, float]]:
        """Return [(other_id, weight), ...] for every edge touching memory_id."""
        rows = self._conn.execute(
            """
            SELECT memory_b, weight FROM hebbian_edges WHERE memory_a = ?
            UNION ALL
            SELECT memory_a, weight FROM hebbian_edges WHERE memory_b = ?
            """,
            (memory_id, memory_id),
        ).fetchall()
        return [(other, float(weight)) for other, weight in rows]

    def decay_all(self, rate: float) -> None:
        """Subtract `rate` from every weight, floored at 0.

        `rate` must be non-negative. A negative rate would inflate every
        weight in a single scheduled batch — silent corruption for
        dream/heartbeat cycles. ValueError guards the sign.
        """
        if rate < 0.0:
            raise ValueError(f"decay rate must be non-negative, got {rate!r}")
        self._conn.execute("UPDATE hebbian_edges SET weight = MAX(weight - ?, 0.0)", (rate,))
        self._conn.commit()

    def garbage_collect(self, threshold: float = 0.01) -> int:
        """Remove edges with weight < threshold. Returns the count removed."""
        cursor = self._conn.execute("DELETE FROM hebbian_edges WHERE weight < ?", (threshold,))
        self._conn.commit()
        return cursor.rowcount

    def spreading_activation(
        self,
        seed_ids: Iterable[str],
        depth: int = 2,
        decay_per_hop: float = 0.5,
    ) -> dict[str, float]:
        """BFS spreading activation from seed_ids, returning activation by id.

        Seed nodes have activation 1.0 and are protected: propagation
        cannot lower them. Each hop multiplies the source activation by
        (edge_weight * decay_per_hop) to produce the neighbour's
        activation. Multi-path arrivals take the max (not sum) — prevents
        activation runaway on densely connected graphs.

        Returns a dict {memory_id: activation}.
        """
        activation: dict[str, float] = {}
        for sid in seed_ids:
            activation[sid] = 1.0

        frontier = set(activation)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour, weight in self.neighbors(node):
                    propagated = activation[node] * weight * decay_per_hop
                    if propagated > activation.get(neighbour, 0.0):
                        activation[neighbour] = propagated
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return activation


def _canonical(a: str, b: str) -> tuple[str, str]:
    """Sort the pair so edge (a, b) and (b, a) hash to the same row."""
    return (a, b) if a <= b else (b, a)
