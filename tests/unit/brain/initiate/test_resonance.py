"""Tests for brain.initiate.resonance — per-memory recall resonance."""
from __future__ import annotations

import math
from pathlib import Path

from brain.initiate.resonance import MemoryActivationBaseline


def test_baseline_first_observation_seeds_with_no_variance(tmp_path: Path):
    db_path = tmp_path / "baseline.db"
    with MemoryActivationBaseline(db_path) as b:
        b.update("mem_1", 1.0, alpha=0.08)
        row = b.get("mem_1")
        assert row is not None
        assert row.ema_mean == 1.0
        assert row.ema_var == 0.0
        assert row.update_count == 1


def test_baseline_ema_converges_to_constant_input(tmp_path: Path):
    db_path = tmp_path / "baseline.db"
    with MemoryActivationBaseline(db_path) as b:
        for _ in range(50):
            b.update("mem_2", 5.0, alpha=0.08)
        row = b.get("mem_2")
        assert row is not None
        # After many constant updates, mean converges to value, var → 0.
        assert math.isclose(row.ema_mean, 5.0, abs_tol=1e-3)
        assert row.ema_var < 0.01
        assert row.update_count == 50


def test_baseline_ema_variance_grows_with_noise(tmp_path: Path):
    db_path = tmp_path / "baseline.db"
    import random
    rng = random.Random(7)
    with MemoryActivationBaseline(db_path) as b:
        for _ in range(50):
            b.update("mem_3", 1.0 + rng.gauss(0, 0.5), alpha=0.08)
        row = b.get("mem_3")
        assert row is not None
        # Noisy input should yield non-trivial variance.
        assert row.ema_var > 0.05


def test_baseline_persists_across_connections(tmp_path: Path):
    db_path = tmp_path / "baseline.db"
    with MemoryActivationBaseline(db_path) as b:
        b.update("mem_4", 3.0, alpha=0.08)
        b.update("mem_4", 4.0, alpha=0.08)
    # Reopen.
    with MemoryActivationBaseline(db_path) as b2:
        row = b2.get("mem_4")
        assert row is not None
        assert row.update_count == 2
        assert 3.0 < row.ema_mean < 4.0


def test_baseline_missing_memory_returns_none(tmp_path: Path):
    db_path = tmp_path / "baseline.db"
    with MemoryActivationBaseline(db_path) as b:
        assert b.get("does_not_exist") is None


def test_compute_current_activation_sums_neighbor_weights(tmp_path):
    from brain.initiate.resonance import compute_current_activation
    from brain.memory.hebbian import HebbianMatrix

    db_path = tmp_path / "hebbian.db"
    h = HebbianMatrix(db_path)
    try:
        h.strengthen("seed", "n1", delta=0.4)
        h.strengthen("seed", "n2", delta=0.3)
        h.strengthen("seed", "n3", delta=0.2)
        activation = compute_current_activation(h, "seed")
        assert abs(activation - 0.9) < 1e-6
    finally:
        h.close()


def test_compute_current_activation_zero_for_isolated_memory(tmp_path):
    from brain.initiate.resonance import compute_current_activation
    from brain.memory.hebbian import HebbianMatrix

    db_path = tmp_path / "hebbian.db"
    h = HebbianMatrix(db_path)
    try:
        activation = compute_current_activation(h, "lonely_memory")
        assert activation == 0.0
    finally:
        h.close()


def test_top_n_most_active_memories(tmp_path):
    from brain.initiate.resonance import top_n_most_active_memories
    from brain.memory.hebbian import HebbianMatrix

    db_path = tmp_path / "hebbian.db"
    h = HebbianMatrix(db_path)
    try:
        h.strengthen("mem_a", "x", delta=0.9)
        h.strengthen("mem_b", "y", delta=0.5)
        h.strengthen("mem_b", "z", delta=0.5)
        h.strengthen("mem_c", "w", delta=0.3)

        top = top_n_most_active_memories(h, n=2)
        ids = [memory_id for memory_id, _ in top]
        # Top-2 by total weight: mem_b (1.0), mem_a (0.9).
        assert "mem_b" in ids and "mem_a" in ids
        assert "mem_c" not in ids
    finally:
        h.close()


def test_top_n_empty_hebbian_returns_empty(tmp_path):
    from brain.initiate.resonance import top_n_most_active_memories
    from brain.memory.hebbian import HebbianMatrix

    db_path = tmp_path / "hebbian.db"
    h = HebbianMatrix(db_path)
    try:
        assert top_n_most_active_memories(h, n=10) == []
    finally:
        h.close()
