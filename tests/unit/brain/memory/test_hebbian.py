"""Tests for brain.memory.hebbian — connection matrix + spreading."""

from __future__ import annotations

import pytest

from brain.memory.hebbian import HebbianMatrix


@pytest.fixture
def matrix() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


def test_matrix_init_creates_schema(matrix: HebbianMatrix) -> None:
    """Fresh matrix has an edges table."""
    cursor = matrix._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='hebbian_edges'"
    )
    assert cursor.fetchone() is not None


def test_strengthen_creates_edge_with_weight(matrix: HebbianMatrix) -> None:
    """strengthen() on a new pair creates an edge with the given delta."""
    matrix.strengthen("a", "b", delta=0.3)
    assert matrix.weight("a", "b") == pytest.approx(0.3)


def test_strengthen_adds_to_existing_weight(matrix: HebbianMatrix) -> None:
    """Repeated strengthen() adds incrementally."""
    matrix.strengthen("a", "b", delta=0.3)
    matrix.strengthen("a", "b", delta=0.2)
    assert matrix.weight("a", "b") == pytest.approx(0.5)


def test_weight_undirected(matrix: HebbianMatrix) -> None:
    """Edge is undirected — weight(a, b) == weight(b, a)."""
    matrix.strengthen("a", "b", delta=0.4)
    assert matrix.weight("a", "b") == matrix.weight("b", "a")


def test_weight_missing_pair_returns_zero(matrix: HebbianMatrix) -> None:
    """Pair with no recorded edge returns weight 0."""
    assert matrix.weight("x", "y") == 0.0


def test_neighbors_returns_connected_ids_with_weights(matrix: HebbianMatrix) -> None:
    """neighbors(id) returns (other_id, weight) for every edge touching id."""
    matrix.strengthen("a", "b", delta=0.3)
    matrix.strengthen("a", "c", delta=0.5)
    matrix.strengthen("d", "e", delta=0.2)

    a_neighbors = matrix.neighbors("a")
    assert sorted(a_neighbors) == [("b", pytest.approx(0.3)), ("c", pytest.approx(0.5))]


def test_neighbors_empty_when_no_edges(matrix: HebbianMatrix) -> None:
    """neighbors of an isolated id returns empty list."""
    assert matrix.neighbors("lonely") == []


def test_decay_all_reduces_every_weight(matrix: HebbianMatrix) -> None:
    """decay_all(rate) subtracts `rate` from every weight (floored at 0)."""
    matrix.strengthen("a", "b", delta=0.5)
    matrix.strengthen("c", "d", delta=0.1)
    matrix.decay_all(rate=0.05)

    assert matrix.weight("a", "b") == pytest.approx(0.45)
    assert matrix.weight("c", "d") == pytest.approx(0.05)


def test_decay_all_floors_at_zero(matrix: HebbianMatrix) -> None:
    """Decay cannot produce negative weights."""
    matrix.strengthen("a", "b", delta=0.05)
    matrix.decay_all(rate=0.1)
    assert matrix.weight("a", "b") == 0.0


def test_garbage_collect_removes_weak_edges(matrix: HebbianMatrix) -> None:
    """garbage_collect deletes edges below the threshold and reports count."""
    matrix.strengthen("a", "b", delta=0.5)
    matrix.strengthen("c", "d", delta=0.005)
    matrix.strengthen("e", "f", delta=0.02)

    removed = matrix.garbage_collect(threshold=0.01)
    assert removed == 1  # only c-d was below 0.01
    assert matrix.weight("c", "d") == 0.0
    assert matrix.weight("a", "b") == pytest.approx(0.5)
    assert matrix.weight("e", "f") == pytest.approx(0.02)


def test_spreading_activation_seeds_at_one(matrix: HebbianMatrix) -> None:
    """Seed nodes have activation 1.0 (baseline)."""
    matrix.strengthen("a", "b", delta=0.5)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    assert act["a"] == pytest.approx(1.0)


def test_spreading_activation_propagates_by_weight(matrix: HebbianMatrix) -> None:
    """Activation propagates to neighbours proportional to edge weight × decay."""
    matrix.strengthen("a", "b", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    # b's activation = 1.0 * 0.8 * 0.5 = 0.4
    assert act["b"] == pytest.approx(0.4)


def test_spreading_activation_respects_depth(matrix: HebbianMatrix) -> None:
    """depth=1 stops at immediate neighbours; no 2-hop activation."""
    matrix.strengthen("a", "b", delta=0.8)
    matrix.strengthen("b", "c", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=1, decay_per_hop=0.5)
    assert "c" not in act


def test_spreading_activation_two_hop(matrix: HebbianMatrix) -> None:
    """depth=2 reaches 2-hop neighbours with compounded decay."""
    matrix.strengthen("a", "b", delta=0.8)
    matrix.strengthen("b", "c", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=2, decay_per_hop=0.5)
    # c via b: b's act = 0.4; c = 0.4 * 0.8 * 0.5 = 0.16
    assert act["c"] == pytest.approx(0.16)


def test_spreading_activation_aggregates_multi_path(matrix: HebbianMatrix) -> None:
    """Node reached from multiple seeds accumulates activation (max, not sum)."""
    matrix.strengthen("a", "x", delta=0.6)
    matrix.strengthen("b", "x", delta=0.8)
    act = matrix.spreading_activation(["a", "b"], depth=1, decay_per_hop=0.5)
    # x's activation: max(1.0*0.6*0.5, 1.0*0.8*0.5) = max(0.3, 0.4) = 0.4
    assert act["x"] == pytest.approx(0.4)
