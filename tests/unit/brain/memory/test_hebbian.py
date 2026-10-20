"""Tests for brain.memory.hebbian — connection matrix + spreading."""

from __future__ import annotations

import logging

import pytest

from brain.memory.hebbian import MAX_WEIGHT, HebbianMatrix
from brain.utils import log_throttle

HEBB_LOG = "brain.memory.hebbian"


@pytest.fixture
def matrix() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


def test_strengthen_caps_single_large_delta_at_max(matrix: HebbianMatrix) -> None:
    matrix.strengthen("a", "b", delta=MAX_WEIGHT + 5.0)
    assert matrix.weight("a", "b") == MAX_WEIGHT


def test_strengthen_repeated_plateaus_at_max(matrix: HebbianMatrix) -> None:
    # 25 * 0.5 = 12.5 uncapped; must plateau at the ceiling.
    for _ in range(25):
        matrix.strengthen("a", "b", delta=0.5)
    assert matrix.weight("a", "b") == MAX_WEIGHT


def test_strengthen_below_cap_adds_exact_delta(matrix: HebbianMatrix) -> None:
    matrix.strengthen("a", "b", delta=0.5)
    matrix.strengthen("a", "b", delta=0.25)
    # Well below MAX_WEIGHT — exact sum, no behavior change.
    assert matrix.weight("a", "b") == 0.75


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


def test_strengthen_self_edge_is_noop(matrix: HebbianMatrix) -> None:
    """strengthen(x, x) is silently ignored — self-edges aren't tracked."""
    matrix.strengthen("x", "x", delta=0.5)
    assert matrix.weight("x", "x") == 0.0
    assert matrix.neighbors("x") == []


def test_strengthen_rejects_non_positive_delta(matrix: HebbianMatrix) -> None:
    """Weights must stay non-negative by module contract; negative delta raises."""
    with pytest.raises(ValueError, match="delta must be positive"):
        matrix.strengthen("a", "b", delta=-0.1)
    with pytest.raises(ValueError, match="delta must be positive"):
        matrix.strengthen("a", "b", delta=0.0)


def test_decay_all_rejects_negative_rate(matrix: HebbianMatrix) -> None:
    """Negative rate would inflate every weight — a sign error in a scheduled
    dream/heartbeat call would silently corrupt the graph. Guard forbids it.
    """
    matrix.strengthen("a", "b", delta=0.3)
    with pytest.raises(ValueError, match="non-negative"):
        matrix.decay_all(rate=-0.1)


def test_decay_all_zero_rate_is_noop(matrix: HebbianMatrix) -> None:
    """rate=0.0 leaves weights untouched."""
    matrix.strengthen("a", "b", delta=0.3)
    matrix.decay_all(rate=0.0)
    assert matrix.weight("a", "b") == pytest.approx(0.3)


def test_spreading_activation_depth_zero_returns_seeds_only(
    matrix: HebbianMatrix,
) -> None:
    """depth=0 skips propagation entirely — only the seeds appear."""
    matrix.strengthen("a", "b", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=0, decay_per_hop=0.5)
    assert act == {"a": 1.0}


def test_spreading_activation_zero_decay_blocks_propagation(
    matrix: HebbianMatrix,
) -> None:
    """decay_per_hop=0 → propagated=0 → nothing beats the default 0.0 guard."""
    matrix.strengthen("a", "b", delta=0.8)
    act = matrix.spreading_activation(["a"], depth=2, decay_per_hop=0.0)
    assert "b" not in act
    assert act["a"] == pytest.approx(1.0)


def test_remove_memory_deletes_all_incident_edges(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    h = HebbianMatrix(tmp_path / "h.db")
    try:
        h.strengthen("a", "b", delta=0.5)
        h.strengthen("a", "c", delta=0.7)
        h.strengthen("b", "c", delta=0.3)
        removed = h.remove_memory("a")
        assert removed == 2
        assert h.neighbors("a") == []
        assert h.weight("b", "c") == 0.3
    finally:
        h.close()


def test_remove_memory_no_edges_is_zero(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    h = HebbianMatrix(tmp_path / "h.db")
    try:
        assert h.remove_memory("ghost") == 0
    finally:
        h.close()


def test_ensure_edge_inserts_once_no_weight_inflation(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    h = HebbianMatrix(tmp_path / "h.db")
    try:
        assert h.ensure_edge("a", "b", 0.4) is True
        assert h.ensure_edge("a", "b", 0.9) is False   # already exists; no change
        assert h.weight("a", "b") == 0.4
        assert h.ensure_edge("b", "a", 0.1) is False    # canonical: same edge
        assert h.ensure_edge("x", "x", 0.5) is False     # self-edge ignored
        assert h.neighbors("x") == []
    finally:
        h.close()


@pytest.fixture
def fresh_throttle(monkeypatch):
    monkeypatch.setattr(log_throttle, "_last", {})
    monkeypatch.setattr(log_throttle, "_counts", {}, raising=False)


def _cap_records(caplog):
    return [r for r in caplog.records if r.name == HEBB_LOG and "MAX_WEIGHT" in r.getMessage()]


def test_strengthen_records_when_the_cap_binds(matrix, caplog, fresh_throttle) -> None:
    # #368: the cap used to bind silently. It only bites a pathological edge,
    # so the first firing warns; later ones log at DEBUG with the count.
    with caplog.at_level(logging.DEBUG, logger=HEBB_LOG):
        matrix.strengthen("a", "b", delta=9.5)  # 9.5: below the cap, no record
        assert _cap_records(caplog) == []
        matrix.strengthen("a", "b", delta=0.5)  # exactly 10.0: reaches, does not exceed
        assert _cap_records(caplog) == []
        matrix.strengthen("b", "a", delta=0.5)  # 10.5 -> 10.0: binds
        matrix.strengthen("x", "y", delta=MAX_WEIGHT + 5.0)  # fresh edge, binds on INSERT
    recs = _cap_records(caplog)
    assert [r.levelno for r in recs] == [logging.WARNING, logging.DEBUG]
    assert "a-b" in recs[0].getMessage() and "10.500" in recs[0].getMessage()
    assert "2 since start" in recs[1].getMessage()
    # Behaviour unchanged: stored weights still saturate at the cap.
    assert matrix.weight("a", "b") == MAX_WEIGHT
    assert matrix.weight("x", "y") == MAX_WEIGHT


def test_set_edge_weight_records_when_the_cap_binds(matrix, caplog, fresh_throttle) -> None:
    with caplog.at_level(logging.DEBUG, logger=HEBB_LOG):
        matrix.set_edge_weight("a", "b", MAX_WEIGHT)  # at the cap: no record
        assert _cap_records(caplog) == []
        matrix.set_edge_weight("d", "c", MAX_WEIGHT + 2.0)  # binds
    (rec,) = _cap_records(caplog)
    assert rec.levelno == logging.WARNING
    assert "c-d" in rec.getMessage() and "12.000" in rec.getMessage() and "set_edge_weight" in rec.getMessage()
    # Behaviour unchanged.
    assert matrix.weight("a", "b") == MAX_WEIGHT
    assert matrix.weight("c", "d") == MAX_WEIGHT


_WRITES = {
    "strengthen": lambda m: m.strengthen("p", "q", delta=0.5),
    "set_edge_weight": lambda m: m.set_edge_weight("p", "q", 1.0),
    "ensure_edge": lambda m: m.ensure_edge("p", "q", 1.0),
    "decay_all": lambda m: m.decay_all(0.1),
    "garbage_collect": lambda m: m.garbage_collect(5.0),
    "remove_memory": lambda m: m.remove_memory("a"),
}


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_a_failed_write_releases_the_write_lock(tmp_path, write) -> None:
    """Regression (found during #368): a HebbianMatrix write that raised left its
    transaction open, holding the SQLite write lock. search_memories swallows the
    error, so every other writer then failed with "database is locked" until this
    connection committed or closed. Every write must roll back what it opened."""
    import sqlite3

    db = tmp_path / "h.db"
    m = HebbianMatrix(db, integrity_check=False)
    m.strengthen("a", "b", delta=1.0)
    m._conn.executescript(
        """
        CREATE TABLE probe (x INTEGER);
        CREATE TRIGGER no_ins BEFORE INSERT ON hebbian_edges BEGIN SELECT RAISE(ABORT, 'boom'); END;
        CREATE TRIGGER no_upd BEFORE UPDATE ON hebbian_edges BEGIN SELECT RAISE(ABORT, 'boom'); END;
        CREATE TRIGGER no_del BEFORE DELETE ON hebbian_edges BEGIN SELECT RAISE(ABORT, 'boom'); END;
        """
    )
    with pytest.raises(sqlite3.DatabaseError, match="boom"):
        _WRITES[write](m)
    assert not m._conn.in_transaction
    other = sqlite3.connect(db, timeout=0.2)
    try:
        other.execute("INSERT INTO probe VALUES (1)")  # fails fast if still locked
        other.commit()
    finally:
        other.close()
    assert m.weight("a", "b") == 1.0  # the failed write changed nothing
    m.close()
