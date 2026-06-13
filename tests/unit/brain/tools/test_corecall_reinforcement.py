from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory
from brain.tools.impls.search_memories import (
    _CORECALL_FANOUT,
    _reinforce_corecall,
)


def _mem(content: str) -> Memory:
    return Memory.create_new(content=content, memory_type="episodic", domain="chat", emotions={})


def test_star_topology_anchor_to_each_not_mesh(tmp_path: Path):
    """Anchor (results[0]) links to each other result; non-anchor pairs are NOT linked."""
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        a, b, c = _mem("dog name"), _mem("the walk"), _mem("the vet")
        _reinforce_corecall(heb, [a, b, c])
        # anchor a is linked to b and c
        assert heb.weight(a.id, b.id) > 0
        assert heb.weight(a.id, c.id) > 0
        # b and c (both non-anchor) are NOT linked to each other
        assert heb.weight(b.id, c.id) == 0
    finally:
        heb.close()


def test_fanout_cap(tmp_path: Path):
    """At most _CORECALL_FANOUT edges are created from the anchor."""
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        mems = [_mem(f"m{i}") for i in range(10)]
        _reinforce_corecall(heb, mems)
        anchor = mems[0]
        linked = sum(1 for m in mems[1:] if heb.weight(anchor.id, m.id) > 0)
        assert linked == _CORECALL_FANOUT
    finally:
        heb.close()


def test_fewer_than_two_results_is_noop(tmp_path: Path):
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        a = _mem("solo")
        _reinforce_corecall(heb, [a])
        assert heb.activation_count(a.id) == 0
        _reinforce_corecall(heb, [])
    finally:
        heb.close()


def test_existing_edge_reraise_is_weight_only_no_new_activation_count(tmp_path: Path):
    """Re-reaching the same cluster raises weight but NOT activation_count (edge already exists)."""
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        a, b = _mem("a"), _mem("b")
        _reinforce_corecall(heb, [a, b])
        count_after_first = heb.activation_count(a.id)
        w1 = heb.weight(a.id, b.id)
        _reinforce_corecall(heb, [a, b])
        assert heb.activation_count(a.id) == count_after_first  # no new edge
        assert heb.weight(a.id, b.id) > w1  # weight rose
    finally:
        heb.close()


def test_fail_soft_on_hebbian_error():
    """A broken hebbian must not raise out of the helper."""

    class _BrokenHebbian:
        def strengthen(self, *_a, **_k):
            raise RuntimeError("boom")

    a, b = _mem("a"), _mem("b")
    _reinforce_corecall(_BrokenHebbian(), [a, b])  # must not raise
