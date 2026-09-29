"""Name-recall fix R3 follow-up (spec §4, S77): each paragraph's 50-candidate
cosine pool is filled genuine-first.

Genuine memories by cosine score first, then monologue-family memories by
cosine score for any places left, so a run of monologue-family memories that
out-score a genuine one can never keep it out of the pool (the coarse cut the
reranker or the cosine path then sees). No score multiplier. One sort over the
scored list, no second scan.

All offline: the R3 scripted embedder and fake rerankers, an in-tmp store,
synthetic data only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.memory import semantic_recall as sr
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.semantic_recall import genuine_first_coarse_cut, run_semantic_recall
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.memory.test_monologue_last import (
    _QUERY,
    _cosine_floor,
    _ids,
    _install_reranker,
    _no_reranker,
    _Recording,
    _rerank_floor,
    _seed,
    _tool,
    _warm,
)

_FAMILY_TYPES = ("monologue", "monologue_trace", "monologue_emotion", "monologue_soul_candidate")


def _pool_of(specs: list[tuple[str, str, float]]):
    """`(id, memory_type, cosine)` specs -> (cosine_scored, pool) for the pure
    helper. Vectors are unused by the helper, so a placeholder stands in."""
    pool = {}
    scored = []
    for mid, mtype, cosine in specs:
        pool[mid] = (Memory.create_new(content=mid, memory_type=mtype, domain="d"), None)
        scored.append((mid, cosine))
    return scored, pool


def _spy_rank_and_gate(monkeypatch: pytest.MonkeyPatch, module: str) -> list[list[tuple[str, float]]]:
    """Record the coarse cut each caller hands `rank_and_gate` (the pool)."""
    seen: list[list[tuple[str, float]]] = []
    real = sr.rank_and_gate

    def _spy(store, query, pool, coarse, **kwargs):
        seen.append(list(coarse))
        return real(store, query, pool, coarse, **kwargs)

    monkeypatch.setattr(f"{module}.rank_and_gate", _spy)
    return seen


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_family_memories_that_outscore_a_genuine_one_do_not_keep_it_out_of_the_pool() -> None:
    specs = [(f"f{i}", _FAMILY_TYPES[i % 4], 0.99 - i * 0.001) for i in range(60)]
    specs += [("g0", "event", 0.30), ("g1", "conversation", 0.20)]
    scored, pool = _pool_of(specs)

    cut = genuine_first_coarse_cut(scored, pool)

    ids = [mid for mid, _ in cut]
    assert len(cut) == CANDIDATE_POOL
    assert ids[:2] == ["g0", "g1"], "genuine first, by cosine"
    assert ids[2:] == [f"f{i}" for i in range(CANDIDATE_POOL - 2)], "family fills the rest by cosine"


def test_fewer_genuine_than_the_pool_fills_the_remainder_with_the_best_family_by_cosine() -> None:
    specs = [("g_lo", "event", 0.10), ("g_hi", "event", 0.40)]
    specs += [(f"f{i}", "monologue", 0.5 + i * 0.001) for i in range(10)]
    scored, pool = _pool_of(specs)

    cut = genuine_first_coarse_cut(scored, pool, size=5)

    assert [mid for mid, _ in cut] == ["g_hi", "g_lo", "f9", "f8", "f7"]


def test_a_pool_of_fifty_genuine_memories_admits_no_family_memory() -> None:
    specs = [(f"g{i}", "event", 0.90 - i * 0.001) for i in range(CANDIDATE_POOL)]
    specs += [(f"f{i}", "monologue", 0.99) for i in range(5)]
    scored, pool = _pool_of(specs)

    cut = genuine_first_coarse_cut(scored, pool)

    assert [mid for mid, _ in cut] == [f"g{i}" for i in range(CANDIDATE_POOL)]


def test_the_pool_is_filled_from_the_scores_not_a_multiplier() -> None:
    """Scores pass through unchanged: the cut hands back each memory's own
    cosine."""
    scored, pool = _pool_of([("f", "monologue", 0.99), ("g", "event", 0.25)])

    assert genuine_first_coarse_cut(scored, pool) == [("g", 0.25), ("f", 0.99)]


def test_a_genuine_memory_far_below_every_family_cosine_still_leads_the_pool() -> None:
    """No hidden multiplier: however low the genuine cosine (near zero, even
    negative) and however high the family cosines, genuine memories come first."""
    specs = [(f"f{i}", "monologue_trace", 0.90 - i * 0.001) for i in range(60)]
    specs += [("g_near_zero", "event", 0.05), ("g_negative", "event", -0.20)]
    scored, pool = _pool_of(specs)

    ids = [mid for mid, _ in genuine_first_coarse_cut(scored, pool)]

    assert ids[:2] == ["g_near_zero", "g_negative"]
    assert len(ids) == CANDIDATE_POOL


def test_the_family_flag_is_computed_once_per_entry_so_the_scan_stays_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [(f"m{i}", "monologue" if i % 2 else "event", i / 100) for i in range(80)]
    scored, pool = _pool_of(specs)
    calls = 0
    real = sr.is_monologue_family

    def _counting(memory):
        nonlocal calls
        calls += 1
        return real(memory)

    monkeypatch.setattr(sr, "is_monologue_family", _counting)

    genuine_first_coarse_cut(scored, pool)

    assert calls == len(scored)


# ---------------------------------------------------------------------------
# Through passive recall and the tool (the two callers of the cut)
# ---------------------------------------------------------------------------


def _flood(store, monkeypatch, n_family: int = 55):
    """Three genuine memories at LOW cosine; `n_family` family memories at
    HIGH cosine, so today's plain top-50 holds only family memories."""
    return _seed(
        store,
        monkeypatch,
        [0.60, 0.55, 0.50],
        [0.99 - i * 0.001 for i in range(n_family)],
    )


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_passive_recall_pool_holds_the_genuine_memories_despite_a_family_flood(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _flood(store, monkeypatch)
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        _warm()
        scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
        _install_reranker(monkeypatch, _Recording(scripted))
    seen = _spy_rank_and_gate(monkeypatch, "brain.memory.semantic_recall")

    result = run_semantic_recall(store, tmp_path, _QUERY)

    (coarse,) = seen
    coarse_ids = [mid for mid, _ in coarse]
    assert len(coarse_ids) == CANDIDATE_POOL
    assert coarse_ids[:3] == [genuine[0].id, genuine[1].id, genuine[2].id]
    assert set(coarse_ids[3:]) <= set(_ids(family))
    assert result is not None and result.path == path
    kept = _ids([*result.full, *result.snippet])
    assert set(_ids(genuine)) <= set(kept)


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_the_tool_pool_holds_the_genuine_memories_despite_a_family_flood(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _flood(store, monkeypatch)
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        _warm()
        scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
        _install_reranker(monkeypatch, _Recording(scripted))
    seen = _spy_rank_and_gate(monkeypatch, "brain.tools.impls.search_memories")

    got = _tool(tmp_path, store, limit=5)

    (coarse,) = seen
    coarse_ids = [mid for mid, _ in coarse]
    assert len(coarse_ids) == CANDIDATE_POOL
    assert coarse_ids[:3] == _ids(genuine)
    assert set(_ids(genuine)) <= set(got)


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_with_fifty_genuine_memories_no_family_memory_surfaces_even_above_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    """S77's literal consequence, pinned as intended: with >= 50 genuine
    memories in the pool the family takes no place in it, so a family memory
    that would clear the floor is not a semantic result (it can still surface
    through the keyword side)."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine_cosines = [0.80 - i * 0.005 for i in range(CANDIDATE_POOL + 2)]
    genuine, family = _seed(store, monkeypatch, genuine_cosines, [0.95, 0.94, 0.93])
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        _warm()
        scripted = {g.content: 5.0 for g in genuine}
        scripted.update({f.content: 50.0 for f in family})
        _install_reranker(monkeypatch, _Recording(scripted))
    seen = _spy_rank_and_gate(monkeypatch, "brain.memory.semantic_recall")

    result = run_semantic_recall(store, tmp_path, _QUERY)

    (coarse,) = seen
    coarse_ids = [mid for mid, _ in coarse]
    assert set(coarse_ids).isdisjoint(_ids(family))
    assert coarse_ids == _ids(genuine[:CANDIDATE_POOL])
    assert result is not None and result.path == path
    assert set(_ids([*result.full, *result.snippet])).isdisjoint(_ids(family))
