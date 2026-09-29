"""Name-recall fix R3 follow-up (spec §4, S77 revised): each query's coarse
cosine pool is the top 50 GENUINE memories by cosine plus the monologue-family
memories that sit in today's plain cosine top 50 (so the pool can exceed 50).

Consequences pinned here: a genuine memory is never displaced by the family
(genuine memories fill their own 50 places); a family memory inside the plain
top 50 stays in the pool, after the genuine ones, so it can still surface
(S16 "can still appear"); a family memory outside the plain top 50 is not
added. Rerank slots go to the pool's prefix, genuine first. No new constant,
no score multiplier, and one pass over the candidates.

All offline: the R3 scripted embedder and fake rerankers, an in-tmp store,
synthetic data only.
"""

from __future__ import annotations

import random
from collections import Counter
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


def _ids_of(cut) -> list[str]:
    return [mid for mid, _ in cut]


def _spy_rank_and_gate(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> list[list[tuple[str, float]]]:
    """Record the coarse cut each caller hands `rank_and_gate` (the pool).
    Since name-recall fix R6 both callers reach it through
    `semantic_recall.search_paragraphs`, so the spy sits there; `module` names
    the caller under test for the reader."""
    seen: list[list[tuple[str, float]]] = []
    real = sr.rank_and_gate

    def _spy(store, query, pool, coarse, **kwargs):
        seen.append(list(coarse))
        return real(store, query, pool, coarse, **kwargs)

    del module
    monkeypatch.setattr(sr, "rank_and_gate", _spy)
    return seen


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_a_family_memory_in_the_plain_top_fifty_is_in_the_pool_alongside_fifty_genuine() -> None:
    """5 family memories at 0.90 and 50 genuine at 0.50-0.75: the plain top 50
    holds the 5 family and 45 genuine, so the pool is ALL 50 genuine plus the
    5 family (55 entries), genuine first."""
    specs = [(f"f{i}", _FAMILY_TYPES[i % 4], 0.90 - i * 0.001) for i in range(5)]
    specs += [(f"g{i}", "event", 0.75 - i * 0.005) for i in range(CANDIDATE_POOL)]
    scored, pool = _pool_of(specs)

    ids = _ids_of(genuine_first_coarse_cut(scored, pool))

    assert len(ids) == CANDIDATE_POOL + 5
    assert ids[:CANDIDATE_POOL] == [f"g{i}" for i in range(CANDIDATE_POOL)]
    assert ids[CANDIDATE_POOL:] == [f"f{i}" for i in range(5)]


def test_a_family_memory_outside_the_plain_top_fifty_is_not_in_the_pool() -> None:
    specs = [(f"g{i}", "event", 0.95 - i * 0.002) for i in range(CANDIDATE_POOL)]
    specs += [("f_low", "monologue", 0.60), ("f_lower", "monologue_trace", 0.10)]
    scored, pool = _pool_of(specs)

    ids = _ids_of(genuine_first_coarse_cut(scored, pool))

    assert ids == [f"g{i}" for i in range(CANDIDATE_POOL)]


def test_family_memories_that_outscore_every_genuine_one_never_displace_a_genuine_one() -> None:
    """60 family at 0.99 and 3 genuine far below: the plain top 50 is all
    family, so the pool is the 3 genuine plus those 50 family (53 entries)."""
    specs = [(f"f{i}", _FAMILY_TYPES[i % 4], 0.99 - i * 0.001) for i in range(60)]
    specs += [("g0", "event", 0.30), ("g1", "conversation", 0.20), ("g2", "event", 0.10)]
    scored, pool = _pool_of(specs)

    ids = _ids_of(genuine_first_coarse_cut(scored, pool))

    assert ids[:3] == ["g0", "g1", "g2"], "every genuine memory, by cosine, first"
    assert ids[3:] == [f"f{i}" for i in range(CANDIDATE_POOL)], "family: the plain top 50 only"
    assert "f50" not in ids


def test_a_genuine_memory_far_below_every_family_cosine_still_leads_the_pool() -> None:
    """No hidden multiplier: near-zero and negative genuine cosines, family at
    0.9, and the genuine memories still come first."""
    specs = [(f"f{i}", "monologue_trace", 0.90 - i * 0.001) for i in range(60)]
    specs += [("g_near_zero", "event", 0.05), ("g_negative", "event", -0.20)]
    scored, pool = _pool_of(specs)

    ids = _ids_of(genuine_first_coarse_cut(scored, pool))

    assert ids[:2] == ["g_near_zero", "g_negative"]


def test_fewer_genuine_than_fifty_still_takes_only_the_plain_top_fifty_family() -> None:
    specs = [("g_lo", "event", 0.10), ("g_hi", "event", 0.40)]
    specs += [(f"f{i}", "monologue", 0.5 + i * 0.001) for i in range(10)]
    scored, pool = _pool_of(specs)

    # size 5: the plain top 5 is f9..f5, so the pool is the 2 genuine then those 5.
    ids = _ids_of(genuine_first_coarse_cut(scored, pool, size=5))

    assert ids == ["g_hi", "g_lo", "f9", "f8", "f7", "f6", "f5"]


def test_scores_pass_through_unchanged() -> None:
    scored, pool = _pool_of([("f", "monologue", 0.99), ("g", "event", 0.25)])

    assert genuine_first_coarse_cut(scored, pool) == [("g", 0.25), ("f", 0.99)]


def _reference(scored, pool, size):
    """The spec sentence, written the slow obvious way (two full scans)."""
    by_cos = sorted(scored, key=lambda p: -p[1])
    fam = lambda p: sr.is_monologue_family(pool[p[0]][0])  # noqa: E731
    genuine = [p for p in by_cos if not fam(p)][:size]
    family = [p for p in by_cos[:size] if fam(p)]
    return genuine + family


def test_the_cut_equals_the_two_scan_reference_including_ties() -> None:
    rng = random.Random(77)
    specs = [
        (
            f"m{i}",
            rng.choice(("event", "monologue", "conversation", "monologue_trace")),
            rng.choice((0.1, 0.2, 0.3, 0.4, 0.5)),  # coarse values force many ties
        )
        for i in range(300)
    ]
    scored, pool = _pool_of(specs)
    for size in (1, 7, CANDIDATE_POOL, 150, 299, 300, 500):
        assert genuine_first_coarse_cut(scored, pool, size=size) == _reference(scored, pool, size)


def test_the_scan_is_one_pass_the_family_flag_is_read_at_most_once_per_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [(f"m{i}", "monologue" if i % 2 else "event", i / 1000) for i in range(400)]
    scored, pool = _pool_of(specs)
    seen: Counter[str] = Counter()
    real = sr.is_monologue_family

    def _counting(memory):
        seen[memory.content] += 1
        return real(memory)

    monkeypatch.setattr(sr, "is_monologue_family", _counting)

    genuine_first_coarse_cut(scored, pool)

    assert max(seen.values()) == 1, "no entry is examined twice"
    assert len(seen) < len(scored), "the walk stops once the pool is complete"


# ---------------------------------------------------------------------------
# Through passive recall and the tool (the two callers of the cut)
# ---------------------------------------------------------------------------


def _flood(store, monkeypatch, n_family: int = 55):
    """Three genuine memories at LOW cosine; `n_family` family memories at
    HIGH cosine, so today's plain top 50 holds only family memories."""
    return _seed(
        store,
        monkeypatch,
        [0.60, 0.55, 0.50],
        [0.99 - i * 0.001 for i in range(n_family)],
    )


def _floors_and_path(store, monkeypatch, genuine, family, path, *, family_scores=None):
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
        return None
    _warm()
    scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
    scripted.update(family_scores or {})
    rec = _Recording(scripted)
    _install_reranker(monkeypatch, rec)
    return rec


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_passive_recall_pool_is_the_genuine_then_the_plain_top_fifty_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _flood(store, monkeypatch)
    _floors_and_path(store, monkeypatch, genuine, family, path)
    seen = _spy_rank_and_gate(monkeypatch, "brain.memory.semantic_recall")

    result = run_semantic_recall(store, tmp_path, _QUERY)

    (coarse,) = seen
    coarse_ids = _ids_of(coarse)
    assert len(coarse_ids) == 3 + CANDIDATE_POOL, "the pool exceeds 50 here"
    assert coarse_ids[:3] == _ids(genuine)
    assert coarse_ids[3:] == _ids(family[:CANDIDATE_POOL]), "family: the plain top 50, by cosine"
    assert result is not None and result.path == path
    assert set(_ids(genuine)) <= set(_ids([*result.full, *result.snippet]))


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_the_tool_pool_is_the_genuine_then_the_plain_top_fifty_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _flood(store, monkeypatch)
    rec = _floors_and_path(store, monkeypatch, genuine, family, path)
    seen = _spy_rank_and_gate(monkeypatch, "brain.tools.impls.search_memories")

    got = _tool(tmp_path, store, limit=5)

    (coarse,) = seen
    coarse_ids = _ids_of(coarse)
    assert coarse_ids[:3] == _ids(genuine)
    assert coarse_ids[3:] == _ids(family[:CANDIDATE_POOL])
    assert set(_ids(genuine)) <= set(got)
    if path == "reranked":
        assert rec.scored_calls(), "the reranked variant must actually have reranked"


def test_a_family_memory_above_the_cosine_floor_still_surfaces_after_fifty_genuine_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S16 "can still appear": 52 genuine memories below the cosine floor and
    3 family memories above it, all inside the plain top 50 (the genuine ones
    sit lower). The pool keeps the family, so the cosine path surfaces them;
    no genuine memory clears the floor, so none is displaced."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store,
        monkeypatch,
        [0.30 - i * 0.001 for i in range(CANDIDATE_POOL + 2)],
        [0.95, 0.94, 0.93],
    )
    _cosine_floor(store, 0.4)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    assert _ids([*result.full, *result.snippet]) == _ids(family)


def test_the_tool_surfaces_the_same_family_memories_on_the_cosine_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store,
        monkeypatch,
        [0.30 - i * 0.001 for i in range(CANDIDATE_POOL + 2)],
        [0.95, 0.94, 0.93],
    )
    _cosine_floor(store, 0.4)
    _no_reranker(monkeypatch)

    # The tool tops the semantic results up from its keyword side, so compare the head.
    assert _tool(tmp_path, store, limit=5)[:3] == _ids(family)


def test_family_memories_take_rerank_slots_only_after_every_genuine_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """First rerank of a process = 5 real candidates (S24). 3 genuine at low
    cosine and 6 family at high cosine (all in the plain top 50): the 5 slots
    go to the 3 genuine, then the 2 best family, and the other family
    candidates are never scored."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.50, 0.40, 0.30], [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
    )
    _rerank_floor(store)
    rec = _Recording()
    _install_reranker(monkeypatch, rec)

    run_semantic_recall(store, tmp_path, _QUERY)

    (scored,) = rec.scored_calls()
    assert scored[:5] == [g.content for g in genuine] + [f.content for f in family[:2]]
    assert not any(f.content in scored for f in family[2:])


def test_with_fifty_genuine_memories_the_rerank_prefix_holds_no_family_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The family stays in the pool (behind the 50 genuine ones) but
    `rerank_for_recall` sees only the first 50 documents, so a family memory is
    never RERANKED when 50 genuine memories exist (its cosine tail, S82, is
    covered in `test_hybrid_family_tail.py`; here the 9-cap is filled by
    reranked genuine results, so no tail result surfaces)."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.80 - i * 0.005 for i in range(CANDIDATE_POOL)], [0.95, 0.94]
    )
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 5.0 for g in genuine}
    scripted.update({f.content: 50.0 for f in family})
    rec = _Recording(scripted)
    _install_reranker(monkeypatch, rec)
    seen = _spy_rank_and_gate(monkeypatch, "brain.memory.semantic_recall")

    result = run_semantic_recall(store, tmp_path, _QUERY)

    (coarse,) = seen
    assert _ids_of(coarse) == _ids(genuine) + _ids(family)
    assert rec.scored_calls() and not any(
        f.content in call for call in rec.scored_calls() for f in family
    )
    assert result is not None and result.path == "reranked"
    assert set(_ids([*result.full, *result.snippet])).isdisjoint(_ids(family))


def test_the_tool_removes_excluded_ids_before_the_cut_so_they_take_no_place_in_the_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """53 genuine memories, the 3 best excluded: the pool is the 50 remaining
    genuine ones (an exclusion applied AFTER the cut would leave only 47) plus
    the plain-top-50 family."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.80 - i * 0.005 for i in range(CANDIDATE_POOL + 3)], [0.95]
    )
    _cosine_floor(store, 0.4)
    _no_reranker(monkeypatch)
    seen = _spy_rank_and_gate(monkeypatch, "brain.tools.impls.search_memories")
    excluded = _ids(genuine[:3])

    _tool(tmp_path, store, limit=5, exclude_ids=excluded)

    (coarse,) = seen
    coarse_ids = _ids_of(coarse)
    assert set(coarse_ids).isdisjoint(excluded)
    assert coarse_ids == _ids(genuine[3:]) + _ids(family)
