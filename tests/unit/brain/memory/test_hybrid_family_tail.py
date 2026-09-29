"""Name-recall fix (spec §4, S82): the cosine tail of a reranked paragraph.

Within a reranked paragraph, the monologue-family candidates that got no
rerank slot are still candidates: they are gated by the COSINE floor and
ranked after the paragraph's reranked results (the S44 "cosine-path
monologue-family" group). One paragraph can yield results on both scales;
each result is gated only by its own scale's floor, and scores are never
compared across scales. Genuine memories are never displaced.

All offline: the R3 scripted embedder and fake rerankers, an in-tmp store,
synthetic data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.memory import semantic_recall as sr
from brain.memory.relevance import CANDIDATE_POOL
from brain.memory.semantic_recall import (
    COSINE_PATH,
    RERANKED_PATH,
    GatedRanking,
    gated_cleared,
    run_semantic_recall,
    select_gated_standouts,
)
from brain.memory.store import COSINE_SCORE_SCALE, MemoryStore
from tests.unit.brain.memory.test_monologue_last import (
    _QUERY,
    _cal_rows,
    _cosine_floor,
    _ids,
    _install_reranker,
    _Recording,
    _rerank_floor,
    _seed,
    _tool,
    _warm,
)

_RERANK_FLOOR = 1.0


def _first_rerank_case(monkeypatch, tmp_path, *, cosine_floor):
    """First rerank of a process = 5 real candidates (S24). 3 genuine at
    LOW cosine + 6 family at HIGH cosine (all in the pool): the 5 slots go to
    the 3 genuine and the 2 best family; family[2:] (4 memories) are unreranked
    and form the cosine tail."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.50, 0.40, 0.30], [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
    )
    _cosine_floor(store, cosine_floor)
    _rerank_floor(store, floor=_RERANK_FLOOR)
    scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
    scripted.update({family[0].content: 4.0, family[1].content: 3.0})
    rec = _Recording(scripted)
    _install_reranker(monkeypatch, rec)
    return store, genuine, family, rec


def test_an_unreranked_family_memory_above_the_cosine_floor_surfaces_after_the_reranked_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    kept = _ids([*result.full, *result.snippet])
    # reranked genuine (3, best score first), reranked family (2), then the cosine tail (4, by cosine)
    assert kept == _ids(reversed(genuine)) + _ids(family[:2]) + _ids(family[2:6])
    assert [h.path for h in result.hits] == [RERANKED_PATH] * 5 + [COSINE_PATH] * 4
    assert [h.monologue_family for h in result.hits] == [False] * 3 + [True] * 6


def test_the_primary_scores_and_the_tail_scores_stay_on_their_own_scales(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert set(result.scores) == set(_ids(genuine)) | set(_ids(family[:2])), "reranked only"
    assert result.scores[family[0].id] == pytest.approx(4.0), "the reranker score, unmodified"
    assert set(result.tail_scores) == set(_ids(family[2:])), "every examined tail candidate"
    assert result.tail_scores[family[2].id] == pytest.approx(0.97, abs=1e-5), "raw cosine"
    assert result.tail_scale == COSINE_SCORE_SCALE
    assert result.tail_pass_mark == pytest.approx(0.5)
    assert result.pass_mark == pytest.approx(_RERANK_FLOOR)
    by_id = {h.memory.id: h for h in result.hits}
    assert by_id[family[2].id].score == pytest.approx(0.97, abs=1e-5)
    assert by_id[family[0].id].score == pytest.approx(4.0)


def test_an_unreranked_family_memory_below_the_cosine_floor_does_not_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cosine floor 0.995 is above every family cosine (0.94-0.99): the tail is
    examined and every candidate fails its own scale's floor."""
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.995)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert _ids([*result.full, *result.snippet]) == _ids(reversed(genuine)) + _ids(family[:2])
    assert {h.path for h in result.hits} == {RERANKED_PATH}
    assert set(result.tail_scores) == set(_ids(family[2:])), "examined, not surfaced"


def test_each_result_is_gated_only_by_its_own_scales_floor_never_the_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No cross-scale comparison: (a) a tail cosine (0.97) far above the rerank
    floor's number (1.0 is > 0.97, so use a rerank floor of 0.1) but below the
    cosine floor 0.99 is NOT admitted; (b) a rerank floor no reranked score
    reaches is covered by its own test below."""
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.99)
    _rerank_floor(store, floor=0.1)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    # cosine 0.97..0.94 pass the RERANK floor's number (0.1) but not the cosine floor (0.99)
    assert {h.path for h in result.hits} == {RERANKED_PATH}
    assert set(_ids([*result.full, *result.snippet])).isdisjoint(_ids(family[2:]))


def test_a_rerank_floor_no_reranked_score_reaches_leaves_the_tail_as_the_whole_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tail is gated by the cosine floor only: with a rerank floor nothing
    reranked can reach, the unreranked family memories (cosine above 0.5) are
    the entire result, on the cosine scale."""
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    _rerank_floor(store, floor=1e9)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert _ids([*result.full, *result.snippet]) == _ids(family[2:])
    assert {h.path for h in result.hits} == {COSINE_PATH}


def test_the_cap_drops_tail_results_before_any_reranked_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warm process reranks all 50 genuine memories (all above the rerank
    floor); the 3 family memories are unreranked, above the cosine floor, and
    cut by the 9-cap: 9 reranked genuine results, no genuine memory displaced."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.80 - i * 0.005 for i in range(CANDIDATE_POOL)], [0.95, 0.94, 0.93]
    )
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=_RERANK_FLOOR)
    _warm()
    _install_reranker(
        monkeypatch, _Recording({g.content: 5.0 + i * 0.01 for i, g in enumerate(genuine)})
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    kept = _ids([*result.full, *result.snippet])
    assert len(kept) == sr.MAX_STANDOUT_COUNT
    assert set(kept) <= set(_ids(genuine))
    assert {h.path for h in result.hits} == {RERANKED_PATH}


def test_a_family_memory_surfaces_semantically_at_fifty_genuine_when_no_genuine_clears(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S16 "can still appear" at real-persona scale on the reranked path: 50
    genuine memories fill the rerank prefix but none clears the rerank floor;
    the unreranked family memories, above the cosine floor, are the result."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.80 - i * 0.005 for i in range(CANDIDATE_POOL)], [0.95, 0.94, 0.93]
    )
    _cosine_floor(store, 0.9)
    _rerank_floor(store, floor=_RERANK_FLOOR)
    _warm()
    _install_reranker(monkeypatch, _Recording({g.content: -3.0 for g in genuine}))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert _ids([*result.full, *result.snippet]) == _ids(family)
    assert {h.path for h in result.hits} == {COSINE_PATH}


def test_a_reranked_family_memory_below_the_rerank_floor_is_not_rescued_by_its_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only UNRERANKED family memories form the tail: a family memory that WAS
    reranked and failed the rerank floor stays out even though its cosine is
    above the cosine floor."""
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    rec._fake = type(rec._fake)(  # noqa: SLF001 - re-script: family[1] scores below the floor
        scores={
            **dict.fromkeys(sr.reranker_mod.ANCHOR_POOL, 0.0),
            **{g.content: 5.0 + i for i, g in enumerate(genuine)},
            family[0].content: 4.0,
            family[1].content: -2.0,
        },
        default=-1000.0,
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    kept = _ids([*result.full, *result.snippet])
    assert family[1].id not in kept
    assert kept == _ids(reversed(genuine)) + _ids([family[0]]) + _ids(family[2:6])


def test_unreranked_genuine_memories_are_not_part_of_the_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S53: candidates beyond the width are dropped for the paragraph. Only the
    family gets a second chance. 8 genuine (cosine above the cosine floor) and
    3 family: the first rerank scores 5 genuine; the 3 unreranked genuine never
    surface, the 3 unreranked family do."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.90 - i * 0.01 for i in range(8)], [0.99, 0.98, 0.97]
    )
    _cosine_floor(store, 0.4)
    _rerank_floor(store, floor=_RERANK_FLOOR)
    _install_reranker(
        monkeypatch, _Recording({g.content: 5.0 - i * 0.1 for i, g in enumerate(genuine)})
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert _ids([*result.full, *result.snippet]) == _ids(genuine[:5]) + _ids(family)
    assert set(_ids(genuine[5:])).isdisjoint(_ids([*result.full, *result.snippet]))


def test_no_tail_when_every_family_memory_was_reranked_and_the_cosine_floor_is_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.50, 0.40, 0.30], [0.99, 0.98])
    _rerank_floor(store, floor=_RERANK_FLOOR)
    scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
    scripted.update({f.content: 4.0 for f in family})
    _install_reranker(monkeypatch, _Recording(scripted))
    reads: list[str] = []
    real = store.get_cosine_floor
    monkeypatch.setattr(store, "get_cosine_floor", lambda m: reads.append(m) or real(m))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert result.tail_scores == {} and result.tail_scale is None
    assert reads == [], "no unreranked family candidate: the cosine floor is not consulted"


def test_a_cosine_floor_failure_keeps_the_reranked_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    def _boom(model_id):
        raise RuntimeError("cosine floor unavailable")

    monkeypatch.setattr(store, "get_cosine_floor", _boom)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == RERANKED_PATH
    assert _ids([*result.full, *result.snippet]) == _ids(reversed(genuine)) + _ids(family[:2])
    assert result.tail_scores == {}


def test_a_passive_turn_logs_one_calibration_row_per_scale_each_stamped_with_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)

    run_semantic_recall(store, tmp_path, _QUERY)

    rows = _cal_rows(store)
    assert [r["score_scale"] for r in rows] == ["normalized", COSINE_SCORE_SCALE]
    normalized_ids = json.loads(rows[0]["candidate_ids"])
    cosine_ids = json.loads(rows[1]["candidate_ids"])
    assert set(normalized_ids) == set(_ids(genuine) + _ids(family[:2]))
    assert cosine_ids == _ids(family[2:6]), "the unreranked family, in cosine order"
    assert set(normalized_ids).isdisjoint(cosine_ids), "each candidate logged on one scale only"


# ---------------------------------------------------------------------------
# The pure selection helpers
# ---------------------------------------------------------------------------


def _ranking(primary, tail=None, floor=1.0, tail_floor=0.5) -> GatedRanking:
    tail_rank = (
        GatedRanking(COSINE_PATH, COSINE_SCORE_SCALE, tail, tail_floor)
        if tail is not None
        else None
    )
    return GatedRanking(RERANKED_PATH, "normalized", primary, floor, tail=tail_rank)


def test_gated_cleared_gates_each_segment_by_its_own_floor_and_keeps_segment_order() -> None:
    gated = _ranking(
        [("g1", 5.0), ("g2", 0.9), ("f1", 3.0)],  # 0.9 < 1.0 fails the rerank floor
        [("t1", 0.9), ("t2", 0.3)],  # 0.9 < the rerank floor's number 1.0 but clears 0.5
    )

    assert gated_cleared(gated) == [
        ("g1", 5.0, RERANKED_PATH),
        ("f1", 3.0, RERANKED_PATH),
        ("t1", 0.9, COSINE_PATH),
    ]


def test_select_gated_standouts_caps_at_nine_dropping_the_tail_first() -> None:
    primary = [(f"g{i}", 5.0) for i in range(8)]
    tail = [(f"t{i}", 0.9) for i in range(4)]

    tiers = select_gated_standouts(_ranking(primary, tail))

    assert tiers is not None
    assert [*tiers.full_ids, *tiers.snippet_ids] == [f"g{i}" for i in range(8)] + ["t0"]
    assert len(tiers.full_ids) == 5


def test_select_gated_standouts_is_none_when_nothing_clears_its_own_floor() -> None:
    assert select_gated_standouts(_ranking([("g", 0.5)], [("t", 0.4)])) is None


def test_a_ranking_without_a_tail_gates_exactly_as_before() -> None:
    gated = _ranking([("g", 5.0), ("f", 0.2)])

    assert gated_cleared(gated) == [("g", 5.0, RERANKED_PATH)]


# ---------------------------------------------------------------------------
# The tool behaves the same (its semantic side is `_semantic_top_k`)
# ---------------------------------------------------------------------------


def _spy_tool_semantic(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture what the tool's semantic side returned (before the keyword side
    is merged below it)."""
    import brain.tools.impls.search_memories as tool

    captured: list = []
    real = tool._semantic_top_k  # noqa: SLF001

    def _spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured.append(out)
        return out

    monkeypatch.setattr(tool, "_semantic_top_k", _spy)
    return captured


def test_the_tool_surfaces_an_unreranked_family_memory_after_the_reranked_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    captured = _spy_tool_semantic(monkeypatch)

    got = _tool(tmp_path, store, limit=9)

    (semantic,) = captured
    assert _ids(semantic) == _ids(reversed(genuine)) + _ids(family[:2]) + _ids(family[2:6])
    assert got[:9] == _ids(semantic)


def test_the_tool_does_not_surface_an_unreranked_family_memory_below_the_cosine_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.995)
    captured = _spy_tool_semantic(monkeypatch)

    _tool(tmp_path, store, limit=9)

    (semantic,) = captured
    assert _ids(semantic) == _ids(reversed(genuine)) + _ids(family[:2])


def test_the_tool_gates_the_tail_by_the_cosine_floor_alone_and_logs_no_calibration_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    _rerank_floor(store, floor=1e9)
    captured = _spy_tool_semantic(monkeypatch)

    _tool(tmp_path, store, limit=9)

    (semantic,) = captured
    assert _ids(semantic) == _ids(family[2:]), "the tail alone, gated by the cosine floor"
    assert _cal_rows(store) == [], "the tool never writes calibration rows (S56)"


def test_the_tool_applies_its_limit_to_the_two_segments_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine, family, rec = _first_rerank_case(monkeypatch, tmp_path, cosine_floor=0.5)
    captured = _spy_tool_semantic(monkeypatch)

    _tool(tmp_path, store, limit=6)

    (semantic,) = captured
    assert _ids(semantic) == _ids(reversed(genuine)) + _ids(family[:2]) + _ids([family[2]])
