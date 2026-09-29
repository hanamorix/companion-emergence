"""Name-recall fix R3 (spec §4; criterion C8, semantic side): the monologue
family ranks last and never displaces a genuine memory.

The kindled's own generated monologue memories (`monologue`, `monologue_trace`,
`monologue_emotion`, `monologue_soul_candidate`) rank AFTER genuine memories
within each path, and genuine candidates are taken first for rerank slots. No
score multiplier. Covered here for passive recall (`run_semantic_recall`, and
once through the rendered block) and the `search_memories` tool, on both the
reranked and the cosine path. Keyword ordering (C8's keyword half) arrives
with R4.

All offline: a scripted embedder, scripted fake rerankers, an in-tmp store.
Synthetic data only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.chat.prompt import _build_recall_block
from brain.dev_constants import MONOLOGUE_FAMILY_TYPES
from brain.memory import reranker as reranker_mod
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import ANCHOR_POOL, FakeRerankerProvider, RerankerProvider
from brain.memory.semantic_recall import (
    MAX_STANDOUT_COUNT,
    SemanticHit,
    genuine_first,
    genuine_first_ranking,
    is_monologue_family,
    run_semantic_recall,
)
from brain.memory.store import COSINE_SCORE_SCALE, Memory, MemoryStore
from brain.tools.dispatch import dispatch

_DIM = 384
_EMBEDDER_ID = "r3-test-embedder"
_RERANKER_ID = "fake-reranker"
_QUERY = "r3 monologue last query"

_FAMILY = ("monologue", "monologue_trace", "monologue_emotion", "monologue_soul_candidate")


class _Embedder(EmbeddingProvider):
    """e0 for the literal query; anything else a zero vector (the pool's
    vectors are seeded directly on the rows)."""

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(_DIM, dtype=np.float32)
        if text == _QUERY:
            v[0] = 1.0
        return v

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def embedding_dim(self) -> int:
        return _DIM

    def model_id(self) -> str:
        return _EMBEDDER_ID


def _vec_with_cosine(c: float) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = c
    v[1] = math.sqrt(max(0.0, 1.0 - c * c))
    return v


class _Recording(RerankerProvider):
    """Scripted scores by content (anchors at 0.0, so normalized == raw);
    records every call so the tests can read the rerank prefix."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = -1_000.0):
        self._fake = FakeRerankerProvider(
            scores={**dict.fromkeys(ANCHOR_POOL, 0.0), **(scores or {})}, default=default
        )
        self.calls: list[list[str]] = []

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append(list(documents))
        return self._fake.rerank(query, documents)

    def model_id(self) -> str:
        return _RERANKER_ID

    def scored_calls(self) -> list[list[str]]:
        """Calls that scored more than the single-document warm-ups."""
        return [c for c in self.calls if len(c) > 1]


def _mem(store: MemoryStore, content: str, memory_type: str, cosine: float) -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="d")
    store.create(m)
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (_vec_with_cosine(cosine).tobytes(), _EMBEDDER_ID, m.id),
    )
    store._conn.commit()  # noqa: SLF001
    return m


def _seed(
    store: MemoryStore,
    monkeypatch: pytest.MonkeyPatch,
    genuine_cosines: list[float],
    family_cosines: list[float],
    *,
    genuine_type: str = "event",
) -> tuple[list[Memory], list[Memory]]:
    """Genuine memories then monologue-family memories (family types
    cycle), each with the scripted cosine to the query."""
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: _Embedder())
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _EMBEDDER_ID)
    genuine = [
        _mem(store, f"r3 genuine memory {i}", genuine_type, c) for i, c in enumerate(genuine_cosines)
    ]
    family = [
        _mem(store, f"r3 monologue memory {i}", _FAMILY[i % len(_FAMILY)], c)
        for i, c in enumerate(family_cosines)
    ]
    return genuine, family


def _rerank_floor(store: MemoryStore, floor: float = 1.0) -> None:
    store.write_reranker_floor(
        _RERANKER_ID, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


def _cosine_floor(store: MemoryStore, floor: float = 0.5) -> None:
    store.write_cosine_floor(
        _EMBEDDER_ID, floor=floor, raw_fit_floor=floor, sample_pairs=10, is_cold_start=False
    )


def _warm() -> None:
    """A process that has already reranked (S24): the width fit then covers
    the whole (<= 50) pool instead of the first-rerank minimum of 5."""
    reranker_mod._record_rerank_cost(_RERANKER_ID, 1_000, 1e-6, None)


def _install_reranker(monkeypatch: pytest.MonkeyPatch, rec: RerankerProvider) -> None:
    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", lambda **kw: rec)


def _no_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**kwargs):
        raise RuntimeError("simulated reranker construction failure")

    monkeypatch.setattr("brain.memory.reranker.build_reranker_provider", _boom)


def _ids(memories) -> list[str]:
    return [m.id for m in memories]


def _cal_rows(store: MemoryStore) -> list:
    return store._conn.execute(  # noqa: SLF001
        "SELECT candidate_ids, reranker_scores, candidate_docs, score_scale FROM calibration_log "
        "ORDER BY id"
    ).fetchall()


def _ctx(tmp_path: Path, store: MemoryStore) -> dict:
    return {"store": store, "hebbian": HebbianMatrix(":memory:"), "persona_dir": tmp_path}


def _tool(tmp_path: Path, store: MemoryStore, **args) -> list[str]:
    out = dispatch("search_memories", {"query": _QUERY, **args}, **_ctx(tmp_path, store))
    assert out["mode"] == "semantic"
    return [m["id"] for m in out["memories"]]


# ---------------------------------------------------------------------------
# P-20: the family set, and how a memory is classified
# ---------------------------------------------------------------------------


def test_the_monologue_family_is_the_four_spec_types() -> None:
    assert MONOLOGUE_FAMILY_TYPES == frozenset(_FAMILY)


@pytest.mark.parametrize("memory_type", _FAMILY)
def test_each_family_type_is_classified_as_monologue_family(memory_type: str) -> None:
    assert is_monologue_family(Memory.create_new(content="x", memory_type=memory_type, domain="d"))


@pytest.mark.parametrize(
    "memory_type", ["event", "conversation", "dream", "consolidated", "meta", "journal_entry"]
)
def test_every_other_type_is_genuine(memory_type: str) -> None:
    assert not is_monologue_family(
        Memory.create_new(content="x", memory_type=memory_type, domain="d")
    )


def test_genuine_first_is_a_stable_partition_not_a_score_change(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    a = Memory.create_new(content="a", memory_type="monologue", domain="d")
    b = Memory.create_new(content="b", memory_type="event", domain="d")
    c = Memory.create_new(content="c", memory_type="monologue_trace", domain="d")
    d = Memory.create_new(content="d", memory_type="conversation", domain="d")
    pool = {m.id: (m, np.zeros(1, dtype=np.float32)) for m in (a, b, c, d)}
    del store
    assert genuine_first([a.id, b.id, c.id, d.id], pool) == [b.id, d.id, a.id, c.id]
    scored = [(a.id, 0.99), (b.id, 0.20), (c.id, 0.98), (d.id, 0.30), (b.id, 0.20)]
    ranked = genuine_first_ranking(scored, pool)
    assert ranked == [(d.id, 0.30), (b.id, 0.20), (b.id, 0.20), (a.id, 0.99), (c.id, 0.98)]


# ---------------------------------------------------------------------------
# C8: the rerank prefix takes every genuine candidate before any monologue one
# ---------------------------------------------------------------------------


def test_the_rerank_prefix_takes_genuine_candidates_before_any_monologue_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """First rerank of a process = 5 real candidates (S24). Three genuine
    memories with LOW cosine and six monologue-family memories with HIGH
    cosine: the five slots go to the three genuine ones, then the two best
    monologue ones (not to five monologue ones, which is what today's
    cosine-only prefix does)."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.50, 0.40, 0.30], [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
    )
    _rerank_floor(store)
    rec = _Recording()
    _install_reranker(monkeypatch, rec)

    run_semantic_recall(store, tmp_path, _QUERY)

    (scored,) = rec.scored_calls()
    assert len(scored) == 5 + 2, "5 real candidates plus k = 5 // 2 anchors"
    assert scored[:5] == [g.content for g in genuine] + [f.content for f in family[:2]]


def test_the_rerank_prefix_keeps_cosine_order_inside_each_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.30, 0.90, 0.60, 0.10, 0.20, 0.80], [0.50, 0.99, 0.70]
    )
    _rerank_floor(store)
    _warm()
    rec = _Recording()
    _install_reranker(monkeypatch, rec)

    run_semantic_recall(store, tmp_path, _QUERY)

    (scored,) = rec.scored_calls()
    n_real = len(genuine) + len(family)
    expected_genuine = [genuine[i].content for i in (1, 5, 2, 0, 4, 3)]  # 0.90 0.80 0.60 0.30 0.20 0.10
    expected_family = [family[i].content for i in (1, 2, 0)]  # 0.99 0.70 0.50
    assert scored[:n_real] == expected_genuine + expected_family


# ---------------------------------------------------------------------------
# C8: reranked path results
# ---------------------------------------------------------------------------


def test_reranked_results_put_every_genuine_memory_above_every_monologue_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The monologue memories score far HIGHER than the genuine ones; they
    still rank after every genuine result, and their scores are reported
    unchanged (no multiplier)."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6] * 5, [0.9] * 5)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 3.0 + i for i, g in enumerate(genuine)}  # 3..7
    scripted.update({f.content: 20.0 + i for i, f in enumerate(family)})  # 20..24
    _install_reranker(monkeypatch, _Recording(scripted))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "reranked"
    order = [*result.full, *result.snippet]
    assert _ids(order) == _ids(reversed(genuine))[:5] + _ids(reversed(family))[:4]
    assert _ids(result.full) == _ids(reversed(genuine)), "the five full slots are all genuine"
    assert result.scores[family[4].id] == pytest.approx(24.0), "no score multiplier"
    assert result.scores[genuine[4].id] == pytest.approx(7.0)


def test_the_cap_drops_monologue_memories_before_any_floor_clearing_genuine_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6] * 6, [0.9] * 6)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 2.0 + i for i, g in enumerate(genuine)}
    scripted.update({f.content: 50.0 + i for i, f in enumerate(family)})
    _install_reranker(monkeypatch, _Recording(scripted))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    kept = _ids([*result.full, *result.snippet])
    assert len(kept) == MAX_STANDOUT_COUNT == 9
    assert set(_ids(genuine)) <= set(kept), "no floor-clearing genuine memory dropped"
    assert kept[6:] == _ids(reversed(family))[:3], "the 3 best monologue memories fill the rest"


def test_more_genuine_than_the_cap_leaves_no_room_for_monologue_memories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6] * 12, [0.95] * 3)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 2.0 + i for i, g in enumerate(genuine)}
    scripted.update({f.content: 90.0 for f in family})
    _install_reranker(monkeypatch, _Recording(scripted))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    kept = _ids([*result.full, *result.snippet])
    assert kept == _ids(reversed(genuine))[:9], "nine best genuine memories, no monologue one"


def test_a_pool_of_only_monologue_memories_still_surfaces_them_by_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ranking last is not exclusion: with no genuine memory in the pool the
    family results are returned, best score first."""
    store = MemoryStore(tmp_path / "memories.db")
    _, family = _seed(store, monkeypatch, [], [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    _rerank_floor(store, floor=1.0)
    _warm()
    _install_reranker(
        monkeypatch, _Recording({f.content: 5.0 + i for i, f in enumerate(family)})
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert _ids([*result.full, *result.snippet]) == _ids(reversed(family))


def test_the_result_carries_one_ordered_tagged_hit_per_surfaced_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6] * 3, [0.9] * 3)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 5.0 + i for i, g in enumerate(genuine)}
    scripted.update({f.content: 9.0 + i for i, f in enumerate(family)})
    _install_reranker(monkeypatch, _Recording(scripted))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    assert all(isinstance(h, SemanticHit) for h in result.hits)
    assert _ids(h.memory for h in result.hits) == _ids([*result.full, *result.snippet])
    assert [h.monologue_family for h in result.hits] == [False] * 3 + [True] * 3
    assert {h.path for h in result.hits} == {"reranked"}
    assert {h.paragraph for h in result.hits} == {0}
    assert [h.score for h in result.hits] == [result.scores[h.memory.id] for h in result.hits]


# ---------------------------------------------------------------------------
# C8: cosine path (no rerank)
# ---------------------------------------------------------------------------


def test_cosine_path_ranks_genuine_memories_first_even_at_lower_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.80, 0.85, 0.90, 0.70, 0.75, 0.60], [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
    )
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"
    kept = _ids([*result.full, *result.snippet])
    # genuine by descending cosine (0.90 0.85 0.80 0.75 0.70 0.60), then the 3 best monologue ones
    assert kept == [genuine[i].id for i in (2, 1, 0, 4, 3, 5)] + _ids(family[:3])
    assert _ids(result.full) == kept[:5]
    assert all(h.path == "cosine" for h in result.hits)


def test_cosine_path_calibration_row_follows_the_path_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S60 / C2e with R3's order: the first min(9, n) candidates in the
    path's own order (genuine first, then monologue-family, each by
    cosine), pass or fail, ids / scores / documents aligned 1:1."""
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(
        store, monkeypatch, [0.60, 0.70, 0.65, 0.55], [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]
    )
    _cosine_floor(store, 0.965)  # only the first three monologue memories would clear
    _no_reranker(monkeypatch)

    run_semantic_recall(store, tmp_path, _QUERY)

    (row,) = _cal_rows(store)
    expected = [genuine[i] for i in (1, 2, 0, 3)] + family[:5]
    assert json.loads(row["candidate_ids"]) == _ids(expected)
    assert json.loads(row["candidate_docs"]) == [m.content for m in expected]
    assert json.loads(row["reranker_scores"]) == pytest.approx(
        [0.70, 0.65, 0.60, 0.55, 0.99, 0.98, 0.97, 0.96, 0.95], abs=1e-5
    )
    assert row["score_scale"] == COSINE_SCORE_SCALE


def test_cosine_path_cap_drops_monologue_memories_before_genuine_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.60] * 7, [0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
    _cosine_floor(store, 0.5)
    _no_reranker(monkeypatch)

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None
    kept = _ids([*result.full, *result.snippet])
    assert set(_ids(genuine)) <= set(kept)
    assert len(kept) == 9 and kept[7:] == _ids(family[:2])


# ---------------------------------------------------------------------------
# C8: the tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_the_tool_never_gives_a_result_slot_to_a_monologue_memory_over_a_genuine_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6, 0.65, 0.7, 0.75, 0.8, 0.85], [0.95] * 6)
    _cosine_floor(store, 0.5)
    _rerank_floor(store, floor=1.0)
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        _warm()
        scripted = {g.content: 2.0 + i for i, g in enumerate(genuine)}
        scripted.update({f.content: 40.0 + i for i, f in enumerate(family)})
        _install_reranker(monkeypatch, _Recording(scripted))
    best_first = _ids(reversed(genuine))  # both paths rank genuine 5 > 4 > ... > 0

    assert _tool(tmp_path, store, limit=4) == best_first[:4]
    assert _tool(tmp_path, store, limit=6) == best_first
    eight = _tool(tmp_path, store, limit=8)
    assert eight[:6] == best_first and set(eight[6:]) <= set(_ids(family)) and len(eight) == 8


@pytest.mark.parametrize("path", ["cosine", "reranked"])
def test_the_tool_removes_excluded_ids_before_ranking_and_still_ranks_genuine_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6, 0.65, 0.7, 0.75, 0.8, 0.85], [0.95] * 4)
    _cosine_floor(store, 0.5)
    _rerank_floor(store, floor=1.0)
    rec = None
    if path == "cosine":
        _no_reranker(monkeypatch)
    else:
        _warm()
        scripted = {g.content: 2.0 + i for i, g in enumerate(genuine)}
        scripted.update({f.content: 40.0 + i for i, f in enumerate(family)})
        rec = _Recording(scripted)
        _install_reranker(monkeypatch, rec)
    excluded = [genuine[5], family[0]]

    got = _tool(tmp_path, store, limit=8, exclude_ids=_ids(excluded))

    assert not set(_ids(excluded)) & set(got)
    assert got[:5] == _ids(reversed(genuine[:5])), "the remaining genuine memories, best first"
    assert set(got[5:]) <= set(_ids(family[1:]))
    if rec is not None:
        (scored,) = rec.scored_calls()
        assert not {m.content for m in excluded} & set(scored), "excluded before prefix selection"


# ---------------------------------------------------------------------------
# C8: passive recall through the rendered block
# ---------------------------------------------------------------------------


def test_the_rendered_block_fills_the_nine_slots_genuine_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    genuine, family = _seed(store, monkeypatch, [0.6] * 6, [0.9] * 6)
    _rerank_floor(store, floor=1.0)
    _warm()
    scripted = {g.content: 2.0 + i for i, g in enumerate(genuine)}
    scripted.update({f.content: 60.0 + i for i, f in enumerate(family)})
    _install_reranker(monkeypatch, _Recording(scripted))

    block = _build_recall_block(store, _QUERY, persona_dir=tmp_path)

    shown = re.findall(r'^    - (\S+): "', block, flags=re.MULTILINE)
    assert set(_ids(genuine)) <= set(shown), "every genuine memory is rendered"
    assert set(shown) & set(_ids(family)) == set(_ids(reversed(family))[:3])
    assert len(shown) == 9
