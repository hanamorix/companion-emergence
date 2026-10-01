"""Name-recall fix R4 (spec §5, S8, S35, S52, S79, S81; plan P-21): `search_memories`

  - semantic mode merges the keyword search in BELOW the semantic results, in
    the slots they leave under the limit (criteria C7, C8, C5b);
  - the keyword side (both modes) sends EVERY word of the query, no stopword
    drop and no cap (S81, superseding S57 for the tool), in two tiers: the
    words the store has always searched (3+ characters), then the 1-2 character
    words, whose hits only fill leftover slots (S79).

Driven through the real `dispatch` path with a scripted embedder and reranker,
like `test_search_memories_mode.py`. Synthetic data only.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from brain.bridge import model_tier
from brain.dev_constants import RERANK_MIN_REAL_CANDIDATES
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.reranker import ANCHOR_POOL, FakeRerankerProvider
from brain.memory.store import Memory, MemoryStore
from brain.tools.dispatch import dispatch

_MODEL_ID = "scripted-test"
_FLOOR = -9.25
_DIM = 384
_LONG = "the quokka photographs were pinned along the studio wall through the long wet autumn"


class _Scripted(EmbeddingProvider):
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        return self._vectors.get(text, np.zeros(_DIM, dtype=np.float32)).astype(np.float32)

    def embedding_dim(self) -> int:
        return _DIM

    def model_id(self) -> str:
        return _MODEL_ID


def _vec(score: float) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = score
    v[1] = math.sqrt(max(0.0, 1.0 - score * score))
    return v


def _query_vec() -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def _mem(store: MemoryStore, content: str, *, memory_type: str = "event", importance: float = 5.0) -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="d", importance=importance)
    store.create(m)
    return m


def _set_vec(store: MemoryStore, mid: str, score: float) -> None:
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
        (_vec(score).tobytes(), _MODEL_ID, mid),
    )
    store._conn.commit()  # noqa: SLF001


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": MemoryStore(tmp_path / "memories.db"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": tmp_path,
    }


def _rc(store: MemoryStore, mid: str) -> float:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _semantic_setup(
    monkeypatch: pytest.MonkeyPatch,
    store: MemoryStore,
    query: str,
    semantic: list[tuple[Memory, float]],
    *,
    extra_pad: int = 0,
) -> None:
    """Give each memory in `semantic` a vector and a floor-clearing rerank score
    (the pair's second element, normalized scale), pad the pool to the rerank minimum,
    and patch the scripted embedder and reranker in."""
    vectors = {query: _query_vec()}
    scripted = dict.fromkeys(ANCHOR_POOL, 0.0)
    for i, (mem, score) in enumerate(semantic):
        _set_vec(store, mem.id, 0.9 - 0.01 * i)
        scripted[mem.content] = score
    for i in range(max(0, RERANK_MIN_REAL_CANDIDATES - len(semantic)) + extra_pad):
        filler = _mem(store, f"zzfiller{i} qqpadding")
        _set_vec(store, filler.id, 0.05)
    monkeypatch.setattr("brain.memory.embeddings.build_embedding_provider", lambda: _Scripted(vectors))
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, _MODEL_ID)
    monkeypatch.setattr(
        "brain.memory.reranker.build_reranker_provider",
        lambda **kw: FakeRerankerProvider(scores=scripted),
    )
    store.write_reranker_floor(
        "fake-reranker", floor=_FLOOR, raw_fit_floor=_FLOOR, sample_pairs=10, is_cold_start=False
    )


def _ids(res: dict) -> list[str]:
    return [m["id"] for m in res["memories"]]


# ---------------------------------------------------------------------------
# C7 (tool): semantic mode merges keyword hits below the semantic results.
# ---------------------------------------------------------------------------


def test_semantic_mode_fills_only_the_slots_semantic_leaves_with_keyword_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = [_mem(store, f"deep breathing eases racing thoughts, note {i}") for i in range(2)]
    kw = [_mem(store, f"{_LONG} (variant {i})") for i in range(6)]
    _semantic_setup(monkeypatch, store, "quokka", [(sem[0], 6.0), (sem[1], 5.0)])

    res = dispatch("search_memories", {"query": "quokka", "limit": 5}, **ctx)

    ids = _ids(res)
    assert res["mode"] == "semantic", "at least one semantic result contributed"
    assert ids[:2] == [sem[0].id, sem[1].id], "semantic results lead, in their own order"
    assert len(ids) == 5, "the limit is the cap: keyword fills 5 - 2 = 3 slots, no more"
    assert set(ids[2:]) <= {m.id for m in kw}


def test_a_memory_found_by_both_paths_appears_once_at_its_semantic_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    other = _mem(store, "deep breathing eases racing thoughts")
    both = _mem(store, f"{_LONG} (found by both paths)")
    [_mem(store, f"{_LONG} (variant {i})") for i in range(4)]
    _semantic_setup(monkeypatch, store, "quokka", [(other, 6.0), (both, 5.0)])

    ids = _ids(dispatch("search_memories", {"query": "quokka", "limit": 5}, **ctx))

    assert ids.count(both.id) == 1
    assert ids.index(both.id) == 1, "at its semantic position, not pushed down by the keyword order"


def test_the_tool_stays_bump_free_and_never_returns_excluded_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = _mem(store, "deep breathing eases racing thoughts")
    excluded_sem = _mem(store, "slow controlled breathing eases panic")
    kw = [_mem(store, f"{_LONG} (variant {i})") for i in range(4)]
    excluded_kw = kw[0]
    _semantic_setup(monkeypatch, store, "quokka", [(sem, 6.0), (excluded_sem, 5.5)], extra_pad=2)
    before = {m.id: _rc(store, m.id) for m in [sem, excluded_sem, *kw]}

    res = dispatch(
        "search_memories",
        {"query": "quokka", "limit": 8, "exclude_ids": [excluded_sem.id, excluded_kw.id]},
        **ctx,
    )
    ids = _ids(res)
    assert excluded_sem.id not in ids and excluded_kw.id not in ids
    assert sem.id in ids
    assert {m.id: _rc(store, m.id) for m in [sem, excluded_sem, *kw]} == before, "retrieval is bump-free"


def test_order_age_and_emotion_apply_to_the_merged_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = _mem(store, "deep breathing eases racing thoughts")
    old_kw = _mem(store, f"{_LONG} (old)")
    new_kw = _mem(store, f"{_LONG} (new)")
    for m, days in ((sem, 5), (old_kw, 30), (new_kw, 1)):
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=days)).isoformat(), m.id),
        )
    store._conn.commit()  # noqa: SLF001
    _semantic_setup(monkeypatch, store, "quokka", [(sem, 6.0)])

    res = dispatch("search_memories", {"query": "quokka", "limit": 3, "order": "age"}, **ctx)
    assert _ids(res) == [new_kw.id, sem.id, old_kw.id], "newest first across the merged list"
    assert res["mode"] == "semantic"


def test_mode_falls_back_to_lexical_when_no_semantic_result_contributes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    kw = _mem(store, f"{_LONG} (only keyword)")
    _semantic_setup(monkeypatch, store, "quokka", [])  # nothing scripted: nothing clears the floor
    res = dispatch("search_memories", {"query": "quokka"}, **ctx)
    assert res["mode"] == "lexical"
    assert _ids(res) == [kw.id]


# ---------------------------------------------------------------------------
# C8 (tool): the monologue family follows genuine keyword hits.
# ---------------------------------------------------------------------------


def test_keyword_monologue_family_hits_follow_genuine_ones_in_both_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    for i in range(4):
        _mem(store, f"{_LONG} (family {i})", memory_type="monologue", importance=9.9)
    genuine = [_mem(store, f"{_LONG} (genuine {i})", importance=1.0) for i in range(3)]
    lexical = _ids(dispatch("search_memories", {"query": "quokka", "mode": "lexical", "limit": 3}, **ctx))
    assert set(lexical) == {m.id for m in genuine}, "the family never takes a slot a genuine hit can fill"

    sem = _mem(store, "deep breathing eases racing thoughts")
    _semantic_setup(monkeypatch, store, "quokka", [(sem, 6.0)])
    merged = _ids(dispatch("search_memories", {"query": "quokka", "limit": 4}, **ctx))
    assert merged[0] == sem.id
    assert set(merged[1:]) == {m.id for m in genuine}


def test_a_family_flood_larger_than_the_candidate_pool_cannot_hide_genuine_hits(tmp_path: Path) -> None:
    """60 monologue-family memories that out-rank 6 genuine ones on bm25 and
    importance fill the ranker's whole 50-row pool unless the pool is family-
    last (spec §4, S16, Acceptance 8): the tool still returns every genuine hit
    first, in lexical mode."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    for i in range(60):
        _mem(store, f"quokka harbour market garden family note {i}", memory_type="monologue", importance=9.5)
    genuine = [_mem(store, f"quokka plain genuine entry {i}", importance=3.0) for i in range(6)]
    res = dispatch(
        "search_memories",
        {"query": "quokka harbour market garden", "mode": "lexical", "limit": 8},
        **ctx,
    )
    ids = _ids(res)
    assert {m.id for m in genuine} == set(ids[:6])
    assert len(ids) == 8


# ---------------------------------------------------------------------------
# C5b (S57): the tool's lexical mode goes through the recall selector, no cap.
# ---------------------------------------------------------------------------

_RARE = ["zephyr", "mosaic", "lantern", "orchid", "pewter", "saffron", "tundra", "velvet", "willow", "yonder"]
_COMMON = ["harbour", "market"]


def _cap_store(store: MemoryStore) -> Memory:
    _mem(store, "decoy " + " ".join(_RARE))
    target = _mem(store, f"the {' '.join(_COMMON)} morning walk with Canary")
    for i in range(7):
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular")
    return target


def test_lexical_mode_sends_every_word_of_the_query_to_the_ranker_in_two_tiers(tmp_path: Path) -> None:
    """S81 + S79: every word, stopwords included, no cap; the words the raw-string
    builder has always searched (3+ characters) first, the 1-2 character words
    as a second ranker call whose hits only follow."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    target = _cap_store(store)
    query = "the " + " ".join(_RARE + _COMMON) + " and of AI"

    import brain.tools.impls.search_memories as tool  # noqa: PLC0415

    real = tool.rank_memories
    seen: list = []

    def spy(st, heb, q, **kwargs):
        seen.append(list(q))
        return real(st, heb, q, **kwargs)

    with patch.object(tool, "rank_memories", spy):
        res = dispatch("search_memories", {"query": query, "mode": "lexical", "limit": 8}, **ctx)

    assert res["mode"] == "lexical"
    assert seen == [["the", *_RARE, *_COMMON, "and"], ["of", "ai"]]
    assert target.id in _ids(res), "a word beyond the old ten-token cap is searched"


def test_short_word_hits_only_follow_tier_one_hits(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    only_long = [_mem(store, f"quokka harbour note {i}") for i in range(3)]
    ai = _mem(store, "an AI notebook, nothing else")
    ids = _ids(dispatch("search_memories", {"query": "quokka AI", "mode": "lexical", "limit": 8}, **ctx))
    assert ids[:3] and set(ids[:3]) == {m.id for m in only_long}
    assert ids[3] == ai.id

    ids2 = _ids(dispatch("search_memories", {"query": "quokka AI", "mode": "lexical", "limit": 3}, **ctx))
    assert ai.id not in ids2, "with tier 1 filling the limit, the short word displaces nothing"


def test_lexical_mode_admits_a_two_letter_acronym(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    ai = _mem(store, "Bob showed Canary an AI notebook about tide tables")
    _mem(store, "harbour filler entry")
    # base behaviour: the raw-string path drops 2-character tokens
    assert store.search_fts_scored("AI", limit=10) == []
    res = dispatch("search_memories", {"query": "AI", "mode": "lexical"}, **ctx)
    assert _ids(res) == [ai.id]


def test_a_stopword_query_and_a_lowercase_unlisted_name_are_searched_in_both_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S81 (supersedes S57 for the tool): no stopword drop, so a lowercase name
    that is also a stopword ('will'), not yet on any known-names list, finds its
    memories in lexical mode and in the keyword side of semantic mode."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    will = _mem(store, "Will came by the workshop with a lantern")
    assert _ids(dispatch("search_memories", {"query": "will", "mode": "lexical"}, **ctx)) == [will.id]
    assert _ids(dispatch("search_memories", {"query": "the and of", "mode": "lexical"}, **ctx))

    sem = _mem(store, "deep breathing eases racing thoughts")
    _semantic_setup(monkeypatch, store, "will", [(sem, 6.0)])
    res = dispatch("search_memories", {"query": "will", "limit": 5}, **ctx)
    assert res["mode"] == "semantic"
    ids = _ids(res)
    assert ids[0] == sem.id and will.id in ids


def test_a_tier_two_genuine_hit_precedes_a_tier_one_family_hit(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    family = _mem(store, "quokka harbour trace note", memory_type="monologue", importance=9.5)
    genuine = _mem(store, "an AI notebook, nothing else")
    ids = _ids(dispatch("search_memories", {"query": "quokka AI", "mode": "lexical", "limit": 8}, **ctx))
    assert ids.index(genuine.id) < ids.index(family.id)


# ---------------------------------------------------------------------------
# Round-4 review: tier-2 extras (found only through 1-2 character words) must
# not displace, out-order or link to what tier 1 (today's search) returns.
# ---------------------------------------------------------------------------


def _short_word_junk(store: MemoryStore, n: int) -> list[Memory]:
    """Newer memories matching only the 1-2 character words of a query."""
    return [_mem(store, f"my dog is s big number {i}") for i in range(n)]


def test_order_age_never_lets_short_word_extras_displace_tier_one_hits(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    targets = [_mem(store, f"quokka rescue report {i}") for i in range(3)]
    _short_word_junk(store, 20)  # created later: an age sort would put them first
    query = "quokka rescue, Canary's is my"
    for order in ("relevance", "age"):
        ids = _ids(dispatch("search_memories", {"query": query, "mode": "lexical", "limit": 5, "order": order}, **ctx))
        assert {m.id for m in targets} <= set(ids[:5]), order
        assert set(ids[:3]) == {m.id for m in targets}, f"tier-1 hits lead under order={order}"


def test_an_emotion_boost_cannot_lift_a_short_word_extra_over_tier_one_hits(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    tier1 = [_mem(store, f"quokka rescue report {i}") for i in range(3)]
    joyful = Memory.create_new(content="an AI notebook", memory_type="event", domain="d", emotions={"joy": 8.0})
    store.create(joyful)
    ids = _ids(
        dispatch("search_memories", {"query": "quokka rescue AI", "mode": "lexical", "limit": 3, "emotion": "joy"}, **ctx)
    )
    assert set(ids) == {m.id for m in tier1}


def test_co_recall_never_links_the_anchor_to_a_short_word_extra(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    _mem(store, "quokka rescue report a")
    _mem(store, "quokka rescue report b")
    junk = _short_word_junk(store, 3)
    edges: list[tuple[str, str]] = []
    real = ctx["hebbian"].strengthen
    ctx["hebbian"].strengthen = lambda a, b, delta=0.1: (edges.append((a, b)), real(a, b, delta))[1]
    dispatch("search_memories", {"query": "quokka rescue is s", "mode": "lexical", "limit": 5}, **ctx)
    assert edges, "the two tier-1 hits are still linked"
    assert not ({b for _, b in edges} & {m.id for m in junk})


def test_exclude_ids_apply_to_tier_two_and_a_hit_in_both_tiers_appears_once(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    both = _mem(store, "quokka rescue and an AI notebook")
    only_short = _mem(store, "an AI notebook")
    excluded = _mem(store, "another AI notebook")
    ids = _ids(
        dispatch(
            "search_memories",
            {"query": "quokka AI", "mode": "lexical", "limit": 8, "exclude_ids": [excluded.id]},
            **ctx,
        )
    )
    assert ids.count(both.id) == 1
    assert only_short.id in ids and excluded.id not in ids


def test_a_family_flood_matching_only_a_short_word_cannot_hide_a_genuine_hit(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    for i in range(60):
        _mem(store, f"ai family note {i}", memory_type="monologue", importance=9.5)
    genuine = _mem(store, "a plain AI entry", importance=3.0)
    ids = _ids(dispatch("search_memories", {"query": "AI", "mode": "lexical", "limit": 8}, **ctx))
    assert ids[0] == genuine.id
