"""Name-recall fix R4 (spec §5, S8, S35, S99; plan P-21): `search_memories`

  - semantic mode merges the keyword search in BELOW the semantic results, in
    the slots they leave under the limit (criteria C7, C8, C5b);
  - the keyword side (both modes) is exactly today's search (S99): the raw
    query string through `rank_memories`, the store's 3-character floor, no
    every-word rule, no tiers, no short-token admission (finer kindled control
    is its own issue, #319); the name query is the only addition (R5, in
    `test_search_memories_name_protection.py`).

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
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, _MODEL_ID)
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
# C5b (S99): the tool's keyword side is today's search, in both modes.
# ---------------------------------------------------------------------------

_RARE = ["zephyr", "mosaic", "lantern", "orchid", "pewter", "saffron", "tundra", "velvet", "willow", "yonder"]
_COMMON = ["harbour", "market"]


def _cap_store(store: MemoryStore) -> Memory:
    _mem(store, "decoy " + " ".join(_RARE))
    target = _mem(store, f"the {' '.join(_COMMON)} morning walk with Canary")
    for i in range(7):
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular")
    return target


def test_lexical_mode_sends_the_raw_query_string_to_the_ranker_once(tmp_path: Path) -> None:
    """S99: no words split out, no tiers, no second call: the ranker gets the
    query exactly as the kindled typed it, as it always did (the store splits it)."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    target = _cap_store(store)
    query = "the " + " ".join(_RARE + _COMMON) + " and of AI"

    import brain.tools.impls.search_memories as tool  # noqa: PLC0415

    real = tool.rank_memories
    seen: list = []

    def spy(st, heb, q, **kwargs):
        seen.append(q)
        return real(st, heb, q, **kwargs)

    with patch.object(tool, "rank_memories", spy):
        res = dispatch("search_memories", {"query": query, "mode": "lexical", "limit": 8}, **ctx)

    assert res["mode"] == "lexical"
    assert seen == [query]
    assert isinstance(seen[0], str)
    assert target.id in _ids(res), "a word beyond the tenth is searched, as the raw query always did"


_QUERIES = [
    "quokka harbour",
    "the and of quokka",
    "quokka AI is my",
    "harbour market notebook tide tables quokka rescue report walk",
    "AI",
    "will",
]


def _base_corpus(store: MemoryStore) -> None:
    for i in range(4):
        _mem(store, f"quokka rescue report {i} at the harbour market")
    _mem(store, "an AI notebook about tide tables, and my walk to the market")
    _mem(store, "Will came by the workshop with a lantern")
    _mem(store, "my dog is s big, the and of it")
    for i in range(3):
        _mem(store, f"harbour filler line {i} about nothing in particular")


@pytest.mark.parametrize("query", _QUERIES)
@pytest.mark.parametrize("limit", [3, 8])
def test_lexical_results_equal_today_s_ranker_output(tmp_path: Path, query: str, limit: int) -> None:
    """The base-equality pin: with no listed name in the query, lexical mode
    returns exactly `rank_memories(store, hebbian, <raw query>, limit=)` (what
    `origin/main` returns), for stopword, short-word, acronym and long queries."""
    from brain.memory.relevance import rank_memories  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    store = ctx["store"]
    _base_corpus(store)
    expected = [m.id for m, _ in rank_memories(store, ctx["hebbian"], query, limit=limit)]
    res = dispatch("search_memories", {"query": query, "mode": "lexical", "limit": limit}, **ctx)
    assert _ids(res) == expected


def test_two_letter_words_are_not_searched_by_the_tool(tmp_path: Path) -> None:
    """The store's 3-character floor stands for the tool (S99): a 2-character
    acronym finds nothing, and beside a longer word it adds no hit."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    only_long = [_mem(store, f"quokka harbour note {i}") for i in range(3)]
    ai = _mem(store, "an AI notebook, nothing else")
    assert _ids(dispatch("search_memories", {"query": "AI", "mode": "lexical"}, **ctx)) == []
    ids = _ids(dispatch("search_memories", {"query": "quokka AI", "mode": "lexical", "limit": 8}, **ctx))
    assert set(ids) == {m.id for m in only_long}
    assert ai.id not in ids, "with room to spare, a short word still pads nothing"


def test_a_three_letter_stopword_query_is_still_searched_as_it_always_was(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    will = _mem(store, "Will came by the workshop with a lantern")
    assert _ids(dispatch("search_memories", {"query": "will", "mode": "lexical"}, **ctx)) == [will.id]
    assert _ids(dispatch("search_memories", {"query": "the and of", "mode": "lexical"}, **ctx))


def test_semantic_mode_keyword_side_is_the_same_raw_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Semantic mode's keyword hits are today's search too: a 2-letter word adds
    none, and the merge leaves the semantic results first."""
    from brain.memory.relevance import rank_memories  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = _mem(store, "deep breathing eases racing thoughts")
    _base_corpus(store)
    query = "quokka rescue AI is"
    expected = [m.id for m, _ in rank_memories(store, ctx["hebbian"], query, limit=50)]
    _semantic_setup(monkeypatch, store, query, [(sem, 6.0)])
    res = dispatch("search_memories", {"query": query, "limit": 6}, **ctx)
    assert res["mode"] == "semantic"
    assert _ids(res) == [sem.id, *expected[:5]]


def test_the_semantic_hit_stays_in_the_result_when_the_query_holds_a_short_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-5 review, finding 1 (MAJOR, at 96493752): in semantic mode a semantic
    result that also matched a 1-2 character word was demoted to a tier-2 'extra'
    and could be dropped from the results. With the tiers gone (S99) the semantic
    hit is sorted like any other result under `order="age"`, whatever short word
    the query holds, and stays in the results."""
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = _mem(store, "deep breathing is easy and calming")
    old = _mem(store, f"{_LONG} (old)")
    new = _mem(store, f"{_LONG} (new)")
    for m, days in ((sem, 5), (old, 30), (new, 1)):
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=days)).isoformat(), m.id),
        )
    store._conn.commit()  # noqa: SLF001
    _semantic_setup(monkeypatch, store, "quokka is", [(sem, 6.0)])
    res = dispatch("search_memories", {"query": "quokka is", "limit": 3, "order": "age"}, **ctx)
    assert res["mode"] == "semantic"
    assert _ids(res) == [new.id, sem.id, old.id]

    # the newest result is the semantic one and three older keyword hits crowd the limit
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ((datetime.now(UTC) - timedelta(hours=1)).isoformat(), sem.id),
    )
    older = [_mem(store, f"{_LONG} (older {i})") for i in range(3)]
    for m in older:
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=40)).isoformat(), m.id),
        )
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ((datetime.now(UTC) - timedelta(days=60)).isoformat(), new.id),
    )
    store._conn.commit()  # noqa: SLF001
    res2 = dispatch("search_memories", {"query": "quokka is", "limit": 3, "order": "age"}, **ctx)
    assert _ids(res2)[0] == sem.id, "the newest result is a semantic one and it is in the results"


def test_the_emotion_boost_applies_to_the_whole_merged_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-5 finding 1, emotion half, and the base tail: the boost partitions the
    merged list, so an emotion-matching KEYWORD hit moves ahead of the semantic
    result, and an emotion-matching semantic result ahead of the keyword hits."""
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    sem = _mem(store, "deep breathing is easy and calming")
    joyful_kw = Memory.create_new(
        content=f"{_LONG} (joyful)", memory_type="event", domain="d", importance=1.0, emotions={"joy": 8.0}
    )
    store.create(joyful_kw)
    kw = [_mem(store, f"{_LONG} (variant {i})") for i in range(3)]
    _semantic_setup(monkeypatch, store, "quokka is", [(sem, 6.0)])
    ids = _ids(dispatch("search_memories", {"query": "quokka is", "limit": 5, "emotion": "joy"}, **ctx))
    assert ids[0] == joyful_kw.id, "the boost lifts the joyful keyword hit over the semantic result"
    assert ids[1] == sem.id
    assert set(ids[2:]) == {m.id for m in kw}


def test_lexical_emotion_boost_moves_matching_hits_first_and_keeps_the_rest_in_order(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    emotions = ["joy", "grief", "calm", "awe"]
    by_emotion: dict[str, str] = {}
    for i, emotion in enumerate(emotions):
        m = Memory.create_new(
            content=f"{_LONG} (entry {i})", memory_type="event", domain="d", importance=5.0, emotions={emotion: 1.0}
        )
        store.create(m)
        by_emotion[emotion] = m.id
    base = _ids(dispatch("search_memories", {"query": "quokka", "mode": "lexical", "limit": 8}, **ctx))
    assert len(base) == 4
    last = base[-1]  # whichever ranks last without the boost
    target = next(e for e, mid in by_emotion.items() if mid == last)
    ids = _ids(
        dispatch("search_memories", {"query": "quokka", "mode": "lexical", "limit": 8, "emotion": target}, **ctx)
    )
    assert ids[0] == last, "the emotion-matching hit moves first"
    assert ids[1:] == base[:-1], "the rest keep their relevance order"


def test_order_age_sorts_the_whole_candidate_pool_not_the_first_few(tmp_path: Path) -> None:
    """`order="age"` widens the lexical fetch to CANDIDATE_POOL (today's rule): the
    newest match wins even when it is the least relevant of 15."""
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    store = ctx["store"]
    mems = [_mem(store, f"{_LONG} (note {i})", importance=9.0 - 0.5 * i) for i in range(15)]
    for i, m in enumerate(mems):  # the less relevant, the newer
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ((datetime.now(UTC) - timedelta(days=100 - 5 * i)).isoformat(), m.id),
        )
    store._conn.commit()  # noqa: SLF001
    ids = _ids(dispatch("search_memories", {"query": "quokka", "mode": "lexical", "limit": 3, "order": "age"}, **ctx))
    assert ids == [mems[14].id, mems[13].id, mems[12].id]


def test_co_recall_links_the_anchor_to_the_first_results_as_before(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    hits = [_mem(store, f"quokka rescue report {i}") for i in range(6)]
    edges: list[tuple[str, str]] = []
    real = ctx["hebbian"].strengthen
    ctx["hebbian"].strengthen = lambda a, b, delta=0.1: (edges.append((a, b)), real(a, b, delta))[1]
    res = dispatch("search_memories", {"query": "quokka rescue is s", "mode": "lexical", "limit": 6}, **ctx)
    ids = _ids(res)
    assert {m.id for m in hits} == set(ids)
    assert [b for _, b in edges] == ids[1:5], "the anchor (first result) links to the next four"
    assert {a for a, _ in edges} == {ids[0]}


# ---------------------------------------------------------------------------
# F7 (R6 review; spec §3): the tool's per-paragraph "keyword token set is
# empty" test uses the words this tool's keyword side actually searches.
# ---------------------------------------------------------------------------


def test_the_tools_paragraph_emptiness_words_are_the_words_its_keyword_side_searches(
    tmp_path: Path,
) -> None:
    import brain.tools.impls.search_memories as tool  # noqa: PLC0415
    from brain.memory.semantic_recall import split_paragraphs  # noqa: PLC0415

    words = tool._keyword_words("Quokka rescue\nok\nlol\nAI is", tmp_path)  # noqa: SLF001
    assert words == {"quokka", "rescue", "lol"}, "3+ character words, no stopword drop, no 1-2 character word"
    kept = split_paragraphs("Quokka rescue\nok\nlol\nAI is", words)
    assert kept == ["Quokka rescue", "lol"], "'ok' and 'AI is' are dropped, exactly as the keyword side would ignore them"
    # no stopword drop (S99): a paragraph of stopwords the raw query still searches is kept
    stop = "the and\nquokka rescue"
    assert tool._keyword_words(stop, tmp_path) == {"the", "and", "quokka", "rescue"}  # noqa: SLF001
    assert split_paragraphs(stop, tool._keyword_words(stop, tmp_path)) == ["the and", "quokka rescue"]  # noqa: SLF001


def test_a_listed_names_words_keep_its_paragraph(tmp_path: Path) -> None:
    """Tested after name protection: a paragraph holding only a listed (2-letter)
    name is kept even though the raw-query floor would ignore the word."""
    import brain.tools.impls.search_memories as tool  # noqa: PLC0415
    from brain.memory import known_names as kn  # noqa: PLC0415
    from brain.memory.semantic_recall import split_paragraphs  # noqa: PLC0415

    persona = tmp_path / "p"
    persona.mkdir()
    assert kn.admit_names(persona, ["Al", "New York"], "tool")
    text = "quokka rescue\nAl\nok\nnew york"
    words = tool._keyword_words(text, persona)  # noqa: SLF001
    assert {"al", "new", "york"} <= words
    assert split_paragraphs(text, words) == ["quokka rescue", "Al", "new york"]


def test_the_tool_passes_that_word_set_to_the_paragraph_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import brain.tools.impls.search_memories as tool  # noqa: PLC0415

    ctx = _ctx(tmp_path)
    sem = _mem(ctx["store"], "deep breathing eases racing thoughts")
    _semantic_setup(monkeypatch, ctx["store"], "quokka rescue\nok", [(sem, 6.0)])
    seen: list = []
    real = tool.search_paragraphs

    def spy(store, query, **kw):
        seen.append(kw["keyword_words"])
        return real(store, query, **kw)

    with patch.object(tool, "search_paragraphs", spy):
        dispatch("search_memories", {"query": "quokka rescue\nok"}, **ctx)
    assert seen == [frozenset({"quokka", "rescue"})]
