"""Name-recall fix R4 (spec §5, S9, S36, S52; plan P-14): the store-side query
builder admits every token of a selector token LIST (2-letter names and
acronyms included), the raw-string path keeps its 3-character minimum, and
`search_with_loss` feeds the graveyard from `lost_query` alone.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from brain.forgetting.recall import search_with_loss
from brain.memory import relevance
from brain.memory.relevance import rank_memories
from brain.memory.store import Memory, MemoryStore, _to_fts_match


def _store_with(*contents: str) -> tuple[MemoryStore, list[Memory]]:
    store = MemoryStore(":memory:")
    mems = []
    for c in contents:
        m = Memory.create_new(content=c, memory_type="conversation", domain="us", importance=5.0)
        store.create(m)
        mems.append(m)
    return store, mems


def test_token_list_admits_every_token_whatever_its_length() -> None:
    assert _to_fts_match(["ai", "no", "cat"]) == '"ai" OR "no" OR "cat"'
    assert _to_fts_match(["x"]) == '"x"'


def test_raw_string_keeps_the_three_character_minimum() -> None:
    assert _to_fts_match("ai no cat") == '"cat"'
    assert _to_fts_match("ai") == ""


def test_token_list_dedups_case_insensitively_and_splits_on_the_word_boundary() -> None:
    assert _to_fts_match(["AI", "ai", "new york"]) == '"AI" OR "new" OR "york"'
    assert _to_fts_match([]) == ""
    assert _to_fts_match(["", "!!"]) == ""


def test_search_fts_scored_and_rank_memories_find_a_two_letter_token_from_a_list() -> None:
    store, (ai, _other) = _store_with("Bob showed Canary an AI notebook", "harbour filler entry")
    assert store.search_fts_scored("ai") == []
    assert [m.id for m, _ in store.search_fts_scored(["ai"])] == [ai.id]
    assert [m.id for m, _ in rank_memories(store, None, ["ai"], limit=5)] == [ai.id]
    assert rank_memories(store, None, "ai", limit=5) == []


def test_ranking_disabled_fallback_joins_a_token_list() -> None:
    store, (mem,) = _store_with("alpha beta gamma")
    with patch.object(relevance, "RELEVANCE_RANKING_ENABLED", False):
        out = rank_memories(store, None, ["alpha", "beta"], limit=5)
    assert [m.id for m, _ in out] == [mem.id]


def test_search_with_loss_feeds_the_graveyard_from_lost_query_only(tmp_path: Path) -> None:
    store, _ = _store_with("alpha beta gamma delta")
    seen: list[str] = []
    with patch("brain.forgetting.recall.graveyard.search", lambda pd, q, **kw: seen.append(q) or []):
        search_with_loss(tmp_path, store, ["alpha", "beta", "gamma"], limit=5, lost_query="alpha")
        search_with_loss(tmp_path, store, ["alpha", "beta", "gamma"], limit=5)
        search_with_loss(tmp_path, store, "alpha beta", limit=5)
    assert seen == ["alpha", "alpha beta gamma", "alpha beta"]


def test_search_with_loss_active_bucket_uses_every_list_token(tmp_path: Path) -> None:
    store, (mem, _other) = _store_with("the AI lab", "harbour entry")
    res = search_with_loss(tmp_path, store, ["ai"], limit=5, lost_query="zzz")
    assert [m.id for m in res.active] == [mem.id]


def test_rank_limit_widens_the_active_bucket_only(tmp_path: Path) -> None:
    """R4: `rank_limit` gives the active bucket the ranker's wider window, while
    the fading bucket keeps the first-`limit` window it always had and the
    graveyard keeps `limit`."""
    store = MemoryStore(":memory:")
    for i in range(6):
        store.create(
            Memory.create_new(content=f"quokka active entry {i}", memory_type="conversation", domain="us", importance=5.0)
        )
    for i in range(6):
        faded = Memory.create_new(
            content=f"quokka faded original {i}", memory_type="conversation", domain="us", importance=9.0
        )
        store.create(faded)
        store.fade(faded.id, summary=f"quokka faded summary {i}")
    narrow = search_with_loss(tmp_path, store, ["quokka"], limit=3)
    wide = search_with_loss(tmp_path, store, ["quokka"], limit=3, rank_limit=50)
    assert len(narrow.fading) == 3 and len(narrow.active) == 0, "the narrow window is all high-importance fading"
    assert len(wide.active) == 6, "active drawn from the wide window"
    assert [m.id for m in wide.fading] == [m.id for m in narrow.fading], "fading window unchanged"
