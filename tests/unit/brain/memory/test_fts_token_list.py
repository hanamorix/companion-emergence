"""Name-recall fix R4 (spec §5, S9, S36, S52; plan P-14): the store-side query
builder admits every token of a selector token LIST (2-letter names and
acronyms included), the raw-string path keeps its 3-character minimum, and
`search_with_loss` feeds the graveyard from `lost_query` alone.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from brain.dev_constants import MONOLOGUE_FAMILY_TYPES
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


def _flood(store: MemoryStore, n_family: int, n_genuine: int) -> tuple[list[Memory], list[Memory]]:
    family = [
        Memory.create_new(
            content=f"quokka harbour market garden family {i}", memory_type="monologue", domain="us", importance=9.5
        )
        for i in range(n_family)
    ]
    genuine = [
        Memory.create_new(content=f"quokka plain entry {i}", memory_type="conversation", domain="us", importance=3.0)
        for i in range(n_genuine)
    ]
    for m in [*family, *genuine]:
        store.create(m)
    return family, genuine


def test_search_fts_scored_family_types_keep_a_family_flood_out_of_the_candidate_pool() -> None:
    store = MemoryStore(":memory:")
    family, genuine = _flood(store, 60, 6)
    query = ["quokka", "harbour", "market", "garden"]
    plain = {m.id for m, _ in store.search_fts_scored(query, limit=50)}
    ordered = store.search_fts_scored(query, limit=50, family_types=MONOLOGUE_FAMILY_TYPES)
    assert not ({m.id for m in genuine} & plain), "premise: a plain bm25 pool of 50 is all family"
    assert {m.id for m in genuine} <= {m.id for m, _ in ordered}, "family-last pool holds every genuine match"
    assert [m.memory_type for m, _ in ordered][:6] == ["conversation"] * 6


def test_rank_memories_genuine_first_puts_the_family_after_every_genuine_memory() -> None:
    store = MemoryStore(":memory:")
    family, genuine = _flood(store, 14, 6)
    out = rank_memories(store, None, ["quokka", "harbour", "market", "garden"], limit=8, genuine_first=True)
    assert {m.id for m, _ in out[:6]} == {m.id for m in genuine}
    assert all(m.memory_type == "monologue" for m, _ in out[6:])
    default = rank_memories(store, None, ["quokka", "harbour", "market", "garden"], limit=8)
    assert all(m.memory_type == "monologue" for m, _ in default), "default ranking is unchanged (family outranks)"


def test_genuine_first_changes_nothing_when_no_family_memory_matches() -> None:
    store = MemoryStore(":memory:")
    for i in range(30):
        store.create(
            Memory.create_new(
                content=f"quokka harbour entry {i} " + "market " * (i % 4),
                memory_type="conversation",
                domain="us",
                importance=float(1 + i % 9),
            )
        )
    q = ["quokka", "harbour", "market"]
    a = [(m.id, s) for m, s in rank_memories(store, None, q, limit=16)]
    b = [(m.id, s) for m, s in rank_memories(store, None, q, limit=16, genuine_first=True)]
    assert [i for i, _ in a] == [i for i, _ in b]
    assert [s for _, s in a] == pytest.approx([s for _, s in b])


def test_search_with_loss_passes_genuine_first_and_keeps_the_bucket_windows(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    family, genuine = _flood(store, 20, 6)
    plain = search_with_loss(tmp_path, store, ["quokka", "harbour"], limit=8)
    genuine_first = search_with_loss(tmp_path, store, ["quokka", "harbour"], limit=8, genuine_first=True)
    assert len(plain.active) <= 8 and len(genuine_first.active) <= 8, "the window is `limit` either way"
    assert {m.id for m in genuine} <= {m.id for m in genuine_first.active}
    assert not ({m.id for m in genuine} <= {m.id for m in plain.active})
