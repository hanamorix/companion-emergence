"""Name-recall fix R5 (spec §5, S47): `FtsPhrases`, the phrase-shaped keyword query
the known-names query sends, and the name query helpers in `relevance`.

A listed name is matched as the name: a multi-word name only as consecutive
words, a 2-letter name is not dropped by the length floor, nothing but letters,
digits and single spaces can reach the FTS expression. Synthetic data only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from brain.memory import known_names as kn
from brain.memory import relevance
from brain.memory.relevance import names_in, rank_memories, rank_name_hits
from brain.memory.store import FtsPhrases, Memory, MemoryStore, _to_fts_match


def _mem(store: MemoryStore, content: str, *, state: str = "active") -> Memory:
    m = Memory.create_new(content=content, memory_type="conversation", domain="us", importance=5.0)
    store.create(m)
    if state != "active":
        store._conn.execute(  # noqa: SLF001
            "UPDATE memories SET state = ?, content_snapshot = content WHERE id = ?", (state, m.id)
        )
        store._conn.commit()  # noqa: SLF001
    return m


# --------------------------------------------------------------------------
# the expression builder
# --------------------------------------------------------------------------


def test_each_element_is_one_quoted_phrase_or_joined() -> None:
    assert _to_fts_match(FtsPhrases(["new york", "pretzel"])) == '"new york" OR "pretzel"'


def test_no_length_floor_and_no_splitting_into_terms() -> None:
    assert _to_fts_match(FtsPhrases(["al"])) == '"al"'
    # the same words as a plain list are separate OR terms; as a phrase they are one
    assert _to_fts_match(["new", "york"]) == '"new" OR "york"'
    assert _to_fts_match(FtsPhrases(["new york"])) == '"new york"'
    # a raw string still drops a 2-letter token (other callers keep the floor)
    assert _to_fts_match("al pretzel") == '"pretzel"'


def test_only_letters_digits_and_single_spaces_reach_the_expression() -> None:
    assert _to_fts_match(FtsPhrases(['x" OR "y', "a   b", "  "])) == '"x OR y" OR "a b"'
    assert _to_fts_match(FtsPhrases([""])) == ""
    assert _to_fts_match(FtsPhrases([])) == ""


def test_duplicate_phrases_are_dropped_case_insensitively() -> None:
    assert _to_fts_match(FtsPhrases(["New York", "new  york", "NEW YORK"])) == '"New York"'


def test_a_phrase_matches_only_consecutive_words_in_a_real_fts_index() -> None:
    store = MemoryStore(":memory:")
    ny = _mem(store, "I moved to New York last spring")
    _mem(store, "a new dress arrived and a postcard of the York minster too")
    _mem(store, "york comes before new in this backwards sentence")
    hits = store.search_fts_scored(FtsPhrases(["new york"]), limit=10)
    assert [m.id for m, _ in hits] == [ny.id]


def test_a_two_letter_phrase_is_searched_where_the_raw_string_builder_drops_it() -> None:
    store = MemoryStore(":memory:")
    al = _mem(store, "Al fixed the ferry engine")
    _mem(store, "harbour filler")
    assert store.search_fts_scored("al", limit=10) == []
    assert [m.id for m, _ in store.search_fts_scored(FtsPhrases(["al"]), limit=10)] == [al.id]


# --------------------------------------------------------------------------
# rank_memories / rank_name_hits
# --------------------------------------------------------------------------


def test_rank_name_hits_covers_active_and_fading_and_ranks_the_family_last() -> None:
    store = MemoryStore(":memory:")
    active = _mem(store, "Pretzel chased the ball across the yard")
    faded = _mem(store, "Pretzel slept by the stove all winter", state="fading")
    fam = Memory.create_new(
        content="Pretzel, I keep wondering about Pretzel", memory_type="monologue", domain="d", importance=9.0
    )
    store.create(fam)
    _mem(store, "an unrelated harbour filler line")
    ranked = rank_name_hits(store, None, ["pretzel"], limit=10)
    ids = [m.id for m, _ in ranked]
    assert set(ids) == {active.id, faded.id, fam.id}
    assert ids[-1] == fam.id, "the monologue family follows every genuine hit"
    states = {m.id: m.state for m, _ in ranked}
    assert states[faded.id] == "fading"


def test_rank_name_hits_with_no_names_runs_no_query() -> None:
    store = MemoryStore(":memory:")
    with patch.object(relevance, "rank_memories") as spy:
        assert rank_name_hits(store, None, [], limit=10) == []
    spy.assert_not_called()


def test_rank_name_hits_honours_exclude_ids() -> None:
    store = MemoryStore(":memory:")
    a = _mem(store, "Pretzel one")
    b = _mem(store, "Pretzel two")
    ranked = rank_name_hits(store, None, ["pretzel"], limit=10, exclude_ids={a.id})
    assert [m.id for m, _ in ranked] == [b.id]


def test_the_ranking_off_fallback_searches_each_phrase() -> None:
    """`RELEVANCE_RANKING_ENABLED = False` (a constant True in production) falls
    back to substring search: phrases are searched one by one, not as one joined
    string that only a memory holding all of them would match."""
    store = MemoryStore(":memory:")
    ny = _mem(store, "I moved to New York last spring")
    pz = _mem(store, "Pretzel chased the ball")
    _mem(store, "an unrelated harbour filler line")
    with patch.object(relevance, "RELEVANCE_RANKING_ENABLED", False):
        ranked = rank_memories(store, None, FtsPhrases(["new york", "pretzel"]), limit=10)
    assert {m.id for m, _ in ranked} == {ny.id, pz.id}
    assert all(score is None for _, score in ranked)


# --------------------------------------------------------------------------
# names_in
# --------------------------------------------------------------------------


def test_names_in_matches_the_raw_words_before_stopword_and_length_rules(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["Pretzel", "Al", "New York"], "tool")
    assert names_in(tmp_path, "Pretzel and AL went to new york") == ["pretzel", "al", "new york"]
    assert names_in(tmp_path, "the new dress and the york minster") == []


def test_names_in_is_empty_without_a_persona_dir_text_or_list(tmp_path: Path) -> None:
    assert names_in(None, "pretzel") == []
    assert names_in(tmp_path, "") == []
    assert names_in(tmp_path, "pretzel") == [], "no list file: nothing to protect"
    assert not kn.known_names_path(tmp_path).exists(), "a lookup never creates the file"


def test_names_in_is_fail_soft(tmp_path: Path) -> None:
    kn.admit_names(tmp_path, ["Pretzel"], "tool")
    with patch.object(relevance, "load_known_names", side_effect=RuntimeError("boom")):
        assert names_in(tmp_path, "pretzel") == []


# --------------------------------------------------------------------------
# lead_with_names (S89)
# --------------------------------------------------------------------------


def _m(store: MemoryStore, content: str) -> Memory:
    return _mem(store, content)


def test_lead_with_names_orders_both_then_name_only_then_general() -> None:
    store = MemoryStore(":memory:")
    both = _m(store, "Pretzel weighs thirty pounds")
    name_only = _m(store, "Pretzel chased the ball")
    general_only = _m(store, "thirty pounds of gravel")
    out = relevance.lead_with_names(
        [name_only, both], [general_only, both], ["pretzel"]
    )
    assert [m.id for m in out] == [both.id, name_only.id, general_only.id]


def test_lead_with_names_decides_membership_from_the_hit_not_the_name_window() -> None:
    store = MemoryStore(":memory:")
    holds_name = _m(store, "Pretzel weighs thirty pounds")  # NOT in the (truncated) name hits
    name_only = _m(store, "Pretzel chased the ball")
    other = _m(store, "thirty pounds of gravel")
    out = relevance.lead_with_names([name_only], [other, holds_name], ["pretzel"])
    assert [m.id for m in out] == [holds_name.id, name_only.id, other.id]


def test_lead_with_names_matches_a_multi_word_name_only_as_consecutive_words() -> None:
    store = MemoryStore(":memory:")
    ny = _m(store, "moved to New York in spring")
    split = _m(store, "a new dress and a York postcard")
    name_hit = _m(store, "New York is loud")
    out = relevance.lead_with_names([name_hit], [split, ny], ["new york"])
    assert [m.id for m in out] == [ny.id, name_hit.id, split.id]


def test_lead_with_names_is_the_identity_without_names_or_name_hits_and_dedups() -> None:
    store = MemoryStore(":memory:")
    a = _m(store, "Pretzel one")
    b = _m(store, "Two things")
    assert [m.id for m in relevance.lead_with_names([], [b, a], ["pretzel"])] == [b.id, a.id]
    assert [m.id for m in relevance.lead_with_names([a], [b, a], [])] == [b.id, a.id]
    out = relevance.lead_with_names([a, a], [a, b, a], ["pretzel"])
    assert [m.id for m in out] == [a.id, b.id]


def test_lead_with_names_trusts_the_name_query_where_the_text_matcher_cannot_see_the_match() -> None:
    """The FTS tokenizer folds accents ("Jose" matches "José"); the ASCII text
    matcher does not. A general hit the name query returned is a name hit either way."""
    store = MemoryStore(":memory:")
    accented = _m(store, "José weighs thirty pounds")
    name_only = _m(store, "Jose chased the ball")
    other = _m(store, "thirty pounds of gravel")
    out = relevance.lead_with_names([name_only, accented], [other, accented], ["jose"])
    assert [m.id for m in out] == [accented.id, name_only.id, other.id]
