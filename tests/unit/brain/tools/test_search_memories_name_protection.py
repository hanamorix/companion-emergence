"""Name-recall fix R5 (spec §5, S27, S35, S47, S99): name protection in the
`search_memories` tool, criterion C5b's name half.

Both modes: the known names in her query (matched on the raw words, before any
stopword or length rule) run ONE extra keyword query, sent as FTS phrases, whose
hits lead the keyword hits. Semantic mode keeps its semantic results first (S42)
and puts the name hits at the head of the keyword hits that fill the leftover
slots. The general keyword search is today's (the raw query, the store's
3-character floor, S99), so a listed name is protected in RANK, and a listed
2-letter name is FOUND at all only through the name query.

Every "leads" assertion has its precondition run on a twin persona with an empty
list. Driven through the real `dispatch` path like
`test_search_memories_keyword_merge.py`. Synthetic data only.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.memory import known_names as kn
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import FtsPhrases, Memory, MemoryStore
from brain.tools.dispatch import dispatch
from tests.unit.brain.tools.test_search_memories_keyword_merge import _semantic_setup


def _mem(store: MemoryStore, content: str, *, memory_type: str = "event") -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="d", importance=5.0)
    store.create(m)
    return m


def _ctx(persona: Path) -> dict:
    persona.mkdir(parents=True, exist_ok=True)
    return {
        "store": MemoryStore(persona / "memories.db"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": persona,
    }


def _ids(res: dict) -> list[str]:
    return [m["id"] for m in res["memories"]]


def _list(persona: Path, *names: str) -> None:
    assert kn.admit_names(persona, list(names), "tool")


def _list_direct(persona: Path, *names: str) -> None:
    """Straight into the file, bypassing the S70 admission filter (it keeps
    stopword strings off the list through every writer)."""
    conn = sqlite3.connect(str(kn.known_names_path(persona)))
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute(kn._SCHEMA)  # noqa: SLF001
        conn.executemany(
            "INSERT OR IGNORE INTO known_names VALUES (?, ?, 'tool', '2026-09-29T00:00:00+00:00')",
            [(kn.normalize_name(n), n) for n in names],
        )
        conn.commit()
    finally:
        conn.close()


def _world(persona: Path):
    """`general` holds two rare words of the queries below, so it out-ranks any
    single-word name memory on BM25 in the general keyword search."""
    ctx = _ctx(persona)
    store = ctx["store"]
    general = _mem(store, "zorblax and quixotic vellum ledger kept by the harbour clerk")
    pretzel = [_mem(store, f"Pretzel is a scruffy terrier mix, walk number {i}") for i in range(3)]
    al = _mem(store, "Al fixed the ferry engine before the storm")
    # "Will" mid-sentence: a stopword-name matches only there (S104/S121).
    will = _mem(store, "then Will brought the lantern to the workshop")
    for i in range(6):
        _mem(store, f"unrelated harbour filler line {i} about tides and nothing else")
    return ctx, general, pretzel, al, will


def _lexical(ctx: dict, query: str, **kw) -> dict:
    return dispatch("search_memories", {"query": query, "mode": "lexical", **kw}, **ctx)


# --------------------------------------------------------------------------
# lexical mode: the name query leads
# --------------------------------------------------------------------------


def test_lexical_a_listed_lowercase_name_leads(tmp_path: Path) -> None:
    ctx0, general0, _, _, _ = _world(tmp_path / "empty")
    assert _ids(_lexical(ctx0, "pretzel zorblax quixotic"))[0] == general0.id, "precondition"

    persona = tmp_path / "listed"
    ctx, general, pretzel, _, _ = _world(persona)
    _list(persona, "Pretzel")
    ids = _ids(_lexical(ctx, "pretzel zorblax quixotic"))
    assert set(ids[:3]) == {m.id for m in pretzel}
    assert ids[3] == general.id
    assert len(ids) == len(set(ids))


def test_lexical_a_name_beyond_the_old_ten_token_window_still_leads(tmp_path: Path) -> None:
    """The raw query has no word cap (S99) and the name query does not depend on
    where the name falls in the query."""
    filler = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"  # 10 words, in no memory
    query = f"{filler} zorblax quixotic pretzel"
    ctx0, general0, _, _, _ = _world(tmp_path / "empty")
    assert _ids(_lexical(ctx0, query))[0] == general0.id, "precondition"
    persona = tmp_path / "listed"
    ctx, _, pretzel, _, _ = _world(persona)
    _list(persona, "Pretzel")
    assert _ids(_lexical(ctx, query))[0] in {m.id for m in pretzel}


def test_lexical_a_listed_two_letter_name_leads(tmp_path: Path) -> None:
    ctx0, general0, _, al0, _ = _world(tmp_path / "empty")
    ids0 = _ids(_lexical(ctx0, "al zorblax quixotic"))
    assert ids0[0] == general0.id and al0.id not in ids0, (
        "precondition: today's search drops the 2-letter word, so the memory is not found at all"
    )
    persona = tmp_path / "listed"
    ctx, _, _, al, _ = _world(persona)
    _list(persona, "Al")
    assert _ids(_lexical(ctx, "al zorblax quixotic"))[0] == al.id


def test_lexical_a_listed_stopword_name_leads(tmp_path: Path) -> None:
    ctx0, general0, _, _, will0 = _world(tmp_path / "empty")
    # S104/S151: the name is written capitalized mid-query ("so" is under the tool's
    # 3-character floor, so the general search is unchanged).
    ids0 = _ids(_lexical(ctx0, "so Will zorblax quixotic"))
    assert ids0[0] == general0.id and will0.id in ids0, "precondition: found as a plain word, but behind"
    persona = tmp_path / "listed"
    ctx, general, _, _, will = _world(persona)
    _list_direct(persona, "will")
    ids = _ids(_lexical(ctx, "so Will zorblax quixotic"))
    assert ids[:2] == [will.id, general.id]
    # a query that STARTS with the name is a name too (S151 supersedes S104's sentence-start rule)
    assert _ids(_lexical(ctx, "Will zorblax quixotic"))[:2] == [will.id, general.id]
    # a lower-case "will" never is
    assert _ids(_lexical(ctx, "will zorblax quixotic"))[0] == general.id


def test_lexical_a_multi_word_name_is_one_phrase(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, general, _, _, _ = _world(persona)
    split = _mem(ctx["store"], "a new dress arrived and a postcard of the York minster too")
    ny = _mem(ctx["store"], "I moved to New York last spring and never looked back")
    _list(persona, "New York")
    seen: list = []
    real = ctx["store"].search_fts_scored

    def spy(query, **kw):
        out = real(query, **kw)
        if isinstance(query, FtsPhrases):
            seen.append((tuple(query), [m.id for m, _ in out]))
        return out

    with patch.object(ctx["store"], "search_fts_scored", spy):
        ids = _ids(_lexical(ctx, "new york zorblax quixotic"))
    assert seen == [(("new york",), [ny.id])], "ONE phrase query; only the consecutive-words memory matches"
    assert ids[0] == ny.id
    assert split.id in ids and ids.index(split.id) > ids.index(general.id), (
        "the split-word memory is only a general keyword hit"
    )


def test_lexical_a_word_not_in_the_list_adds_no_query_and_no_reordering(tmp_path: Path) -> None:
    ctx0, *_ = _world(tmp_path / "empty")
    base = _ids(_lexical(ctx0, "kettle zorblax quixotic"))
    persona = tmp_path / "listed"
    ctx, *_ = _world(persona)
    _list(persona, "Pretzel")
    queries: list = []
    real = ctx["store"].search_fts_scored

    def spy(query, **kw):
        queries.append(query)
        return real(query, **kw)

    with patch.object(ctx["store"], "search_fts_scored", spy):
        ids = _ids(_lexical(ctx, "kettle zorblax quixotic"))
    assert not any(isinstance(q, FtsPhrases) for q in queries)
    assert len(ids) == len(base)
    # same store contents up to ids: compare the rendered content order
    contents = [m["content"] for m in _lexical(ctx, "kettle zorblax quixotic")["memories"]]
    contents0 = [m["content"] for m in _lexical(ctx0, "kettle zorblax quixotic")["memories"]]
    assert contents == contents0


def test_lexical_exclude_ids_apply_to_the_name_query(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, _, pretzel, _, _ = _world(persona)
    _list(persona, "Pretzel")
    ids = _ids(_lexical(ctx, "pretzel zorblax quixotic", exclude_ids=[pretzel[0].id]))
    assert pretzel[0].id not in ids
    assert ids[0] in {pretzel[1].id, pretzel[2].id}


def test_lexical_a_name_hit_is_an_ordinary_result_under_order_age(tmp_path: Path) -> None:
    """Under `order="age"` a name hit sorts with the other results by age (no
    hit is set apart, S99); without the listing today's search never finds the
    2-letter name's memory."""
    query = "al zorblax quixotic"

    def build(name: str, listed: bool):
        persona = tmp_path / name
        ctx = _ctx(persona)
        general = _mem(ctx["store"], "zorblax and quixotic vellum ledger kept by the harbour clerk")
        al = _mem(ctx["store"], "Al fixed the ferry engine before the storm")  # newer than general
        for i in range(3):
            _mem(ctx["store"], f"unrelated harbour filler line {i}")
        if listed:
            _list(persona, "Al")
        return ctx, general, al

    ctx0, general0, al0 = build("empty", False)
    ids0 = _ids(_lexical(ctx0, query, order="age", limit=5))
    assert ids0 == [general0.id], "precondition: today's search drops the 2-letter word"
    ctx1, general1, al1 = build("listed", True)
    ids1 = _ids(_lexical(ctx1, query, order="age", limit=5))
    assert ids1 == [al1.id, general1.id], "listed: the newer memory leads the age sort"


def test_lexical_family_name_hits_follow_genuine_hits(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, general, pretzel, _, _ = _world(persona)
    fam = _mem(ctx["store"], "Pretzel, I keep circling Pretzel in my head", memory_type="monologue")
    _list(persona, "Pretzel")
    ids = _ids(_lexical(ctx, "pretzel zorblax quixotic"))
    assert ids.index(fam.id) > ids.index(general.id), "spec §4: keyword family after every keyword genuine hit"
    assert ids.index(fam.id) > max(ids.index(m.id) for m in pretzel)


# --------------------------------------------------------------------------
# semantic mode: semantic first, then name hits, then the general keyword hits
# --------------------------------------------------------------------------


def test_semantic_name_hits_follow_the_semantic_results_and_lead_the_keyword_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "pretzel zorblax quixotic"
    persona = tmp_path / "listed"
    ctx, general, pretzel, _, _ = _world(persona)
    sem = _mem(ctx["store"], "the sunlit dock where the boats moor at low tide")
    _list(persona, "Pretzel")
    _semantic_setup(monkeypatch, ctx["store"], query, [(sem, 6.0)])
    res = dispatch("search_memories", {"query": query, "limit": 6}, **ctx)
    ids = _ids(res)
    assert res["mode"] == "semantic"
    assert ids[0] == sem.id, "semantic results stay first (S42)"
    assert set(ids[1:4]) == {m.id for m in pretzel}, "then the name hits"
    assert ids[4] == general.id


def test_semantic_precondition_without_the_list_general_hits_lead_the_keyword_fill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "pretzel zorblax quixotic"
    ctx, general, _, _, _ = _world(tmp_path / "empty")
    sem = _mem(ctx["store"], "the sunlit dock where the boats moor at low tide")
    _semantic_setup(monkeypatch, ctx["store"], query, [(sem, 6.0)])
    ids = _ids(dispatch("search_memories", {"query": query, "limit": 6}, **ctx))
    assert ids[:2] == [sem.id, general.id]


def test_semantic_name_hits_fill_only_the_slots_semantic_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "pretzel zorblax quixotic"
    persona = tmp_path / "listed"
    ctx, _, pretzel, _, _ = _world(persona)
    sems = [_mem(ctx["store"], f"deep breathing eases racing thoughts, note {i}") for i in range(3)]
    _list(persona, "Pretzel")
    _semantic_setup(monkeypatch, ctx["store"], query, [(m, 6.0 - i) for i, m in enumerate(sems)])
    ids = _ids(dispatch("search_memories", {"query": query, "limit": 4}, **ctx))
    assert ids[:3] == [m.id for m in sems]
    assert len(ids) == 4 and ids[3] in {m.id for m in pretzel}, "no slot is reserved for name hits"


def test_the_tool_stays_bump_free_with_name_hits(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, _, pretzel, _, _ = _world(persona)
    _list(persona, "Pretzel")
    _lexical(ctx, "pretzel zorblax quixotic")
    rows = ctx["store"]._conn.execute("SELECT SUM(recall_count) FROM memories").fetchone()  # noqa: SLF001
    assert rows[0] == 0
    del pretzel


def test_a_failing_names_lookup_leaves_the_tool_working(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, general, _, _, _ = _world(persona)
    _list(persona, "Pretzel")
    with patch("brain.memory.relevance.load_known_names", side_effect=RuntimeError("boom")):
        ids = _ids(_lexical(ctx, "pretzel zorblax quixotic"))
    assert ids[0] == general.id


def test_the_name_query_gets_the_tools_hebbian_handle_and_orders_by_blended_score(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx = _ctx(persona)
    low = _mem(ctx["store"], "Pretzel is a scruffy terrier mix, the quiet one")
    high = _mem(ctx["store"], "Pretzel is a scruffy terrier mix, the loud one")
    ctx["store"]._conn.execute(  # noqa: SLF001
        "UPDATE memories SET importance = 8.0 WHERE id = ?", (high.id,)
    )
    ctx["store"]._conn.execute(  # noqa: SLF001
        "UPDATE memories SET importance = 2.0 WHERE id = ?", (low.id,)
    )
    ctx["store"]._conn.commit()  # noqa: SLF001
    _list(persona, "Pretzel")
    import brain.tools.impls.search_memories as tool  # noqa: PLC0415

    handles: list = []
    real = tool.rank_name_hits

    def spy(store, hebbian, names, **kw):
        handles.append(hebbian)
        return real(store, hebbian, names, **kw)

    with patch.object(tool, "rank_name_hits", spy):
        ids = _ids(_lexical(ctx, "pretzel"))
    assert handles == [ctx["hebbian"]]
    assert ids == [high.id, low.id]


# --------------------------------------------------------------------------
# S89: name hits the general search also found lead, then name-only hits,
# then the general hits
# --------------------------------------------------------------------------


def _many_pretzels(persona: Path, n: int = 60):
    ctx = _ctx(persona)
    store = ctx["store"]
    target = _mem(store, "Pretzel weighs about thirty pounds and is a scruffy terrier mix, heavier than he looks")
    _mem(store, "zorblax quixotic vellum ledger kept by the harbour clerk")
    for i in range(n):
        _mem(store, f"Pretzel chased the ball across the yard, walk number {i}")
    return ctx, target


def test_a_name_with_many_memories_no_longer_pushes_out_a_memory_matching_the_rest_of_the_query(
    tmp_path: Path,
) -> None:
    query = "how heavy is pretzel, is he a terrier, thirty pounds?"
    ctx0, target0 = _many_pretzels(tmp_path / "empty")
    assert _ids(_lexical(ctx0, query, limit=5))[0] == target0.id, "precondition: base ranks it first"
    persona = tmp_path / "listed"
    ctx, target = _many_pretzels(persona)
    _list(persona, "Pretzel")
    ids = _ids(_lexical(ctx, query, limit=5))
    assert ids[0] == target.id
    assert len(ids) == 5


def _will_world(persona: Path):
    ctx = _ctx(persona)
    store = ctx["store"]
    # "Will" mid-sentence in every memory: S104/S121.
    a = _mem(store, "so Will noted that zorblax and quixotic vellum stayed in the ledger all through the storm season")
    b1 = _mem(store, "then Will waved from the pier")
    b2 = _mem(store, "and Will sang on the ferry")
    c = _mem(store, "zorblax and quixotic vellum ledger kept by the harbour clerk")
    store._conn.execute("UPDATE memories SET importance = 9.0 WHERE id = ?", (b1.id,))  # noqa: SLF001
    store._conn.execute("UPDATE memories SET importance = 8.0 WHERE id = ?", (b2.id,))  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    return ctx, a, b1, b2, c


def test_lexical_name_and_query_hits_lead_then_name_only_then_general(tmp_path: Path) -> None:
    # Today's general search drops the 2-letter word: A and C match the rest of the query,
    # so group 1 (name and rest) holds A, group 2 (name only) B1, group 3 C.
    persona = tmp_path / "listed"
    ctx = _ctx(persona)
    store = ctx["store"]
    a = _mem(store, "Al noted that zorblax and quixotic vellum stayed in the ledger all through the storm season")
    b1 = _mem(store, "Al waved from the pier")
    c = _mem(store, "zorblax and quixotic vellum ledger kept by the harbour clerk")
    store._conn.execute("UPDATE memories SET importance = 9.0 WHERE id = ?", (b1.id,))  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    _list(persona, "Al")
    ids = _ids(_lexical(ctx, "al zorblax quixotic", limit=8))
    assert ids == [a.id, b1.id, c.id]


def test_lexical_all_general_hits_holding_the_name_precede_the_name_only_ones(tmp_path: Path) -> None:
    persona = tmp_path / "listed"
    ctx, a, b1, b2, c = _will_world(persona)
    _list_direct(persona, "will")
    ids = _ids(_lexical(ctx, "so Will zorblax quixotic", limit=8))
    assert ids[0] == a.id, "name + rest of the query leads, whatever the name query's own order says"
    assert set(ids) == {a.id, b1.id, b2.id, c.id}
    assert ids.index(c.id) > ids.index(a.id), "the general-only hit follows the name + query hit"


def test_semantic_the_same_order_holds_behind_the_semantic_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    query = "al zorblax quixotic"
    persona = tmp_path / "listed"
    ctx = _ctx(persona)
    store = ctx["store"]
    a = _mem(store, "Al noted that zorblax and quixotic vellum stayed in the ledger all through the storm season")
    b1 = _mem(store, "Al waved from the pier")
    c = _mem(store, "zorblax and quixotic vellum ledger kept by the harbour clerk")
    sem = _mem(store, "the sunlit dock where the boats moor at low tide")
    _list(persona, "Al")
    _semantic_setup(monkeypatch, store, query, [(sem, 6.0)])
    ids = _ids(dispatch("search_memories", {"query": query, "limit": 8}, **ctx))
    assert ids == [sem.id, a.id, b1.id, c.id]


def test_a_query_of_only_two_letter_words_returns_only_the_name_hits(tmp_path: Path) -> None:
    """Today's search drops every 2-letter word, so 'al ai' reaches memories only
    through the listed name: the 'Al' memories, never a memory that only holds
    the acronym 'AI'."""
    persona = tmp_path / "listed"
    ctx = _ctx(persona)
    store = ctx["store"]
    al = [_mem(store, f"Al chased the ball, walk {i}") for i in range(3)]
    ai_only = _mem(store, "an AI notebook about tide tables")
    assert _ids(_lexical(ctx, "al ai", limit=8)) == [], "precondition: nothing without the listing"
    _list(persona, "Al")
    ids = _ids(_lexical(ctx, "al ai", limit=8))
    assert set(ids) == {m.id for m in al}
    assert ai_only.id not in ids
