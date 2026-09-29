"""Name-recall fix R5 (spec §5, S27, S35, S47, S79, S81): name protection in the
`search_memories` tool, criterion C5b's name half.

Both modes: the known names in her query (matched on the raw words, before any
stopword or length rule) run ONE extra keyword query, sent as FTS phrases, whose
hits lead the keyword hits. Semantic mode keeps its semantic results first (S42)
and puts the name hits at the head of the keyword hits that fill the leftover
slots. S81 (every word of her query is searched) is unchanged: a listed name is
protected in RANK, not merely found.

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
    will = _mem(store, "Will brought the lantern to the workshop")
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
    """S81 sends every word; the name query does not depend on where the name
    falls in the query."""
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
    assert ids0[0] == general0.id and ids0.index(al0.id) > 0, "precondition: the 2-letter word only follows"
    persona = tmp_path / "listed"
    ctx, _, _, al, _ = _world(persona)
    _list(persona, "Al")
    assert _ids(_lexical(ctx, "al zorblax quixotic"))[0] == al.id


def test_lexical_a_listed_stopword_name_leads(tmp_path: Path) -> None:
    ctx0, general0, _, _, will0 = _world(tmp_path / "empty")
    ids0 = _ids(_lexical(ctx0, "will zorblax quixotic"))
    assert ids0[0] == general0.id and will0.id in ids0, "precondition: S81 finds it, but behind"
    persona = tmp_path / "listed"
    ctx, general, _, _, will = _world(persona)
    _list_direct(persona, "will")
    ids = _ids(_lexical(ctx, "will zorblax quixotic"))
    assert ids[:2] == [will.id, general.id]


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


def test_lexical_a_name_hit_that_is_also_a_short_word_is_not_a_tier_two_extra(tmp_path: Path) -> None:
    """Under `order="age"` the tier-2 extras (found only through the 1-2
    character words) follow everything else. A hit the NAME query found is not
    an extra: it sorts with the other results by age."""
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
    assert ids0.index(general0.id) < ids0.index(al0.id), "precondition: the extra follows despite being newer"
    ctx1, _, al1 = build("listed", True)
    ids1 = _ids(_lexical(ctx1, query, order="age", limit=5))
    assert ids1[0] == al1.id, "listed: an ordinary result, so the newer memory leads the age sort"


def test_lexical_family_name_hits_follow_genuine_hits_across_all_tiers(tmp_path: Path) -> None:
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
