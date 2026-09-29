"""Name-recall fix R5 (spec §5, S27, S35, S36, S47, S79; plan P-14, P-15): name
protection wired into passive recall.

Covers criterion C5 (passive half): a listed name typed in lower case triggers
ONE extra keyword query whose hits rank ahead of the general keyword hits in the
active and fading buckets; a word not in the list triggers no name query and no
reordering; a listed 2-letter name and a listed name that is also a stopword
survive the stopword and length rules; a listed "new york" is matched only as
consecutive raw words and sent as ONE FTS phrase. Also pins the graveyard
exclusion (P-14: the name query never feeds the lost bucket) and the
"not recognised" list's use of matched names (P-15).

Every "leads" assertion has its precondition run on a twin world with an empty
list, so the test cannot pass by the ordering already being that way.

Drives the REAL `_build_recall_block` against a real store and a real
known-names file. Synthetic data only (user "Bob", persona label "Canary").
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from brain.chat.prompt import _build_recall_block, _extract_recall_tokens, _legacy_capped_tokens
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import graveyard as gv
from brain.forgetting.salience import SalienceInputs
from brain.memory import known_names as kn
from brain.memory.semantic_recall import SemanticRecallResult
from brain.memory.store import FtsPhrases, Memory, MemoryStore, _to_fts_match

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _mem(store: MemoryStore, content: str, *, importance: float = 5.0) -> Memory:
    m = Memory.create_new(content=content, memory_type="conversation", domain="us", importance=importance)
    store.create(m)
    return m


def _fade(store: MemoryStore, mem: Memory) -> None:
    """A direct state update: `store.fade` would embed and load a model."""
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET state = 'fading', content_snapshot = content WHERE id = ?", (mem.id,)
    )
    store._conn.commit()  # noqa: SLF001


def _list_names(persona: Path, *names: str) -> None:
    """Put names on the list through the real admission function."""
    admitted = kn.admit_names(persona, list(names), "tool")
    assert admitted, f"{names} were not admitted"


def _list_names_direct(persona: Path, *names: str) -> None:
    """Write entries straight into the file, bypassing the S70 admission filter
    (which keeps stopword strings out of the list through every writer): this
    exercises the matcher and the recall wiring, not the admission."""
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


def _rows(block: str) -> list[tuple[str, str]]:
    """(memory id, rendered body) of the ACTIVE section, in render order."""
    rows: list[tuple[str, str]] = []
    in_active = False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped == "active:":
            in_active = True
            continue
        if stripped.startswith(("softened", "lost", "not recognised")):
            in_active = False
            continue
        if in_active:
            m = re.match(r'^    - (\S+): "(.*)"$', line)
            if m:
                rows.append((m.group(1), m.group(2)))
    return rows


def _ids(block: str) -> list[str]:
    return [r[0] for r in _rows(block)]


def _softened(block: str) -> list[str]:
    out: list[str] = []
    on = False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("softened"):
            on = True
            continue
        if stripped.startswith(("lost", "not recognised")):
            on = False
            continue
        if on:
            m = re.match(r'^    - "(.*)"  \[state: fading\]$', line)
            if m:
                out.append(m.group(1))
    return out


def _not_recognised(block: str) -> list[str]:
    out: list[str] = []
    on = False
    for line in block.splitlines():
        if line.strip().startswith("not recognised"):
            on = True
            continue
        if on:
            m = re.match(r"^    - (.+)$", line)
            if m:
                out.append(m.group(1))
    return out


def _world(persona: Path) -> SimpleNamespace:
    """A small kennel. `general` matches two rare words of the messages below
    (so it out-ranks any name memory on BM25 in the general query); the name
    memories each match only the name."""
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
    store = MemoryStore(":memory:")
    general = _mem(store, "zorblax and quixotic vellum ledger kept by the harbour clerk")
    pretzel = [_mem(store, f"Pretzel is a scruffy terrier mix, walk number {i}") for i in range(3)]
    al = _mem(store, "Al fixed the ferry engine before the storm")
    will = _mem(store, "Will brought the lantern to the workshop")
    for i in range(6):
        _mem(store, f"unrelated harbour filler line {i} about tides and nothing else")
    return SimpleNamespace(store=store, general=general, pretzel=pretzel, al=al, will=will)


def _render(w: SimpleNamespace, message: str, persona: Path, *, semantic=None, **kwargs) -> str:
    with patch("brain.chat.prompt.run_semantic_recall", return_value=semantic):
        return _build_recall_block(w.store, message, persona_dir=persona, **kwargs)


def _twin(tmp_path: Path, name: str, message: str, *, list_names=(), direct=(), **kwargs):
    """Render `message` in a fresh world: (world, block)."""
    persona = tmp_path / name
    persona.mkdir()
    w = _world(persona)
    if list_names:
        _list_names(persona, *list_names)
    if direct:
        _list_names_direct(persona, *direct)
    return w, _render(w, message, persona, **kwargs)


# --------------------------------------------------------------------------
# C5: a listed lower-case name triggers the name query; its hits lead
# --------------------------------------------------------------------------


def test_a_listed_lowercase_name_leads_the_general_keyword_hits(tmp_path: Path) -> None:
    message = "pretzel zorblax quixotic"
    w0, before = _twin(tmp_path, "empty", message)
    assert _ids(before)[0] == w0.general.id, "precondition: without the list the rare-word hit leads"

    w1, after = _twin(tmp_path, "listed", message, list_names=["Pretzel"])
    ids = _ids(after)
    pretzel_ids = {m.id for m in w1.pretzel}
    assert set(ids[:3]) == pretzel_ids, "the name's hits rank ahead of every general hit"
    assert ids[3] == w1.general.id
    assert len(ids) == len(set(ids)), "a memory found by both queries appears once"


def test_the_name_query_hits_lead_in_the_softened_section_too(tmp_path: Path) -> None:
    message = "pretzel zorblax quixotic"

    def build(name: str, listed: bool):
        persona = tmp_path / name
        persona.mkdir()
        w = _world(persona)
        f_gen = _mem(w.store, "a faded zorblax quixotic tale of a ledger")
        f_name = _mem(w.store, "Pretzel faded into the old summer")
        _fade(w.store, f_gen)
        _fade(w.store, f_name)
        if listed:
            _list_names(persona, "Pretzel")
        return _render(w, message, persona)

    before = _softened(build("empty", False))
    assert before[0].startswith("a faded zorblax"), "precondition: the general fading hit leads"
    after = _softened(build("listed", True))
    assert after[0].startswith("Pretzel faded"), "the name's fading hit leads the softened section"
    assert len(after) == len(before) == 2


def test_name_hits_follow_the_semantic_results_and_lead_the_keyword_hits(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    sem = _mem(w.store, "the sunlit dock where the boats moor at low tide")
    _list_names(persona, "Pretzel")
    semantic = SemanticRecallResult(full=[sem], snippet=[], scores={sem.id: 5.0})
    ids = _ids(_render(w, "pretzel zorblax quixotic", persona, semantic=semantic))
    assert ids[0] == sem.id, "semantic results stay first (S42)"
    assert set(ids[1:4]) == {m.id for m in w.pretzel}, "then the name hits (S79), no slot reserved"
    assert ids[4] == w.general.id


def test_a_word_not_in_the_list_triggers_no_name_query_and_no_reordering(tmp_path: Path) -> None:
    message = "kettle zorblax quixotic"
    persona_a = tmp_path / "a"
    persona_a.mkdir()
    wa = _world(persona_a)
    _mem(wa.store, "the copper kettle whistles at dawn")
    baseline = _render(wa, message, persona_a)

    persona_b = tmp_path / "b"
    persona_b.mkdir()
    wb = _world(persona_b)
    _mem(wb.store, "the copper kettle whistles at dawn")
    _list_names(persona_b, "Pretzel")  # a list exists, but "kettle" is not on it
    calls: list = []
    real_search = wb.store.search_fts_scored

    def spy(query, **kw):
        calls.append(query)
        return real_search(query, **kw)

    with patch.object(wb.store, "search_fts_scored", spy):
        block = _render(wb, message, persona_b)

    assert not any(isinstance(q, FtsPhrases) for q in calls), "no name query was issued"
    assert [r[1] for r in _rows(block)] == [r[1] for r in _rows(baseline)], "no reordering"


def test_an_empty_or_missing_list_adds_no_query_at_all(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    with patch("brain.chat.prompt.rank_name_hits") as spy:
        _render(w, "pretzel zorblax quixotic", persona)
    # the helper is called only when a name matched
    spy.assert_not_called()


# --------------------------------------------------------------------------
# C5: the stopword and length rules do not apply to a listed name
# --------------------------------------------------------------------------


def test_a_listed_two_letter_name_leads(tmp_path: Path) -> None:
    message = "al zorblax quixotic"
    w0, before = _twin(tmp_path, "empty", message)
    ids0 = _ids(before)
    assert ids0[0] == w0.general.id and ids0.index(w0.al.id) > 0, (
        "precondition: without the list the 2-letter word only follows (tier 2)"
    )
    w1, after = _twin(tmp_path, "listed", message, list_names=["Al"])
    assert _ids(after)[0] == w1.al.id


def test_a_listed_name_that_is_also_a_stopword_survives_the_stopword_rule(tmp_path: Path) -> None:
    message = "will zorblax quixotic"
    w0, before = _twin(tmp_path, "empty", message)
    assert w0.will.id not in _ids(before), "precondition: 'will' is dropped by the selector"
    # inserted directly: admission (S70) keeps stopword strings out of the list
    w1, after = _twin(tmp_path, "listed", message, direct=["will"])
    ids = _ids(after)
    assert ids[0] == w1.will.id and ids[1] == w1.general.id


def test_a_message_that_is_only_a_listed_stopword_name_still_recalls(tmp_path: Path) -> None:
    w0, before = _twin(tmp_path, "empty", "will")
    assert before == "", "precondition: the selector drops it, so nothing is searched"
    w1, after = _twin(tmp_path, "listed", "will", direct=["will"])
    assert _ids(after) == [w1.will.id]


def test_a_listed_multi_word_name_matches_only_consecutive_words_as_one_phrase(tmp_path: Path) -> None:
    def build(name: str, listed: bool):
        persona = tmp_path / name
        persona.mkdir()
        w = _world(persona)
        split = _mem(w.store, "a new dress arrived and a postcard of the York minster too")
        ny = _mem(w.store, "I moved to New York last spring and never looked back")
        if listed:
            _list_names(persona, "New York")
        return w, split, ny, persona

    w, split, ny, persona = build("empty", False)
    ids0 = _ids(_render(w, "new york zorblax quixotic", persona))
    assert ids0[0] == w.general.id, "precondition: without the list the rare-word hit leads"

    w, split, ny, persona = build("listed", True)
    seen: list = []
    real = w.store.search_fts_scored

    def spy(query, **kw):
        out = real(query, **kw)
        if isinstance(query, FtsPhrases):
            seen.append((tuple(query), _to_fts_match(query), [m.id for m, _ in out]))
        return out

    with patch.object(w.store, "search_fts_scored", spy):
        ids = _ids(_render(w, "new york zorblax quixotic", persona))
    assert seen == [(("new york",), '"new york"', [ny.id])], "ONE query, ONE phrase, consecutive words only"
    assert ids[0] == ny.id, "the phrase hit leads"
    assert ids.index(ny.id) < ids.index(w.general.id)


def test_name_words_are_matched_on_the_raw_words_case_insensitively(tmp_path: Path) -> None:
    w, block = _twin(tmp_path, "listed", "PRETZEL? pretzel! zorblax quixotic", list_names=["pretzel"])
    assert {m.id for m in w.pretzel} == set(_ids(block)[:3])


# --------------------------------------------------------------------------
# P-14: the name query never feeds the graveyard; one hebbian handle
# --------------------------------------------------------------------------


def _bury(persona: Path, content: str, mem_id: str) -> None:
    lost = Memory.create_new(content=content, memory_type="episodic", domain="memory", emotions={"joy": 8.5})
    object.__setattr__(lost, "id", mem_id)
    gv.append(
        persona,
        memory=lost,
        salience_at_drop=0.6,
        inputs=SalienceInputs(emotion=0.85, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.1),
        lived_age_hours=24.0,
        reason="test-seed",
    )


def test_the_name_query_does_not_feed_the_graveyard(tmp_path: Path) -> None:
    """A listed name the selector drops ('will') is searched by the name query
    but NOT handed to `graveyard.search`: the lost bucket keeps today's feed
    (P-14, the F11 interim), so a name whose only memory is lost surfaces no
    lost hit through this query."""
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    _list_names_direct(persona, "will")
    _bury(persona, "Will once waved from the pier at dusk", "mem-lost-will")
    queries: list[str] = []
    real = gv.search

    def spy(persona_dir, query, **kw):
        queries.append(query)
        return real(persona_dir, query, **kw)

    with (
        patch("brain.forgetting.recall.graveyard.search", spy),
        patch("brain.grief.handle_recall_touch") as touch,
    ):
        block = _render(w, "will zorblax quixotic", persona)

    legacy = " ".join(_legacy_capped_tokens(_extract_recall_tokens("will zorblax quixotic", w.store)))
    assert "will" not in legacy.split(), "precondition: the selector drops the name"
    assert queries == [legacy], "the graveyard gets today's capped token string only"
    assert "lost (no longer in active memory)" not in block
    assert touch.call_count == 0
    assert _ids(block)[0] == w.will.id, "the active memory is still found and leads"


def test_the_name_query_shares_the_turns_one_hebbian_handle(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    _list_names(persona, "Pretzel")
    import brain.chat.prompt as prompt_mod  # noqa: PLC0415
    from brain.memory import hebbian as heb_mod  # noqa: PLC0415

    opened: list = []
    real_cls = heb_mod.HebbianMatrix

    class Spy(real_cls):
        def __init__(self, *a, **kw):
            opened.append(self)
            super().__init__(*a, **kw)

    handles: list = []
    real_rank = prompt_mod.rank_name_hits

    def rank_spy(store, hebbian, names, **kw):
        handles.append(hebbian)
        return real_rank(store, hebbian, names, **kw)

    with (
        patch("brain.memory.hebbian.HebbianMatrix", Spy),
        patch.object(prompt_mod, "rank_name_hits", rank_spy),
    ):
        _render(w, "pretzel zorblax quixotic", persona)
    assert len(opened) == 1, "one HebbianMatrix per turn"
    assert handles == [opened[0]], "the name query used that same handle"


# --------------------------------------------------------------------------
# P-15: matched names join the "not recognised" candidates
# --------------------------------------------------------------------------


def test_a_matched_name_the_store_has_never_seen_is_listed_as_not_recognised(tmp_path: Path) -> None:
    _, before = _twin(tmp_path, "empty", "zed zorblax")
    assert "zed" in _not_recognised(before), "precondition: a 3-letter unknown is listed today"

    # 'ed' is dropped by the selector (2 letters, lower case, df 0) so it is
    # listed only because the name list matched it.
    _, without = _twin(tmp_path, "without", "ed zorblax")
    assert "ed" not in _not_recognised(without)
    _, with_name = _twin(tmp_path, "with", "ed zorblax", list_names=["Ed"])
    assert "ed" in _not_recognised(with_name)


def test_a_matched_name_the_store_knows_is_not_listed_as_not_recognised(tmp_path: Path) -> None:
    """A listed name with memories (df > 0) is not 'not recognised', however it
    is typed: the lookup must include the names' own words."""
    _, block = _twin(tmp_path, "listed", "will zorblax quixotic", direct=["will"])
    assert "will" not in _not_recognised(block)
    _, block2 = _twin(tmp_path, "listed2", "al zorblax quixotic", list_names=["Al"])
    assert "al" not in _not_recognised(block2)


def test_a_single_word_name_that_is_also_a_selector_token_is_listed_once(tmp_path: Path) -> None:
    _, block = _twin(tmp_path, "listed", "glimmer zorblax", list_names=["Glimmer"])
    assert _not_recognised(block).count("glimmer") == 1


def test_a_multi_word_name_is_listed_when_any_word_is_unknown_not_only_the_first(tmp_path: Path) -> None:
    # 'harbour' is in the store (df > 0); 'glimmer' is not
    _, block = _twin(tmp_path, "second", "harbour glimmer zorblax", list_names=["Harbour Glimmer"])
    assert "harbour glimmer" in _not_recognised(block)
    # every word known: the phrase is not listed
    _, block2 = _twin(tmp_path, "known", "ferry engine zorblax", list_names=["Ferry Engine"])
    assert "ferry engine" not in _not_recognised(block2)


def test_a_multi_word_name_with_an_unknown_word_is_listed_by_its_phrase(tmp_path: Path) -> None:
    _, block = _twin(tmp_path, "with", "glimmer harbour zorblax", list_names=["Glimmer Harbour"])
    listed = _not_recognised(block)
    assert "glimmer harbour" in listed, "one word of the phrase is unknown (df 0)"


# --------------------------------------------------------------------------
# fail-soft and the no-persona_dir path
# --------------------------------------------------------------------------


def test_a_failing_names_lookup_is_no_protection_not_a_failed_recall(tmp_path: Path) -> None:
    message = "pretzel zorblax quixotic"
    w0, baseline = _twin(tmp_path, "empty", message)
    persona = tmp_path / "boom"
    persona.mkdir()
    w1 = _world(persona)
    _list_names(persona, "Pretzel")
    with patch("brain.memory.relevance.load_known_names", side_effect=RuntimeError("disk on fire")):
        block = _render(w1, message, persona)
    assert [r[1] for r in _rows(block)] == [r[1] for r in _rows(baseline)]


def test_a_failing_name_query_leaves_the_rest_of_the_turn_intact(tmp_path: Path) -> None:
    message = "pretzel zorblax quixotic"
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    _list_names(persona, "Pretzel")
    with patch("brain.chat.prompt.rank_name_hits", side_effect=RuntimeError("ranker down")):
        block = _render(w, message, persona)
    assert _ids(block)[0] == w.general.id, "the general keyword search still ran"


def test_no_persona_dir_means_no_name_protection_and_no_crash(tmp_path: Path) -> None:
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    _list_names(persona, "Pretzel")
    # (the no-persona_dir block has a different bullet shape; compare by content)
    block = _build_recall_block(w.store, "pretzel zorblax quixotic", persona_dir=None)
    lines = [ln for ln in block.splitlines() if ln.startswith("- [importance")]
    assert "zorblax" in lines[0], "the general hit still leads: no list is read without a persona_dir"


def test_a_name_hit_that_is_also_a_short_word_hit_is_not_demoted_in_the_importance_quota(
    tmp_path: Path,
) -> None:
    """Keyword-only turn: three importance-9 general hits and an importance-9
    memory the NAME query found ('al', which the general search only reaches as
    a tier-2 short word). The name hit leads, so it is not a tier-2 extra: it
    shares the quota of full renders with tier 1 by importance, in list order,
    and renders in full; only tier-2-only hits wait behind tier 1 for the quota."""
    message = "al zorblax quixotic vellum"

    def build(name: str, listed: bool):
        persona = tmp_path / name
        persona.mkdir()
        persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
        store = MemoryStore(":memory:")
        general = [
            _mem(store, f"zorblax quixotic vellum ledger entry number {i} kept by the clerk", importance=9.0)
            for i in range(3)
        ]
        al = _mem(store, "Al fixed the ferry engine before the storm, a long enough body to cut", importance=9.0)
        if listed:
            _list_names(persona, "Al")
        w = SimpleNamespace(store=store)
        return w, general, al, persona

    w, general, al, persona = build("empty", False)
    rows0 = dict(_rows(_render(w, message, persona)))
    assert al.id in rows0, "precondition: the 2-letter word is searched (tier 2)"
    w, general, al, persona = build("listed", True)
    rows = _rows(_render(w, message, persona))
    assert rows[0][0] == al.id and not rows[0][1].endswith("…"), "the leading name hit renders in full"


def test_a_message_that_is_only_a_listed_name_never_queries_the_graveyard_with_nothing(tmp_path: Path) -> None:
    """The name-only path has no selector token, so the general query is empty.
    An empty lost query would match EVERY graveyard entry (and fire grief
    touches): the general search is skipped and the graveyard is not called."""
    persona = tmp_path / "p"
    persona.mkdir()
    w = _world(persona)
    _list_names_direct(persona, "will")
    _bury(persona, "A summary of some long forgotten harbour afternoon", "mem-lost-a")
    _bury(persona, "Another lost note about the ferry timetable", "mem-lost-b")
    queries: list[str] = []
    real = gv.search

    def spy(persona_dir, query, **kw):
        queries.append(query)
        return real(persona_dir, query, **kw)

    with (
        patch("brain.forgetting.recall.graveyard.search", spy),
        patch("brain.grief.handle_recall_touch") as touch,
    ):
        block = _render(w, "will", persona)
    assert queries == [], "the graveyard is not called at all on a name-only turn"
    assert "lost (no longer in active memory)" not in block
    assert touch.call_count == 0
    assert _ids(block) == [w.will.id]


def test_name_hits_are_ordered_among_themselves_by_blended_score(tmp_path: Path) -> None:
    def build(name: str):
        persona = tmp_path / name
        persona.mkdir()
        persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
        store = MemoryStore(":memory:")
        low = _mem(store, "Pretzel is a scruffy terrier mix, the quiet one", importance=2.0)
        high = _mem(store, "Pretzel is a scruffy terrier mix, the loud one", importance=8.0)
        low_f = _mem(store, "Quiet Pretzel faded into the old summer", importance=2.0)
        high_f = _mem(store, "Loud Pretzel faded into the old summer", importance=8.0)
        _fade(store, low_f)
        _fade(store, high_f)
        _list_names(persona, "Pretzel")
        w = SimpleNamespace(store=store)
        return w, low, high, persona

    w, low, high, persona = build("p")
    block = _render(w, "pretzel", persona)
    assert _ids(block) == [high.id, low.id], "the better blended score leads within the name hits"
    softened = _softened(block)
    assert softened[0].startswith("Loud") and softened[1].startswith("Quiet")
