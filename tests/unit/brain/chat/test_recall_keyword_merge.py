"""Name-recall fix R4 (spec §3/§5, plan P-10, P-11, P-14, P-25, P-31): the keyword
search is ALWAYS merged into passive recall, below the semantic results.

Covers criteria C7 (dedup favors semantic, one bump), C8 (keyword side: the
monologue family follows genuine hits), C9c (block frame), C9e ("not recognised"
keeps today's size), C9f (the graveyard is fed exactly today's capped token
set) and the cap removal / short-token admission (S9, S36, S52).

Drives the REAL `_build_recall_block` against a real store; the semantic result
is a hand-built `SemanticRecallResult` (patched in as `run_semantic_recall`'s
return value) so the assembly is exercised on exact, deterministic inputs.
Synthetic data only (user "Bob", persona label "Canary").
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.chat.prompt import (
    _RECALL_SNIPPET_INVITATION,
    _RECALL_TOKEN_LIMIT,
    _build_recall_block,
    _extract_recall_tokens,
    _legacy_capped_tokens,
)
from brain.dev_constants import MONOLOGUE_FAMILY_TYPES
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import graveyard as gv
from brain.forgetting.salience import SalienceInputs
from brain.memory.pending import PendingQueue
from brain.memory.relevance import FULL_INJECT_MAX, SNIPPET_COUNT
from brain.memory.semantic_recall import (
    FULL_INJECT_STANDOUT_MAX,
    MAX_STANDOUT_COUNT,
    SemanticRecallResult,
)
from brain.memory.store import Memory, MemoryStore

# A body long enough that the ~20% snippet cut is visible (a snippet ends with
# an ellipsis, a full render does not): 24 words, well past the 20-char floor.
_LONG = (
    "the quokka photographs were pinned along the studio wall through the whole "
    "long wet autumn while nobody looked at them again until spring"
)


def _rc(store: MemoryStore, mid: str) -> float:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _mem(
    store: MemoryStore,
    content: str,
    *,
    importance: float = 5.0,
    memory_type: str = "conversation",
) -> Memory:
    m = Memory.create_new(
        content=content, memory_type=memory_type, domain="us", importance=importance
    )
    store.create(m)
    return m


def _semantic(*, full: list[Memory], snippet: list[Memory] | None = None) -> SemanticRecallResult:
    snippet = snippet or []
    scores = {m.id: 5.0 - i for i, m in enumerate([*full, *snippet])}
    return SemanticRecallResult(full=full, snippet=snippet, scores=scores)


def _active_rows(block: str) -> list[tuple[str, str]]:
    """(memory id, rendered body) for each row of the ACTIVE section, in render
    order."""
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


def _is_snippet(body: str) -> bool:
    return body.endswith("…")


def _render(store: MemoryStore, message: str, tmp_path: Path, semantic, **kwargs) -> str:
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    with patch("brain.chat.prompt.run_semantic_recall", return_value=semantic):
        return _build_recall_block(store, message, persona_dir=tmp_path, **kwargs)


def _seed_keyword_hits(store: MemoryStore, n: int, *, importance: float = 5.0) -> list[Memory]:
    """`n` keyword-only hits for the token "quokka" (no semantic relationship)."""
    return [_mem(store, f"{_LONG} (variant {i})", importance=importance) for i in range(n)]


# ---------------------------------------------------------------------------
# C7: keyword hits fill only the slots semantic leaves, under the cap of 9,
# taking the next positions; first 5 positions full, later ones snippets.
# ---------------------------------------------------------------------------


def test_keyword_hits_fill_the_slots_semantic_leaves_and_tier_by_position(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"unrelated harbour note {i} " + _LONG.replace("quokka", "gull")) for i in range(3)]
    kw = _seed_keyword_hits(store, 10)
    block = _render(store, "quokka", tmp_path, _semantic(full=sem))

    rows = _active_rows(block)
    ids = [r[0] for r in rows]
    # semantic first, in the semantic order, then keyword hits only
    assert ids[:3] == [m.id for m in sem]
    assert len(ids) == MAX_STANDOUT_COUNT, "semantic 3 + keyword fills up to the cap of 9, no more"
    assert set(ids[3:]) <= {m.id for m in kw}
    # full vs snippet BY POSITION across the merged list: first 5 full, rest snippet
    for pos, (_, body) in enumerate(rows):
        if pos < FULL_INJECT_STANDOUT_MAX:
            assert not _is_snippet(body), f"position {pos + 1} must render full"
        else:
            assert _is_snippet(body), f"position {pos + 1} must render as a snippet"


def test_full_semantic_cap_leaves_no_slot_for_keyword_hits(tmp_path: Path) -> None:
    """No slots are reserved for keyword hits (S40): nine semantic results fill
    the block and the keyword hit waits."""
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(9)]
    kw = _seed_keyword_hits(store, 2)
    result = _semantic(full=sem[:5], snippet=sem[5:])
    ids = [r[0] for r in _active_rows(_render(store, "quokka", tmp_path, result))]
    assert ids == [m.id for m in sem]
    assert not ({m.id for m in kw} & set(ids))


def test_importance_nine_keyword_hit_renders_full_beyond_position_five_at_most_three(
    tmp_path: Path,
) -> None:
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(5)]
    hi = [_mem(store, f"{_LONG} (important {i})", importance=9.0 + i * 0.1) for i in range(4)]
    result = _semantic(full=sem)
    block = _render(store, "quokka", tmp_path, result)
    rows = _active_rows(block)
    assert [r[0] for r in rows[:5]] == [m.id for m in sem]
    keyword_rows = rows[5:]
    assert len(keyword_rows) == 4, "9 - 5 semantic = 4 slots"
    full_keyword = [rid for rid, body in keyword_rows if not _is_snippet(body)]
    snippet_keyword = [rid for rid, body in keyword_rows if _is_snippet(body)]
    assert len(full_keyword) == FULL_INJECT_MAX, "at most 3 importance-9 hits render full"
    assert len(snippet_keyword) == 1
    # the three highest-importance hits are the full ones (today's rule)
    top3 = {m.id for m in sorted(hi, key=lambda m: -m.importance)[:3]}
    assert set(full_keyword) == top3
    # more than 5 full in the block overall (5 semantic + 3): the rule holds
    assert sum(1 for _, body in rows if not _is_snippet(body)) == FULL_INJECT_STANDOUT_MAX + 3


def test_hit_found_by_both_paths_appears_once_at_its_semantic_position_with_one_bump(
    tmp_path: Path,
) -> None:
    store = MemoryStore(":memory:")
    sem_full = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(5)]
    both = _mem(store, f"{_LONG} (found by both paths)")
    kw_only = _seed_keyword_hits(store, 3)
    result = _semantic(full=sem_full, snippet=[both])
    PendingQueue(tmp_path).drain()
    before = {m.id: _rc(store, m.id) for m in [both, *kw_only]}
    block = _render(store, "quokka", tmp_path, result)

    rows = _active_rows(block)
    ids = [r[0] for r in rows]
    assert ids.count(both.id) == 1, "a memory found by both paths appears once"
    assert ids.index(both.id) == 5, "at its semantic position (the first snippet slot)"
    assert _is_snippet(rows[5][1]), "and at its semantic tier (snippet), not promoted by the keyword path"
    # one bump: the semantic snippet's top-rank amount (0.8), not doubled
    assert _rc(store, both.id) - before[both.id] == pytest.approx(0.8)
    # enqueued for reappraisal exactly once
    queued = [r["memory_id"] for r in PendingQueue(tmp_path).drain() if r.get("_route") == "reappraise_importance"]
    assert queued.count(both.id) == 1


def test_recall_count_bump_follows_the_render_for_keyword_hits(tmp_path: Path) -> None:
    """A keyword hit in the first 5 positions renders full and takes the full
    +1.0 tick; one at position 6+ renders a snippet and takes a fractional
    rank-weighted tick; a keyword hit that is cut by the cap is never bumped."""
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(3)]
    kw = _seed_keyword_hits(store, 8)
    block = _render(store, "quokka", tmp_path, _semantic(full=sem))
    rows = _active_rows(block)
    rendered = {r[0] for r in rows}
    for pos, (rid, _body) in enumerate(rows[len(sem) :], start=len(sem)):
        bump = _rc(store, rid)
        if pos < FULL_INJECT_STANDOUT_MAX:
            assert bump == pytest.approx(1.0), "full keyword row takes the +1.0 tick"
        else:
            assert 0.0 < bump < 1.0, "snippet keyword row takes a fractional tick"
    for m in kw:
        if m.id not in rendered:
            assert _rc(store, m.id) == 0.0, "an unrendered keyword hit is never bumped"


# ---------------------------------------------------------------------------
# No semantic result: keyword-only under today's lexical caps (P-12).
# ---------------------------------------------------------------------------


def test_keyword_only_turn_keeps_the_lexical_caps_and_full_inject_rule(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    hi = _mem(store, f"{_LONG} (very important)", importance=9.5)
    _seed_keyword_hits(store, 12)
    block = _render(store, "quokka", tmp_path, None)
    rows = _active_rows(block)
    assert len(rows) == SNIPPET_COUNT, "keyword-only renders the lexical cap (8), not 9"
    fulls = [rid for rid, body in rows if not _is_snippet(body)]
    assert fulls == [hi.id], "only the importance >= 9 hit is full: no position rule without a semantic result"


# ---------------------------------------------------------------------------
# C8 keyword side + P-11: the monologue family never renders above a genuine
# memory, and the semantic snippet tier is not re-sorted by importance.
# ---------------------------------------------------------------------------


def test_keyword_monologue_family_hits_follow_genuine_hits(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    family = [
        _mem(store, f"{_LONG} (family {t})", importance=9.9, memory_type=t)
        for t in sorted(MONOLOGUE_FAMILY_TYPES)
    ]
    genuine = _seed_keyword_hits(store, 3, importance=1.0)
    ids = [r[0] for r in _active_rows(_render(store, "quokka", tmp_path, None))]
    family_ids = {m.id for m in family}
    positions = {mid: i for i, mid in enumerate(ids)}
    assert all(
        positions[g.id] < positions[f_id] for g in genuine for f_id in family_ids if f_id in positions
    ), "every genuine keyword hit ranks above every monologue-family hit, whatever the importance"


def test_merged_fill_takes_genuine_keyword_hits_before_family_ones(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(5)]
    family = [_mem(store, f"{_LONG} (family {i})", importance=9.9, memory_type="monologue") for i in range(4)]
    genuine = _seed_keyword_hits(store, 2, importance=1.0)
    ids = [r[0] for r in _active_rows(_render(store, "quokka", tmp_path, _semantic(full=sem)))]
    assert ids[5:7] == sorted([g.id for g in genuine], key=ids.index)
    assert {ids[5], ids[6]} == {g.id for g in genuine}
    assert set(ids[7:]) <= {m.id for m in family}


def _family_flood_store(n_family: int) -> tuple[MemoryStore, list[Memory], list[Memory]]:
    """`n_family` monologue-family memories matching 4 query words at importance
    9.5 and 6 genuine ones matching 1 word at importance 3.0. The family
    outranks the genuine ones on every signal (bm25, importance), and at 60 it
    outnumbers the ranker's whole 50-row candidate pool."""
    store = MemoryStore(":memory:")
    family = [
        _mem(store, f"quokka harbour market garden family note {i}", importance=9.5, memory_type="monologue")
        for i in range(n_family)
    ]
    genuine = [_mem(store, f"quokka plain genuine entry {i}", importance=3.0) for i in range(6)]
    return store, family, genuine


@pytest.mark.parametrize("n_family", [14, 44, 45, 60])
def test_family_flood_cannot_crowd_genuine_keyword_hits_out_of_the_limit(tmp_path: Path, n_family: int) -> None:
    """Floods of 14 (past the old 2*limit window), 45 and 60 (past the ranker's
    50-row candidate pool): every genuine match still renders, ahead of every
    family match (spec §4, S16, Acceptance 8)."""
    store, family, genuine = _family_flood_store(n_family)
    ids = [r[0] for r in _active_rows(_render(store, "quokka harbour market garden", tmp_path, None))]
    assert len(ids) == SNIPPET_COUNT
    assert {m.id for m in genuine} <= set(ids), "every genuine hit the store holds is rendered"
    assert {*ids[: len(genuine)]} == {m.id for m in genuine}, "and they lead the family hits"


@pytest.mark.parametrize("n_family", [14, 60])
def test_family_flood_on_the_no_persona_dir_path(n_family: int) -> None:
    store, family, genuine = _family_flood_store(n_family)
    block = _build_recall_block(store, "quokka harbour market garden", persona_dir=None)
    assert all(m.id in block for m in genuine)


def test_full_inject_slots_go_to_genuine_memories_before_monologue_family_ones(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(5)]
    family = _mem(store, f"{_LONG} (family)", importance=9.9, memory_type="monologue_trace")
    genuine = [_mem(store, f"{_LONG} (genuine {i})", importance=9.0) for i in range(3)]
    rows = _active_rows(_render(store, "quokka", tmp_path, _semantic(full=sem)))
    full_keyword = {rid for rid, body in rows[5:] if not _is_snippet(body)}
    assert full_keyword == {m.id for m in genuine}, "the 3 promoted-full slots are genuine; the family hit is a snippet"
    assert family.id in {rid for rid, _ in rows}


def test_semantic_snippet_tier_renders_in_path_order_not_resorted_by_importance(tmp_path: Path) -> None:
    """P-11 (R3's hand-off): a high-importance monologue-family memory in the
    semantic snippet tier must not render above a genuine snippet the path
    ranked first, and the top rank-weighted bump goes to the first-ranked."""
    store = MemoryStore(":memory:")
    sem_full = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(5)]
    genuine_snip = _mem(store, f"{_LONG} (genuine snippet)", importance=1.0)
    family_snip = _mem(store, f"{_LONG} (family snippet)", importance=9.9, memory_type="monologue_trace")
    result = _semantic(full=sem_full, snippet=[genuine_snip, family_snip])
    block = _render(store, "zzzunrelated", tmp_path, result)
    ids = [r[0] for r in _active_rows(block)]
    assert ids[5:7] == [genuine_snip.id, family_snip.id]
    assert _rc(store, genuine_snip.id) == pytest.approx(0.8), "top rank-weighted bump: the genuine snippet"
    assert _rc(store, family_snip.id) == pytest.approx(0.1), "bottom rank-weighted bump: the family snippet"


# ---------------------------------------------------------------------------
# S9 / S36 / S52: the 10-token cap is gone from searching; short tokens the
# selector keeps reach the store.
# ---------------------------------------------------------------------------

_RARE = ["zephyr", "mosaic", "lantern", "orchid", "pewter", "saffron", "tundra", "velvet", "willow", "yonder"]
_COMMON = ["harbour", "market"]


def _cap_fixture(store: MemoryStore) -> tuple[Memory, str]:
    """A message of 12 salient tokens: 10 rare ones (they rank first, so they
    ARE the old top 10) that all live in ONE decoy memory, plus 2 common ones
    (ranked 11th-12th, beyond the old cap) that are the only route to `target`."""
    _mem(store, "decoy " + " ".join(_RARE))
    target = _mem(store, f"the {' '.join(_COMMON)} morning walk with Canary")
    for i in range(7):
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular")
    message = " ".join(_RARE + _COMMON)
    return target, message


def test_a_token_beyond_the_old_ten_token_cap_is_searched(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    target, message = _cap_fixture(store)
    tokens = _extract_recall_tokens(message, store)
    assert len(tokens) == 12, "the selector returns every survivor, no cap"
    assert set(_legacy_capped_tokens(tokens)) == set(_RARE), "the old top 10 are the rare tokens"
    # base behaviour (searching only the old top 10) cannot reach the target ...
    old = store.search_fts_scored(_legacy_capped_tokens(tokens), limit=50)
    assert target.id not in {m.id for m, _ in old}
    # ... this build's passive recall does
    ids = [r[0] for r in _active_rows(_render(store, message, tmp_path, None))]
    assert target.id in ids


def test_legacy_capped_tokens_is_exactly_the_first_ten_of_the_ranked_tokens() -> None:
    assert _legacy_capped_tokens([str(i) for i in range(25)]) == [str(i) for i in range(10)]
    assert _RECALL_TOKEN_LIMIT == 10


def test_a_two_letter_acronym_the_selector_keeps_reaches_the_store(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    ai = _mem(store, "Bob showed Canary an AI notebook about tide tables")
    for i in range(3):
        _mem(store, f"harbour filler entry {i}")
    assert _extract_recall_tokens("AI", store) == ["ai"]
    # the raw-string path (other callers) keeps its 3-character minimum ...
    assert store.search_fts_scored("ai", limit=10) == []
    # ... a selector token list admits it, in passive recall too
    ids = [r[0] for r in _active_rows(_render(store, "AI", tmp_path, None))]
    assert ids == [ai.id]
    # and the no-persona_dir legacy path
    block = _build_recall_block(store, "AI", persona_dir=None)
    assert ai.id in block


# ---------------------------------------------------------------------------
# C9e (S71, REVIEW-PENDING): "not recognised" keeps today's size, chosen from
# what the old 10-token selector picked; the uncapped search still gets every
# token.
# ---------------------------------------------------------------------------

_UNKNOWN = [
    "Alderwick", "Brimstone", "Cobblewood", "Dunmarrow", "Eskerfield", "Fenwater", "Glimmerton",
    "Hollowmere", "Ironvale", "Jasperdown", "Kettlemoor", "Larkspire", "Marrowfen", "Nettlebrook",
]


def test_not_recognised_list_is_chosen_from_the_old_top_ten(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    _mem(store, "one filler memory so the store is not empty")
    message = "Bob asked about " + " ".join(_UNKNOWN)
    tokens = _extract_recall_tokens(message, store)
    assert len(tokens) > _RECALL_TOKEN_LIMIT
    legacy = _legacy_capped_tokens(tokens)
    queries: list = []
    original = MemoryStore.search_fts_scored

    def spy(self, query, **kwargs):
        queries.append(list(query) if not isinstance(query, str) else query)
        return original(self, query, **kwargs)

    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    with (
        patch.object(MemoryStore, "search_fts_scored", spy),
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
    ):
        block = _build_recall_block(store, message, persona_dir=tmp_path)

    listed = [ln.strip()[2:] for ln in block.splitlines() if ln.startswith("    - ") and "\"" not in ln]
    # today's size: exactly the old picks (all df 0, all capitalized), never more
    assert listed == legacy
    assert len(listed) == _RECALL_TOKEN_LIMIT
    assert not (set(tokens[_RECALL_TOKEN_LIMIT:]) & set(listed))
    # S79: tier 1 = today's search (the old top 10 joined into one raw string);
    # tier 2 = the tokens beyond it, so together every token is still searched
    assert queries[0] == " ".join(legacy)
    assert set(queries[1]) == set(tokens[_RECALL_TOKEN_LIMIT:])
    assert set(legacy) | set(queries[1]) == set(tokens)


# ---------------------------------------------------------------------------
# C9f (Q16 interim, PARKED for the owner): the graveyard is fed exactly what
# today's code feeds it; grief-touch does not fire more often than today.
# ---------------------------------------------------------------------------


def _bury(persona_dir: Path, content: str, mem_id: str) -> None:
    lost = Memory.create_new(
        content=content, memory_type="episodic", domain="memory", emotions={"joy": 8.5}
    )
    object.__setattr__(lost, "id", mem_id)
    gv.append(
        persona_dir,
        memory=lost,
        salience_at_drop=0.6,
        inputs=SalienceInputs(emotion=0.85, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.1),
        lived_age_hours=24.0,
        reason="test-seed",
    )


def _graveyard_fixture(tmp_path: Path, *, buried_word: str) -> tuple[MemoryStore, str, list[str]]:
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    store = MemoryStore(":memory:")
    # 12 known words: 10 rare (old top 10) and 2 common (beyond the old cap)
    _mem(store, "decoy " + " ".join(_RARE))
    for i in range(6):
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular")
    message = " ".join(_RARE + _COMMON)
    _bury(tmp_path, f"a summary that mentions the {buried_word} once", "mem-buried")
    tokens = _extract_recall_tokens(message, store)
    return store, message, _legacy_capped_tokens(tokens)


def test_graveyard_is_fed_the_old_capped_token_string_and_entry_beyond_the_cap_is_not_returned(
    tmp_path: Path,
) -> None:
    store, message, legacy = _graveyard_fixture(tmp_path, buried_word="harbour")  # 11th-12th token
    seen_queries: list[str] = []
    real_search = gv.search

    def spy(persona_dir, query, **kw):
        seen_queries.append(query)
        return real_search(persona_dir, query, **kw)

    with (
        patch("brain.forgetting.recall.graveyard.search", spy),
        patch("brain.grief.handle_recall_touch") as touch,
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
    ):
        block = _build_recall_block(store, message, persona_dir=tmp_path)

    assert seen_queries == [" ".join(legacy)], "graveyard.search gets exactly the base code's string"
    assert "lost (no longer in active memory)" not in block, "the entry beyond the old top 10 is NOT returned"
    assert touch.call_count == 0, "grief-touch is not called more often than on the base (0 for this fixture)"


def test_graveyard_entry_inside_the_old_top_ten_is_still_returned_and_touches_once(tmp_path: Path) -> None:
    store, message, legacy = _graveyard_fixture(tmp_path, buried_word="zephyr")  # inside the old top 10
    with (
        patch("brain.grief.handle_recall_touch") as touch,
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
    ):
        block = _build_recall_block(store, message, persona_dir=tmp_path)
    assert "lost (no longer in active memory)" in block
    assert touch.call_count == 1


def test_tier_one_query_is_todays_capped_string_and_feeds_the_graveyard_too(tmp_path: Path) -> None:
    """S79: the active/fading search's first tier is today's query (the old top
    10 joined) and the graveyard receives exactly that string; the tokens
    beyond it go to a second, graveyard-free tier."""
    store, message, legacy = _graveyard_fixture(tmp_path, buried_word="harbour")
    calls: list[dict] = []
    import brain.chat.prompt as prompt_mod  # noqa: PLC0415
    from brain.forgetting import recall as recall_mod  # noqa: PLC0415

    real = recall_mod.search_with_loss

    def spy(persona_dir, st, query, **kwargs):
        calls.append({"query": query, "lost_query": kwargs.get("lost_query")})
        return real(persona_dir, st, query, **kwargs)

    with (
        patch("brain.forgetting.recall.search_with_loss", spy),
        patch.object(prompt_mod, "run_semantic_recall", return_value=None),
    ):
        _build_recall_block(store, message, persona_dir=tmp_path)
    assert len(calls) == 1, "one search_with_loss call: the graveyard is fed once"
    assert calls[0]["query"] == " ".join(legacy)
    assert calls[0]["lost_query"] == " ".join(legacy)


# ---------------------------------------------------------------------------
# C9c: the recall block frame is unchanged on a merged turn.
# ---------------------------------------------------------------------------


def test_block_frame_and_section_order_are_preserved_on_a_merged_turn(tmp_path: Path) -> None:
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    store = MemoryStore(":memory:")
    sem = _mem(store, "the sunlit dock where the boats moor at low tide")
    _mem(store, f"{_LONG} (keyword)")
    fading = _mem(store, "the full original quokka story since forgotten in detail")
    store.fade(fading.id, summary="quokka days, long since faded from that summer")
    _bury(tmp_path, "the quokka rooftop morning before the cold rain", "mem-quokka-lost")
    with patch("brain.chat.prompt.run_semantic_recall", return_value=_semantic(full=[sem])):
        block = _build_recall_block(store, "quokka Zorblaxian", persona_dir=tmp_path)
    lines = block.splitlines()
    assert lines[0] == _RECALL_SNIPPET_INVITATION, "snippet invitation first when an active section exists"
    assert lines[1] == "recall", "then the header"
    order = [
        next(i for i, ln in enumerate(lines) if ln.strip() == "active:"),
        next(i for i, ln in enumerate(lines) if ln.strip().startswith("softened")),
        next(i for i, ln in enumerate(lines) if ln.strip().startswith("lost")),
        next(i for i, ln in enumerate(lines) if ln.strip().startswith("not recognised")),
    ]
    assert order == sorted(order)
    assert any(re.match(r'^    - \S+: "', ln) for ln in lines), "bullet format `    - <id>: \"<text>\"`"


# ---------------------------------------------------------------------------
# S79: keyword order = name query (R5 seam), today's capped selection, then the
# remaining tokens filling only the leftover slots.
# ---------------------------------------------------------------------------


def test_tier_two_hits_follow_every_tier_one_hit(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    target, message = _cap_fixture(store)
    rows = _active_rows(_render(store, message, tmp_path, None))
    ids = [r[0] for r in rows]
    decoy_id = next(mid for mid, body in rows if body.startswith("decoy"))
    assert ids[0] == decoy_id, "tier 1 (today's capped search) leads"
    assert target.id in ids[1:], "tier 2 (the tokens beyond the cap) follows and adds"


def test_tier_two_only_fills_leftover_slots(tmp_path: Path) -> None:
    """Enough tier-1 hits to fill the cap leave no slot for tier 2: nothing
    today's search finds is displaced by the extra tokens (S79, C12)."""
    store = MemoryStore(":memory:")
    target, message = _cap_fixture(store)
    tier_one = [_mem(store, f"{_RARE[i % 10]} {_RARE[(i + 1) % 10]} tier one entry {i}") for i in range(10)]
    ids = [r[0] for r in _active_rows(_render(store, message, tmp_path, None))]
    assert len(ids) == SNIPPET_COUNT
    assert target.id not in ids
    tier_one_ids = {m.id for m in tier_one} | {
        mid for mid, body in _active_rows(_render(store, message, tmp_path, None)) if body.startswith("decoy")
    }
    assert set(ids) <= tier_one_ids, "every rendered hit is a tier-1 hit"


def test_tier_one_alone_equals_todays_search_when_nothing_is_beyond_it(tmp_path: Path) -> None:
    """A message of 10 or fewer salient tokens, all 3+ characters, has no tier 2:
    exactly one FTS query, the raw string base code sent."""
    store = MemoryStore(":memory:")
    for i in range(3):
        _mem(store, f"quokka harbour entry {i}")
    queries: list = []
    original = MemoryStore.search_fts_scored

    def spy(self, query, **kwargs):
        queries.append(query)
        return original(self, query, **kwargs)

    with patch.object(MemoryStore, "search_fts_scored", spy):
        _render(store, "quokka harbour", tmp_path, None)
    assert queries == [" ".join(_legacy_capped_tokens(_extract_recall_tokens("quokka harbour", store)))]


def test_a_two_letter_token_in_the_old_top_ten_is_searched_in_tier_two(tmp_path: Path) -> None:
    """The store's 3-character floor stays in tier 1 (so it equals base); a
    2-letter acronym the selector kept still reaches the store, via tier 2 (S36)."""
    store = MemoryStore(":memory:")
    ai = _mem(store, "Bob showed Canary an AI notebook about tide tables")
    for i in range(3):
        _mem(store, f"harbour filler entry {i}")
    ids = [r[0] for r in _active_rows(_render(store, "AI harbour", tmp_path, None))]
    assert ai.id in ids


# ---------------------------------------------------------------------------
# S80: the "up to 3" counts only hits shown full BECAUSE of the importance rule.
# ---------------------------------------------------------------------------


def test_importance_rule_quota_ignores_hits_already_full_by_position(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    sem = [_mem(store, f"harbour gull entry {i} " + _LONG.replace("quokka", "gull")) for i in range(3)]
    for i in range(6):
        _mem(store, f"{_LONG} (important {i})", importance=9.5 - 0.1 * i)
    rows = _active_rows(_render(store, "quokka", tmp_path, _semantic(full=sem)))
    assert len(rows) == 9
    fulls = sum(1 for _, body in rows if not _is_snippet(body))
    # 3 semantic + keyword positions 4 and 5 by position + 3 promoted by importance beyond position 5
    assert fulls == 3 + 2 + FULL_INJECT_MAX
    assert _is_snippet(rows[-1][1])


def test_a_fading_memory_reachable_only_through_tier_two_is_still_softened_in(tmp_path: Path) -> None:
    """Tier 2 adds to the softened (fading) section too, after tier 1's hits.
    (The fade is a direct state update: `store.fade` would embed.)"""
    store = MemoryStore(":memory:")
    _target, message = _cap_fixture(store)
    faded = _mem(store, "an old harbour morning long since faded from that summer")
    store._conn.execute(  # noqa: SLF001
        "UPDATE memories SET state = 'fading', content_snapshot = content WHERE id = ?", (faded.id,)
    )
    store._conn.commit()  # noqa: SLF001
    block = _render(store, message, tmp_path, None)
    assert "softened (fading" in block
    assert "an old harbour" in block


def test_importance_quota_is_spent_on_tier_one_hits_before_tier_two(tmp_path: Path) -> None:
    """The extra tokens never take a full render (or the full bump) from a hit
    today's search already surfaces: five importance-9 tier-1 hits use the quota
    of 3 before three importance-10 tier-2 hits are considered."""
    store = MemoryStore(":memory:")
    tier1 = [_mem(store, f"{' '.join(_RARE)} tier one {i}", importance=9.0 + 0.1 * i) for i in range(5)]
    tier2 = [_mem(store, f"{' '.join(_COMMON)} tier two {i}", importance=10.0) for i in range(3)]
    for i in range(12):  # make the common words frequent, so they rank beyond the old top 10
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular", importance=4.0)
    message = " ".join(_RARE + _COMMON)
    rows = _active_rows(_render(store, message, tmp_path, None))
    full = {rid for rid, body in rows if not _is_snippet(body)}
    top3_tier1 = {m.id for m in sorted(tier1, key=lambda m: -m.importance)[:3]}
    assert full == top3_tier1
    assert not ({m.id for m in tier2} & full)


def _family_in_tier_one_store() -> tuple[MemoryStore, Memory, Memory, str]:
    store = MemoryStore(":memory:")
    family = _mem(store, "decoy " + " ".join(_RARE), memory_type="monologue", importance=9.5)
    target = _mem(store, f"the {' '.join(_COMMON)} morning walk with Canary")
    for i in range(3):
        _mem(store, f"{_COMMON[i % 2]} filler line {i} about nothing in particular", importance=4.0)
    return store, family, target, " ".join(_RARE + _COMMON)


def test_a_tier_two_genuine_hit_precedes_a_tier_one_family_hit(tmp_path: Path) -> None:
    store, family, target, message = _family_in_tier_one_store()
    ids = [r[0] for r in _active_rows(_render(store, message, tmp_path, None))]
    assert ids.index(target.id) < ids.index(family.id)


def test_a_tier_two_genuine_hit_precedes_a_tier_one_family_hit_without_persona_dir() -> None:
    store, family, target, message = _family_in_tier_one_store()
    block = _build_recall_block(store, message, persona_dir=None)
    assert block.index(target.id) < block.index(family.id)
