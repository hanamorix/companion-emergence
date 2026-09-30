"""Long-paste increment (S103, S131-S134): passive recall's tier-2 extra-words
step inside the REAL `_build_recall_block`, against a real store.

Criteria: L1 (all words fit = today's search), L2 (a long paste is cut to what
fits; names and tier 1 always run; the inherent effects), L3 (an exhausted
budget), L6 (the tier-1 seed in the real flow, the residue), L7 (the search tool
is untouched), L9 (the no-persona_dir path), L11 (call order), L13 (composition
with the semantic step, S132).

A scripted clock (`keyword_budget._clock`) is advanced INSIDE each patched step
(the semantic step, tier 1, the name query, the tier-2 search), so the span from
the top of `_build_recall_block` to the tier-2 decision is observed, not assumed.
All times are dyadic fractions of the budget `B`, so "strictly inside" and
"exactly at" fixtures are exact. The roles in the synthetic message:

  tier 1   : "Quill" (a listed name) + tierword1..9, each in one memory (T1A, T1B),
  fill     : n-1 words that share ONE memory (BULK),
  K, D     : the word the cut keeps (index n-1 of tier 2) and the word it drops (index n),
  unknowns : words no memory has (df 0), the long tail of the paste.

Synthetic data only (user "Bob", persona label "Canary").
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.chat.prompt import (
    _build_recall_block,
    _extract_recall_tokens,
    _keyword_tiers,
    _legacy_capped_tokens,
)
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.memory import keyword_budget, relevance
from brain.memory import known_names as kn
from brain.memory.cost_period import LinearFit
from brain.memory.store import FtsPhrases, Memory, MemoryStore

B = keyword_budget.recall_budget_seconds()
OVH = B / 256  # the seeded fit's overhead
RATE = B / 1024  # the seeded fit's per-word rate
T1 = 8 * RATE  # scripted seconds of tier 1
T_NAME = 8 * RATE  # scripted seconds of the name query
BASE_KWARGS = {"limit": 16, "include_fading": True, "genuine_first": True}


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _mem(store: MemoryStore, content: str, *, importance: float = 5.0, memory_type: str = "conversation") -> Memory:
    m = Memory.create_new(content=content, memory_type=memory_type, domain="us", importance=importance)
    store.create(m)
    return m


def _row_ids(block: str) -> list[str]:
    """Memory ids of the ACTIVE section, in render order."""
    ids: list[str] = []
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
            m = re.match(r'^    - (\S+): "', line)
            if m:
                ids.append(m.group(1))
    return ids


def make_world(
    tmp_path: Path,
    *,
    n: int = 40,
    unknown: int | None = None,
    names: bool = True,
    family: bool = False,
    acronym: bool = False,
    al_pair: bool = False,
):
    """The synthetic paste and store. `n` = words of tier 2 that fit in the cut
    fixtures: tier 2 is [fill x (n-1), K, D, (al), unknowns...]."""
    persona = tmp_path / "persona"
    persona.mkdir(parents=True, exist_ok=True)
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
    store = MemoryStore(":memory:")
    if names:
        assert kn.admit_names(persona, ["Quill"], "tool")
    lead = "Quill" if names else "tierword0"
    tier1 = [lead] + [f"tierword{i}" for i in range(1, 10)]
    if family:
        tier1 = [lead, "famwordxyz"] + [f"tierword{i}" for i in range(1, 9)]
    if acronym:
        tier1 = [lead, "AB"] + [f"tierword{i}" for i in range(1, 9)]
    fill = [f"fillw{i:03d}" for i in range(n - 1)]
    k_word, d_word = "keepword", "dropword"
    tail = ["pierword", "Al"] if al_pair else []
    count_unknown = unknown if unknown is not None else max(0, 300 - (len(tier1) + len(fill) + 2 + len(tail)))
    unknowns = [f"unkn{i:03d}" for i in range(count_unknown)]
    paste = " ".join([*tier1, *fill, k_word, d_word, *tail, *unknowns])

    mems = SimpleNamespace()
    in_t1_memories = [t for t in tier1 if t != "famwordxyz"]  # the family memory is that word's only memory
    half = len(in_t1_memories) // 2
    mems.t1a = _mem(store, " ".join(in_t1_memories[:half]) + " noted on the harbour wall")
    mems.t1b = _mem(store, " ".join(in_t1_memories[half:]) + " logged by the quay")
    mems.bulk = _mem(store, " ".join(fill) + " summed up in the ledger") if fill else None
    mems.keep = _mem(store, f"{k_word} kept note")
    mems.drop = _mem(store, f"{d_word} dropped note")
    if family:
        mems.fam = _mem(store, "famwordxyz pondering on the tide", memory_type="monologue")
    if al_pair:
        mems.al_both = _mem(store, "Al and pierword together on the pier", importance=5.0)
        mems.al_only = _mem(store, "Al chased the ball down the pier", importance=9.0)
        assert kn.admit_names(persona, ["Al"], "tool")
    tokens = _extract_recall_tokens(paste, store)
    legacy = _legacy_capped_tokens(tokens)
    capped, words = _keyword_tiers(tokens, legacy)
    assert k_word in words and d_word in words, "fixture: K and D must be tier-2 words"
    if not (family or acronym or al_pair):
        assert words.index(d_word) == n and words.index(k_word) == n - 1, (words[:5], words.index(k_word))
    if al_pair:
        assert words.index("pierword") == n + 1 and words.index(d_word) == n, words[n - 1 : n + 3]
    return SimpleNamespace(
        store=store,
        persona=persona,
        paste=paste,
        tokens=tokens,
        legacy=legacy,
        capped=capped,
        words=words,
        mems=mems,
        n=n,
    )


def seed_fit(store: MemoryStore, overhead: float = OVH, rate: float = RATE) -> LinearFit:
    """Sizes 4 and 8: the OLS denominator is 16, so the fit is exactly (overhead, rate)."""
    key = str(store.db_path)
    for x in (4, 8):
        keyword_budget._keyword_cost.record(key, x, overhead + x * rate)  # noqa: SLF001
    fit = keyword_budget._keyword_cost.fit(key)  # noqa: SLF001
    assert (fit.overhead, fit.rate) == (overhead, rate)
    return fit


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", c)
    return c


def wire(monkeypatch: pytest.MonkeyPatch, clock: Clock, *, t_sem: float, t1: float = T1, t_name: float = T_NAME,
         tier2_overhead: float = OVH, tier2_rate: float = RATE):
    """Patch the semantic step, tier 1, the name query and the tier-2 search so each
    advances the scripted clock and records its call. Returns the call log."""
    calls: list[tuple] = []

    def fake_semantic(*_a, **_k):
        clock.advance(t_sem)
        return None

    monkeypatch.setattr("brain.chat.prompt.run_semantic_recall", fake_semantic)

    import brain.forgetting.recall as recall

    real_swl = recall.search_with_loss

    def spy_swl(*a, **k):
        calls.append(("tier1", a, k))
        clock.advance(t1)
        return real_swl(*a, **k)

    monkeypatch.setattr(recall, "search_with_loss", spy_swl)

    real_rank = relevance.rank_memories

    def spy_rank(store, heb, query, **k):
        if isinstance(query, FtsPhrases):
            calls.append(("name", tuple(query), k))
            clock.advance(t_name)
        elif isinstance(query, list):
            calls.append(("tier2", list(query), k))
            clock.advance(tier2_overhead + tier2_rate * len(query))
        else:
            calls.append(("tier1-legacy", query, k))
            clock.advance(t1)
        return real_rank(store, heb, query, **k)

    monkeypatch.setattr(relevance, "rank_memories", spy_rank)
    return calls


def by_role(block: str, w) -> str:
    """The block with each memory id replaced by its role name (worlds have different uuids)."""
    for role, mem in vars(w.mems).items():
        if mem is not None:
            block = block.replace(mem.id, f"<{role}>")
    return block


def render(w, *, persona: bool = True, limit: int = 8) -> str:
    return _build_recall_block(w.store, w.paste, persona_dir=w.persona if persona else None, limit=limit)


def kinds(calls: list[tuple]) -> list[str]:
    return [c[0] for c in calls]


def tier2(calls: list[tuple]) -> list[tuple]:
    return [c for c in calls if c[0] == "tier2"]


def t_sem_for(left_target: float, *, t1: float = T1, t_name: float = T_NAME) -> float:
    """The scripted semantic seconds that leave exactly `left_target` when tier 2 is decided."""
    return B - left_target - t1 - t_name


# ---------------------------------------------------------------------------
# L1: all words fit: today's search.
# ---------------------------------------------------------------------------


def _bypass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        keyword_budget, "search_extra_words", lambda store, words, started, search, seed: search(words)
    )


@pytest.mark.parametrize("variant", ["seeded", "fresh-seed", "frozen-zero"])
def test_when_every_extra_word_fits_tier_two_is_todays_single_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock, variant: str
) -> None:
    w = make_world(tmp_path / "a", n=2, unknown=1)  # a short remainder: [fill0, K, D, unkn000]
    assert 3 <= len(w.words) <= 4
    if variant == "seeded":
        seed_fit(w.store)
    calls = wire(
        monkeypatch,
        clock,
        t_sem=0.0,
        t1=0.0 if variant == "frozen-zero" else T1,
        t_name=0.0 if variant == "frozen-zero" else T_NAME,
        tier2_overhead=0.0,
        tier2_rate=0.0,
    )
    block = render(w)
    t2 = tier2(calls)
    assert len(t2) == 1, "exactly one tier-2 search"
    assert t2[0][1] == w.words, "every word, in the selector's order"
    assert t2[0][2] == BASE_KWARGS, "the keyword arguments of BASE"
    # the byte-for-byte comparison with the pre-increment code path
    w2 = make_world(tmp_path / "b", n=2, unknown=1)
    if variant == "seeded":
        seed_fit(w2.store)
    _bypass(monkeypatch)
    wire(monkeypatch, Clock(), t_sem=0.0, t1=0.0, t_name=0.0, tier2_overhead=0.0, tier2_rate=0.0)
    assert by_role(render(w2), w2) == by_role(block, w)
    assert w.mems.keep.id in block


def test_no_persona_dir_path_when_everything_fits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock) -> None:
    w = make_world(tmp_path, n=2, unknown=1, names=False)
    seed_fit(w.store)
    calls = wire(monkeypatch, clock, t_sem=0.0, t1=0.0, tier2_overhead=0.0, tier2_rate=0.0)
    block = render(w, persona=False)
    t2 = tier2(calls)
    assert len(t2) == 1 and t2[0][1] == w.words
    assert t2[0][2] == {"limit": 8, "genuine_first": True}, "the legacy path's own arguments"
    assert w.mems.keep.id in block


# ---------------------------------------------------------------------------
# L2 / L11: a long paste, the budget cuts the extras.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offset", "kept_delta"),
    [(RATE / 2, 0), (0.0, 0), (-RATE / 2, -1)],
    ids=["strictly-inside", "exactly-at-predict(n)", "just-below-predict(n)"],
)
def test_a_long_paste_is_cut_to_the_words_that_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock, offset: float, kept_delta: int
) -> None:
    n = 40
    w = make_world(tmp_path, n=n)
    assert len(w.paste.split()) >= 300 and len(w.words) >= 250
    fit = seed_fit(w.store)
    left_target = fit.predict(n) + offset
    calls = wire(monkeypatch, clock, t_sem=t_sem_for(left_target))
    block = render(w)
    want = n + kept_delta
    t2 = tier2(calls)
    assert len(t2) == 1, "one tier-2 search"
    assert t2[0][1] == w.words[:want], "exactly the first n words, selector order"
    assert t2[0][2] == BASE_KWARGS
    # the first dropped word is absent, the last kept word is present
    if want == n:
        assert w.mems.keep.id in block, "the last kept word (index n-1) is found"
    else:
        assert w.mems.keep.id not in block
    assert w.mems.drop.id not in block, "the first dropped word (index n) is not"
    # names and today's 10 words always run, with their usual arguments, in order
    assert kinds(calls) == ["tier1", "name", "tier2"], "L11: tier 1, the name query, then tier 2"
    tier1 = calls[0]
    assert tier1[1][2] == w.capped and tier1[2]["limit"] == 16 and tier1[2]["lost_names"] == ["quill"]
    assert tier1[2]["lost_query"] == " ".join(w.legacy) and tier1[2]["genuine_first"] is True
    assert calls[1][1] == ("quill",)


def test_the_cut_changes_only_the_tier_two_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock) -> None:
    n = 40
    cut_world = make_world(tmp_path / "cut", n=n)
    fit = seed_fit(cut_world.store)
    wire(monkeypatch, clock, t_sem=t_sem_for(fit.predict(n) + RATE / 2))
    cut_ids = _row_ids(render(cut_world))

    full_world = make_world(tmp_path / "full", n=n)
    keyword_budget._reset_keyword_cost()  # noqa: SLF001
    seed_fit(full_world.store)
    clock2 = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", clock2)
    wire(monkeypatch, clock2, t_sem=0.0)
    full_ids = _row_ids(render(full_world))

    assert full_world.mems.drop.id in full_ids and cut_world.mems.drop.id not in cut_ids
    # map the two worlds' memories onto each other by role
    roles = ("t1a", "t1b", "bulk", "keep", "drop")
    to_role_full = {getattr(full_world.mems, r).id: r for r in roles}
    to_role_cut = {getattr(cut_world.mems, r).id: r for r in roles}
    assert [to_role_cut[i] for i in cut_ids] == [to_role_full[i] for i in full_ids if to_role_full[i] != "drop"]


def test_inherent_effect_a_name_and_tier_two_double_hit_falls_to_the_name_only_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    """Pinned as today's rule (S89) applied to a shorter tier 2, not a regression: a memory
    found by the name query AND the general search leads the name-only ones; cut the tier-2 word
    that found it and it drops to the name-only group, ordered by its own score."""
    n = 40
    w = make_world(tmp_path, n=n, al_pair=True)
    fit = seed_fit(w.store)
    # up to and including "pierword" (index n+1), but not the name's own token "al" (index n+2):
    # the name token would find BOTH Al memories through the general search
    calls = wire(monkeypatch, clock, t_sem=t_sem_for(fit.predict(n + 2) + RATE / 2))
    ids_fit = _row_ids(render(w))
    assert tier2(calls)[0][1] == w.words[: n + 2] and "al" not in tier2(calls)[0][1]
    both, only = w.mems.al_both.id, w.mems.al_only.id
    assert ids_fit.index(both) < ids_fit.index(only), "found by both: leads the name-only hit"

    w2 = make_world(tmp_path / "second", n=n, al_pair=True)
    keyword_budget._reset_keyword_cost()  # noqa: SLF001
    fit2 = seed_fit(w2.store)
    clock2 = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", clock2)
    wire(monkeypatch, clock2, t_sem=t_sem_for(fit2.predict(n + 1) + RATE / 2))  # "pierword" (index n+1) dropped
    ids_cut = _row_ids(render(w2))
    both2, only2 = w2.mems.al_both.id, w2.mems.al_only.id
    assert ids_cut.index(only2) < ids_cut.index(both2), "both are name-only now: the higher-scored one leads"


def test_inherent_effect_a_tier_one_family_hit_reappears_when_tier_two_hits_no_longer_fill_the_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    """Pinned as today's rule (S94): genuine hits, tier 2 included, rank before a tier-1
    monologue-family hit, so cutting tier-2 hits can let the family hit back in."""
    n = 40
    w = make_world(tmp_path, n=n, family=True)
    fit = seed_fit(w.store)
    wire(monkeypatch, clock, t_sem=0.0)
    ids_fit = _row_ids(render(w, limit=5))
    assert w.mems.fam.id not in ids_fit, "five genuine hits fill the five slots"

    w2 = make_world(tmp_path / "second", n=n, family=True)
    keyword_budget._reset_keyword_cost()  # noqa: SLF001
    fit2 = seed_fit(w2.store)
    clock2 = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", clock2)
    wire(monkeypatch, clock2, t_sem=t_sem_for(fit2.predict(n) + RATE / 2))  # D dropped
    ids_cut = _row_ids(render(w2, limit=5))
    assert w2.mems.fam.id in ids_cut, "four genuine hits: the family hit takes the fifth slot"
    assert fit.rate == fit2.rate


# ---------------------------------------------------------------------------
# L3: the budget is already spent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["seeded", "fresh"])
@pytest.mark.parametrize(
    "case", ["over-budget", "left-exactly-zero", "one-word-does-not-fit"]
)
def test_when_nothing_fits_no_extra_word_is_searched_and_the_rest_still_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock, mode: str, case: str
) -> None:
    w = make_world(tmp_path, n=40)
    if mode == "seeded":
        seed_fit(w.store)
    left = {"over-budget": -RATE, "left-exactly-zero": 0.0, "one-word-does-not-fit": OVH + RATE / 2}[case]
    if mode == "fresh" and case == "one-word-does-not-fit":
        # the seed rate is T1 / 10 words: pick a left below one word of the seed
        left = (T1 / 10) / 2
    calls = wire(monkeypatch, clock, t_sem=t_sem_for(left))
    block = render(w)
    assert tier2(calls) == [], "no tier-2 search of any size (no probe, no partial)"
    state = keyword_budget._keyword_cost.state(str(w.store.db_path))  # noqa: SLF001
    assert (state is None) if mode == "fresh" else (state.sums.n == 2), "no sample recorded"
    assert kinds(calls) == ["tier1", "name"], "tier 1 and the name query still run"
    assert w.mems.t1a.id in block, "the block renders from tier 1"


# ---------------------------------------------------------------------------
# L6: the tier-1 seed in the real flow; the residue.
# ---------------------------------------------------------------------------


def test_a_fresh_process_sizes_tier_two_from_the_tier_one_seed_with_the_floor_filtered_word_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    """Tier 1's top 10 holds a 2-letter acronym the store's 3-character floor drops:
    the seed's word count is 9, not 10."""
    w = make_world(tmp_path, n=40, acronym=True)
    assert "ab" in w.legacy
    rate = B / 512
    t1 = 9 * rate
    left_target = 40.5 * rate
    calls = wire(monkeypatch, clock, t_sem=t_sem_for(left_target, t1=t1), t1=t1)
    render(w)
    t2 = tier2(calls)
    assert len(t2) == 1 and t2[0][1] == w.words[:40], "floor(left / (t1 / 9)) = 40; a count of 10 would give 45"
    assert keyword_budget._keyword_cost.state(str(w.store.db_path)) is None, "S134: first sample discarded"  # noqa: SLF001


def test_residue_tier_one_raised_and_no_estimate_means_no_extras(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    w = make_world(tmp_path, n=40)
    calls = wire(monkeypatch, clock, t_sem=0.0, t1=0.0, t_name=0.0)
    import brain.forgetting.recall as recall

    def boom(*_a, **_k):
        raise RuntimeError("tier 1 failed")

    monkeypatch.setattr(recall, "search_with_loss", boom)
    render(w)
    assert tier2(calls) == [], "never unbounded: no seed, no fit, no extras"
    # with a measured estimate the same message does search
    seed_fit(w.store)
    render(w)
    assert len(tier2(calls)) == 1


def test_residue_a_message_whose_tier_one_words_are_all_under_the_floor_has_no_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    """"AI ML": the store's floor leaves tier 1 with no word, so no seed exists. In a fresh
    process with no estimate the two words are not searched; once the series has an
    estimate they are. (Owner FYI: S131 says a fresh process always has a seed; this is
    the one message shape where it does not.)"""
    persona = tmp_path / "p"
    persona.mkdir(parents=True, exist_ok=True)
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), persona)
    store = MemoryStore(":memory:")
    target = _mem(store, "AI ML notes from the harbour course")
    calls = wire(monkeypatch, clock, t_sem=0.0, t1=0.0)

    def go() -> str:
        return _build_recall_block(store, "AI ML", persona_dir=persona)

    block = go()
    assert tier2(calls) == [] and target.id not in block
    seed_fit(store)
    block = go()
    assert len(tier2(calls)) == 1 and target.id in block


# ---------------------------------------------------------------------------
# L9: the no-persona_dir path.
# ---------------------------------------------------------------------------


def test_no_persona_dir_path_cuts_and_skips_like_the_persona_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    n = 40
    w = make_world(tmp_path, n=n, names=False)
    fit = seed_fit(w.store)
    left_target = fit.predict(n) + RATE / 2
    t1 = B - left_target
    calls = wire(monkeypatch, clock, t_sem=0.0, t1=t1)
    render(w, persona=False)
    t2 = tier2(calls)
    assert len(t2) == 1 and t2[0][1] == w.words[:n]
    assert t2[0][2] == {"limit": 8, "genuine_first": True}
    assert kinds(calls) == ["tier1-legacy", "tier2"]
    state = keyword_budget._keyword_cost.state(str(w.store.db_path)).sums  # noqa: SLF001
    assert state.n == 2, "the discarded first sample leaves the two seeded ones"
    # exhausted: nothing is searched
    keyword_budget._reset_keyword_cost()  # noqa: SLF001
    seed_fit(w.store)
    clock.t = 0.0
    calls.clear()
    wire_calls = wire(monkeypatch, clock, t_sem=0.0, t1=B)
    render(w, persona=False)
    assert tier2(wire_calls) == []


# ---------------------------------------------------------------------------
# L13: composition with the semantic step (S132).
# ---------------------------------------------------------------------------


def test_extras_share_the_one_budget_with_the_semantic_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: Clock
) -> None:
    """S132 (owner FYI): "what is left" is literal, after reranking: a message whose semantic
    step spends its whole share gets no extras that turn, and the first message after a
    restart has little left. Alternative on the owner's list: reserve the extras' predicted
    cost before sizing the rerank (changes R6's formula)."""
    seen: list[int] = []
    for c in (B, B - 16 * RATE - OVH, 0.75 * B, 0.5 * B, 0.0):
        keyword_budget._reset_keyword_cost()  # noqa: SLF001
        ck = Clock()
        monkeypatch.setattr(keyword_budget, "_clock", ck)
        w = make_world(tmp_path / f"c{c}", n=40)
        fit = seed_fit(w.store)
        calls = wire(monkeypatch, ck, t_sem=c)
        render(w)
        t2 = tier2(calls)
        left = B - (c + T1 + T_NAME)
        want = 0 if left <= 0 else min(len(w.words), keyword_budget.words_that_fit(fit, len(w.words), left))
        assert (len(t2[0][1]) if t2 else 0) == want
        assert ck.t <= B + 1e-9 or want == 0, "the cut message never overruns the budget"
        seen.append(want)
    assert seen[0] == 0 and seen == sorted(seen), "monotone in what the semantic step left"
    assert seen[-1] > seen[1]


# ---------------------------------------------------------------------------
# L7: the search tool is not affected (S99).
# ---------------------------------------------------------------------------


def test_the_tool_never_touches_the_keyword_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from brain.tools.dispatch import dispatch
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _ctx
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _mem as tool_mem

    def forbidden(*_a, **_k):
        raise AssertionError("the search tool must not use the keyword budget")

    for name in ("search_extra_words", "timed_tier1", "record_tier2", "timed", "now"):
        monkeypatch.setattr(keyword_budget, name, forbidden)
    ctx = _ctx(tmp_path)
    store = ctx["store"]
    tool_mem(store, "quokka harbour market garden notes from last autumn")
    queries: list = []
    real = MemoryStore.search_fts_scored

    def spy(self, query, **k):
        queries.append(query)
        return real(self, query, **k)

    monkeypatch.setattr(MemoryStore, "search_fts_scored", spy)
    long_query = "quokka harbour market garden " + " ".join(f"word{i:03d}" for i in range(300))
    res = dispatch("search_memories", {"query": long_query, "mode": "lexical", "limit": 8}, **ctx)
    assert res["memories"], "the tool still searches and finds"
    assert queries == [long_query], "the raw query, every word, exactly as today"
    assert keyword_budget._keyword_cost.state(str(store.db_path)) is None  # noqa: SLF001
