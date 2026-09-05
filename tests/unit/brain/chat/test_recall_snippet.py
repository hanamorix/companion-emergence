"""Assembled recall-block behaviour for P2 — C6/C8/C9/C11/C12/C13/C19/C20/C21/C22.

All oracles exercise the REAL `_build_recall_block` (and the static/volatile
builders), not an isolated `rank_memories`, so the assembled-path guarantees the
round-4 review flagged are actually verified.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.chat.prompt import (
    _EPISTEMIC_INSTRUCTION,
    _build_recall_block,
    build_static_system_message,
    build_volatile_context,
)
from brain.engines.daemon_state import DaemonState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.relevance import FULL_INJECT_MAX, SNIPPET_COUNT
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore
from brain.tools.impls.search_memories import search_memories

# Invariants the epistemic instruction must preserve, checked in
# test_c13_unfamiliar_bucket_and_epistemic_instruction below instead of a
# byte-for-byte frozen copy (which went stale under ef173846's reword and
# would go stale again on the next wording pass).


def _store() -> MemoryStore:
    return MemoryStore(":memory:")


def _mem(
    store: MemoryStore,
    content: str,
    *,
    importance: float = 0.0,
    age_days: float = 0.0,
) -> Memory:
    m = Memory(
        id=str(uuid.uuid4()),
        content=content,
        memory_type="event",
        domain="d",
        created_at=datetime.now(UTC) - timedelta(days=age_days),
        importance=importance,
    )
    store.create(m)
    return m


def _rc(store: MemoryStore, mid: str) -> int:
    return store._conn.execute("SELECT recall_count FROM memories WHERE id = ?", (mid,)).fetchone()[0]


def _la(store: MemoryStore, mid: str):
    return store._conn.execute(
        "SELECT last_accessed_at FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _active_lines(block: str) -> list[str]:
    # Active bullets now read '    - <id>: "…"' (memory id precedes the
    # quoted snippet so read_full_memory(<id>) is callable) instead of the
    # old bare '    - "…"'. Match the id-prefixed quote instead of the
    # literal '    - "' — a stricter startswith would miss every active line.
    import re

    return [ln for ln in block.splitlines() if re.match(r'^    - \S+: "', ln)]


def _display_ids(block: str) -> list[str]:
    """Ids of the rendered 'active:' bullets, in DISPLAY (top-to-bottom) order."""
    import re

    ids = []
    for line in _active_lines(block):
        m = re.match(r'^    - (\S+): "', line)
        if m:
            ids.append(m.group(1))
    return ids


def test_c6_surfacing_bumps_recall_count_fractionally_by_display_rank(
    tmp_path: Path,
) -> None:
    """ND-1 revision (recall-reinforcement G1): passive surfacing now bumps
    recall_count FRACTIONALLY by DISPLAY rank (top ~0.8, bottom ~0.1,
    monotonically decreasing) instead of not at all. Drives the real
    _build_recall_block assembly end-to-end (not a standalone amount(i)
    helper): the DISPLAY order is read back from the block's own rendered
    bullets, then each surfaced memory's recall_count delta is checked
    against that order.

    explicit search_memories stays bump-free (G7 — see
    test_search_memories_ranked.py for the dedicated assertion).
    """
    store = _store()
    heb = HebbianMatrix(":memory:")
    mems = [_mem(store, f"jordan harbor chapter {i}", importance=float(i)) for i in range(4)]
    before = {m.id: _rc(store, m.id) for m in mems}

    block = _build_recall_block(store, "jordan harbor", persona_dir=tmp_path)
    ids_in_order = _display_ids(block)
    assert len(ids_in_order) == 4, "all 4 low-importance memories render as snippets"

    deltas = [_rc(store, mid) - before[mid] for mid in ids_in_order]
    assert deltas[0] == pytest.approx(0.8, abs=0.01), "top surfaced memory bumped ~0.8"
    assert deltas[-1] == pytest.approx(0.1, abs=0.01), "bottom surfaced memory bumped ~0.1"
    for a, b in zip(deltas, deltas[1:], strict=False):
        assert a >= b - 1e-9, "bump amount is non-increasing by display rank"

    # search_memories (explicit query) stays bump-free — no double-credit.
    before_search = {mid: _rc(store, mid) for mid in ids_in_order}
    search_memories("jordan", store=store, hebbian=heb, persona_dir=tmp_path)
    for mid in ids_in_order:
        assert _rc(store, mid) == before_search[mid], "search_memories must not bump recall_count"


def test_g2_single_surfaced_memory_bumped_at_top_amount(tmp_path: Path) -> None:
    """G2: N=1 surfaced memory is bumped by the TOP amount (~0.8), not the
    bottom amount (~0.1) and not zero."""
    store = _store()
    m = _mem(store, "solitary lighthouse at midnight", importance=1.0)
    before = _rc(store, m.id)

    block = _build_recall_block(store, "solitary lighthouse", persona_dir=tmp_path)
    assert block.strip() != ""
    assert _rc(store, m.id) - before == pytest.approx(0.8, abs=0.01)


def test_full_inject_active_rows_receive_no_passive_bump(tmp_path: Path) -> None:
    """G1 (full-inject exclusion): a high-importance memory rendered IN FULL
    (not as a snippet) receives no passive bump — it is already maximally
    salient by importance. A snippet-rendered sibling in the same block still
    gets bumped normally."""
    store = _store()
    hi = _mem(store, "beacon " + ("H" * 300), importance=9.5)  # full-inject
    lo = _mem(store, "beacon low importance detail", importance=1.0)  # snippet
    before_hi, before_lo = _rc(store, hi.id), _rc(store, lo.id)

    block = _build_recall_block(store, "beacon", persona_dir=tmp_path)
    assert ("H" * 300) in block, "hi renders in full (untruncated) — confirms it is full-inject"

    assert _rc(store, hi.id) == before_hi, "full-inject rows get no passive bump"
    assert _rc(store, lo.id) > before_lo, "snippet-rendered rows are still bumped"


def test_lost_bucket_entries_are_never_bumped(tmp_path: Path) -> None:
    """G1 (lost-bucket exclusion): graveyard 'lost' entries are dicts with no
    live store row — there is nothing to bump, and the block assembly must
    not crash reaching for one. A live sibling memory in the same block is
    still bumped normally."""
    from brain.forgetting import graveyard
    from brain.forgetting.salience import SalienceInputs

    store = _store()
    live = _mem(store, "beacon still active detail", importance=1.0)
    before_live = _rc(store, live.id)

    lost_mem = Memory.create_new(
        content="beacon old lost studio address",
        memory_type="episodic",
        domain="d",
        emotions={},
    )
    graveyard.append(
        tmp_path,
        memory=lost_mem,
        salience_at_drop=0.05,
        inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
        lived_age_hours=200.0,
        reason="test-seed",
    )

    block = _build_recall_block(store, "beacon", persona_dir=tmp_path)
    assert "forgotten" in block.lower()
    assert _rc(store, live.id) > before_live, "the live sibling is still bumped normally"


def test_g19_fading_memory_surfaced_as_snippet_is_bumped_last_accessed_unchanged(
    tmp_path: Path,
) -> None:
    """G19: a fading memory surfaced as a snippet IS bumped, but the passive
    bump leaves last_accessed_at unchanged (bump_recall touches recall_count
    only)."""
    store = _store()
    m = _mem(store, "harbor lantern original long-form detail here", importance=3.0)
    store.fade(m.id, summary="harbor lantern moment")
    before_rc, before_la = _rc(store, m.id), _la(store, m.id)
    assert before_la is None

    block = _build_recall_block(store, "harbor lantern", persona_dir=tmp_path)
    assert "softened (fading" in block

    assert _rc(store, m.id) > before_rc, "fading memory surfaced as a snippet is bumped"
    assert _la(store, m.id) == before_la, "passive bump must not touch last_accessed_at"


def test_g18_multi_turn_bump_accumulates_additively_and_stays_bounded(
    tmp_path: Path,
) -> None:
    """G18: passive recall fires every turn — the bump must accumulate
    ADDITIVELY across turns (surfacing the same top-ranked memory across 3
    calls yields recall_count ~= 2.4), and the salience effect stays BOUNDED
    by the clamped recall arm (recall_count/10, capped 1.0) — a much larger
    recall_count cannot push the recall arm's salience contribution past its
    cap."""
    from brain.forgetting import salience
    from brain.forgetting.salience import DEFAULT_WEIGHTS

    store = _store()
    m = _mem(store, "orchid signal beacon", importance=1.0)

    for _ in range(3):
        block = _build_recall_block(store, "orchid signal beacon", persona_dir=tmp_path)
        assert block.strip() != ""

    rc = _rc(store, m.id)
    assert rc == pytest.approx(2.4, abs=0.02), "3 top-rank (N=1) surfacings accumulate additively"

    heb = HebbianMatrix(":memory:")
    restored = store.get(m.id, bump=False)
    s = salience.score(
        restored, store=store, hebbian=heb, felt_time_state=None, soul_linked_ids=set()
    )
    # felt_time_state=None -> cold-start freshness=1.0 (fixed); the recall arm
    # contributes weights["recall"] * clamp(recall_count/10, 0, 1).
    expected = DEFAULT_WEIGHTS["freshness"] * 1.0 + DEFAULT_WEIGHTS["recall"] * min(rc / 10.0, 1.0)
    assert s == pytest.approx(expected, abs=0.01)

    # Bounded: even a much larger recall_count cannot push the recall arm's
    # contribution past its clamp.
    store.bump_recall(m.id, 50.0)
    restored2 = store.get(m.id, bump=False)
    s2 = salience.score(
        restored2, store=store, hebbian=heb, felt_time_state=None, soul_linked_ids=set()
    )
    max_possible = DEFAULT_WEIGHTS["freshness"] * 1.0 + DEFAULT_WEIGHTS["recall"] * 1.0
    assert s2 == pytest.approx(max_possible, abs=0.01)


def test_c8_high_importance_full_low_importance_snippet(tmp_path: Path) -> None:
    store = _store()
    hi_body = "jordan " + ("H" * 300)
    lo_body = "jordan " + ("L" * 300)
    _mem(store, hi_body, importance=9.5)
    _mem(store, lo_body, importance=1.0)

    block = _build_recall_block(store, "jordan", persona_dir=tmp_path)
    assert ("H" * 300) in block, "high-importance memory renders in full (untruncated)"
    assert ("L" * 300) not in block, "low-importance long body is truncated to a snippet"
    assert "…" in block


def test_c9_assembled_surfaces_up_to_snippet_count(tmp_path: Path) -> None:
    store = _store()
    for i in range(10):
        _mem(store, f"jordan moment number {i}")
    block = _build_recall_block(store, "jordan", persona_dir=tmp_path)
    assert len(_active_lines(block)) == SNIPPET_COUNT == 8  # up from the stale limit=5


def test_c12_recall_fires_on_matching_input(tmp_path: Path) -> None:
    store = _store()
    _mem(store, "jordan was here that summer", importance=3.0)
    block = _build_recall_block(store, "what about jordan", persona_dir=tmp_path)
    assert block.strip() != ""
    # Proportional snippet length (CHANGE 2) truncates this 28-char body to
    # its floored 20-char snippet, so the FULL string no longer appears
    # verbatim — only the surviving prefix does (recall fired; the exact
    # truncation formula is covered by test_snippet_length.py).
    assert "jordan was here" in block


def test_c13_unfamiliar_bucket_and_epistemic_instruction(tmp_path: Path) -> None:
    store = _store()
    block = _build_recall_block(store, "who is marcus and lisbon", persona_dir=tmp_path)
    assert "not recognised (searched; no memory found)" in block
    assert "marcus" in block

    # Epistemic instruction invariants (not a byte-for-byte frozen copy, which
    # drifted under ef173846's reword). ef173846's point was that
    # read_full_memory is now named AHEAD of search_memories, so assert both
    # presence and ordering.
    instr = _EPISTEMIC_INSTRUCTION
    i_read_full = instr.find("read_full_memory")
    i_search = instr.find("search_memories")
    assert i_read_full != -1, "must instruct opening surfaced memories via read_full_memory"
    assert i_search != -1, "must instruct search_memories for unsurfaced info"
    assert i_read_full < i_search, "read_full_memory must be named ahead of search_memories"

    # Content the frozen copy was guarding: the search-before-denying rule and
    # the exact "not recognised" wording the recall block emits (asserted
    # against `block` above), plus the never-invent-familiarity distinction.
    assert "call search_memories" in instr
    assert 'Never say "I don\'t remember" without searching first' in instr
    assert '"not recognised (searched; no memory found)"' in instr
    assert '"I never knew this"' in instr and '"I don\'t remember"' in instr
    assert "Do not invent familiarity" in instr


def test_c11_recall_absent_from_static_present_in_volatile(tmp_path: Path) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    store = _store()
    _mem(store, "jordan and the long walk", importance=3.0)
    soul = SoulStore(":memory:")

    static = build_static_system_message(persona_dir, voice_md="")
    assert "── recall" not in static
    assert "recall\n  active:" not in static

    volatile = build_volatile_context(
        persona_dir,
        voice_md="",
        daemon_state=DaemonState(),
        soul_store=soul,
        store=store,
        user_input="tell me about jordan",
    )
    assert "recall\n  active:" in volatile


def test_c20_recall_block_full_inject_is_bounded(tmp_path: Path) -> None:
    store = _store()
    for i in range(6):
        _mem(store, f"jordan detail {i} " + ("Z" * 200), importance=9.5)
    block = _build_recall_block(store, "jordan", persona_dir=tmp_path)
    lines = _active_lines(block)
    full = [ln for ln in lines if "…" not in ln]
    snippets = [ln for ln in lines if "…" in ln]
    assert len(full) <= FULL_INJECT_MAX, "at most FULL_INJECT_MAX full-injects"
    assert len(snippets) >= 1, "further imp≥9 candidates fall back to snippets"


def test_c19_c21_single_hardened_hebbian_open(monkeypatch, tmp_path: Path) -> None:
    import brain.memory.hebbian as hebmod

    calls = {"init": 0, "integrity": [], "closed": 0}
    real_init = hebmod.HebbianMatrix.__init__
    real_close = hebmod.HebbianMatrix.close

    def spy_init(self, db_path, *, integrity_check=True):
        calls["init"] += 1
        calls["integrity"].append(integrity_check)
        real_init(self, db_path, integrity_check=integrity_check)

    def spy_close(self):
        calls["closed"] += 1
        real_close(self)

    monkeypatch.setattr(hebmod.HebbianMatrix, "__init__", spy_init)
    monkeypatch.setattr(hebmod.HebbianMatrix, "close", spy_close)

    store = _store()
    for token in ("jordan", "marcus", "lisbon"):
        _mem(store, f"{token} shared a memory here", importance=3.0)

    _build_recall_block(store, "jordan marcus lisbon", persona_dir=tmp_path)

    # C21: exactly one open regardless of the 3 matching tokens.
    assert calls["init"] == 1
    # C19: opened with integrity_check=False and closed (no leaked handle).
    assert calls["integrity"] == [False]
    assert calls["closed"] >= 1


def test_c22_assembled_orders_by_blended_relevance(tmp_path: Path) -> None:
    store = _store()
    # Shared token "signal": strong match, low importance, old (low recency).
    _mem(store, "alpha signal signal signal signal signal", importance=1.0, age_days=200)
    # Weak match (one mention in filler), high importance, new.
    _mem(
        store,
        "beta signal among a great many other filler words padding here around it",
        importance=9.0,
        age_days=0,
    )
    block = _build_recall_block(store, "signal marcus", persona_dir=tmp_path)
    i_strong = block.find("alpha")
    i_weak = block.find("beta")
    # Blended relevance puts the strong-match/low-importance memory ABOVE the
    # weak-match/high-importance one. Under a (-importance,-ts) merge, "beta"
    # (importance 9) would rank first — this is the discriminating assertion.
    assert 0 <= i_strong < i_weak
