"""tests/unit/brain/chat/test_stage3_defect3_defect4.py

Stage-3 Defect-3 + Defect-4 fix — regression coverage.

Defect 3 (a cold red-team finding): a semantic-conclusive turn used to
return from `_build_recall_block` immediately via
`_render_semantic_recall_block`, silently skipping the fading/lost
partition, the "not recognised" signal, and the grief-touch breadcrumb
(`handle_recall_touch` on graveyard hits) — machinery the old lexical path
ALWAYS ran. Roy's Option-1 ruling: keep that machinery, always run it;
semantic conclusiveness only decides which ACTIVE memories are
selected/surfaced. These tests seed a REAL fading memory (via
`store.fade`) and a REAL graveyard (lost) entry with high emotion
(guarantees grief-touch clears its intensity threshold), patch
`run_semantic_recall` to return a canned CONCLUSIVE result for an
unrelated active memory, and confirm `_build_recall_block` still surfaces
fading/lost/not-recognised and fires `handle_recall_touch` exactly once. A
companion test proves the SAME graveyard hit still fires grief-touch
exactly once on an INCONCLUSIVE (`semantic_result=None`) turn — no
double-fire, no skip, across the fork.

Defect 4 (bundled): the semantic render's own snippet-tier fractional
recall_count bump + `enqueue_reappraisals` call ran UNCONDITIONALLY, unlike
the lexical path's `SNIPPET_MODE_ENABLED` gate (dormant today since that
flag is hardcoded True, but inconsistent — when snippet mode is off,
`_recall_snippet` renders full untruncated bodies, so a "snippet" isn't
actually a snippet and must not get the fractional bump). Proves that with
`SNIPPET_MODE_ENABLED` patched False, the semantic snippet tier gets
neither the bump nor the enqueue, while the (unconditional, spec-mandated)
full-tier bump still fires; a contrast test proves the gate only changes
behaviour when the flag is off.

Confirmed pre-fix failure (at fac9b2b3, before this change lands): the
Defect-3 tests below assert "not recognised"/fading/lost/grief-touch on a
semantic-conclusive turn — pre-fix, `_build_recall_block` returned directly
from `_render_semantic_recall_block` and none of that machinery ever ran,
so these assertions failed. The Defect-4 test asserts no bump/enqueue with
snippet mode off — pre-fix, `_render_semantic_recall_block` ran its
snippet-tier bump + enqueue unconditionally, so that assertion failed too.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from brain.chat.prompt import _build_recall_block
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import graveyard as gv
from brain.forgetting.salience import SalienceInputs
from brain.memory.pending import PendingQueue
from brain.memory.semantic_recall import SemanticRecallResult
from brain.memory.store import Memory, MemoryStore


def _rc(store: MemoryStore, mid: str) -> float:
    return store._conn.execute(  # noqa: SLF001
        "SELECT recall_count FROM memories WHERE id = ?", (mid,)
    ).fetchone()[0]


def _grief_event_rows(store: MemoryStore) -> list[dict]:
    rows = store._conn.execute(  # noqa: SLF001
        "SELECT metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    return [json.loads(r["metadata_json"]) for r in rows]


def _seed_fixture(tmp_path: Path) -> tuple[MemoryStore, Memory, Memory]:
    """Real fading memory + real graveyard (lost) entry + an unrelated
    active memory (the semantic standout). Shared by both Defect-3 tests
    (and the #231 Fix 1 fading-bump-parity tests below) so the only thing
    that differs between callers is whether semantic recall is conclusive.
    Returns (store, active_mem, fading_mem).
    """
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    store = MemoryStore(":memory:")

    # The semantic-conclusive turn's ACTIVE standout — deliberately
    # unrelated (no lexical overlap) to the fading/lost content below, so
    # the lexical search_with_loss call (which still always runs) does NOT
    # pick it up as an active hit of its own; the point is that this
    # candidate makes the turn "conclusive" for reasons having nothing to
    # do with the graveyard/fading content.
    active_mem = Memory.create_new(
        content="the sunlit dock where the boats moor at low tide",
        memory_type="event",
        domain="d",
    )
    store.create(active_mem)

    # A fading memory whose post-fade (summary) content carries "workshop"
    # — a lexical hit for the query tokens below.
    fading_mem = Memory.create_new(
        content="the full original workshop story, since forgotten in detail",
        memory_type="episodic",
        domain="d",
    )
    store.create(fading_mem)
    # "workshop" placed at the very start of the summary so it survives the
    # snippet-mode proportional truncation (~20% of body length) intact —
    # a truncation point mid-word would make the "workshop" substring
    # assertion below a false negative, not a real regression.
    store.fade(fading_mem.id, summary="workshop days, long since faded from that summer")

    # A graveyard (lost) entry with high emotion so handle_recall_touch's
    # intensity threshold clears and a grief breadcrumb is guaranteed —
    # same recipe as tests/grief/test_integration_chat_prompt.py's
    # test_chat_prompt_recall_writes_grief_breadcrumb_on_lost_hit.
    lost_mem = Memory.create_new(
        content="the rooftop morning before the cold rain hit",
        memory_type="episodic",
        domain="memory",
        emotions={"joy": 8.5},
    )
    object.__setattr__(lost_mem, "id", "mem-rooftop")
    gv.append(
        tmp_path,
        memory=lost_mem,
        salience_at_drop=0.6,
        inputs=SalienceInputs(emotion=0.85, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.1),
        lived_age_hours=24.0,
        reason="test-seed",
    )

    return store, active_mem, fading_mem


def _canned_conclusive_result(active_mem: Memory) -> SemanticRecallResult:
    return SemanticRecallResult(full=[active_mem], snippet=[], scores={active_mem.id: 0.91})


# ---------------------------------------------------------------------------
# Defect 3 — semantic-conclusive turn must NOT suppress fading/lost/
# not-recognised/grief-touch.
# ---------------------------------------------------------------------------


def test_semantic_conclusive_turn_still_surfaces_fading_lost_and_grief_touch(tmp_path: Path) -> None:
    store, active_mem, _fading_mem = _seed_fixture(tmp_path)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    # The semantic standout still surfaces as the "active:" section — the
    # fix must not have broken the conclusive-semantic surfacing itself.
    assert active_mem.id in block

    # Defect 3: the fading memory still surfaces even though semantic was
    # conclusive for an unrelated active candidate.
    assert "softened (fading" in block
    assert "workshop" in block

    # Defect 3: the graveyard (lost) hit still surfaces.
    assert "lost (no longer in active memory)" in block
    assert "rooftop morning" in block

    # Defect 3: "not recognised" is still computed on a conclusive turn.
    assert "not recognised" in block
    assert "Marcus" in block

    # Defect 3: grief-touch fires EXACTLY ONCE for the graveyard hit — not
    # suppressed by the conclusive semantic branch, and not double-fired.
    grief_rows = _grief_event_rows(store)
    assert len(grief_rows) == 1
    assert grief_rows[0]["grief_referent_id"] == "mem-rooftop"
    assert grief_rows[0]["grief_subtype"] == "recall_touch"


def test_inconclusive_turn_still_fires_grief_touch_exactly_once(tmp_path: Path) -> None:
    """Companion to the above: the SAME graveyard hit, on an INCONCLUSIVE
    (semantic_result=None) turn, must still fire grief-touch exactly once —
    proving the restructure didn't introduce a double-fire (or a skip) on
    the branch that already ran this machinery pre-fix."""
    store, _active_mem, _fading_mem = _seed_fixture(tmp_path)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    assert "lost (no longer in active memory)" in block
    assert "rooftop morning" in block

    grief_rows = _grief_event_rows(store)
    assert len(grief_rows) == 1
    assert grief_rows[0]["grief_referent_id"] == "mem-rooftop"


# ---------------------------------------------------------------------------
# Defect 4 — semantic snippet-tier bump + enqueue_reappraisals gated under
# SNIPPET_MODE_ENABLED, matching the lexical path's CHANGE-1/CHANGE-3 gate.
# ---------------------------------------------------------------------------


def _semantic_full_plus_snippet(store: MemoryStore) -> tuple[SemanticRecallResult, Memory, Memory]:
    full_mem = Memory.create_new(
        content="the lighthouse beam sweeping the bay", memory_type="event", domain="d"
    )
    snippet_mem = Memory.create_new(
        content="a quieter memory of the same lighthouse evening", memory_type="event", domain="d"
    )
    store.create(full_mem)
    store.create(snippet_mem)
    result = SemanticRecallResult(
        full=[full_mem],
        snippet=[snippet_mem],
        scores={full_mem.id: 0.9, snippet_mem.id: 0.7},
    )
    return result, full_mem, snippet_mem


def test_semantic_full_render_bumps_and_enqueues_when_snippet_mode_disabled(tmp_path: Path) -> None:
    """#231 flag-off correctness (FLIPPED from the old Defect-4 gated-off test).

    Pre-consolidation this test PINNED the flag-off ZERO-bump: the semantic
    snippet tier's bump + enqueue were gated behind `SNIPPET_MODE_ENABLED`, so
    with the flag off a "snippet"-tier row was rendered in FULL (because
    `_recall_snippet` returns the untruncated body) yet got NO bump and NO
    enqueue — the exact defect the `open_memory` consolidation closes. Post-fix,
    a row rendered in full is OPENED through the one door regardless of the flag:
    full +1.0 recall bump AND a reappraisal enqueue, on both tiers."""
    store = MemoryStore(":memory:")
    result, full_mem, snippet_mem = _semantic_full_plus_snippet(store)
    before_full = _rc(store, full_mem.id)
    before_snip = _rc(store, snippet_mem.id)
    PendingQueue(tmp_path).drain()

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=result),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", False),
    ):
        block = _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert full_mem.id in block
    assert snippet_mem.id in block

    # Both rows render FULL when snippet mode is off, so BOTH get the full +1.0
    # open bump — the snippet tier is no longer silently skipped.
    assert _rc(store, full_mem.id) - before_full == pytest.approx(1.0)
    assert _rc(store, snippet_mem.id) - before_snip == pytest.approx(1.0)

    # Both are now enqueued for reappraisal (pre-fix: enqueue entirely skipped).
    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(full_mem.id) == 1
    assert ids.count(snippet_mem.id) == 1


def test_semantic_snippet_bump_and_enqueue_apply_when_snippet_mode_enabled(tmp_path: Path) -> None:
    """Contrast case (prod path, flag ON): the snippet tier still gets the
    rank-weighted fractional bump and the full tier still gets +1.0, and every
    surfaced id is enqueued exactly once. This is a byte-identical parity guard
    for the consolidation — outcomes here are unchanged from pre-fix; only the
    enqueue call GROUPING changed (full ids now enqueue per-id through the door,
    snippet ids in one batch), so this asserts the OUTCOME, not the call count."""
    store = MemoryStore(":memory:")
    result, full_mem, snippet_mem = _semantic_full_plus_snippet(store)
    before_snip = _rc(store, snippet_mem.id)
    PendingQueue(tmp_path).drain()

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=result),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", True),
    ):
        _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    # Snippet tier: rank-weighted fractional bump (a lone snippet -> 0.8).
    assert _rc(store, snippet_mem.id) - before_snip == pytest.approx(0.8)
    # Full tier: +1.0.
    assert _rc(store, full_mem.id) == pytest.approx(1.0)
    # Both enqueued exactly once (full via the door, snippet via the batch).
    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(full_mem.id) == 1
    assert ids.count(snippet_mem.id) == 1


# ---------------------------------------------------------------------------
# #231 Fix 1 — fading recall-bump is PATH-INDEPENDENT (owner ruling: "if it
# surfaces it gets the full +1 ... doesn't matter how it got surfaced").
# fading_top renders identically (always snippet-level, `full=False`) on
# both the semantic-conclusive and inconclusive branches, so its
# recall_count bump must fire on both branches too — previously it was
# gated under `semantic_result is None`, so a conclusive turn's fading rows
# were rendered but never bumped (fading decaying faster on semantic turns).
# ---------------------------------------------------------------------------


def test_fading_bump_fires_on_conclusive_semantic_turn(tmp_path: Path) -> None:
    """Pre-fix (path-parity bug): the fading memory renders under "softened
    (fading...)" on a conclusive-semantic turn but its recall_count is left
    untouched, because the old CHANGE-1 bump block was gated entirely on
    `semantic_result is None`. Post-fix: the bump fires here too."""
    store, active_mem, fading_mem = _seed_fixture(tmp_path)
    before = _rc(store, fading_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    assert "softened (fading" in block
    # A lone surfaced fading memory gets the top-rank fractional tick (0.8),
    # matching the same rank-weighted scheme the lexical path already uses.
    assert _rc(store, fading_mem.id) - before == pytest.approx(0.8)


def test_fading_bump_matches_across_conclusive_and_inconclusive_turns(tmp_path: Path) -> None:
    """The core path-parity assertion: the SAME fading memory, surfaced on a
    CONCLUSIVE-semantic turn, gets the identical bump amount as on an
    INCONCLUSIVE turn — proving the bump no longer depends on which
    retrieval path produced the surfacing."""
    store_a, active_mem, fading_a = _seed_fixture(tmp_path)
    before_a = _rc(store_a, fading_a.id)
    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        _build_recall_block(store_a, "workshop rooftop Marcus", persona_dir=tmp_path)
    bump_conclusive = _rc(store_a, fading_a.id) - before_a

    store_b, _active_mem_b, fading_b = _seed_fixture(tmp_path)
    before_b = _rc(store_b, fading_b.id)
    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        _build_recall_block(store_b, "workshop rooftop Marcus", persona_dir=tmp_path)
    bump_inconclusive = _rc(store_b, fading_b.id) - before_b

    assert bump_conclusive == pytest.approx(bump_inconclusive)
    assert bump_conclusive == pytest.approx(0.8)


def test_active_top_bump_stays_gated_to_inconclusive_branch(tmp_path: Path) -> None:
    """Guard against over-correcting Fix 1: `active_top` (the lexical
    active-selection candidates, computed via the always-run
    `search_with_loss` call but NOT rendered under "active:" on a
    conclusive turn — only the semantic result is) must still NOT be
    bumped on a conclusive turn. Only the semantic-active section's own
    bump (`_render_semantic_active_lines`) and the now-path-independent
    fading bump apply there.
    """
    store, active_mem, fading_mem = _seed_fixture(tmp_path)

    # A THIRD memory: stays ACTIVE (never faded), lexically matches
    # "workshop" so search_with_loss's always-run partition puts it in
    # active_hits/active_top — but it is unrelated to the semantic
    # standout (`active_mem`), so it is never chosen/rendered on a
    # conclusive turn.
    lexical_active_mem = Memory.create_new(
        content="the workshop schedule pinned by the door", memory_type="event", domain="d"
    )
    store.create(lexical_active_mem)
    before_lexical = _rc(store, lexical_active_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    # The semantic-conclusive standout gets its OWN bump (full-tier, +1,
    # from _render_semantic_active_lines) — unrelated to the lexical
    # active_top gating this test is checking.
    assert _rc(store, active_mem.id) == pytest.approx(1.0)
    # fading_mem gets its now-path-independent bump.
    assert _rc(store, fading_mem.id) > 0

    # The lexical active_top candidate is computed (it lexically matches
    # "workshop") but never rendered under "active:" on this conclusive
    # turn, so it must NOT be bumped — bumping an unrendered row would
    # reinforce a memory the user never actually saw.
    assert lexical_active_mem.id not in block
    assert _rc(store, lexical_active_mem.id) == before_lexical


# ---------------------------------------------------------------------------
# #231 Fix 2 — reappraisal-ENQUEUE is now PATH-INDEPENDENT (owner ruling: "if
# a memory gets opened, it doesn't matter how it surfaced, it's been opened,
# so treat it as a memory that's been opened ... it goes back into the
# reappraisal queue"). Fix 1 (above) made the fading recall_count BUMP
# path-independent but left the reappraisal ENQUEUE gated to
# `semantic_result is None` — a surfaced fading memory on a CONCLUSIVE turn
# got bumped but was never queued for reappraisal. These tests prove the
# enqueue now fires on both branches, matching the bump's parity, and that a
# conclusive turn's fading id is enqueued exactly once (no double-enqueue
# against `_render_semantic_active_lines`'s own enqueue for the semantic
# active ids).
# ---------------------------------------------------------------------------


def _reappraisal_ids(rows: list[dict]) -> list[str]:
    return [r["memory_id"] for r in rows if r.get("_route") == "reappraise_importance"]


def test_reappraisal_enqueue_fires_on_conclusive_semantic_turn(tmp_path: Path) -> None:
    """Pre-fix (enqueue path-parity bug): the fading memory is bumped (Fix 1)
    on a conclusive-semantic turn but never enqueued for reappraisal, because
    the enqueue call was gated entirely on `semantic_result is None`.
    Post-fix: the fading id is enqueued here too, exactly once — alongside
    (not instead of) the semantic active id's own enqueue from
    `_render_semantic_active_lines`."""
    store, active_mem, fading_mem = _seed_fixture(tmp_path)
    PendingQueue(tmp_path).drain()  # clear anything a fixture helper may have queued

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    assert "softened (fading" in block

    reappraisal_ids = _reappraisal_ids(PendingQueue(tmp_path).drain())

    # The fading memory is now enqueued for reappraisal on the conclusive
    # branch, matching the inconclusive branch's existing behaviour.
    assert reappraisal_ids.count(fading_mem.id) == 1

    # The semantic-active standout is ALSO enqueued (its own pre-existing
    # `_render_semantic_active_lines` enqueue) — both ids present, neither
    # duplicated: no double-enqueue between the two enqueue call sites.
    assert reappraisal_ids.count(active_mem.id) == 1
    assert len(reappraisal_ids) == len(set(reappraisal_ids))


def test_reappraisal_enqueue_matches_across_conclusive_and_inconclusive_turns(tmp_path: Path) -> None:
    """The core enqueue path-parity assertion: the SAME fading memory,
    surfaced on a CONCLUSIVE-semantic turn, is enqueued for reappraisal just
    as it is on an INCONCLUSIVE turn — proving the enqueue no longer depends
    on which retrieval path produced the surfacing."""
    store_a, active_mem, fading_a = _seed_fixture(tmp_path)
    PendingQueue(tmp_path).drain()
    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=_canned_conclusive_result(active_mem)),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        _build_recall_block(store_a, "workshop rooftop Marcus", persona_dir=tmp_path)
    ids_conclusive = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert fading_a.id in ids_conclusive

    store_b, _active_mem_b, fading_b = _seed_fixture(tmp_path)
    PendingQueue(tmp_path).drain()
    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
    ):
        _build_recall_block(store_b, "workshop rooftop Marcus", persona_dir=tmp_path)
    ids_inconclusive = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert fading_b.id in ids_inconclusive


# ---------------------------------------------------------------------------
# #231 Fix 3 — an OPENED / full-rendered memory gets the FULL recall bump on
# EVERY path (owner ruling: "if it gets opened, it gets the full bump,
# doesn't matter how it got opened"). On the lexical (inconclusive) path the
# importance-threshold `full_ids` memories are rendered untruncated, i.e.
# OPENED, so they get store.bump_recall(id, 1.0) at full strength, NOT the
# fractional snippet rate, and are no longer EXCLUDED from every bump. That
# full-open bump is ungated on SNIPPET_MODE_ENABLED, mirroring the semantic
# full-inject tier. Pre-fix these lexical full_ids memories got NO bump at
# all (the fractional loop explicitly skipped them, "already maximally
# salient"), so every "== 1.0" assertion below FAILS against the pre-fix code.
# ---------------------------------------------------------------------------


def _seed_lexical_full_and_snippet(store: MemoryStore) -> tuple[Memory, Memory]:
    """A high-importance memory (importance >= FULL_INJECT_IMPORTANCE, so it
    is rendered in FULL and lands in `full_ids`) plus a low-importance one
    (rendered as a snippet), both lexically matching "lighthouse" so the
    always-run lexical partition puts both into active_hits/active_top on an
    inconclusive turn.
    """
    full_mem = Memory.create_new(
        content="the lighthouse keeper's oath, never once broken",
        memory_type="event",
        domain="d",
        importance=9.5,
    )
    snippet_mem = Memory.create_new(
        content="a passing glance at the lighthouse from the far shore",
        memory_type="event",
        domain="d",
        importance=2.0,
    )
    store.create(full_mem)
    store.create(snippet_mem)
    return full_mem, snippet_mem


def test_lexical_full_open_gets_full_bump_not_fractional(tmp_path: Path) -> None:
    """On an INCONCLUSIVE (lexical) turn, a full-rendered `full_ids` memory
    gets the FULL +1.0 recall bump, not the fractional snippet rate and not
    zero. Pre-fix it was EXCLUDED from every bump, so this "+1.0" delta fails
    against the old code; the low-importance snippet keeps its fractional
    bump, proving full-open is treated differently from a snippet."""
    store = MemoryStore(":memory:")
    full_mem, snippet_mem = _seed_lexical_full_and_snippet(store)
    before_full = _rc(store, full_mem.id)
    before_snip = _rc(store, snippet_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", True),
    ):
        block = _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert full_mem.id in block
    assert snippet_mem.id in block

    # The opened (full-rendered) memory gets the FULL bump.
    assert _rc(store, full_mem.id) - before_full == pytest.approx(1.0)

    # The non-full snippet memory keeps its fractional bump: strictly less
    # than the full tick and non-zero (a lone snippet-tier row -> 0.8).
    snip_bump = _rc(store, snippet_mem.id) - before_snip
    assert 0 < snip_bump < 1.0


def test_lexical_full_open_bump_fires_when_snippet_mode_off(tmp_path: Path) -> None:
    """The full-open bump is UNGATED on SNIPPET_MODE_ENABLED. With snippet
    mode OFF, `full_ids` is empty but every active row is rendered in full
    (opened), so both active memories get the FULL +1.0 bump. Pre-fix the
    lexical path did NO bumping at all when snippet mode was off (both the
    fading and the fractional loops are gated on SNIPPET_MODE_ENABLED), so
    these "+1.0" deltas fail against the old code."""
    store = MemoryStore(":memory:")
    full_mem, snippet_mem = _seed_lexical_full_and_snippet(store)
    before_full = _rc(store, full_mem.id)
    before_snip = _rc(store, snippet_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", False),
    ):
        block = _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert full_mem.id in block
    assert snippet_mem.id in block

    # Snippet mode off -> every active row is fully opened, so both get the
    # full +1.0 bump (full-open == full bump, ungated).
    assert _rc(store, full_mem.id) - before_full == pytest.approx(1.0)
    assert _rc(store, snippet_mem.id) - before_snip == pytest.approx(1.0)


def test_lexical_full_open_no_double_bump(tmp_path: Path) -> None:
    """A memory opened once is bumped exactly once. The full-open loop adds
    the id to `seen_bump` before the fractional loop runs, so the fractional
    loop skips it: its recall_count reflects a single +1.0, never +1.0 stacked
    with a fractional tick (which would read ~1.8)."""
    store = MemoryStore(":memory:")
    full_mem, _snippet_mem = _seed_lexical_full_and_snippet(store)
    before_full = _rc(store, full_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", True),
    ):
        _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    # Exactly one full tick, no fractional tick stacked on top.
    assert _rc(store, full_mem.id) - before_full == pytest.approx(1.0)


def test_lexical_full_open_batches_multiple_ids_into_one_enqueue_call(tmp_path: Path) -> None:
    """#231 follow-up (restored batching): when a single turn full-opens
    MULTIPLE memories (both importance >= FULL_INJECT_IMPORTANCE, so both land
    in `full_ids`), the passive pass issues exactly ONE `enqueue_reappraisals`
    file-lock/write for all of them — not one per id. Pre-follow-up, each
    full-open enqueued through `open_memory` individually, so this lock-count
    assertion would fail (2 acquisitions instead of 1) against that code.
    The enqueued id-SET stays exactly the two full-open ids either way."""
    from brain.utils.file_lock import file_lock as real_file_lock

    store = MemoryStore(":memory:")
    full_mem_a = Memory.create_new(
        content="the lighthouse keeper's first oath, never once broken",
        memory_type="event",
        domain="d",
        importance=9.5,
    )
    full_mem_b = Memory.create_new(
        content="the lighthouse keeper's second oath, kept just as well",
        memory_type="event",
        domain="d",
        importance=9.2,
    )
    store.create(full_mem_a)
    store.create(full_mem_b)
    PendingQueue(tmp_path).drain()

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", True),
        patch("brain.memory.pending.file_lock", wraps=real_file_lock) as mock_lock,
    ):
        block = _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert full_mem_a.id in block
    assert full_mem_b.id in block

    # Both full-opened -> ONE lock acquisition covers both (no snippet-tier
    # ids this turn, so this is the full-open flush's lock count in isolation).
    assert mock_lock.call_count == 1

    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(full_mem_a.id) == 1
    assert ids.count(full_mem_b.id) == 1
    assert len(ids) == 2  # exactly these two ids, nothing extra, nothing dropped


# ---------------------------------------------------------------------------
# #231 consolidation — the FADING tier's flag-off correctness. The fading tier
# always renders `_recall_snippet(mem, full=False)`, so when SNIPPET_MODE_ENABLED
# is off it renders the untruncated body (a full open). Pre-consolidation its
# bump AND enqueue were both gated behind SNIPPET_MODE_ENABLED, so a fading row
# rendered in full got NEITHER — the exact flag-off gap the one door closes.
# ---------------------------------------------------------------------------


def test_fading_full_render_bumps_and_enqueues_when_snippet_mode_off(tmp_path: Path) -> None:
    """With snippet mode OFF, a surfaced fading memory is rendered in full, so
    it is OPENED through the one door: full +1.0 bump (not the fractional 0.8
    rate) AND a reappraisal enqueue. Pre-fix it got zero bump and no enqueue."""
    store, _active_mem, fading_mem = _seed_fixture(tmp_path)
    before = _rc(store, fading_mem.id)
    PendingQueue(tmp_path).drain()

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=None),
        patch(
            "brain.chat.prompt._extract_recall_tokens",
            return_value=["workshop", "rooftop", "Marcus"],
        ),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", False),
    ):
        block = _build_recall_block(store, "workshop rooftop Marcus", persona_dir=tmp_path)

    assert "softened (fading" in block
    # Full render (flag off) -> the FULL +1.0 open bump, not the fractional rate.
    assert _rc(store, fading_mem.id) - before == pytest.approx(1.0)
    # And enqueued for reappraisal (was skipped entirely pre-fix).
    ids = _reappraisal_ids(PendingQueue(tmp_path).drain())
    assert ids.count(fading_mem.id) == 1
