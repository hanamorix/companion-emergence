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


def test_semantic_snippet_bump_and_enqueue_gated_off_when_snippet_mode_disabled(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    result, full_mem, snippet_mem = _semantic_full_plus_snippet(store)
    before_snip = _rc(store, snippet_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=result),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", False),
        patch("brain.memory.pending.PendingQueue.enqueue_reappraisals") as mock_enqueue,
    ):
        block = _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert full_mem.id in block
    assert snippet_mem.id in block

    # Full-tier bump is UNCONDITIONAL (spec-mandated, unaffected by Defect
    # 4) — still fires even with snippet mode off.
    assert _rc(store, full_mem.id) == pytest.approx(1.0)

    # Defect 4: the snippet-tier bump must NOT apply when snippet mode is off.
    assert _rc(store, snippet_mem.id) == before_snip

    # Defect 4: no reappraisal enqueue at all when snippet mode is off.
    mock_enqueue.assert_not_called()


def test_semantic_snippet_bump_and_enqueue_apply_when_snippet_mode_enabled(tmp_path: Path) -> None:
    """Contrast case: with snippet mode ON (today's default), the snippet
    tier DOES get bumped and enqueued — proves the Defect-4 gate only
    changes behaviour when the flag is off, not a blanket removal."""
    store = MemoryStore(":memory:")
    result, _full_mem, snippet_mem = _semantic_full_plus_snippet(store)
    before_snip = _rc(store, snippet_mem.id)

    with (
        patch("brain.chat.prompt.run_semantic_recall", return_value=result),
        patch("brain.chat.prompt._extract_recall_tokens", return_value=["lighthouse"]),
        patch("brain.chat.prompt.SNIPPET_MODE_ENABLED", True),
        patch("brain.memory.pending.PendingQueue.enqueue_reappraisals") as mock_enqueue,
    ):
        _build_recall_block(store, "lighthouse", persona_dir=tmp_path)

    assert _rc(store, snippet_mem.id) - before_snip == pytest.approx(0.8)
    mock_enqueue.assert_called_once()


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
