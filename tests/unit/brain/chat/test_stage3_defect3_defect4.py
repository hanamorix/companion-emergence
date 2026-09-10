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


def _seed_fixture(tmp_path: Path) -> tuple[MemoryStore, Memory]:
    """Real fading memory + real graveyard (lost) entry + an unrelated
    active memory (the semantic standout). Shared by both Defect-3 tests so
    the only thing that differs between them is whether semantic recall is
    conclusive.
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

    return store, active_mem


def _canned_conclusive_result(active_mem: Memory) -> SemanticRecallResult:
    return SemanticRecallResult(full=[active_mem], snippet=[], scores={active_mem.id: 0.91})


# ---------------------------------------------------------------------------
# Defect 3 — semantic-conclusive turn must NOT suppress fading/lost/
# not-recognised/grief-touch.
# ---------------------------------------------------------------------------


def test_semantic_conclusive_turn_still_surfaces_fading_lost_and_grief_touch(tmp_path: Path) -> None:
    store, active_mem = _seed_fixture(tmp_path)

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
    store, _active_mem = _seed_fixture(tmp_path)

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
