"""test_recall_query_tier1.py — Stage 1.5 gating criteria for the Tier-1
passive-recall query-construction fix (`changes/recall-query-tier1/`).

No live/integration/requires_claude_cli markers → runs in default CI. Fast
(":memory:" SQLite only).
"""

from __future__ import annotations

import re
from pathlib import Path

from brain.chat.prompt import _build_recall_block
from brain.forgetting import graveyard
from brain.forgetting.salience import SalienceInputs
from brain.memory.relevance import RELEVANCE_RANKING_ENABLED
from brain.memory.store import Memory, MemoryStore
from tests.memory.recall_eval import (
    COMMON_STORE_FRACTION,
    MSG_BURIED,
    MSG_EARLY,
    RARE_STORE_FRACTION,
    K,
    fts_call_count,
    metrics,
    not_recognised_tokens,
    seed_store,
    surfaced_ids,
)

# ---------------------------------------------------------------------------
# C1 — pool fix / buried≈early parity, on both the rare-target and
# common-target (incident-representative) stores.
# ---------------------------------------------------------------------------


def test_buried_reaches_early_parity_rare(tmp_path: Path) -> None:
    store, target_ids = seed_store(RARE_STORE_FRACTION)
    try:
        early = metrics(surfaced_ids(store, MSG_EARLY, tmp_path), target_ids, K)
        buried = metrics(surfaced_ids(store, MSG_BURIED, tmp_path), target_ids, K)
        assert buried["recall@k"] > 0, "buried trigger must surface at least one target"
        assert buried["recall@k"] >= 0.8 * early["recall@k"], (
            f"buried recall@k={buried['recall@k']} below 0.8*early={0.8 * early['recall@k']}"
        )
    finally:
        store.close()


def test_buried_reaches_early_parity_common(tmp_path: Path) -> None:
    """Incident-representative gate: target term seeded common (~25-30% of
    the store, LOW IDF) — the direction the diagnosis says actually failed."""
    store, target_ids = seed_store(COMMON_STORE_FRACTION)
    try:
        early = metrics(surfaced_ids(store, MSG_EARLY, tmp_path), target_ids, K)
        buried = metrics(surfaced_ids(store, MSG_BURIED, tmp_path), target_ids, K)
        assert buried["recall@k"] > 0, "buried trigger must surface at least one target"
        assert buried["recall@k"] >= 0.8 * early["recall@k"], (
            f"buried recall@k={buried['recall@k']} below 0.8*early={0.8 * early['recall@k']}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# C2 — query consolidation: exactly ONE search_fts_scored call per
# _build_recall_block invocation, on both Path A and Path B.
# ---------------------------------------------------------------------------


def test_single_fts_query_path_a(tmp_path: Path) -> None:
    assert RELEVANCE_RANKING_ENABLED is True, (
        "C2's call-count proxy assumes ranking is on; the search_text fallback "
        "path issues zero search_fts_scored calls."
    )
    store, _ = seed_store(RARE_STORE_FRACTION)
    try:
        count = fts_call_count(store, MSG_BURIED, persona_dir=None)
        assert count == 1
    finally:
        store.close()


def test_single_fts_query_path_b(tmp_path: Path) -> None:
    assert RELEVANCE_RANKING_ENABLED is True
    store, _ = seed_store(RARE_STORE_FRACTION)
    try:
        count = fts_call_count(store, MSG_BURIED, persona_dir=tmp_path)
        assert count == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# C4 — render-structure preservation: active + fading + lost/graveyard all
# populated, still parseable in their existing per-section formats, section
# order intact.
# ---------------------------------------------------------------------------


def test_render_structure_preserved(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    try:
        active_mem = Memory.create_new(
            content="harborlight beacon note still active",
            memory_type="conversation",
            domain="us",
            importance=3.0,
        )
        store.create(active_mem)

        fading_mem = Memory.create_new(
            content="harborlight beacon note before fading",
            memory_type="conversation",
            domain="us",
            importance=3.0,
        )
        store.create(fading_mem)
        store.fade(fading_mem.id, summary="harborlight beacon summary")

        lost_mem = Memory.create_new(
            content="harborlight beacon lost entry",
            memory_type="conversation",
            domain="us",
        )
        graveyard.append(
            tmp_path,
            memory=lost_mem,
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )

        block = _build_recall_block(store, "harborlight beacon", persona_dir=tmp_path)

        assert "  active:" in block
        assert "  softened (fading; original detail gone):" in block
        assert "  lost (no longer in active memory):" in block

        # Section order intact.
        i_active = block.index("  active:")
        i_softened = block.index("  softened (fading; original detail gone):")
        i_lost = block.index("  lost (no longer in active memory):")
        assert i_active < i_softened < i_lost

        # Active bullets: '    - <uuid>: "..."'.
        active_re = re.compile(r'^ {4}- [0-9a-f-]{36}: ".*"$')
        active_section = block[i_active:i_softened]
        active_lines = [ln for ln in active_section.splitlines() if ln.strip().startswith("- ")]
        assert active_lines, "expected at least one active bullet"
        assert all(active_re.match(ln) for ln in active_lines)

        # Fading bullets: '    - "..."  [state: fading]' (no id).
        fading_re = re.compile(r'^ {4}- ".*"  \[state: fading\]$')
        fading_section = block[i_softened:i_lost]
        fading_lines = [ln for ln in fading_section.splitlines() if ln.strip().startswith("- ")]
        assert fading_lines, "expected at least one fading bullet"
        assert all(fading_re.match(ln) for ln in fading_lines)

        # Lost bullets: '    - "..."  [forgotten — reason]' (no id).
        lost_re = re.compile(r'^ {4}- ".*"  \[forgotten')
        lost_section = block[i_lost:]
        # Stop before a possible "not recognised" section.
        if "not recognised" in lost_section:
            lost_section = lost_section[: lost_section.index("not recognised")]
        lost_lines = [ln for ln in lost_section.splitlines() if ln.strip().startswith("- ")]
        assert lost_lines, "expected at least one lost bullet"
        assert all(lost_re.match(ln) for ln in lost_lines)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# C9 — "not recognised" display reflects the store: a selected token with no
# matching memory is listed; one with a matching memory is not.
# ---------------------------------------------------------------------------


def test_not_recognised_reflects_store(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="a story about a lighthouse on the point",
                memory_type="conversation",
                domain="us",
            )
        )
        block = _build_recall_block(store, "lighthouse zephyrion", persona_dir=tmp_path)
        absent = not_recognised_tokens(block)
        assert "zephyrion" in absent
        assert "lighthouse" not in absent
    finally:
        store.close()


# ---------------------------------------------------------------------------
# #145 — when MORE than five query tokens are unrecognised, the display trims
# to the proper-noun-shaped (capitalised) ones. That branch compared
# ``t[0].isupper()`` on already-lowercased tokens and always emptied the list.
# ---------------------------------------------------------------------------


def test_not_recognised_over_five_keeps_capitalised_tokens(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    try:
        block = _build_recall_block(
            store,
            "we met Zephyrion and Quillabar near glimmerholt, vantorix, brindlewick, oskaline",
            persona_dir=tmp_path,
        )
        absent = not_recognised_tokens(block)
        assert "zephyrion" in absent
        assert "quillabar" in absent
        assert "glimmerholt" not in absent
    finally:
        store.close()
