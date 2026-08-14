"""Assembled recall-block behaviour for P2 — C6/C8/C9/C11/C12/C13/C19/C20/C21/C22.

All oracles exercise the REAL `_build_recall_block` (and the static/volatile
builders), not an isolated `rank_memories`, so the assembled-path guarantees the
round-4 review flagged are actually verified.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

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

# Byte-for-byte copy of the source literal — a future edit to the epistemic
# instruction (which references the exact "not recognised" wording) fails here.
_EXPECTED_EPISTEMIC = (
    "If asked about something you might have stored — a name, a fact, a shared "
    "moment — and it isn't in the context you can see, call search_memories "
    'before answering. Never say "I don\'t remember" without searching first. '
    'When names or entities appear under "not recognised (searched; no memory '
    'found)", acknowledge the gap honestly. Distinguish "I never knew this" '
    'from "I don\'t remember". Do not invent familiarity.'
)


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
    return [ln for ln in block.splitlines() if ln.startswith('    - "')]


def test_c6_surfacing_does_not_bump_recall_count(tmp_path: Path) -> None:
    store = _store()
    heb = HebbianMatrix(":memory:")
    m = _mem(store, "jordan and the harbor at dawn", importance=5.0)
    before_rc, before_la = _rc(store, m.id), _la(store, m.id)

    _build_recall_block(store, "tell me about jordan", persona_dir=tmp_path)
    search_memories("jordan", store=store, hebbian=heb, persona_dir=tmp_path)

    assert _rc(store, m.id) == before_rc, "surfacing must not bump recall_count"
    assert _la(store, m.id) == before_la, "surfacing must not touch last_accessed_at"


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
    assert "jordan was here that summer" in block


def test_c13_unfamiliar_bucket_and_epistemic_instruction(tmp_path: Path) -> None:
    store = _store()
    block = _build_recall_block(store, "who is marcus and lisbon", persona_dir=tmp_path)
    assert "not recognised (searched; no memory found)" in block
    assert "marcus" in block
    # Epistemic instruction (which references that exact wording) is byte-equal.
    assert _EPISTEMIC_INSTRUCTION == _EXPECTED_EPISTEMIC


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
