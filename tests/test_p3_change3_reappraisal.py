"""Tests for P3 retention rework, Change 3 — importance re-rating on recall,
via the pending queue.

Covers C3.1-C3.3 from changes/p3-retention/1.5-criteria.md.

Naming: synthetic user = Bob, persona = Canary, model = Claude. No Phoebe.
"""

from __future__ import annotations

import inspect

import pytest

from brain.chat import prompt as prompt_mod
from brain.chat.prompt import _build_recall_block
from brain.engines.consolidation import run_consolidation
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore


def _mem(content: str, *, importance: float = 3.0) -> Memory:
    return Memory.create_new(
        content=content, memory_type="conversation", domain="us", importance=importance
    )


# --------------------------------------------------------------------------- C3.1
def test_c3_1_reappraise_updates_existing_row_no_new_row(tmp_path):
    """Enqueue an existing-memory re-appraise item, run the gate with an
    injected fake appraiser returning a known importance, assert the
    existing row's importance is updated to that value AND no new row was
    created (row count unchanged). Fail-test: an INSERT path (new row) or an
    unchanged importance would flag."""
    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem("Bob's home address is 12 Elm Street", importance=3.0)
    store.create(mem)
    before_count = store.count(active_only=False)

    PendingQueue(tmp_path).enqueue_reappraisal(mem.id, source="recall")

    known_value = 7.25

    def _fake_reappraiser(m: Memory) -> float:
        return known_value

    result = run_consolidation(
        store, persona_dir=tmp_path,
        reappraiser=_fake_reappraiser,
    )
    assert result.reappraised == 1

    got = store.get(mem.id, bump=False)
    assert got.importance == pytest.approx(known_value)
    assert store.count(active_only=False) == before_count  # no INSERT — same row count
    # A normal reappraise update does not clobber unrelated fields.
    assert got.content == "Bob's home address is 12 Elm Street"
    assert got.emotions == {}
    store.close()


def test_c3_1_default_noop_reappraiser_leaves_importance_unchanged(tmp_path):
    """With no provider and no injected reappraiser, the default is a no-op
    (STAGE-3 CORRECTION finding #2) — importance is unchanged, not ratcheted."""
    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem("a stable fact", importance=4.5)
    store.create(mem)
    PendingQueue(tmp_path).enqueue_reappraisal(mem.id, source="recall")

    result = run_consolidation(store, persona_dir=tmp_path)
    assert result.reappraised == 1
    got = store.get(mem.id, bump=False)
    assert got.importance == pytest.approx(4.5)
    store.close()


# --------------------------------------------------------------------------- C3.2
def test_c3_2_row_deleted_mid_window_no_resurrection_no_crash(tmp_path):
    """A re-appraise item whose target row is hard-deleted BETWEEN the
    handler's store.get and its store.update (injected deterministically via
    the fake reappraiser's own side effect, which runs in that exact
    window) does not resurrect the row and does not crash — the update is
    skipped. Fail-test: a version that INSERTs on missing target
    (resurrection) or raises an unhandled KeyError would flag."""
    store = MemoryStore(tmp_path / "memories.db")
    mem = _mem("a memory that will be deleted mid-appraisal", importance=2.0)
    store.create(mem)
    before_count = store.count(active_only=False)
    PendingQueue(tmp_path).enqueue_reappraisal(mem.id, source="recall")

    def _deleting_reappraiser(m: Memory) -> float:
        # Fires strictly between the handler's store.get (already returned
        # `m`) and its store.update — the HARDER window (finding #3).
        store.hard_delete(m.id)
        return 9.9

    result = run_consolidation(
        store, persona_dir=tmp_path,
        reappraiser=_deleting_reappraiser,
    )
    # No crash (the try/except KeyError around store.update absorbed it).
    assert result.reappraised == 0  # the update was skipped, not counted
    assert store.get(mem.id, bump=False) is None  # not resurrected
    assert store.count(active_only=False) == before_count - 1  # genuinely gone, no phantom row
    store.close()


def test_c3_2_row_already_missing_at_read_is_skipped(tmp_path):
    """A re-appraise item whose target was already gone BEFORE the read
    (the simpler window) is also skipped, not resurrected."""
    store = MemoryStore(tmp_path / "memories.db")
    PendingQueue(tmp_path).enqueue_reappraisal("nonexistent-id", source="recall")

    def _fake_reappraiser(m: Memory) -> float:
        raise AssertionError("reappraiser must never be called for a missing row")

    result = run_consolidation(
        store, persona_dir=tmp_path,
        reappraiser=_fake_reappraiser,
    )
    assert result.reappraised == 0
    assert store.get("nonexistent-id", bump=False) is None
    store.close()


# --------------------------------------------------------------------------- C3.3
def test_c3_3_recall_hook_source_never_references_a_provider():
    """Static check: the recall-block hook's source never references a
    provider/appraiser — it can only enqueue. Fail-test: an inline-appraise
    implementation would need a provider/reappraiser call at this site."""
    src = inspect.getsource(prompt_mod._build_recall_block)
    assert ".generate(" not in src  # the LLMProvider call surface
    assert "reappraiser" not in src  # no inline appraiser reference at all
    assert "enqueue_reappraisal" in src  # the hook DOES enqueue


def test_c3_3_recall_enqueues_for_full_inject_and_snippet_union_not_inline(tmp_path):
    """Exercising the recall-block hook enqueues a re-appraise request for
    BOTH a full-inject (full_ids) and a snippet (bump_targets) surfaced
    memory id (finding #5), and does NOT invoke any appraiser/provider
    inline: importance is bit-for-bit unchanged immediately after the call
    (appraisal only ever happens later, at the gate's own tick)."""
    store = MemoryStore(tmp_path / "memories.db")
    hi = _mem("lighthouse beacon primary marker signal", importance=9.5)  # -> full_ids
    lo = _mem("lighthouse beacon secondary marker signal", importance=2.0)  # -> bump_targets
    store.create(hi)
    store.create(lo)

    block = _build_recall_block(store, "lighthouse beacon marker", persona_dir=tmp_path)
    assert block.strip() != ""

    # Enqueue-only: no inline appraisal happened — importance untouched.
    assert store.get(hi.id, bump=False).importance == pytest.approx(9.5)
    assert store.get(lo.id, bump=False).importance == pytest.approx(2.0)

    entries = PendingQueue(tmp_path).drain()
    reappraise_ids = {
        e["memory_id"] for e in entries if e.get("_route") == "reappraise_importance"
    }
    assert hi.id in reappraise_ids
    assert lo.id in reappraise_ids
    store.close()
