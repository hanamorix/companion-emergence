"""Emotion rework B1 PR 6: the remaining writers keep feelings on the 0-10 scale."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.no_base_failure(reason="pins ingest's existing 0-10 clamp (#466) on the path into the pending queue")
def test_ingest_item_reaches_the_queue_on_the_zero_to_ten_scale(tmp_path: Path):
    from brain.ingest.commit import commit_item
    from brain.ingest.types import ExtractedItem
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        item = ExtractedItem(text="she said she was proud", label="feeling",
                             emotions={"pride": 14.0, "fear": float("nan"), "joy": 3.0}).normalize()
        commit_item(item, session_id="s1", store=store, hebbian=hebbian)
        queued = PendingQueue(tmp_path).read_recent("feeling", limit=5)
    finally:
        store.close()
        hebbian.close()
    assert [m._stored_emotions for m in queued] == [{"pride": 10.0, "joy": 3.0}]


@pytest.mark.no_base_failure(reason="pins the file-commit memory's existing empty feelings")
def test_file_commit_memory_stores_no_feelings(tmp_path: Path, monkeypatch):
    from datetime import UTC, datetime

    from brain.files import pending
    from brain.files.commit import commit_write
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    persona = tmp_path / "persona"  # a sibling of home: the guard refuses paths inside the persona
    target = tmp_path / "home" / "out" / "n.md"
    rid = pending.create(persona, op="create", resolved_path=str(target.resolve()), content="body", now=datetime.now(UTC))
    store = MemoryStore(persona / "memories.db")
    try:
        assert commit_write(persona, rid, store=store)["ok"]
        queued = PendingQueue(persona).read_recent("file_write", limit=5)
    finally:
        store.close()
    assert [m._stored_emotions for m in queued] == [{}]


def test_grief_breadcrumb_residue_is_held_to_ten(tmp_path: Path):
    """Dream rows store unclamped sums (switch-off dreams still do), so a dropped memory's strongest feeling can be above 20. The drop reads
    the memory through the engine's one read, which holds it to 10 first, so the residue the breadcrumb carries (half of the held feeling)
    is 5 and never above 10."""
    from brain.grief import handle_drop
    from brain.memory.store import Memory, MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    try:
        dropped = Memory.create_new(content="an old dream", memory_type="dream", domain="us", emotions={"desire": 24.0})
        handle_drop(memory=dropped, persona_dir=tmp_path, store=store)
        crumbs = store.list_by_type("grief_event", active_only=True)
    finally:
        store.close()
    assert len(crumbs) == 1
    assert crumbs[0]._stored_emotions["desire"] == 5.0
