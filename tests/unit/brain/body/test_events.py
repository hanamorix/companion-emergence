"""Unit tests for brain/body/events.py — record_climax_event helper.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.body.events import record_climax_event
from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "p"
    p.mkdir()
    return p


def _seed_originating(store: MemoryStore, *, climax: float = 8.0) -> Memory:
    """Create + commit a memory with high climax intensity. Return the
    persisted Memory (with id populated)."""
    mem = Memory.create_new(
        memory_type="conversation",
        content="a long erotic scene cresting at the end",
        emotions={"climax": climax, "arousal": 8.0, "desire": 8.0},
        domain="general",
    )
    store.create(mem)
    return mem


def test_writes_journal_entry_with_correct_metadata(store, persona_dir):
    origin = _seed_originating(store)
    new_id = record_climax_event(
        originating_memory=origin, store=store, persona_dir=persona_dir,
    )
    assert new_id is not None

    entries = store.list_by_type("journal_entry", active_only=True)
    assert len(entries) == 1
    j = entries[0]
    assert j.metadata["private"] is True
    assert j.metadata["source"] == "climax_event"
    assert j.metadata["auto_generated"] is True
    assert j.metadata["originating_memory_id"] == origin.id
    assert j.metadata["reflex_arc_name"] is None


def test_journal_carries_originating_emotions(store, persona_dir):
    origin = _seed_originating(store)
    record_climax_event(
        originating_memory=origin, store=store, persona_dir=persona_dir,
    )
    j = store.list_by_type("journal_entry", active_only=True)[0]
    assert j.emotions["climax"] == 8.0
    assert j.emotions["arousal"] == 8.0
    assert j.emotions["desire"] == 8.0


def test_journal_content_includes_originating_snippet(store, persona_dir):
    origin = _seed_originating(store)
    record_climax_event(
        originating_memory=origin, store=store, persona_dir=persona_dir,
    )
    j = store.list_by_type("journal_entry", active_only=True)[0]
    assert "the body crested" in j.content
    assert "context:" in j.content
    # snippet of originating memory shows up
    assert "long erotic scene" in j.content


def test_emits_behavioral_log_entry(store, persona_dir):
    origin = _seed_originating(store)
    record_climax_event(
        originating_memory=origin, store=store, persona_dir=persona_dir,
    )
    log_path = persona_dir / "behavioral_log.jsonl"
    assert log_path.exists()
    entries = list(read_jsonl_skipping_corrupt(log_path))
    assert len(entries) == 1
    e = entries[0]
    assert e["kind"] == "climax_event"
    assert e["source"] == "climax_event"
    assert e["reflex_arc_name"] is None
    assert e["emotional_state"]["climax"] == 8.0


def test_returns_none_on_store_failure(persona_dir, monkeypatch):
    """Fail-soft: when journal_entry write itself raises, returns None and
    logs warn — never propagates."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        s = MemoryStore(Path(td) / "memories.db")
        try:
            origin = _seed_originating(s)

            def boom(*a, **k):
                raise RuntimeError("simulated db failure")
            monkeypatch.setattr(s, "create", boom)

            # Should NOT raise
            new_id = record_climax_event(
                originating_memory=origin, store=s, persona_dir=persona_dir,
            )
            assert new_id is None
        finally:
            s.close()


def test_does_not_recurse_when_journal_itself_has_climax(store, persona_dir):
    """The journal_entry we create has climax in its emotions, but we hook
    in `add_memory` (not in store.create). So creating the journal directly
    via store.create from inside record_climax_event does NOT re-trigger
    record_climax_event. This test pins that behavior: exactly ONE
    journal_entry per call."""
    origin = _seed_originating(store)
    record_climax_event(
        originating_memory=origin, store=store, persona_dir=persona_dir,
    )
    entries = store.list_by_type("journal_entry", active_only=True)
    assert len(entries) == 1
