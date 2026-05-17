"""brain.migrator.og_journal_dna — migrate OG creative_dna + reflex_journal memories."""

from __future__ import annotations

import json
from pathlib import Path

from brain.creative.dna import load_creative_dna
from brain.memory.store import Memory, MemoryStore
from brain.migrator.og_journal_dna import (
    migrate_creative_dna,
    migrate_journal_memories,
)


def test_migrate_creative_dna_from_og_dict_schema(tmp_path: Path):
    """OG newer schema: tendencies as {active, emerging, fading} dict."""
    og_root = tmp_path / "og"
    og_data = og_root / "data"
    og_data.mkdir(parents=True)
    og_dna = {
        "version": "1.0",
        "writing_style": {
            "core_voice": "literary, sensory-dense",
            "strengths": ["power dynamics"],
            "tendencies": {
                "active": ["ending on physical action", "italic NPC thoughts"],
                "emerging": ["sentence fragments"],
                "fading": ["ending on questions"],
            },
            "influences": ["clarice lispector"],
            "avoid": ["hypophora"],
        },
    }
    (og_data / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    result = migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    assert result is True

    new_dna = load_creative_dna(persona_dir)
    assert new_dna["core_voice"] == "literary, sensory-dense"
    assert new_dna["strengths"] == ["power dynamics"]

    active_names = [t["name"] for t in new_dna["tendencies"]["active"]]
    assert active_names == ["ending on physical action", "italic NPC thoughts"]

    # Per-tendency dicts have biographical metadata
    first = new_dna["tendencies"]["active"][0]
    assert "added_at" in first
    assert "reasoning" in first
    assert first["reasoning"] == "imported from OG NellBrain on migration"

    emerging_names = [t["name"] for t in new_dna["tendencies"]["emerging"]]
    assert emerging_names == ["sentence fragments"]

    fading_names = [t["name"] for t in new_dna["tendencies"]["fading"]]
    assert fading_names == ["ending on questions"]


def test_migrate_creative_dna_from_og_list_schema(tmp_path: Path):
    """OG older schema: tendencies as plain string list (treated as active)."""
    og_root = tmp_path / "og"
    og_data = og_root / "data"
    og_data.mkdir(parents=True)
    og_dna = {
        "version": "1.0",
        "writing_style": {
            "core_voice": "v",
            "strengths": [],
            "tendencies": ["habit one", "habit two"],
            "influences": [],
            "avoid": [],
        },
    }
    (og_data / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)

    new_dna = load_creative_dna(persona_dir)
    active_names = [t["name"] for t in new_dna["tendencies"]["active"]]
    assert active_names == ["habit one", "habit two"]


def test_migrate_creative_dna_idempotent(tmp_path: Path):
    """Re-migration produces deterministic same output."""
    og_root = tmp_path / "og"
    (og_root / "data").mkdir(parents=True)
    og_dna = {
        "version": "1.0",
        "writing_style": {
            "core_voice": "v",
            "strengths": [],
            "tendencies": [],
            "influences": [],
            "avoid": [],
        },
    }
    (og_root / "data" / "nell_creative_dna.json").write_text(json.dumps(og_dna))

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    first = (persona_dir / "creative_dna.json").read_text()
    migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    second = (persona_dir / "creative_dna.json").read_text()
    assert first == second


def test_migrate_creative_dna_no_og_file(tmp_path: Path):
    """No OG file → return False, no creative_dna.json written."""
    og_root = tmp_path / "og"
    og_root.mkdir()
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    result = migrate_creative_dna(persona_dir=persona_dir, og_root=og_root)
    assert result is False
    assert not (persona_dir / "creative_dna.json").exists()


def test_migrate_journal_memories_changes_memory_type(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        # Seed two old reflex_journal memories
        for i in range(2):
            mem = Memory.create_new(
                content=f"old journal {i}",
                memory_type="reflex_journal",
                domain="self",
                emotions={},
                metadata={"reflex_arc_name": f"arc_{i}"},
            )
            store.create(mem)

        migrated = migrate_journal_memories(persona_dir=persona_dir, store=store)
        assert migrated == 2

        # Verify old type is gone
        assert store.list_by_type("reflex_journal") == []
        # New type populated
        new_journal = store.list_by_type("journal_entry")
        assert len(new_journal) == 2
        for m in new_journal:
            assert m.metadata["private"] is True
            assert m.metadata["source"] == "reflex_arc"
            assert m.metadata["auto_generated"] is True
            assert m.metadata["reflex_arc_name"].startswith("arc_")
    finally:
        store.close()


def test_migrate_journal_memories_idempotent(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        mem = Memory.create_new(
            content="x",
            memory_type="reflex_journal",
            domain="self",
            emotions={},
        )
        store.create(mem)

        migrated_first = migrate_journal_memories(persona_dir=persona_dir, store=store)
        migrated_second = migrate_journal_memories(persona_dir=persona_dir, store=store)
        assert migrated_first == 1
        assert migrated_second == 0
    finally:
        store.close()
