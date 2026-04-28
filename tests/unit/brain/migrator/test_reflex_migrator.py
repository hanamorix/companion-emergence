"""Tests for reflex migrator's Phase 2 created_by/created_at stamping."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from brain.migrator.og_reflex import migrate_reflex_arcs


def test_migrator_stamps_og_migration_on_all_arcs(tmp_path: Path):
    """After migration, every arc in reflex_arcs.json carries created_by='og_migration'."""
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()
    migrate_reflex_arcs(persona_dir=persona_dir)

    arcs_file = persona_dir / "reflex_arcs.json"
    assert arcs_file.exists()
    data = json.loads(arcs_file.read_text())
    arcs = data["arcs"]
    assert len(arcs) >= 4
    for arc in arcs:
        assert arc["created_by"] == "og_migration", f"arc {arc['name']!r} missing stamp"
        parsed = datetime.fromisoformat(arc["created_at"])
        assert parsed.tzinfo is not None


def test_migrator_idempotent_no_double_stamp(tmp_path: Path):
    """Re-migrating doesn't change created_at on existing arcs."""
    persona_dir = tmp_path / "test-persona"
    persona_dir.mkdir()

    migrate_reflex_arcs(persona_dir=persona_dir)
    first = json.loads((persona_dir / "reflex_arcs.json").read_text())
    first_created_at = {arc["name"]: arc["created_at"] for arc in first["arcs"]}

    migrate_reflex_arcs(persona_dir=persona_dir)
    second = json.loads((persona_dir / "reflex_arcs.json").read_text())

    for arc in second["arcs"]:
        assert arc["created_by"] == "og_migration"
        assert arc["created_at"] == first_created_at[arc["name"]], (
            f"re-migration changed created_at for {arc['name']!r} — should be idempotent"
        )


def test_migrator_from_og_source(tmp_path: Path):
    """When given an og_reflex_engine_path, arcs are extracted + stamped."""
    import textwrap

    og_path = tmp_path / "reflex_engine.py"
    og_path.write_text(textwrap.dedent("""\
        REFLEX_ARCS = {
            "creative_pitch": {
                "trigger": {"creative_hunger": 9},
                "days_since_min": 0,
                "action": "generate_story_pitch",
                "output": "gifts",
                "cooldown_hours": 48,
                "description": "desc",
                "prompt_template": "You are Nell."
            }
        }
    """), encoding="utf-8")

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    arcs = migrate_reflex_arcs(persona_dir=persona_dir, og_reflex_engine_path=og_path)

    assert len(arcs) == 1
    assert arcs[0]["created_by"] == "og_migration"
    parsed = datetime.fromisoformat(arcs[0]["created_at"])
    assert parsed.tzinfo is not None

    # Also verify the written file
    data = json.loads((persona_dir / "reflex_arcs.json").read_text())
    assert data["arcs"][0]["name"] == "creative_pitch"


def test_migrator_corrupt_existing_file_restamps(tmp_path: Path):
    """If existing reflex_arcs.json is corrupt, a fresh stamp is written."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "reflex_arcs.json").write_text("{corrupt{{", encoding="utf-8")

    arcs = migrate_reflex_arcs(persona_dir=persona_dir)
    assert len(arcs) >= 4
    for arc in arcs:
        assert arc["created_by"] == "og_migration"
        datetime.fromisoformat(arc["created_at"])  # must parse
