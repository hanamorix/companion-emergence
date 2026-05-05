"""brain.migrator.og_journal_dna — migrate OG NellBrain three-stream data.

Per spec §6: two migrations.

  1. migrate_creative_dna — convert OG nell_creative_dna.json (two schema
     variants) to companion-emergence schema with biographical metadata.
  2. migrate_journal_memories — change memory_type='reflex_journal' →
     'journal_entry' on existing memories in the persona's MemoryStore.

Both idempotent: re-running on already-migrated data is a no-op.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.creative.dna import save_creative_dna
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def migrate_creative_dna(*, persona_dir: Path, og_root: Path) -> bool:
    """Convert OG nell_creative_dna.json to the new schema. Returns True if
    migration ran, False if the OG file was missing.

    Handles both OG schema variants:
      - older: tendencies = list[str] (treated as active)
      - newer: tendencies = {active, emerging, fading}
    """
    og_path = og_root / "data" / "nell_creative_dna.json"
    if not og_path.exists():
        return False
    try:
        og = json.loads(og_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("og creative_dna read failed: %s", exc)
        return False

    style = og.get("writing_style", {})
    file_mtime = datetime.fromtimestamp(og_path.stat().st_mtime, tz=UTC).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    reasoning = "imported from OG NellBrain on migration"

    new = {
        "version": 1,
        "core_voice": style.get("core_voice", ""),
        "strengths": list(style.get("strengths", [])),
        "tendencies": _migrate_tendencies(style.get("tendencies", []), file_mtime, reasoning),
        "influences": list(style.get("influences", [])),
        "avoid": list(style.get("avoid", [])),
    }
    save_creative_dna(persona_dir, new)
    return True


def _migrate_tendencies(og_tendencies: Any, mtime: str, reasoning: str) -> dict[str, list]:
    """Coerce both OG schema variants to {active, emerging, fading}.

    Defensive against malformed OG payloads: int / str / None / anything
    other than list-or-dict returns the empty shape rather than crashing
    the migration mid-run with AttributeError. The migrator's outer
    try/except (OSError, JSONDecodeError, ValueError) does NOT catch
    AttributeError, so a single corrupted file would have killed an
    otherwise-healthy migration.
    """
    if isinstance(og_tendencies, list):
        return {
            "active": [
                {
                    "name": name,
                    "added_at": mtime,
                    "reasoning": reasoning,
                    "evidence_memory_ids": [],
                }
                for name in og_tendencies
            ],
            "emerging": [],
            "fading": [],
        }
    if not isinstance(og_tendencies, dict):
        return {"active": [], "emerging": [], "fading": []}
    return {
        "active": [
            {
                "name": name,
                "added_at": mtime,
                "reasoning": reasoning,
                "evidence_memory_ids": [],
            }
            for name in og_tendencies.get("active", [])
        ],
        "emerging": [
            {
                "name": name,
                "added_at": mtime,
                "reasoning": reasoning,
                "evidence_memory_ids": [],
            }
            for name in og_tendencies.get("emerging", [])
        ],
        "fading": [
            {
                "name": name,
                "demoted_to_fading_at": mtime,
                "last_evidence_at": mtime,
                "reasoning": reasoning,
            }
            for name in og_tendencies.get("fading", [])
        ],
    }


def migrate_journal_memories(*, persona_dir: Path, store: MemoryStore) -> int:
    """Change memory_type='reflex_journal' to 'journal_entry' on existing
    memories. Set metadata.private=True, source='reflex_arc',
    auto_generated=True. Returns count of migrated memories.

    Idempotent: re-running finds nothing to migrate.
    """
    migrated = 0
    for memory in store.list_by_type("reflex_journal", active_only=True):
        new_metadata = dict(memory.metadata or {})
        new_metadata["private"] = True
        new_metadata["source"] = "reflex_arc"
        new_metadata["auto_generated"] = True
        # Preserve existing reflex_arc_name if present, else "unknown"
        if "reflex_arc_name" not in new_metadata:
            new_metadata["reflex_arc_name"] = "unknown"

        store.update(
            memory.id,
            memory_type="journal_entry",
            metadata=new_metadata,
        )
        migrated += 1
    return migrated
