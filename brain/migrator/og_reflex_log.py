"""brain.migrator.og_reflex_log — schema migration for OG nell_reflex_log.json.

Reads og_data_dir/nell_reflex_log.json (no version wrapper, fires array
with extra OG-only fields), strips the OG-only fields (output_preview,
output_type, days_since_human, description), wraps with
{version: 1, fires: [...]} and writes to persona_dir/reflex_log.json.
Idempotent.

The new framework's reflex_log expects an output_memory_id reference per
fire; OG didn't have that concept (it had inline output_preview text).
Migrated fires get output_memory_id=None. The full OG fire detail is
preserved verbatim in legacy/nell_reflex_log.json by Layer 2A.

Fires with missing arc or fired_at are dropped (cannot meaningfully
migrate without those).

See docs/superpowers/specs/2026-05-05-migrator-soul-candidates-and-reflex-log-design.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.health.attempt_heal import save_with_backup_text

logger = logging.getLogger(__name__)


def migrate_reflex_log(
    *,
    og_data_dir: Path,
    persona_dir: Path,
) -> int:
    """Read og_data_dir/nell_reflex_log.json, transform, write to persona_dir/reflex_log.json.

    Returns:
        migrated_fires — count of fires migrated.

    Returns 0 silently if the OG file is missing or has no fires.
    """
    src = og_data_dir / "nell_reflex_log.json"
    if not src.exists():
        return 0

    try:
        og = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("nell_reflex_log.json malformed: %s", exc)
        return 0

    og_fires = og.get("fires", [])
    if not isinstance(og_fires, list):
        logger.warning("nell_reflex_log.json 'fires' is not a list; skipping")
        return 0

    migrated_fires: list[dict[str, Any]] = []
    for og_fire in og_fires:
        if not isinstance(og_fire, dict):
            continue
        if "arc" not in og_fire or "fired_at" not in og_fire:
            continue
        new_fire = {
            "arc": og_fire["arc"],
            "fired_at": og_fire["fired_at"],
            "trigger_state": og_fire.get("trigger_state", {}),
            "output_memory_id": None,
        }
        migrated_fires.append(new_fire)

    if migrated_fires:
        dest = persona_dir / "reflex_log.json"
        payload = {"version": 1, "fires": migrated_fires}
        save_with_backup_text(dest, json.dumps(payload, ensure_ascii=False, indent=2))

    return len(migrated_fires)
