"""brain.migrator.og_legacy — verbatim preservation of OG NellBrain files.

Copies a positive list of biographical OG files to persona/<name>/legacy/<basename>
during nell migrate. Pure verbatim byte copy — no JSON parsing, no schema
validation, no transformation. The whole point is "do not lose content";
broken files preserve their broken bytes for future migrator-authors.

Files in LEGACY_FILES that are missing from the OG dir are silently skipped
and counted in the migration report as 'missing'. This is the typical case
for forkers who didn't use every OG NellBrain feature.

Idempotent in the simple sense: re-running overwrites with the same content.
The migrator's --force flag handles the broader install-rerun case at a
higher level (backup + atomic rename of the whole persona dir).

See docs/superpowers/specs/2026-05-04-migrator-legacy-preservation-design.md.
"""
from __future__ import annotations

from pathlib import Path


LEGACY_FILES: tuple[str, ...] = (
    # Tier 1 — biographical, no migrator, no surface (11)
    "nell_journal.json",
    "nell_growth.json",
    "nell_gifts.json",
    "nell_narratives.json",
    "nell_surprises.json",
    "nell_outbox.json",
    "nell_personality.json",
    "emotion_blends.json",
    "nell_emotion_vocabulary.json",
    "nell_body_state.json",
    "nell_heartbeat_log.json",
    # Tier 1 supplements (3)
    "nell_growth_log.jsonl",
    "behavioral_log.jsonl",
    "soul_audit.jsonl",
    # Tier 3 — regenerable, but historical snapshot is biographical (2)
    "self_model.json",
    "nell_style_fingerprint.json",
)


def migrate_legacy_files(
    *,
    og_data_dir: Path,
    persona_dir: Path,
) -> tuple[list[str], list[str]]:
    """Copy each LEGACY_FILES entry from og_data_dir to persona_dir/legacy/.

    Args:
        og_data_dir: The OG NellBrain `data/` directory (where memories_v2.json lives).
        persona_dir: The new framework persona dir (where memories.db will live).

    Returns:
        (preserved, missing) — both lists of basenames. Order matches LEGACY_FILES
        for deterministic test assertions.

    Raises:
        OSError if persona_dir is unwritable (e.g. permissions). Otherwise no
        exceptions; missing OG files are silently counted in `missing`.
    """
    legacy_dir = persona_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    preserved: list[str] = []
    missing: list[str] = []
    for name in LEGACY_FILES:
        src = og_data_dir / name
        if not src.exists():
            missing.append(name)
            continue
        dest = legacy_dir / name
        dest.write_bytes(src.read_bytes())
        preserved.append(name)
    return preserved, missing
