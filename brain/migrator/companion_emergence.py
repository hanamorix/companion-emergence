"""companion-emergence → companion-emergence migrator.

For v0.0.12 → v0.0.18 upgrades, or migrating a persona between machines.
The format hasn't broken since v0.0.12 — this migrator is a validated
forward-copy, not a schema transformation. Memories.db, hebbian.db,
crystallizations.db, and persona_config.json all travel as-is.

What we do:
- Validate the source dir actually IS a companion-emergence persona
  (memories.db present, persona_config.json parses).
- Detect "user pointed at the /personas parent" and suggest subdirs.
- Copy the dir with --force-backup support.
- Write source-manifest.json (forensic record of what was copied).
- Write app_config.json if NellFace doesn't have a selection yet.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import app_config
from brain.migrator.report import MigrationReport
from brain.paths import get_persona_dir


@dataclass(frozen=True)
class CompanionEmergenceMigrateArgs:
    input_dir: Path
    install_as: str
    force: bool


def _iter_persona_dirs(parent: Path) -> Iterator[Path]:
    if not parent.is_dir():
        return
    for child in sorted(parent.iterdir()):
        if child.is_dir() and (child / "memories.db").is_file():
            yield child


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _count_rows(db: Path, table: str) -> int:
    if not db.is_file():
        return 0
    try:
        conn = sqlite3.connect(db)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return 0


def _read_source_lived_age(input_dir: Path) -> float:
    """Read lived_age_hours from the source's felt_time_state.json without
    mutating it. 0.0 if absent or unreadable (fresh-brain default)."""
    p = input_dir / "felt_time_state.json"
    if not p.is_file():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("lived_age_hours", 0.0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def preflight_companion_emergence(input_dir: Path) -> dict[str, Any]:
    """Inspect input_dir. Pure read. Returns ExistingCePreflight-shaped dict."""
    result: dict[str, Any] = {
        "ok": False,
        "persona_name": None,
        "imported_user_name": None,
        "imported_voice_template": None,
        "memory_count": None,
        "crystallization_count": None,
        "hebbian_edge_count": None,
        "source_size_bytes": 0,
        "errors": [],
        "warnings": [],
    }

    if not input_dir.exists():
        result["errors"].append({"code": "input_missing", "message": f"No folder at {input_dir}."})
        return result
    if not input_dir.is_dir():
        result["errors"].append({"code": "input_not_dir", "message": f"{input_dir} exists but isn't a folder."})
        return result

    memories_db = input_dir / "memories.db"
    if not memories_db.is_file():
        subdirs = list(_iter_persona_dirs(input_dir))
        if subdirs:
            result["errors"].append({
                "code": "pointed_at_parent",
                "message": (
                    "That's the folder that contains your Kindled, not a single Kindled. "
                    "Pick the subfolder named after your companion."
                ),
                "detail": {"suggested_subdirs": [d.name for d in subdirs]},
            })
        else:
            result["errors"].append({
                "code": "no_memories_db",
                "message": "Doesn't look like a companion-emergence persona — there's no `memories.db` here.",
            })
        return result

    config_path = input_dir / "persona_config.json"
    if not config_path.is_file():
        result["errors"].append({
            "code": "no_persona_config",
            "message": "Persona folder missing `persona_config.json`. This usually means a copy was interrupted.",
        })
        return result

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["errors"].append({
            "code": "bad_persona_config",
            "message": f"`persona_config.json` is not valid JSON: {exc}",
        })
        return result

    result["persona_name"] = input_dir.name
    result["imported_user_name"] = config.get("user_name")
    result["imported_voice_template"] = config.get("voice_template")
    result["memory_count"] = _count_rows(memories_db, "memories")
    # Hebbian edges table is named `hebbian_edges` (brain/memory/hebbian.py)
    result["hebbian_edge_count"] = _count_rows(input_dir / "hebbian.db", "hebbian_edges")
    # Soul crystallizations table is named `crystallizations` (brain/soul/store.py)
    result["crystallization_count"] = _count_rows(input_dir / "crystallizations.db", "crystallizations")
    result["source_size_bytes"] = _dir_size_bytes(input_dir)
    result["ok"] = True
    return result


def migrate_companion_emergence(args: CompanionEmergenceMigrateArgs) -> MigrationReport:
    started = time.monotonic()

    preflight = preflight_companion_emergence(args.input_dir)
    if not preflight["ok"]:
        first = preflight["errors"][0]
        raise ValueError(f"{first['code']}: {first['message']}")

    target_dir = get_persona_dir(args.install_as)
    if target_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"A Kindled named `{args.install_as}` already exists at {target_dir}. "
                "Re-run with --force to back up and overwrite."
            )
        backup = target_dir.with_name(
            f"{target_dir.name}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        )
        shutil.move(str(target_dir), str(backup))

    shutil.copytree(args.input_dir, target_dir, dirs_exist_ok=False)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "migrated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_kind": "companion-emergence",
        "source_input_dir": str(args.input_dir),
        "source_size_bytes": preflight["source_size_bytes"],
        "memory_count": preflight["memory_count"],
        "crystallization_count": preflight["crystallization_count"],
        "hebbian_edge_count": preflight["hebbian_edge_count"],
        "lived_age_hours_at_migration": _read_source_lived_age(args.input_dir),
    }
    (target_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    app_config.write_if_missing(args.install_as)

    elapsed = time.monotonic() - started
    return MigrationReport(
        memories_migrated=preflight["memory_count"] or 0,
        memories_skipped=[],
        edges_migrated=preflight["hebbian_edge_count"] or 0,
        edges_skipped=0,
        elapsed_seconds=elapsed,
        source_manifest=[],
        next_steps_inspect_cmds=[
            f"nell status --persona {args.install_as}",
            f"nell memory list --persona {args.install_as} --limit 10",
        ],
        next_steps_install_cmd="(none — copy complete; relaunch Companion Emergence)",
        crystallizations_migrated=preflight["crystallization_count"] or 0,
        crystallizations_skipped_reason=None,
        creative_dna_migrated=False,
        creative_dna_skipped_reason="forward-copy preserves whatever existed",
        legacy_files_preserved=0,
        legacy_files_missing=0,
        legacy_skipped_reason=None,
        bytes_copied=preflight["source_size_bytes"],
        source_kind="companion-emergence",
    )
