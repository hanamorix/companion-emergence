"""emergence-kit → companion-emergence migrator.

Imports a brain authored with the lighter `emergence-kit
<https://github.com/hanamorix/emergence-kit>`_ kit (a Python
``my_brain.py`` + a handful of JSON files) into a fresh
companion-emergence persona.

The two formats overlap heavily — emergence-kit uses
``memories_v2.json`` (or the older ``memories.json``) with the same
per-memory dict shape OG NellBrain uses (id, content, memory_type,
domain, emotions, importance, tags, active, created_at, ...) — so
the existing :func:`brain.migrator.transform.transform_memory`
function does the heavy lifting.

What emergence-kit *doesn't* ship that the OG-NellBrain migrator
otherwise expects, all handled gracefully here:

* No ``connection_matrix.npy`` — no Hebbian edges. Hebbian builds
  back up over time as the brain runs; the missing matrix isn't a
  loss, just a fresh edges database.
* No ``creative_dna.json`` — emergence-kit doesn't track a writing
  voice fingerprint.
* No ``reflex_engine.py`` / ``reflex_log.json`` — emergence-kit
  doesn't have reflex arcs.
* No ``nell_interests.json`` — emergence-kit doesn't track a
  research-interest set.

What we do read:

* ``memories_v2.json`` (or fallback ``memories.json``) — the canonical
  memory list. Every dict goes through ``transform_memory``.
* ``soul_template.json`` — emergence-kit's soul file. Schema:
  ``{"crystallizations": [...], "soul_truth": str, "first_love": ...,
  "version": int}``. Each crystallization gets imported into the
  new persona's soul store.
* ``personality.json`` (if present) — copied verbatim into the
  persona dir as ``personality.json`` for future hooks; the brain
  doesn't currently consume it but preserving it keeps the carry-
  over honest.

The result is a persona dir with memories.db, hebbian.db (empty
but valid), crystallizations.db, and a copy of the personality
file. Same shape as a fresh OG-NellBrain migration produces; the
brain doesn't care which kit the data came from.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.migrator.report import MigrationReport
from brain.migrator.transform import SkippedMemory, transform_memory
from brain.paths import get_persona_dir
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmergenceKitMigrateArgs:
    """Validated argument bundle for ``migrate_emergence_kit``."""

    input_dir: Path
    output_dir: Path | None
    install_as: str | None
    force: bool

    def __post_init__(self) -> None:
        if (self.output_dir is None) == (self.install_as is None):
            raise ValueError("Exactly one of --output or --install-as must be provided.")


def _read_emergence_kit_memories(input_dir: Path) -> list[dict[str, Any]]:
    """Read memories from the kit dir.

    emergence-kit's ``my_brain.py`` writes ``memories_v2.json`` by
    default. The pre-v2 layout used ``memories.json``; try that as a
    fallback so older kit snapshots still import cleanly.
    """
    for filename in ("memories_v2.json", "memories.json"):
        path = input_dir / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"emergence-kit memory file at {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(
                f"emergence-kit memory file at {path} is not a JSON list "
                f"(got {type(data).__name__})"
            )
        return data
    raise FileNotFoundError(
        f"No memories_v2.json or memories.json under {input_dir}. "
        "Point --input at the directory that contains the kit's brain JSON files."
    )


def _read_emergence_kit_soul(input_dir: Path) -> dict[str, Any] | None:
    """Read ``soul_template.json`` if present."""
    path = input_dir / "soul_template.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("emergence-kit: soul_template.json unreadable: %s", exc)
        return None


def _import_kit_crystallizations(
    soul: dict[str, Any] | None,
    target_db: Path,
) -> tuple[int, str | None]:
    """Convert emergence-kit crystallizations → SoulStore rows.

    emergence-kit's crystallization shape is best-effort. We import
    any entry with at least an ``id`` and ``moment``; other fields
    default sensibly so a sparse import still produces valid rows.
    """
    if soul is None:
        return 0, "no soul_template.json found"
    crystals = soul.get("crystallizations")
    if not isinstance(crystals, list) or not crystals:
        return 0, "no crystallizations to import"

    imported = 0
    skipped_reasons: list[str] = []
    soul_store = SoulStore(db_path=target_db)
    try:
        for raw in crystals:
            if not isinstance(raw, dict):
                skipped_reasons.append("non-dict entry")
                continue
            crystal_id = raw.get("id") or raw.get("crystal_id")
            moment = raw.get("moment") or raw.get("text") or raw.get("content")
            if not crystal_id or not moment:
                skipped_reasons.append("missing id or moment")
                continue
            try:
                ts_raw = raw.get("crystallized_at") or raw.get("created_at")
                if isinstance(ts_raw, str):
                    if ts_raw.endswith("Z"):
                        ts_raw = ts_raw[:-1] + "+00:00"
                    crystallized_at = datetime.fromisoformat(ts_raw)
                else:
                    crystallized_at = datetime.now(UTC)
                if crystallized_at.tzinfo is None:
                    crystallized_at = crystallized_at.replace(tzinfo=UTC)
            except (TypeError, ValueError):
                crystallized_at = datetime.now(UTC)
            try:
                resonance_val = int(round(float(raw.get("resonance", 8))))
            except (TypeError, ValueError):
                resonance_val = 8
            crystal = Crystallization(
                id=str(crystal_id),
                moment=str(moment),
                love_type=str(raw.get("love_type") or "tender"),
                why_it_matters=str(raw.get("why_it_matters") or ""),
                crystallized_at=crystallized_at,
                who_or_what=str(raw.get("who_or_what") or ""),
                resonance=max(1, min(10, resonance_val)),
                permanent=bool(raw.get("permanent", True)),
            )
            try:
                soul_store.create(crystal)
                imported += 1
            except Exception as exc:  # noqa: BLE001
                skipped_reasons.append(f"create error: {exc}")
    finally:
        soul_store.close()
    skipped_summary = (
        f"skipped {len(skipped_reasons)}: {skipped_reasons[:3]}"
        if skipped_reasons
        else None
    )
    return imported, skipped_summary


def _copy_personality_file(input_dir: Path, work_dir: Path) -> bool:
    """Copy ``personality.json`` verbatim into the new persona dir."""
    src = input_dir / "personality.json"
    if not src.is_file():
        return False
    dst = work_dir / "personality.json"
    try:
        shutil.copy2(src, dst)
        return True
    except OSError as exc:
        logger.warning("emergence-kit: personality.json copy failed: %s", exc)
        return False


def _ensure_clobber_safe(path: Path, force: bool, kind: str) -> None:
    """Refuse to overwrite a non-empty target unless --force was given."""
    if not path.exists():
        return
    if any(path.iterdir()):
        if not force:
            raise FileExistsError(
                f"{kind} {path} already exists and is non-empty. "
                "Pass --force to back up and overwrite."
            )


def migrate_emergence_kit(args: EmergenceKitMigrateArgs) -> MigrationReport:
    """Orchestrate an emergence-kit → companion-emergence migration."""
    started = time.monotonic()

    if args.output_dir is not None:
        work_dir = args.output_dir
        _ensure_clobber_safe(work_dir, args.force, kind="output directory")
        work_dir.mkdir(parents=True, exist_ok=True)
        finalise_target: Path | None = None
    else:
        assert args.install_as is not None
        final_dir = get_persona_dir(args.install_as)
        if final_dir.exists() and not args.force:
            raise FileExistsError(
                f"Persona directory already exists: {final_dir}. "
                "Pass --force to back up and overwrite."
            )
        work_dir = final_dir.with_name(f"{args.install_as}.new")
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        finalise_target = final_dir

    kit_memories = _read_emergence_kit_memories(args.input_dir)
    store = MemoryStore(db_path=work_dir / "memories.db")
    migrated_count = 0
    skipped: list[SkippedMemory] = []
    seen_ids: set[str] = set()
    try:
        for raw in kit_memories:
            mem, sk = transform_memory(raw)
            if sk is not None:
                skipped.append(sk)
                continue
            assert mem is not None
            if mem.id in seen_ids:
                skipped.append(
                    SkippedMemory(
                        id=mem.id,
                        reason="duplicate_id",
                        field="id",
                        raw_snippet=mem.content[:120],
                    )
                )
                continue
            seen_ids.add(mem.id)
            store.create(mem)
            migrated_count += 1
    finally:
        store.close()

    # Empty Hebbian DB so the schema is in place; matrix builds back
    # up as the brain runs.
    hebbian = HebbianMatrix(db_path=work_dir / "hebbian.db")
    hebbian.close()

    soul = _read_emergence_kit_soul(args.input_dir)
    crystals_imported, soul_skipped_reason = _import_kit_crystallizations(
        soul, work_dir / "crystallizations.db"
    )

    personality_copied = _copy_personality_file(args.input_dir, work_dir)

    if finalise_target is not None:
        if finalise_target.exists() and args.force:
            backup = finalise_target.with_name(
                f"{finalise_target.name}.backup-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            )
            shutil.move(str(finalise_target), str(backup))
        os.replace(work_dir, finalise_target)
        result_dir = finalise_target
    else:
        result_dir = work_dir

    # Write a small source-manifest with which kit + input dir was
    # used. Different from the OG migrator's per-file SHA manifest;
    # emergence-kit only has ~3 files, recording them by name + size
    # is enough forensic detail.
    _write_kit_source_manifest(
        result_dir / "source-manifest.json",
        input_dir=args.input_dir,
        memories_imported=migrated_count,
        memories_skipped=len(skipped),
        crystallizations_imported=crystals_imported,
        personality_copied=personality_copied,
    )

    elapsed = time.monotonic() - started
    return MigrationReport(
        memories_migrated=migrated_count,
        memories_skipped=skipped,
        edges_migrated=0,
        edges_skipped=0,
        elapsed_seconds=elapsed,
        source_manifest=[],
        next_steps_inspect_cmds=[
            f"nell status --persona {args.install_as or '<persona>'}",
            f"nell memory list --persona {args.install_as or '<persona>'} --limit 10",
        ],
        next_steps_install_cmd=(
            f"nell init --persona {args.install_as} --fresh"
            if args.install_as
            else "Inspect the output dir, then nell init --persona <name> --fresh"
        ),
        crystallizations_migrated=crystals_imported,
        crystallizations_skipped_reason=soul_skipped_reason,
        creative_dna_migrated=False,
        creative_dna_skipped_reason="emergence-kit doesn't ship creative_dna.json",
        legacy_files_preserved=1 if personality_copied else 0,
        legacy_files_missing=0 if personality_copied else 1,
        legacy_skipped_reason=(
            None if personality_copied else "personality.json not present in source"
        ),
    )


def _write_kit_source_manifest(
    path: Path,
    *,
    input_dir: Path,
    memories_imported: int,
    memories_skipped: int,
    crystallizations_imported: int,
    personality_copied: bool,
) -> None:
    """Forensic record of the import: which kit, which dir, what landed."""
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_kit": "emergence-kit",
        "source_input_dir": str(input_dir),
        "summary": {
            "memories_imported": memories_imported,
            "memories_skipped": memories_skipped,
            "crystallizations_imported": crystallizations_imported,
            "personality_copied": personality_copied,
        },
    }
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("emergence-kit: source-manifest write failed: %s", exc)
