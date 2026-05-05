"""brain.migrator.og_soul_candidates — schema migration for OG soul_candidates.jsonl.

Reads og_data_dir/soul_candidates.jsonl line-by-line, transforms each
record to the new framework's schema, writes to persona_dir/soul_candidates.jsonl.

Idempotent: re-running overwrites with the same content (the outer --force
flag handles install-rerun versioning at the persona-dir level).

Schema deltas handled:
- importance: OG 0-100 scale → new 0-10 scale via round(og/10) clamped to [0, 10].
- decided_at + status="accepted" → accepted_at.
- decided_at + status="rejected" → rejected_at.
- rejection_reason → reason.
- source: dropped (framework-internal in OG, not in new schema).
- crystallization_id: kept (only on accepted candidates).
- session_id: kept (None if missing).

Skipping rule: candidates without a memory_id are skipped (cannot link
to source memory). Counted in skipped_missing_memory_id tally.

See docs/superpowers/specs/2026-05-05-migrator-soul-candidates-and-reflex-log-design.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from brain.health.attempt_heal import save_with_backup_text

logger = logging.getLogger(__name__)

_DEFAULT_IMPORTANCE = 8


def migrate_soul_candidates(
    *,
    og_data_dir: Path,
    persona_dir: Path,
) -> tuple[int, int]:
    """Read og_data_dir/soul_candidates.jsonl, transform, write to persona_dir.

    Args:
        og_data_dir: The OG NellBrain `data/` directory.
        persona_dir: The new framework persona dir.

    Returns:
        (migrated, skipped_missing_memory_id) — counts.

    Returns (0, 0) silently if the OG file is missing.
    """
    src = og_data_dir / "soul_candidates.jsonl"
    if not src.exists():
        return (0, 0)

    migrated_records: list[dict[str, Any]] = []
    skipped_missing_memory_id = 0

    for line_num, raw in enumerate(
        src.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            og = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "soul_candidates.jsonl line %d malformed: %s", line_num, exc,
            )
            continue

        if not og.get("memory_id"):
            skipped_missing_memory_id += 1
            continue

        new = _transform(og)
        migrated_records.append(new)

    if migrated_records:
        dest = persona_dir / "soul_candidates.jsonl"
        text = "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in migrated_records
        )
        save_with_backup_text(dest, text)

    return (len(migrated_records), skipped_missing_memory_id)


def _transform(og: dict[str, Any]) -> dict[str, Any]:
    """Convert one OG candidate dict to new-schema dict."""
    importance_raw = og.get("importance", _DEFAULT_IMPORTANCE * 10)
    try:
        og_imp = float(importance_raw)
    except (TypeError, ValueError):
        og_imp = _DEFAULT_IMPORTANCE * 10
    importance = max(0, min(10, round(og_imp / 10)))

    new: dict[str, Any] = {
        "memory_id": og["memory_id"],
        "text": og.get("text", ""),
        "label": og.get("label", ""),
        "importance": importance,
        "session_id": og.get("session_id"),
        "queued_at": og.get("queued_at", ""),
        "status": og.get("status", "pending"),
    }

    decided_at = og.get("decided_at")
    status = new["status"]
    if status == "accepted" and decided_at:
        new["accepted_at"] = decided_at
        if og.get("crystallization_id"):
            new["crystallization_id"] = og["crystallization_id"]
    elif status == "rejected" and decided_at:
        new["rejected_at"] = decided_at
        if og.get("rejection_reason"):
            new["reason"] = og["rejection_reason"]

    return new
