"""graveyard.py — forgotten_memories.jsonl writer + reader per spec §5.

Mirrors brain/growth/arc_storage.py removed_arcs.jsonl exactly:
  - file_lock + os.fsync atomic append (Windows-safe)
  - iter_jsonl_skipping_corrupt for tolerant reads

Spec: docs/superpowers/specs/2026-05-18-forgetting-design.md §5
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from brain.forgetting.salience import SalienceInputs
from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.memory.store import Memory
from brain.utils.file_lock import file_lock

GRAVEYARD_FILENAME = "forgotten_memories.jsonl"


def append(
    persona_dir: Path,
    *,
    memory: Memory,
    salience_at_drop: float,
    inputs: SalienceInputs,
    lived_age_hours: float,
    reason: str,
) -> None:
    """Atomic append to forgotten_memories.jsonl.

    Mirrors brain/growth/arc_storage.append_removed_arc exactly:
    file_lock + open(path, 'ab') + os.fsync for Windows-safe concurrent
    appends (POSIX has atomic O_APPEND; Windows doesn't).

    Graveyard write MUST happen before hard_delete — per spec §4
    "write JSONL entry first".
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "memory_id": memory.id,
        "forgotten_at_iso": datetime.now(UTC).isoformat(),
        "lived_age_hours_at_forgetting": lived_age_hours,
        "domain": memory.domain,
        "memory_type": memory.memory_type,
        "created_at_iso": memory.created_at.isoformat() if memory.created_at else None,
        "summary": memory.content,  # the fading body IS the summary at drop time
        "emotion_at_ingest": dict(memory.emotions),
        "salience_at_drop": salience_at_drop,
        "salience_inputs_at_drop": asdict(inputs),
        "graveyard_reason": reason,
    }
    path = persona_dir / GRAVEYARD_FILENAME
    line = (json.dumps(entry) + "\n").encode("utf-8")
    with file_lock(path):
        with open(path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def search(persona_dir: Path, query: str, *, limit: int = 5) -> list[dict]:
    """Case-insensitive substring match against summary + domain.

    Returns most-recent first (sorted by forgotten_at_iso lexicographically —
    ISO 8601 sorts lexically). Tolerates corrupt JSONL lines via
    iter_jsonl_skipping_corrupt. Respects limit.
    """
    path = persona_dir / GRAVEYARD_FILENAME
    if not path.exists():
        return []
    needle = query.lower()
    hits: list[dict] = []
    for entry in iter_jsonl_skipping_corrupt(path):
        summary = (entry.get("summary") or "").lower()
        domain = (entry.get("domain") or "").lower()
        if needle in summary or needle in domain:
            hits.append(entry)
    # Most-recent first via forgotten_at_iso (ISO 8601 lexicographic order is correct).
    hits.sort(key=lambda e: e.get("forgotten_at_iso") or "", reverse=True)
    return hits[:limit]


def read_all(persona_dir: Path) -> list[dict]:
    """All graveyard entries in append order.

    Used by the ambient prompt block to count recent losses. Pure read —
    no mutation, no sorting. Returns empty list if file absent.
    """
    path = persona_dir / GRAVEYARD_FILENAME
    if not path.exists():
        return []
    return list(iter_jsonl_skipping_corrupt(path))
