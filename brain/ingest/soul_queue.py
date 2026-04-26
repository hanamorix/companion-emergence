"""SP-4 SOUL stage — soul candidate queue for high-importance memories.

When an extracted item has importance >= DEFAULT_SOUL_THRESHOLD (8), it is
appended to <persona_dir>/soul_candidates.jsonl.

Principle note: this queue is NOT a human-approval gate. It is deferred work
for SP-5 (soul module) to consume autonomously. The brain decides which
candidates to crystallize. SP-4 just records them with status="auto_pending"
so SP-5 has data to work from when it lands.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.ingest.types import ExtractedItem

logger = logging.getLogger(__name__)

DEFAULT_SOUL_THRESHOLD = 8

_SOUL_CANDIDATES_FILENAME = "soul_candidates.jsonl"


def _soul_candidates_path(persona_dir: Path) -> Path:
    return persona_dir / _SOUL_CANDIDATES_FILENAME


def queue_soul_candidate(
    persona_dir: Path,
    *,
    memory_id: str,
    item: ExtractedItem,
    session_id: str,
) -> None:
    """Append a soul candidate record to <persona_dir>/soul_candidates.jsonl.

    The record shape matches the OG nell_conversation_ingest.py schema with
    one field change: status is "auto_pending" (was "pending" in OG) to
    signal that SP-5 owns the crystallization decision.

    Record schema:
        memory_id:   str   — the newly committed memory's id
        text:        str   — the item's content
        label:       str   — memory_type / label
        importance:  int   — 1-10 extraction importance
        session_id:  str   — source session
        queued_at:   str   — ISO-8601 UTC timestamp
        status:      str   — "auto_pending" (SP-5 consumes and updates this)
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "memory_id": memory_id,
        "text": item.text,
        "label": item.label,
        "importance": item.importance,
        "session_id": session_id,
        "queued_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "auto_pending",
    }
    path = _soul_candidates_path(persona_dir)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("queue_soul_candidate: failed to write to %s: %s", path, exc)


def list_soul_candidates(persona_dir: Path) -> list[dict]:
    """Read all candidates from soul_candidates.jsonl, skipping malformed lines."""
    return read_jsonl_skipping_corrupt(_soul_candidates_path(persona_dir))
