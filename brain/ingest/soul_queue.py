"""SP-4 SOUL stage — soul candidate queue for high-importance memories.

When an extracted item has importance >= DEFAULT_SOUL_THRESHOLD (8), it is
appended to <persona_dir>/soul_candidates.jsonl.

Principle note: this queue is NOT a human-approval gate. It is deferred work
for SP-5 (soul module) to consume autonomously. The brain decides which
candidates to crystallize. SP-4 just records them with status="auto_pending"
so SP-5 has data to work from when it lands.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import Recovery, append_jsonl, read_jsonl
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
) -> bool:
    """Append a soul candidate record to <persona_dir>/soul_candidates.jsonl.

    The record shape matches the OG nell_conversation_ingest.py schema with
    one field change: status is "auto_pending" (was "pending" in OG) to
    signal that SP-5 owns the crystallization decision.

    Record schema:
        memory_id:    str   — the newly committed memory's id
        text:         str   — the item's content
        label:        str   — memory_type / label
        importance:   int   — 1-10 extraction importance
        session_id:   str   — source session
        queued_at:    str   — ISO-8601 UTC timestamp
        status:       str   — lifecycle: "auto_pending" → "accepted" | "rejected" |
                              "expired".  SP-5 owns the transition.
        defer_count:  int   — incremented on each defer; when it reaches
                              brain.soul.review._MAX_DEFERS (3) the candidate is
                              retired as "expired" without another LLM call.
                              Absent on records queued before v0.0.34 — treated as 0.
        expired_at:   str   — ISO-8601 UTC, set when status→"expired".
    """
    record = {
        "memory_id": memory_id,
        "text": item.text,
        "label": item.label,
        "importance": item.importance,
        "session_id": session_id,
        "queued_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "auto_pending",
    }
    # Audit 2026-05-07 P2-2: the append takes the same exclusive file lock the review path holds for its
    # read-modify-rewrite cycle, so candidates queued between review's read and replace are not dropped;
    # fsync keeps a SIGKILL from leaving a torn line.
    return append_jsonl(
        _soul_candidates_path(persona_dir), record, fsync=True, on_error="log", nonfinite="null"
    ) is not None


def is_soul_candidate_queued(persona_dir: Path, *, session_id: str, text: str) -> bool:
    """True when a candidate with this session id and text is already in the queue, whatever its status or memory id.

    The memory id cannot tell a repeat apart: an extraction that runs again over the same rows (a run killed after its commits
    but before its progress was saved) mints a new id for the same text. Tolerates a malformed line and a missing file.
    """
    return any(
        rec.get("session_id") == session_id and rec.get("text") == text
        for rec in read_jsonl(_soul_candidates_path(persona_dir))
        if isinstance(rec, dict)
    )


def list_soul_candidates(persona_dir: Path) -> list[dict]:
    """Read all candidates from soul_candidates.jsonl, skipping malformed lines."""
    return read_jsonl(_soul_candidates_path(persona_dir), recover=Recovery.RAISE)
