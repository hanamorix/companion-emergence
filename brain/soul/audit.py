"""Soul audit log — append + read soul_audit.jsonl.

Every autonomous soul decision (accept / reject / defer) is permanently
logged here. The audit trail is the safety rail — humans can see every
decision the brain made about its own soul.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.soul.review import Decision

logger = logging.getLogger(__name__)


def append_audit_entry(
    persona_dir: Path,
    decision: Decision,
    candidate: dict,
    related: list[str],
    emotional_summary: str,
    crystallization_id: str | None,
    dry_run: bool,
) -> bool:
    """Append one decision entry to <persona_dir>/soul_audit.jsonl.

    Never raises — a write failure logs a warning and returns False.
    The audit trail is permanent; entries are never deleted or modified.
    """
    audit_path = persona_dir / "soul_audit.jsonl"

    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "candidate_id": decision.candidate_id,
        "candidate_text": candidate.get("text", "")[:300],
        "candidate_source": candidate.get("source") or candidate.get("session_id") or "?",
        "candidate_label": candidate.get("label", "?"),
        "decision": decision.decision,
        "confidence": decision.confidence,
        "love_type": decision.love_type,
        "resonance": decision.resonance,
        "reasoning": decision.reasoning,
        "why_it_matters": decision.why_it_matters,
        "related_memories": related,
        "emotional_state": emotional_summary,
        "crystallization_id": crystallization_id,
        "dry_run": dry_run,
        "parse_error": decision.parse_error,
        "forced_defer_reason": decision.forced_defer_reason,
    }

    try:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        # fsync after append: the audit trail IS the safety rail. A torn
        # final line means the brain made a decision and lost the audit
        # entry that explained it — exactly the failure mode this log
        # exists to prevent.
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return True
    except OSError as exc:
        logger.warning("soul audit log write failed for %s: %s", audit_path, exc)
        return False


def read_audit_log(persona_dir: Path, *, limit: int | None = None) -> list[dict]:
    """Read soul_audit.jsonl entries via read_jsonl_skipping_corrupt.

    Returns entries oldest-first. If limit is provided, returns the last N
    entries (i.e. the most recent N, in chronological order).
    """
    from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

    audit_path = persona_dir / "soul_audit.jsonl"
    entries = read_jsonl_skipping_corrupt(audit_path)

    if limit is not None:
        entries = entries[-limit:]

    return entries
