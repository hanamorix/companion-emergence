"""Soul audit log — append + read soul_audit.jsonl.

Every autonomous soul decision (accept / reject / defer) is permanently
logged here. The audit trail is the safety rail — humans can see every
decision the brain made about its own soul.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from brain.health.json_io import Recovery, append_jsonl, iter_jsonl, read_jsonl

if TYPE_CHECKING:
    from brain.soul.review import Decision

logger = logging.getLogger(__name__)

_ARCHIVE_PATTERN = re.compile(r"^soul_audit\.(\d{4})\.jsonl\.gz$")


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

    # fsync after append: the audit trail IS the safety rail. A torn
    # final line means the brain made a decision and lost the audit
    # entry that explained it — exactly the failure mode this log
    # exists to prevent.
    return append_jsonl(audit_path, entry, fsync=True, on_error="log", nonfinite="null") is not None


def read_audit_log(persona_dir: Path, *, limit: int | None = None) -> list[dict]:
    """Read soul_audit.jsonl entries via read_jsonl.

    Returns entries oldest-first. If limit is provided, returns the last N
    entries (i.e. the most recent N, in chronological order).
    """
    entries = read_jsonl(persona_dir / "soul_audit.jsonl", recover=Recovery.RAISE)

    if limit is not None:
        entries = entries[-limit:]

    return entries


def iter_audit_full(persona_dir: Path) -> Iterator[dict]:
    """Yield every audit entry across active + yearly archives, chronologically.

    Walks ``soul_audit.YYYY.jsonl.gz`` archives oldest year → newest year,
    then the active ``soul_audit.jsonl``. Per-line resilience matches
    :func:`read_audit_log` — malformed lines are skipped with a warning.

    Hana's spec requirement (2026-05-11): every decision ever made about
    Nell's soul must remain reachable. This is the reader that honours
    that — used by ``nell soul audit --full`` and any future caller that
    needs the full history. Streaming, so memory stays bounded regardless
    of how many years have accumulated.
    """
    # 1. Yearly archives, oldest first.
    if persona_dir.exists():
        archives: list[tuple[int, Path]] = []
        for child in persona_dir.iterdir():
            m = _ARCHIVE_PATTERN.match(child.name)
            if m:
                archives.append((int(m.group(1)), child))
        archives.sort(key=lambda t: t[0])
        for _year, archive_path in archives:
            yield from iter_jsonl(archive_path, recover=Recovery.RAISE)

    # 2. Active log.
    yield from iter_jsonl(persona_dir / "soul_audit.jsonl", recover=Recovery.RAISE)
