"""Deterministic candidate emission — no LLM, no cost.

Event sources call emit_initiate_candidate() with structured metadata.
Idempotent on source_id: re-emission of the same source is a no-op.

The queue file is initiate_candidates.jsonl in the persona dir. Append
contract: every writer reopens (per the v0.0.8 retention contract).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt
from brain.initiate.schemas import (
    CandidateKind,
    CandidateSource,
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
    make_candidate_id,
)

logger = logging.getLogger(__name__)


def emit_initiate_candidate(
    persona_dir: Path,
    *,
    kind: CandidateKind,
    source: CandidateSource,
    source_id: str,
    semantic_context: SemanticContext,
    emotional_snapshot: EmotionalSnapshot | None = None,
    proposal: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> None:
    """Append one candidate to <persona_dir>/initiate_candidates.jsonl.

    Idempotent on (kind, source, source_id): re-emission is a no-op.
    Creates the queue file if it doesn't exist. Never raises on disk
    error — logs a warning and continues.
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    queue = persona_dir / "initiate_candidates.jsonl"

    # Dedupe check against existing queue contents.
    for existing in iter_jsonl_skipping_corrupt(queue):
        if (
            existing.get("kind") == kind
            and existing.get("source") == source
            and existing.get("source_id") == source_id
        ):
            return

    now = now or datetime.now(UTC)
    candidate = InitiateCandidate(
        candidate_id=make_candidate_id(now),
        ts=now.isoformat(),
        kind=kind,
        source=source,
        source_id=source_id,
        emotional_snapshot=emotional_snapshot,
        semantic_context=semantic_context,
        claimed_at=None,
        proposal=proposal,
    )

    try:
        with queue.open("a", encoding="utf-8") as f:
            f.write(candidate.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate candidate emit failed for %s: %s", queue, exc)


def read_candidates(persona_dir: Path) -> list[InitiateCandidate]:
    """Return all queued candidates (oldest first)."""
    queue = persona_dir / "initiate_candidates.jsonl"
    out: list[InitiateCandidate] = []
    for raw in iter_jsonl_skipping_corrupt(queue):
        # Reconstruct via from_jsonl roundtrip for type safety.
        out.append(InitiateCandidate.from_jsonl(json.dumps(raw)))
    return out


def remove_candidate(persona_dir: Path, candidate_id: str) -> None:
    """Atomically remove one candidate from the queue by ID.

    Rewrites the queue file without the target row. Used after a
    candidate has been processed (decision written to audit).

    Side effect: corrupt rows already in the queue are dropped during
    the rewrite. iter_jsonl_skipping_corrupt warns on each at read time;
    no separate audit is written here. For a queue file this is the
    correct behaviour (a corrupt candidate row is unrecoverable), but
    callers should be aware that rewriting the queue is also a cleanup
    operation.
    """
    queue = persona_dir / "initiate_candidates.jsonl"
    if not queue.exists():
        return
    surviving = [c for c in read_candidates(persona_dir) if c.candidate_id != candidate_id]
    tmp = queue.with_suffix(queue.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for c in surviving:
                f.write(c.to_jsonl() + "\n")
        tmp.replace(queue)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("initiate candidate remove failed for %s: %s", queue, exc)
