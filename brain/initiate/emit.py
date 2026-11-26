"""Deterministic candidate emission — no LLM, no cost.

Event sources call emit_initiate_candidate() with structured metadata.
Idempotent on source_id: re-emission of the same source is a no-op.

The queue file is initiate_candidates.jsonl in the persona dir. Append
contract: every writer reopens (per the v0.0.8 retention contract).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.health.json_io import Recovery, append_jsonl, iter_jsonl, write_jsonl
from brain.initiate.schemas import (
    CandidateKind,
    CandidateSource,
    EmotionalSnapshot,
    InitiateCandidate,
    SemanticContext,
    make_candidate_id,
)
from brain.utils.file_lock import file_lock

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
    queue = persona_dir / "initiate_candidates.jsonl"

    # Dedupe check against existing queue contents.
    for existing in iter_jsonl(queue, recover=Recovery.RAISE):
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

    append_jsonl(queue, candidate.to_dict(), on_error="log", nonfinite="null")


def read_candidates(persona_dir: Path) -> list[InitiateCandidate]:
    """Return all queued candidates (oldest first)."""
    queue = persona_dir / "initiate_candidates.jsonl"
    out: list[InitiateCandidate] = []
    # RAISE: remove_candidate rewrites the queue from this read, so a failed read must not look like an empty queue.
    for raw in iter_jsonl(queue, recover=Recovery.RAISE):
        out.append(InitiateCandidate.from_dict(raw))
    return out


def remove_candidate(persona_dir: Path, candidate_id: str) -> None:
    """Atomically remove one candidate from the queue by ID.

    Rewrites the queue file without the target row. Used after a
    candidate has been processed (decision written to audit).

    Side effect: corrupt rows already in the queue are dropped during
    the rewrite. iter_jsonl warns on each at read time;
    no separate audit is written here. For a queue file this is the
    correct behaviour (a corrupt candidate row is unrecoverable), but
    callers should be aware that rewriting the queue is also a cleanup
    operation.
    """
    queue = persona_dir / "initiate_candidates.jsonl"
    if not queue.exists():
        return
    # The file lock makes read-and-rewrite one step against emit_initiate_candidate's append: an append
    # landing between the read and the replace used to be lost.
    with file_lock(queue):
        surviving = [c for c in read_candidates(persona_dir) if c.candidate_id != candidate_id]
        write_jsonl(queue, [c.to_dict() for c in surviving], fsync=False, on_error="log", nonfinite="null")
