"""SP-4 pipeline driver — orchestrates all 8 ingest stages.

close_session()       — run all 8 stages for one session, delete its buffer.
close_stale_sessions()— iterate active sessions, close any past the silence window.

Stage flow:
  BUFFER  → read turns from <persona_dir>/active_conversations/<session_id>.jsonl
  CLOSE   → guard: if no turns, unlink buffer and return empty report
  EXTRACT → format transcript, call LLM, get ExtractedItems
  SCORE   → normalize each item (label coercion, importance clamp, text strip)
  DEDUPE  → cosine similarity against EmbeddingCache (opt-in; None = skip)
  COMMIT  → direct write to MemoryStore + auto-Hebbian
  SOUL    → queue high-importance items to soul_candidates.jsonl
  LOG     → emit structured log event with counts
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import (
    delete_session_buffer,
    list_active_sessions,
    read_session,
    session_silence_minutes,
)
from brain.ingest.commit import commit_item
from brain.ingest.dedupe import DEFAULT_DEDUP_THRESHOLD, is_duplicate
from brain.ingest.extract import extract_items, format_transcript
from brain.ingest.soul_queue import DEFAULT_SOUL_THRESHOLD, queue_soul_candidate
from brain.ingest.types import IngestReport
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def _load_user_name(persona_dir: Path) -> str | None:
    """Look up PersonaConfig.user_name; None if missing or unset.

    Best-effort — never raises. PersonaConfig is the home for the
    user's name; we read it here once per session close. If the
    config file is corrupt or the field is missing, the extractor
    falls back to the legacy "user:" / "assistant:" prompt path.
    """
    try:
        from brain.persona_config import PersonaConfig

        return PersonaConfig.load(persona_dir / "persona_config.json").user_name
    except Exception:  # noqa: BLE001
        return None


def close_session(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache | None = None,
    config: dict | None = None,
) -> IngestReport:
    """Run the full 8-stage ingest pipeline on one session and delete its buffer.

    Parameters
    ----------
    persona_dir:
        Root directory for this persona (contains active_conversations/, etc.)
    session_id:
        The session to close and process.
    store:
        SQLite-backed MemoryStore to commit new memories into.
    hebbian:
        HebbianMatrix to strengthen connections between related memories.
    provider:
        LLMProvider used for EXTRACT stage (generate() surface).
    embeddings:
        Optional EmbeddingCache for DEDUPE. When None, dedupe is skipped.
    config:
        Optional dict of pipeline knobs:
          extraction_max_retries: int = 1
          max_transcript_tokens: int = 6000
          dedup_threshold: float = 0.88
          crystallize_threshold: int = 8

    Returns
    -------
    IngestReport with counts of what happened.
    """
    cfg = config or {}
    report = IngestReport(session_id=session_id)

    # ── BUFFER: read the session turns ──────────────────────────────────────
    turns = read_session(persona_dir, session_id)

    # ── CLOSE guard: empty session ───────────────────────────────────────────
    if not turns:
        delete_session_buffer(persona_dir, session_id)
        return report

    # ── EXTRACT ──────────────────────────────────────────────────────────────
    # Bug A (audit-3): pass speaker names so the LLM extractor can
    # disambiguate the current user from historical figures the assistant
    # may reference. user_name comes from PersonaConfig.user_name (None
    # when unset → legacy unnamed prompt). assistant_name is the persona
    # name (always known — it's the persona dir's basename).
    user_name = _load_user_name(persona_dir)
    assistant_name = persona_dir.name
    transcript = format_transcript(
        turns,
        max_tokens=int(cfg.get("max_transcript_tokens", 6000)),
        user_name=user_name,
        assistant_name=assistant_name,
    )
    items = extract_items(
        transcript,
        provider=provider,
        max_retries=int(cfg.get("extraction_max_retries", 1)),
        user_name=user_name,
        assistant_name=assistant_name,
    )
    report.extracted = len(items)

    # ── SCORE (normalize at the boundary) ────────────────────────────────────
    items = [it.normalize() for it in items if it.text]

    # ── DEDUPE + COMMIT + SOUL ────────────────────────────────────────────────
    dedup_threshold = float(cfg.get("dedup_threshold", DEFAULT_DEDUP_THRESHOLD))
    crystallize_threshold = int(cfg.get("crystallize_threshold", DEFAULT_SOUL_THRESHOLD))

    for item in items:
        # DEDUPE
        if is_duplicate(item.text, store=store, threshold=dedup_threshold, embeddings=embeddings):
            report.deduped += 1
            continue

        # COMMIT
        mem_id = commit_item(item, session_id=session_id, store=store, hebbian=hebbian)
        if mem_id is None:
            report.errors += 1
            continue

        report.committed += 1
        report.memory_ids.append(mem_id)

        # SOUL
        if item.importance >= crystallize_threshold:
            queued = queue_soul_candidate(
                persona_dir,
                memory_id=mem_id,
                item=item,
                session_id=session_id,
            )
            if queued:
                report.soul_candidates += 1
            else:
                report.soul_queue_errors += 1

    # ── LOG ──────────────────────────────────────────────────────────────────
    logger.info(
        "conversation_ingested session=%s turns=%d extracted=%d committed=%d "
        "deduped=%d soul_candidates=%d soul_queue_errors=%d errors=%d",
        session_id,
        len(turns),
        report.extracted,
        report.committed,
        report.deduped,
        report.soul_candidates,
        report.soul_queue_errors,
        report.errors,
    )

    # ── DELETE buffer ─────────────────────────────────────────────────────────
    delete_session_buffer(persona_dir, session_id)

    return report


def close_stale_sessions(
    persona_dir: Path,
    *,
    silence_minutes: float = 5.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache | None = None,
    config: dict | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; close any whose last turn is older than silence_minutes.

    Empty sessions (no turns) are cleaned up silently without running the
    full pipeline.

    Returns reports for closed sessions only (skips fresh sessions).
    """
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        from brain.ingest.buffer import read_session as _read

        turns = _read(persona_dir, sid)
        if not turns:
            # Ghost file — clean it up without generating a report.
            delete_session_buffer(persona_dir, sid)
            continue
        age = session_silence_minutes(turns)
        if age >= silence_minutes:
            report = close_session(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=provider,
                embeddings=embeddings,
                config=config,
            )
            reports.append(report)
    return reports
