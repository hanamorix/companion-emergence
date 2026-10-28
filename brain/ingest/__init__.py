"""brain.ingest — SP-4 conversation ingest pipeline.

Pipeline: BUFFER → CHUNK → EXTRACT → SCORE → DEDUPE → COMMIT → SOUL → LOG.

Chat turns are buffered per-session to JSONL files. A per-session cursor records how far
extraction got. An extraction pass reads the rows after the cursor, splits them into chunks
of whole rows no larger than ``ingest.extraction_chunk_chars`` (#523), and extracts each
chunk, oldest first, into candidate memories via an LLM; they are deduplicated against
existing embeddings, committed (gated labels go to the pending queue), high-importance
items are queued for SP-5 (soul module), and the cursor moves past each committed chunk.

Public API:
    ingest_turn(persona_dir, turn) -> str
    close_session(persona_dir, session_id, *, store, hebbian, provider, ...) -> IngestReport
        Extracts every chunk after the cursor, archives the whole buffer, then deletes it (CLI --close, /sessions/close).
    close_stale_sessions(persona_dir, *, silence_minutes, store, ...) -> list[IngestReport]
        Destructive sweep (close_session per silent session); explicit close callers only (a bridge stop runs no sweep).
    extract_session_snapshot(persona_dir, session_id, *, store, hebbian, provider, ...) -> IngestReport
        Non-destructive cursor-driven snapshot — leaves the buffer intact; ``max_chunks`` bounds a pass.
    snapshot_stale_sessions(persona_dir, *, silence_minutes, store, ...) -> list[IngestReport]
        Periodic non-destructive sweep, one chunk per session per pass — used by the supervisor's snapshot job.
    finalize_stale_sessions(persona_dir, *, finalize_after_hours, store, ...) -> list[IngestReport]
        Long-silence (default 24h) extraction-only sweep, one chunk per session per run — used by supervisor.
    list_active_sessions(persona_dir) -> list[str]
    list_soul_candidates(persona_dir) -> list[dict]
    IngestReport
    ExtractedItem
"""

from brain.ingest.buffer import ingest_turn, list_active_sessions
from brain.ingest.pipeline import (
    close_session,
    close_stale_sessions,
    extract_session_snapshot,
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
from brain.ingest.soul_queue import list_soul_candidates
from brain.ingest.types import ExtractedItem, IngestReport

__all__ = [
    "ingest_turn",
    "close_session",
    "close_stale_sessions",
    "extract_session_snapshot",
    "finalize_stale_sessions",
    "snapshot_stale_sessions",
    "list_active_sessions",
    "list_soul_candidates",
    "ExtractedItem",
    "IngestReport",
]
