"""brain.ingest — SP-4 conversation ingest pipeline.

8-stage pipeline: BUFFER → CLOSE → EXTRACT → SCORE → DEDUPE → COMMIT → SOUL → LOG.

Chat turns are buffered per-session to JSONL files. When a session closes
(explicit or stale silence), the full transcript is extracted into candidate
memories via an LLM, deduplicated against existing embeddings, committed
directly to the MemoryStore, and high-importance items are queued for SP-5
(soul module) to consume.

Public API:
    ingest_turn(persona_dir, turn) -> str
    close_session(persona_dir, session_id, *, store, hebbian, provider, ...) -> IngestReport
    close_stale_sessions(persona_dir, *, silence_minutes, store, ...) -> list[IngestReport]
    list_active_sessions(persona_dir) -> list[str]
    list_soul_candidates(persona_dir) -> list[dict]
    IngestReport
    ExtractedItem
"""

from brain.ingest.buffer import ingest_turn, list_active_sessions
from brain.ingest.pipeline import close_session, close_stale_sessions
from brain.ingest.soul_queue import list_soul_candidates
from brain.ingest.types import ExtractedItem, IngestReport

__all__ = [
    "ingest_turn",
    "close_session",
    "close_stale_sessions",
    "list_active_sessions",
    "list_soul_candidates",
    "ExtractedItem",
    "IngestReport",
]
