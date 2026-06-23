"""brain.notes.runner — the note-making closure fired by run_notes_tick.

Holds the background throttle slot, gathers her interior (recent dreams, current
emotional read, the last session she shared with the user), composes one note in
her own voice, and writes it — folder-bounded, create-only — into the authorized
folder. Budget + cooldown are already cleared by the tick before this fires.

Phase 4 (NOT YET) adds the wire-backs after a successful write: a note-state
initiate memory (so she mentions it next chat), an emotion delta, and a feed
entry. The seam is marked below.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.bridge import cli_throttle as _cli_throttle
from brain.notes import compose as _compose
from brain.notes import write as _write

logger = logging.getLogger(__name__)

_DREAM_LIMIT = 3


def _dreams_summary(store: Any) -> str:
    try:
        dreams = store.list_by_type("dream", active_only=True, limit=_DREAM_LIMIT)
    except Exception:
        logger.exception("notes.runner: listing dreams failed")
        return ""
    return " / ".join(d.content for d in dreams if getattr(d, "content", ""))


def _emotion_summary(store: Any) -> str:
    try:
        from brain.maker.sources import current_emotional_intensity
        peak = current_emotional_intensity(store)
    except Exception:
        logger.exception("notes.runner: emotion read failed")
        return ""
    return f"peak intensity {peak:.1f}" if peak else ""


def _last_session_summary(persona_dir: Path) -> str:
    try:
        from brain.ingest.buffer import list_active_sessions, read_session
        sessions = list_active_sessions(persona_dir)
        if not sessions:
            return ""
        turns = read_session(persona_dir, sessions[-1])
    except Exception:
        logger.exception("notes.runner: reading last session failed")
        return ""
    texts = [str(t.get("content", "")) for t in turns if t.get("content")]
    joined = " ".join(texts)[-400:]
    return joined.strip()


def make_note_and_wire(*, persona_dir: Path, config: Any, provider: Any,
                       now: datetime | None = None) -> None:
    """Compose one note from her interior and write it into the authorized
    folder. Through-path: gather → compose → commit_note. Phase 4 wires the
    memory/feed/emotion feeds after the write."""
    now = now or datetime.now(UTC)
    folder_raw = getattr(config, "notes_folder", None)
    if not folder_raw:
        return
    folder = Path(folder_raw)
    user_name = getattr(config, "user_name", None) or "you"

    # Gather her interior first — cheap local DB reads, no LLM, no slot needed.
    from brain.memory.store import MemoryStore
    store = MemoryStore(persona_dir / "memories.db")
    try:
        dreams_summary = _dreams_summary(store)
        emotion_summary = _emotion_summary(store)
    finally:
        store.close()
    last_session_summary = _last_session_summary(persona_dir)

    # Compose (the LLM call) runs INSIDE the held background slot so the
    # concurrency cap + chat-yield apply to it — mirroring maker (defer #57).
    # If the slot is unavailable, raise the distinct signal so run_notes_tick
    # treats it as a quiet retry, NOT a failure (no cooldown advance, no error).
    with _cli_throttle.background_slot() as slot:
        if not slot:
            logger.info("notes: throttle slot unavailable — deferring note")
            raise _cli_throttle.ThrottleDeferred("throttle deferred")
        note = _compose.make_note(
            provider,
            user_name=user_name,
            dreams_summary=dreams_summary,
            emotion_summary=emotion_summary,
            last_session_summary=last_session_summary,
        )
    path = _write.commit_note(persona_dir, folder, note, now=now)
    if path is None:
        logger.warning("notes: commit_note refused the write (guard) — nothing written")
        return
    logger.info("notes: wrote %r → %s", note.subject, path)

    # --- Phase 4 wiring (Task 9): wire the note back into her loops ---
    # A note-state initiate memory names the folder so she mentions it next chat,
    # and carries a small tenderness delta (vocab-filtered inside the writer). The
    # feed reads these note-state memories via brain.bridge.feed.build_note_entries.
    _wire_note_back(persona_dir, note=note, folder=folder, user_name=user_name, now=now)


def _wire_note_back(persona_dir: Path, *, note: Any, folder: Path, user_name: str,
                    now: datetime) -> None:
    """Write the note-state initiate memory + emotion delta after a successful
    write. Fail-soft: the file is already on disk; a missing memory only degrades
    ambient mention, never the write itself."""
    from uuid import uuid4

    from brain.initiate.memory import write_initiate_memory
    from brain.memory.store import MemoryStore

    store = MemoryStore(persona_dir / "memories.db")
    try:
        write_initiate_memory(
            store,
            audit_id=uuid4().hex,
            subject=note.subject,
            message=f"I left {user_name} a note in {folder}",
            state="note",
            ts=now.isoformat(),
            user_name=user_name,
            reach_emotions={"tenderness": 0.12},
        )
    except Exception:
        logger.exception("notes.runner: wiring the note-state memory failed")
    finally:
        store.close()
