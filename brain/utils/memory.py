"""Shared memory helpers used by multiple engines."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def days_since_human(
    store: MemoryStore,
    now: datetime,
    *,
    persona_dir: Path | None = None,
) -> float:
    """Days since the user last spoke. 999.0 if no signal anywhere.

    Cascades through the real data sources:

    1. Active session buffers — `<persona_dir>/active_conversations/*.jsonl`
       — find the newest user-speaker turn timestamp. This catches mid-
       session and just-closed conversations before ingest runs.
    2. Closed-session extractions — memories whose
       `metadata_json.source_summary` starts with `"conversation:"`.
       These came from a chat buffer that had user turns by definition;
       the memory's `created_at` is within minutes of the last user
       turn (when the ingest pipeline ran on session close).
    3. Fallback — 999.0.

    persona_dir is optional for backward-compat with legacy callers, but
    when omitted the function can only see closed-session signal — fresh
    chats won't update the count until the session closes.

    Used by reflex + research engines to gate on persona-silence duration,
    by get_body_state/boot brain-tools, and by the /persona/state
    aggregator. NOTE: the original implementation queried
    memory_type="conversation" — but no production code path writes that
    type (extracted memories are tagged observation/feeling/decision/
    question/fact/note). The query always returned [] → 999.0 fallback,
    silently breaking long-silence reflex arcs and body-state day-counts.
    Same root cause as the M-1 audit-2 fix to body/words.py.
    """
    candidates: list[datetime] = []

    # 1. Active session buffers — the freshest possible signal
    if persona_dir is not None:
        latest_buffer_ts = _latest_user_turn_in_buffers(persona_dir)
        if latest_buffer_ts is not None:
            candidates.append(latest_buffer_ts)

    # 2. Closed-session extractions — memories tagged conversation:<sid>
    latest_convo_memory_ts = _latest_conversation_memory_ts(store)
    if latest_convo_memory_ts is not None:
        candidates.append(latest_convo_memory_ts)

    if not candidates:
        return 999.0

    most_recent = max(candidates)
    if most_recent.tzinfo is None:
        most_recent = most_recent.replace(tzinfo=UTC)
    return (now - most_recent).total_seconds() / 86400.0


def _latest_user_turn_in_buffers(persona_dir: Path) -> datetime | None:
    """Most-recent user-turn timestamp across all active_conversations/*.jsonl."""
    active_dir = persona_dir / "active_conversations"
    if not active_dir.exists():
        return None
    latest: datetime | None = None
    try:
        for jsonl_path in active_dir.glob("*.jsonl"):
            try:
                with jsonl_path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            turn = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if turn.get("speaker") != "user":
                            continue
                        ts_raw = turn.get("ts")
                        if not ts_raw:
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_raw)
                        except ValueError:
                            continue
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)
                        if latest is None or ts > latest:
                            latest = ts
            except OSError:
                continue
    except OSError:
        return None
    return latest


def _latest_conversation_memory_ts(store: MemoryStore) -> datetime | None:
    """Most-recent created_at among memories whose source_summary marks them
    as extracted from a chat buffer (`conversation:<session_id>`).

    SQL LIKE on the metadata_json column — small persona DBs are fast;
    the ingest pipeline writes one batch of these per closed session,
    so even years of chats stay in the low thousands.
    """
    try:
        row = store._conn.execute(  # noqa: SLF001 — same-tier read
            """
            SELECT created_at FROM memories
            WHERE active = 1
              AND metadata_json LIKE '%"source_summary": "conversation:%'
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    except Exception:  # noqa: BLE001
        logger.warning("days_since_human: conversation memory scan failed", exc_info=True)
        return None
    if row is None:
        return None
    raw = row["created_at"] if hasattr(row, "keys") else row[0]
    try:
        ts = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts
