"""Conversation compaction — the one core that fades old history into a
persisted, archived summary block at the head of the buffer.

Three callers drive it (see changes/timed-conversation-compaction/1-spec.md):
  * the Kindled ``compact_history`` tool  — fold_existing_summary=False (append)
  * the daily supervisor cadence          — fold_existing_summary=True  (fade)
  * the apply_budget backstop              — fold_existing_summary=True  (fade)

Design invariants this module upholds:
  * **Lossless before lossy.** Raw turns (and, when folding, the old summary)
    are written to the append-only archive and verified BEFORE the live buffer
    is rewritten — an archive failure leaves the buffer untouched (no data loss).
  * **Never drop the un-extracted.** Only raw turns at/before the ingest cursor
    (``ts <= cursor``) are removable, so a turn is never compacted away before it
    becomes a memory. A ``None`` cursor (nothing extracted yet) is a hard no-op.
  * **Stable prefix.** The summary block is a persisted record rendered by
    ``_buffer_turns_to_messages`` as a head system message; between compactions
    the buffer only grows at the tail, so the replayed prefix is byte-stable
    (the cache side effect).
  * **Idempotent.** No removable raw turns ⇒ hard no-op; the existing summary is
    never re-faded with no new input, regardless of the fold flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import (
    acquire_compaction_lock,
    append_archive,
    read_cursor,
    read_session,
    release_compaction_lock,
    rewrite_session_atomic,
)

logger = logging.getLogger(__name__)

# Reuses the budget.py wording so a faded block reads the same whether produced
# here or by the (now-delegating) apply_budget backstop.
_COMPACTION_PROMPT = """Summarize the following conversation for context preservation.
Preserve: names of people and places, decisions made, emotional beats,
unresolved threads, anything that would be referenced later.
Drop: pleasantries, repetition, formatting noise.
Output prose only, no headers or lists.

CONVERSATION:
{transcript}

SUMMARY:"""


@dataclass
class CompactionResult:
    """Outcome of one compact_conversation call."""

    compacted: bool          # did the buffer actually change?
    compacted_n: int         # raw turns moved to archive
    new_gen: int             # gen of the summary now at the head (0 if none)
    fell_soft: bool          # provider failed → deterministic note used
    reaped_stale_lock: bool  # a crashed predecessor's lock was reaped
    reason: str = ""         # why a no-op happened (for the log)


def _parse_ts(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _split_buffer(turns: list[dict]) -> tuple[dict | None, list[dict]]:
    """Return (existing_summary_row_or_None, raw_turns_in_order).

    The first ``speaker=="summary"`` row is the existing summary; raw turns are
    the user/assistant rows in file order. Defensive against a stray second
    summary row (ignored here; the assembler hoists/drops, and the writer keeps
    at most one)."""
    existing: dict | None = None
    raw: list[dict] = []
    for t in turns:
        if t.get("speaker") == "summary":
            if existing is None:
                existing = t
            continue
        # Any non-summary row is a real conversation turn (user/assistant or an
        # arbitrary name-speaker) — never dropped, so a rewrite can't lose data.
        raw.append(t)
    return existing, raw


def _summary_text(row: dict | None) -> str:
    return (row or {}).get("text", "") or ""


def _summary_gen(row: dict | None) -> int:
    if not row:
        return 0
    try:
        return int(row.get("compaction", {}).get("gen", 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def compact_conversation(
    persona_dir,
    session_id: str,
    *,
    older_than: timedelta,
    fold_existing_summary: bool,
    provider: LLMProvider,
    min_keep_tail: int = 40,
    now: datetime | None = None,
    lock_stale_s: float = 600.0,
) -> CompactionResult:
    """Fade in-range, already-extracted raw turns into the head summary block.

    See module docstring for the invariants. Returns a CompactionResult; a
    no-op (locked / None cursor / nothing aged) returns ``compacted=False`` with
    a ``reason``.
    """
    now = now or datetime.now(UTC)

    reaped = False
    if not acquire_compaction_lock(persona_dir, session_id, stale_s=lock_stale_s):
        return CompactionResult(False, 0, 0, False, False, reason="locked")
    # We acquired (possibly after reaping); we can't cheaply tell here whether a
    # reap happened, so surface it via the lock module if needed later. Keep
    # False — reaping is logged inside acquire on the rare path.
    try:
        turns = read_session(persona_dir, session_id)
        existing_summary, raw_turns = _split_buffer(turns)

        # --- Cursor guard: only compact what is provably extracted -------------
        cursor = read_cursor(persona_dir, session_id)
        if cursor is None:
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), False, reaped,
                reason="cursor_none",
            )
        cursor_dt = _parse_ts(cursor)
        cutoff = now - older_than
        if cursor_dt is not None and cursor_dt < cutoff:
            cutoff = cursor_dt

        # Protect the most-recent min_keep_tail raw turns regardless of age.
        protected = set(range(max(0, len(raw_turns) - min_keep_tail), len(raw_turns)))
        removable: list[dict] = []
        retained: list[dict] = []
        for i, t in enumerate(raw_turns):
            ts = _parse_ts(t.get("ts"))
            if i not in protected and ts is not None and ts <= cutoff:
                removable.append(t)
            else:
                retained.append(t)

        if not removable:
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), False, reaped,
                reason="nothing_aged",
            )

        # --- Summarize ---------------------------------------------------------
        fell_soft = False
        if fold_existing_summary and existing_summary is not None:
            transcript_rows = [existing_summary, *removable]
        else:
            transcript_rows = removable
        transcript = "\n".join(
            f"{r.get('speaker', '?')}: {r.get('text', '')}" for r in transcript_rows
        )
        try:
            new_part = provider.generate(
                prompt=_COMPACTION_PROMPT.format(transcript=transcript)
            ).strip()
        except Exception:
            logger.exception(
                "compaction: provider summarisation failed session=%s; falling back",
                session_id,
            )
            new_part = f"[truncated {len(removable)} earlier messages]"
            fell_soft = True

        if fold_existing_summary or existing_summary is None:
            # Fade (or first-ever summary): the new text supersedes the old.
            new_text = new_part
        else:
            # Tool append: keep the existing summary verbatim, append the new part.
            prior = _summary_text(existing_summary)
            new_text = f"{prior}\n\n{new_part}" if prior else new_part

        new_gen = _summary_gen(existing_summary) + 1
        covers_until = removable[-1].get("ts") or cutoff.isoformat()
        summary_row = {
            "session_id": session_id,
            "speaker": "summary",
            "text": new_text,
            "ts": now.isoformat(timespec="seconds"),
            "compaction": {
                "covers_until_ts": covers_until,
                "folded": bool(fold_existing_summary),
                "gen": new_gen,
            },
        }

        # --- Archive BEFORE mutating the live buffer (lossless-before-lossy) ----
        # Archive the removed raw turns and, when folding, the old summary being
        # superseded (so the provenance chain keeps every faded version).
        archive_records = list(removable)
        if fold_existing_summary and existing_summary is not None:
            archive_records.append(existing_summary)
        try:
            written = append_archive(persona_dir, session_id, archive_records)
            if written <= 0 and archive_records:
                raise OSError("archive append wrote zero bytes")
        except Exception:
            logger.exception(
                "compaction: archive write failed session=%s; buffer left untouched",
                session_id,
            )
            return CompactionResult(
                False, 0, _summary_gen(existing_summary), fell_soft, reaped,
                reason="archive_failed",
            )

        # --- Install [summary, *retained] atomically ---------------------------
        rewrite_session_atomic(persona_dir, session_id, [summary_row, *retained])
        logger.info(
            "compaction: session=%s gen=%d folded=%s compacted_n=%d fell_soft=%s",
            session_id, new_gen, fold_existing_summary, len(removable), fell_soft,
        )
        return CompactionResult(
            True, len(removable), new_gen, fell_soft, reaped, reason="ok"
        )
    finally:
        release_compaction_lock(persona_dir, session_id)
