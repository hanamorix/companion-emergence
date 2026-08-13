"""Session rollover (1c) — start a fresh session seeded with the previous one's
post-compaction form, archiving the rest.

Two triggers, one core (``perform_rollover``):

  * **1c-A — >24h idle-gap stale resume (SYNC).** ``seed_mode="summary_only"``:
    fold the *entire* old conversation (``min_keep_tail=0``) and seed the new
    session with the summary alone (a >24h-old "recent 40" is not recent).
  * **1c-B — weekly cap, on the daily supervisor tick.** ``seed_mode="tiers_plus_tail"``:
    run the age-gated cascade to bring the 3 tiers current, then seed the new
    session with those 3 tiers + the 40 most-recent raw messages, carrying the
    old session's extraction cursor so the carried raw tail is neither
    re-extracted nor lost (C18).

The rollover OWNS the old buffer's lifecycle: after extract → fold → seed it
deletes the old buffer + cursor, writes the ``rolled_to`` successor pointer
(under the compaction lock, BEFORE the delete, so the old sid never resolves to
nothing), and evicts the in-memory registry entry. The finalize tick no longer
deletes buffers (see brain/ingest/pipeline.finalize_stale_sessions).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.chat.compaction import (
    build_compaction_provider,
    cascade_conversation,
    compact_conversation,
)
from brain.chat.session import create_session, remove_session
from brain.ingest.buffer import (
    acquire_compaction_lock,
    delete_backoff,
    delete_cursor,
    delete_session_buffer,
    read_cursor,
    read_session,
    release_compaction_lock,
    rewrite_session_atomic,
    write_cursor,
    write_rolled_to,
)

logger = logging.getLogger(__name__)

_ROLLOVER_TAIL = 40  # 1c-B seed carries the 40 most-recent raw messages


def _split_summary_and_raw(turns: list[dict]) -> tuple[dict | None, list[dict]]:
    summary: dict | None = None
    raw: list[dict] = []
    for t in turns:
        if t.get("speaker") == "summary":
            if summary is None:
                summary = t
        else:
            raw.append(t)
    return summary, raw


def perform_rollover(
    persona_dir: Path,
    old_sid: str,
    persona_name: str,
    *,
    seed_mode: str,
    now: datetime | None = None,
    provider: LLMProvider | None = None,
    store=None,
    hebbian=None,
    embeddings=None,
    config: dict | None = None,
) -> str | None:
    """Roll ``old_sid`` over into a fresh session. Returns the new session id, or
    None if there is nothing to seed / the session is busy.

    ``seed_mode`` is ``"summary_only"`` (1c-A) or ``"tiers_plus_tail"`` (1c-B).
    Memory extraction of the old buffer runs first (best-effort) when ``store`` +
    ``hebbian`` are supplied — the seed is the immediate recap, memory is long-term
    recall.
    """
    now = now or datetime.now(UTC)
    persona_dir = Path(persona_dir)

    # 1. Final memory extraction of the old buffer (best-effort). Advances the
    #    cursor so the carried tail's extraction state is current (C18).
    if store is not None and hebbian is not None and provider is not None:
        try:
            from brain.ingest.pipeline import extract_session_snapshot

            extract_session_snapshot(
                persona_dir, old_sid,
                store=store, hebbian=hebbian, provider=provider,
                embeddings=embeddings, config=config,
            )
        except Exception:
            logger.exception("rollover: extraction failed session=%s (continuing)", old_sid)

    # 2. Fold the old conversation into its post-compaction form.
    comp_provider = build_compaction_provider(persona_dir)
    if seed_mode == "summary_only":
        # Fold everything (a >24h-old "recent 40" is not recent).
        compact_conversation(
            persona_dir, old_sid,
            older_than=timedelta(0), fold_existing_summary=True,
            provider=comp_provider, min_keep_tail=0, now=now,
        )
    elif seed_mode == "tiers_plus_tail":
        # Bring the 3 tiers current, keeping the 40-msg raw tail live.
        cascade_conversation(
            persona_dir, old_sid, provider=comp_provider, now=now,
            min_keep_tail=_ROLLOVER_TAIL,
        )
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown seed_mode {seed_mode!r}")

    # 3-6. Seed the new session + reap the old buffer, under the compaction lock.
    #      Re-read the current committed row under the lock (interposition-safe;
    #      apply_budget may have touched the 24h tier between fold and here).
    if not acquire_compaction_lock(persona_dir, old_sid):
        return None  # busy → defer to the next cycle
    try:
        turns = read_session(persona_dir, old_sid)
        summary_row, raw = _split_summary_and_raw(turns)
        if seed_mode == "summary_only":
            seed_rows: list[dict] = [summary_row] if summary_row else []
        else:
            tail = raw[-_ROLLOVER_TAIL:]
            seed_rows = ([summary_row] if summary_row else []) + tail
        if not seed_rows:
            return None  # nothing to seed → abort (nothing lost; old buffer stays)

        old_cursor = read_cursor(persona_dir, old_sid)

        new_sess = create_session(persona_name)
        new_sid = new_sess.session_id
        # Re-stamp the seed rows' session_id to the new session so downstream
        # readers see a consistent id.
        reseeded = [{**r, "session_id": new_sid} for r in seed_rows]
        rewrite_session_atomic(persona_dir, new_sid, reseeded)
        # Carry the old session's extraction cursor so already-extracted carried
        # tail messages are NOT re-extracted while unextracted ones still are (C18).
        if old_cursor:
            write_cursor(persona_dir, new_sid, old_cursor)

        # Successor pointer BEFORE the delete (old sid never resolves to nothing).
        write_rolled_to(persona_dir, old_sid, new_sid)
        delete_session_buffer(persona_dir, old_sid)
        delete_cursor(persona_dir, old_sid)
        delete_backoff(persona_dir, old_sid)
        remove_session(old_sid)  # registry evict (no file delete — already handled)
        logger.info(
            "rollover: session=%s → %s mode=%s seeded=%d",
            old_sid, new_sid, seed_mode, len(reseeded),
        )
        return new_sid
    finally:
        release_compaction_lock(persona_dir, old_sid)


def maybe_weekly_rollover(
    persona_dir: Path,
    session_id: str,
    persona_name: str,
    *,
    weekly_age: timedelta,
    quiet_gap: timedelta,
    now: datetime | None = None,
    provider: LLMProvider | None = None,
    store=None,
    hebbian=None,
    embeddings=None,
    config: dict | None = None,
) -> str | None:
    """1c-B check, run on the daily tick AFTER the cascade fold. Fire the weekly
    swap iff the session age (oldest turn → now) ≥ ``weekly_age`` AND its last turn
    is older than ``quiet_gap`` (never mid-exchange). Returns the new sid on a swap,
    else None (defer)."""
    now = now or datetime.now(UTC)
    turns = read_session(persona_dir, session_id)
    raw = [t for t in turns if t.get("speaker") != "summary"]
    if not raw:
        return None

    oldest, newest = _ts_span(raw)
    if oldest is None or (now - oldest) < weekly_age:
        return None  # not old enough yet

    # Quiet-gap: defer if the last turn is within the quiet window (mid-exchange).
    if newest is None or (now - newest) < quiet_gap:
        return None

    return perform_rollover(
        persona_dir, session_id, persona_name,
        seed_mode="tiers_plus_tail", now=now, provider=provider,
        store=store, hebbian=hebbian, embeddings=embeddings, config=config,
    )


def _ts_span(raw: list[dict]) -> tuple[datetime | None, datetime | None]:
    """(oldest, newest) parsed ts across raw turns; (None, None) if none parse."""
    oldest: datetime | None = None
    newest: datetime | None = None
    for t in raw:
        raw_ts = t.get("ts")
        if not raw_ts:
            continue
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        if oldest is None or dt < oldest:
            oldest = dt
        if newest is None or dt > newest:
            newest = dt
    return oldest, newest
