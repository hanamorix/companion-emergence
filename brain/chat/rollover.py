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
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.chat.compaction import (
    build_compaction_provider,
    cascade_conversation,
    compact_conversation,
)
from brain.chat.session import create_session, registry_lock, remove_session
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

    # 2. Fold the old conversation into its post-compaction form. NOTE: this runs
    #    AFTER step-1 extraction advanced the cursor, so it is NOT redundant with
    #    the daily tick's cascade — the tick cascades BEFORE any extraction (cursor
    #    guard → folds nothing), so this post-extraction fold is what actually
    #    populates the tiers the seed re-reads. (Stage-6 L2 proposed removing it as
    #    "redundant"; the C9 real-tick test proves it load-bearing — reverted.)
    comp_provider = build_compaction_provider(persona_dir)
    if seed_mode == "summary_only":
        # Fold everything (a >24h-old "recent 40" is not recent).
        compact_conversation(
            persona_dir, old_sid,
            older_than=timedelta(0), fold_existing_summary=True,
            provider=comp_provider, min_keep_tail=0, now=now,
        )
    elif seed_mode == "tiers_plus_tail":
        # Bring the 3 tiers current (post-extraction), keeping the 40-msg raw tail.
        cascade_conversation(
            persona_dir, old_sid, provider=comp_provider, now=now,
            min_keep_tail=_ROLLOVER_TAIL,
        )
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown seed_mode {seed_mode!r}")

    # 3-6. Seed the new session + reap the old buffer. The compaction lock guards
    #      against a concurrent cascade; the registry lock (below) guards the
    #      seed-read↔pointer-write against a concurrent live-turn persist.
    if not acquire_compaction_lock(persona_dir, old_sid):
        return None  # busy → defer to the next cycle
    try:
        # The destructive critical section runs under registry_lock() — the SAME
        # lock brain.chat.session.persist_turns_following_successor holds for a live
        # turn persist. This closes the resolve-persist race by construction (stage-6
        # r4): a concurrent persist for old_sid either lands BEFORE the seed re-read
        # below (and is carried into the successor seed) or, once the rolled_to
        # pointer is written, is redirected into the live successor buffer — it can
        # never resurrect the deleted old buffer. Both the persist worker thread and
        # this supervisor thread are ordinary OS threads, so an RLock is the right
        # cross-thread primitive (the async in_flight_locks cannot span the boundary).
        #
        # Re-read the committed row under the lock (interposition-safe; apply_budget
        # may have touched the 24h tier between fold and here — and a persist that
        # slipped in before us is now visible and captured).
        with registry_lock():
            turns = read_session(persona_dir, old_sid)
            summary_row, raw = _split_summary_and_raw(turns)
            if seed_mode == "summary_only":
                seed_rows: list[dict] = [summary_row] if summary_row else []
            else:
                tail = raw[-_ROLLOVER_TAIL:]
                seed_rows = ([summary_row] if summary_row else []) + tail
            if not seed_rows:
                # Nothing to seed → abort (nothing lost; old buffer stays).
                return None

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

            # Successor pointer goes down FIRST — under the lock — so from this
            # instant any resolve OR persist of old_sid redirects to the successor
            # (C-1 resolve-side redirect + the persist-side redirect above). Then
            # evict the stale registry entry. get_or_hydrate_session consults the
            # pointer before the registry regardless; evict-before-delete just clears
            # the stale entry promptly as belt-and-braces.
            write_rolled_to(persona_dir, old_sid, new_sid)
            remove_session(old_sid)  # registry evict (no file delete — handled next)

        # Reap the old buffer/cursor/backoff OUTSIDE the registry lock: the pointer
        # is durably in place, so every resolve/persist of old_sid already redirects
        # to the successor and nothing can append to the old buffer anymore — the
        # delete only reclaims the now-orphaned file. Keeping the (possibly
        # retry-looping) unlink out of the lock avoids stalling registry ops on
        # other sessions.
        delete_session_buffer(persona_dir, old_sid)
        delete_cursor(persona_dir, old_sid)
        delete_backoff(persona_dir, old_sid)
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
    is_session_busy: Callable[[str], bool] | None = None,
) -> str | None:
    """1c-B check — the weekly swap fires ONLY when the session is IDLE, the same as
    every other automatic tick (owner ruling 2026-08-13). Two idle conditions must
    both hold: (1) the last completed turn is older than ``quiet_gap`` (user-quiet,
    never mid-exchange) AND (2) there is NO in-flight request for the session
    (``is_session_busy`` belt). Fires iff session age ≥ ``weekly_age`` AND both idle
    conditions hold. Returns the new sid on a swap, else None (defer).

    These two checks are a best-effort EFFICIENCY/UX belt — they avoid swapping a
    session out from under an active user. They are NOT what makes the swap safe: the
    resolve-persist race is closed inside ``perform_rollover``, whose destructive
    section holds ``registry_lock()`` across its seed re-read → ``rolled_to`` pointer
    write, serializing it against a concurrent live-turn persist (which takes the same
    lock). So even if a request registers in the check-then-act gap after (2) passed,
    its turn is captured into the successor seed or redirected to the successor buffer
    — never orphaned onto a deleted old buffer."""
    now = now or datetime.now(UTC)
    turns = read_session(persona_dir, session_id)
    raw = [t for t in turns if t.get("speaker") != "summary"]
    if not raw:
        return None

    oldest, newest = _ts_span(raw)
    if oldest is None or (now - oldest) < weekly_age:
        return None  # not old enough yet

    # Idle condition 1 (user-quiet): defer if the last turn is within the quiet
    # window (mid-exchange).
    if newest is None or (now - newest) < quiet_gap:
        return None

    # Idle condition 2 (best-effort belt): defer if a request is in-flight for this
    # session. A multi-second tool-loop can be active even when the last COMPLETED
    # turn is old (a request that started after a long quiet gap), so the quiet-gap
    # alone does not catch it — the in-flight check does. This is an efficiency/UX
    # guard (don't swap under an active user), NOT the race-safety mechanism: the
    # check-then-act gap between here and the buffer delete is closed structurally by
    # registry_lock() inside perform_rollover (see its destructive section).
    # Best-effort cross-thread read of the async in_flight_locks.
    if is_session_busy is not None and is_session_busy(session_id):
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
