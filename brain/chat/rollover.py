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

Every raw turn of the old buffer is carried into the seed or archived: the
cascade archives only extracted turns older than 24h, the 1c-B seed carries only
the last 40, so the rollover archives, durably and before it creates the
successor, the raw turns it does not carry (#556), and then drops those rows
from the old buffer so no turn is in both. The old buffer's append lock
(``session_append_lock``) is held from the final read to the unlink, so a row
appended in between cannot be deleted unseen.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.chat.compaction import (
    build_compaction_provider,
    cascade_conversation,
    compact_conversation,
    summary_gen,
)
from brain.chat.session import create_session, registry_lock, remove_session
from brain.ingest.buffer import (
    acquire_compaction_lock,
    archive_once,
    delete_backoff,
    delete_cursor,
    delete_session_buffer,
    read_cursor,
    read_session,
    release_compaction_lock,
    rewrite_session_atomic,
    session_append_lock,
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


def _extraction_allows_rollover(
    persona_dir: Path,
    old_sid: str,
    *,
    store,
    hebbian,
    provider,
    config: dict | None,
) -> bool:
    """Run the final extraction of the old buffer; True only when the rollover may go on to fold and DELETE it (#477, #473, #523).

    The extraction runs to the end of the buffer, chunk by chunk (no rollover bounds it). It may go on when that extraction
    completed (the cursor reached the buffer's end), or when the session's extraction has failed persistently: the failing chunk
    is set aside in ``poison/`` (kept, never lost), and the extraction then runs once more over the chunks after it, which were
    never tried; the rollover goes on only if that pass completes. It may not, and the buffer stays for the next cycle, when:
    * another pass is running (busy) or the model call was refused as "not now" (deferred; also when it is RAISED): neither is a
      failure and neither is ever counted. A refusal because the bridge is stopping (``ShutdownDeferred``, #456) is
      re-raised instead, so the caller's job ends with its cadence unchanged;
    * the pass returned inside the backoff window without calling the provider (``backoff_skipped``): no attempt was made, so
      nothing is counted and nothing is set aside;
    * the extraction failed (a failed pass, or an ordinary exception out of it, which is recorded as a failure like a failed pass)
      and the session is still below the 3-failure threshold the stale sweep uses;
    * it is at the threshold but the failing chunk could not be set aside, or another pass holds the extraction lock;
    * the extraction after the set-aside did not complete (deferred, failed again, busy, or raised): the remaining chunks are
      extractable and stay in the buffer for the next cycle.
    """
    from brain.bridge.cli_throttle import ShutdownDeferred, ThrottleDeferred
    from brain.ingest.pipeline import (
        extract_session_snapshot,
        extraction_failures_exhausted,
        poison_unextracted_span,
        record_extraction_failure,
    )

    try:
        report = extract_session_snapshot(
            persona_dir, old_sid, store=store, hebbian=hebbian, provider=provider, config=config
        )
    except ShutdownDeferred:
        # #456 (C38 d): the bridge is stopping: the refusal ends the caller's job (the compaction tick re-raises it, so
        # its cadence is not advanced); the old buffer is kept.
        raise
    except ThrottleDeferred:
        logger.info("rollover: extraction deferred session=%s; keeping the old buffer", old_sid)
        return False
    except Exception:
        # An ordinary exception out of the extraction (a locked store, a malformed turn) counts like a failed pass, so a
        # persistent one reaches the same threshold instead of holding the buffer forever.
        logger.exception("rollover: extraction raised session=%s; counted as a failure, keeping the old buffer", old_sid)
        try:
            record_extraction_failure(persona_dir, old_sid)
        except Exception:
            logger.exception("rollover: could not record the failure session=%s", old_sid)
            return False
    else:
        if report.completed:
            return True
        if report.busy or report.deferred or report.backoff_skipped:
            logger.info(
                "rollover: extraction did not run to the end session=%s busy=%s deferred=%s backoff_skipped=%s; keeping the old buffer",
                old_sid, report.busy, report.deferred, report.backoff_skipped,
            )
            return False
        # a real attempt that failed: its failure is already recorded in the sidecar by the pass
    if not extraction_failures_exhausted(persona_dir, old_sid):
        logger.info("rollover: extraction failed session=%s below the poison threshold; keeping the old buffer", old_sid)
        return False
    try:
        set_aside = poison_unextracted_span(persona_dir, old_sid)
    except Exception:
        logger.exception("rollover: could not set the failing chunk aside session=%s; keeping the old buffer", old_sid)
        return False
    if set_aside is None:
        logger.info("rollover: another extraction holds the session session=%s; keeping the old buffer", old_sid)
        return False
    # #523: only the failing chunk was set aside; the chunks after it were never tried. Extract them before the fold and
    # the delete, and go on only if that pass reaches the end.
    try:
        again = extract_session_snapshot(
            persona_dir, old_sid, store=store, hebbian=hebbian, provider=provider, config=config
        )
    except ShutdownDeferred:
        raise
    except ThrottleDeferred:
        logger.info("rollover: extraction after the set-aside deferred session=%s; keeping the old buffer", old_sid)
        return False
    except Exception:
        logger.exception("rollover: extraction after the set-aside raised session=%s; counted as a failure, keeping the old buffer", old_sid)
        try:
            record_extraction_failure(persona_dir, old_sid)
        except Exception:
            logger.exception("rollover: could not record the failure session=%s", old_sid)
        return False
    if again.completed:
        return True
    logger.info(
        "rollover: extraction after the set-aside did not reach the end session=%s busy=%s deferred=%s errors=%d; keeping the old buffer",
        old_sid, again.busy, again.deferred, again.errors,
    )
    return False


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
    config: dict | None = None,
) -> str | None:
    """Roll ``old_sid`` over into a fresh session. Returns the new session id, or
    None if there is nothing to seed / the session is busy.

    ``seed_mode`` is ``"summary_only"`` (1c-A) or ``"tiers_plus_tail"`` (1c-B).
    Memory extraction of the old buffer runs first when ``store`` + ``hebbian`` are supplied: every
    chunk after the cursor, to the end (#523; a long unextracted buffer makes one model call per
    chunk), and the rollover goes on only if it completed (a busy, deferred or failed extraction
    returns None with the buffer kept and the cursor after the last committed chunk). The seed is
    the immediate recap, memory is long-term recall.

    The raw turns the seed does not carry (in ``tiers_plus_tail``, those before the last 40 that the cascade
    has not archived) are archived, fsynced, before the successor, the pointer or the delete exist. An archive
    failure raises before any of those: the old buffer, its cursor and the session list stay as they were. Once
    the archive is durable the archived rows are dropped from the old buffer, so a failure at any later step
    leaves no turn in both files and a retry (whatever changed meanwhile) archives nothing twice; until a rollover
    succeeds the still-live old session shows only its summary and the carried tail. The repeats left: a crash or
    an error between the archive append and its commit marker, or between the marker and that buffer rewrite, may
    archive the batch again on a later run (an immediate retry is skipped by the marker); a turn is never missing.
    The old buffer's append lock is held from the final read to the unlink.
    """
    now = now or datetime.now(UTC)
    persona_dir = Path(persona_dir)

    # 1. Final memory extraction of the old buffer. Advances the
    #    cursor so the carried tail's extraction state is current (C18).
    if store is not None and hebbian is not None and provider is not None:
        if not _extraction_allows_rollover(
            persona_dir, old_sid, store=store, hebbian=hebbian, provider=provider, config=config
        ):
            return None

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
        with ExitStack() as buffer_guard:
            with registry_lock():
                # The old buffer's append lock is taken inside the registry lock (the order a live persist
                # uses: registry lock, then the append lock) and held past the registry lock's exit, to the
                # unlink below: a row an appender that does not go through the registry adds between this
                # read and the unlink waits instead of being deleted unseen.
                buffer_guard.enter_context(session_append_lock(persona_dir, old_sid))
                turns = read_session(persona_dir, old_sid)
                summary_row, raw = _split_summary_and_raw(turns)
                not_carried: list[dict] = []  # raw turns the seed leaves out; archived below
                if seed_mode == "summary_only":
                    if summary_row is None:
                        # No summary produced (the fold folded nothing / the content is
                        # not foldable) → nothing to seed. Abort below and leave the old
                        # buffer INTACT — a stale session that can't be summarised is not
                        # attach-eligible, and any turn sitting in the buffer stays there
                        # (not lost). Preserves the pre-existing "skip stale" behaviour.
                        seed_rows: list[dict] = []
                    else:
                        # A summary EXISTS, so the rollover WILL proceed and delete the old
                        # buffer. Carry the summary PLUS any raw turns still present after
                        # the fold: the fold (min_keep_tail=0, older_than=0) normally
                        # empties raw, so this is usually just [summary]; but a live turn
                        # that raced in after the fold (before this seed re-read under the
                        # lock) — or a turn the cursor-guarded fold left unfolded — is a
                        # genuinely FRESH turn. Dropping it while deleting the old buffer
                        # would lose it permanently (never folded, never archived) (r4).
                        seed_rows = [summary_row] + raw
                else:
                    tail = raw[-_ROLLOVER_TAIL:]
                    seed_rows = ([summary_row] if summary_row else []) + tail
                    # The raw turns before the tail that the cascade did not archive (it archives only
                    # extracted turns older than 24h) are in neither the seed nor the archive, and the
                    # delete below would be their end (#556).
                    not_carried = raw[:-_ROLLOVER_TAIL]
                if not seed_rows:
                    # Nothing to seed → abort (nothing lost; old buffer stays).
                    return None

                # Archive what the seed does not carry BEFORE the old buffer, the successor or the pointer change:
                # the archive write is fsynced (and de-duplicated across an immediate retry by its commit marker),
                # so a failure here raises with the old buffer, its cursor and the session list untouched, and a
                # crash after it leaves every turn in the buffer or the archive.
                if not_carried:
                    written = archive_once(persona_dir, old_sid, not_carried, new_gen=summary_gen(summary_row))
                    if written <= 0:
                        raise OSError("archive append wrote zero bytes")
                    # Only now that the batch is durable, drop it from the old buffer (the append lock is held, so
                    # `turns` is the whole buffer): a later step that raises leaves no row in both the buffer and the
                    # archive, so a retry, or the daily tick's cascade on this still-live session, cannot archive it
                    # again whatever changed in between.
                    archived = {id(t) for t in not_carried}
                    rewrite_session_atomic(persona_dir, old_sid, [t for t in turns if id(t) not in archived])

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
            # to the successor, so no live-turn persist reaches the old buffer anymore — the
            # delete only reclaims the now-orphaned file. Keeping the (possibly
            # retry-looping) unlink out of the registry lock avoids stalling registry ops on
            # other sessions; the old buffer's own append lock is still held, so an appender
            # that bypasses the registry waits for the unlink instead of racing it.
            delete_session_buffer(persona_dir, old_sid)
            delete_cursor(persona_dir, old_sid)
            delete_backoff(persona_dir, old_sid)
        logger.info(
            "rollover: session=%s → %s mode=%s seeded=%d archived=%d",
            old_sid, new_sid, seed_mode, len(reseeded), len(not_carried),
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

    new_sid = perform_rollover(
        persona_dir, session_id, persona_name,
        seed_mode="tiers_plus_tail", now=now, provider=provider,
        store=store, hebbian=hebbian, config=config,
    )

    # #231 RERANKER RE-ARCHITECTURE (2026-09-10): the Stage-4 per-persona
    # semantic floor/gap recalibration pass that used to run here (spec
    # decision 5's "Calibration" bullet, brain/memory/semantic_calibration.py)
    # is REMOVED — the cold red-team proved deriving a cosine floor/gap from
    # the corpus's own inter-memory spread doesn't generalize. Replaced by a
    # cross-encoder reranker + a floor gate (brain/memory/semantic_recall.py's
    # `select_standouts`) — originally a fixed module constant, cut over by
    # F2a inc8 (#250 §7/§8) to a floor the DAILY calibration tick derives and
    # persists (brain/memory/floor_calibration.py), so this rollover path
    # still needs no per-persona recalibration pass of its own here.

    return new_sid


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
