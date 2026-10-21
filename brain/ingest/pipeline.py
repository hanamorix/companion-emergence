"""SP-4 pipeline driver: extracts a session's conversation rows into memories.

close_session()             extract every row after the cursor, chunk by chunk, then delete the buffer, cursor and backoff.
extract_session_snapshot()  non-destructive: extract the rows after the cursor sidecar, chunk by chunk; the buffer stays.
close_stale_sessions()      destructive sweep (close_session for every silent session); for explicit close callers only.
snapshot_stale_sessions()   non-destructive sweep of the supervisor's session-snapshot job, one chunk per session per pass.
finalize_stale_sessions()   24h extraction-only sweep, one chunk per session per run; dead-letters a buffer that keeps failing.

Stage flow of one pass:
  BUFFER  → read the rows after the cursor from <persona_dir>/active_conversations/<session_id>.jsonl (summary rows dropped)
  CHUNK   → split them, oldest first, into chunks of whole rows no larger than ``ingest.extraction_chunk_chars``
            (brain.ingest.ingest_utils.chunk_turns, #523)
  then per chunk, in ``run_chunk_stages``:
  EXTRACT → format the chunk's transcript, call the LLM, get ExtractedItems
  SCORE   → normalize each item (label coercion, importance clamp, text strip)
  DEDUPE  → cosine similarity against row/warm-matrix vectors (always attempted, fail-soft)
  COMMIT  → direct write to MemoryStore + auto-Hebbian
  SOUL    → queue high-importance items to soul_candidates.jsonl
  and after a chunk whose commits all succeeded, the cursor moves to that chunk's end, so a pass that stops early (a
  deferral, a failure, a chunk limit) resumes at the next chunk and no row is skipped.
  LOG     → emit structured log events with counts
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.cli_throttle import ThrottleDeferred
from brain.bridge.provider import LLMProvider
from brain.health.json_io import append_jsonl
from brain.ingest.buffer import (
    delete_backoff,
    delete_cursor,
    delete_session_buffer,
    extract_lock_path,
    list_active_sessions,
    read_backoff,
    read_cursor,
    read_session,
    read_session_after,
    session_silence_minutes,
    write_backoff,
    write_cursor,
)
from brain.ingest.commit import commit_item
from brain.ingest.dedupe import DEFAULT_DEDUP_THRESHOLD, is_duplicate
from brain.ingest.extract import extract_items_with_status
from brain.ingest.ingest_utils import (
    chunk_turns,
    extraction_chunk_chars,
    load_emotion_vocab,
    load_user_name,
    load_user_pronouns,
)
from brain.ingest.soul_queue import DEFAULT_SOUL_THRESHOLD, queue_soul_candidate
from brain.ingest.types import IngestReport
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import GATE_BYPASS_TYPES
from brain.memory.store import MemoryStore
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import count_throttled

logger = logging.getLogger(__name__)


# F-011 — extraction backoff knobs. After this many recorded extraction
# failures for a given session (record_extraction_failure), pause
# extraction for `_BACKOFF_WINDOW_MINUTES` so we stop burning LLM calls
# on a wedged buffer. Success clears the state.
_BACKOFF_FAILURE_THRESHOLD = 3
_BACKOFF_WINDOW_MINUTES = 10.0


def extraction_failures_exhausted(persona_dir: Path, session_id: str) -> bool:
    """True when the session's recorded extraction failures reached the poison threshold (the same one the stale sweep uses)."""
    state = read_backoff(persona_dir, session_id)
    return state is not None and state["failures"] >= _BACKOFF_FAILURE_THRESHOLD


def record_extraction_failure(persona_dir: Path, session_id: str) -> None:
    """Record one failed extraction attempt in the session's backoff sidecar (F-011).

    The first failure in a new window anchors ``first_failure_at`` to now; later ones increment the count without moving the anchor,
    so the window measures from the *first* failure. Three of them move a silent buffer to ``poison/`` (the stale sweep) and let the
    rollover set the failing chunk aside. Never call it for a deferral or a busy pass: those are not failures.
    """
    existing = read_backoff(persona_dir, session_id)
    if existing is None:
        write_backoff(persona_dir, session_id, failures=1, first_failure_at=datetime.now(UTC).isoformat())
    else:
        write_backoff(
            persona_dir,
            session_id,
            failures=existing["failures"] + 1,
            first_failure_at=existing["first_failure_at"],
        )


def poison_unextracted_span(persona_dir: Path, session_id: str) -> int | None:
    """Set aside the chunk whose extraction keeps failing in ``active_conversations/poison/<sid>.span.jsonl`` and move the cursor
    past it, so the rest of the buffer can be extracted and the buffer folded and reaped (#477, #523).

    Only the FIRST chunk after the cursor is set aside: it is the one whose failures reached the threshold (a pass stops at its
    first failing chunk, and the chunking is deterministic for the same rows and limit). The chunks after it were never tried
    and stay in the buffer after the moved cursor for the next pass. The stale sweep moves a whole buffer to ``poison/`` after
    the same 3 recorded failures, but only once the session has been silent for 24 hours; a session in daily use with a
    persistent extraction failure would otherwise grow without bound once the rollover refuses to reap unextracted turns. The
    rows are KEPT (appended verbatim, never silently lost) and the failure record is cleared.

    Runs under the session's non-blocking extraction lock: returns None when another pass holds it (busy: nothing is changed and the
    caller must not reap); otherwise how many rows were set aside. Raises when they cannot be written, in which case nothing has
    been changed. A failing CURSOR write happens after the rows were appended, so they may then exist twice after a retry
    (duplicates, never a loss); the caller still must not reap.
    """
    with file_lock(extract_lock_path(persona_dir, session_id), blocking=False) as acquired:
        if not acquired:
            return None
        cursor = read_cursor(persona_dir, session_id)
        turns = [t for t in read_session_after(persona_dir, session_id, cursor) if t.get("speaker") != "summary"]
        if not turns:
            delete_backoff(persona_dir, session_id)
            return 0
        failing = chunk_turns(
            turns,
            max_chars=extraction_chunk_chars(),
            user_name=load_user_name(persona_dir),
            assistant_name=persona_dir.name,
        )[0]
        poison_dir = persona_dir / "active_conversations" / "poison"
        poison_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(poison_dir / f"{session_id}.span.jsonl", list(failing.turns), fsync=True)
        if failing.cursor_ts:
            write_cursor(persona_dir, session_id, failing.cursor_ts)
        delete_backoff(persona_dir, session_id)
        logger.warning(
            "conversation_span_poisoned session=%s turns=%d: extraction of this chunk kept failing; the turns are kept in poison/",
            session_id,
            len(failing.turns),
        )
        return len(failing.turns)


class ChunkResult(enum.Enum):
    """What ``run_chunk_stages`` did with one chunk."""

    OK = "ok"  # extracted, and every item deduped or committed
    DEFERRED = "deferred"  # the model call was refused as "not now" (#473): nothing tried, nothing failed
    FAILED = "failed"  # the extraction failed (provider or parse failures exhausted the retries)
    COMMIT_FAILED = "commit_failed"  # extracted, but at least one item could not be committed


def run_chunk_stages(
    persona_dir: Path,
    session_id: str,
    transcript: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    cfg: dict,
    report: IngestReport,
    user_name: str | None,
    assistant_name: str | None,
    valid_emotions: set[str],
    emotion_vocab: str,
    user_pronouns: dict | None,
    queue_soul: bool = True,
    extra_metadata: dict | None = None,
    created_at: datetime | None = None,
) -> ChunkResult:
    """EXTRACT → SCORE → DEDUPE → COMMIT → SOUL for one chunk's transcript; the one copy of these stages (#523).

    Called per chunk by the snapshot, by ``close_session`` and by the extraction backfill. Adds its counts to ``report``
    (``chunks`` += 1 for the extraction call). Never touches the cursor, the buffer or the backoff sidecar: the caller decides
    what the result means for them.

    ``cfg`` knobs: ``extraction_max_retries`` (1), ``dedup_threshold`` (0.88), ``crystallize_threshold`` (8).
    ``queue_soul``: queue high-importance items as soul candidates (the backfill turns it off under ``--all``).
    ``extra_metadata`` / ``created_at``: passed to ``commit_item`` (the backfill's source marker and back-date).
    A bridge that is stopping (``ShutdownDeferred``) is raised out of the extraction, not returned.
    """
    report.chunks += 1
    extraction = extract_items_with_status(
        transcript,
        provider=provider,
        max_retries=int(cfg.get("extraction_max_retries", 1)),
        user_name=user_name,
        assistant_name=assistant_name,
        emotion_vocab=emotion_vocab,
        user_pronouns=user_pronouns,
        persona_dir=persona_dir,
    )
    if extraction.deferred:
        return ChunkResult.DEFERRED
    if extraction.failed:
        report.errors += 1
        logger.warning(
            "conversation_extract_failed session=%s chars=%d error=%s",
            session_id,
            len(transcript),
            extraction.error or "unknown extraction failure",
        )
        return ChunkResult.FAILED

    items = [it.normalize(valid_emotions=valid_emotions, source=f"session={session_id}") for it in extraction.items if it.text]
    report.extracted += len(items)

    dedup_threshold = float(cfg.get("dedup_threshold", DEFAULT_DEDUP_THRESHOLD))
    crystallize_threshold = int(cfg.get("crystallize_threshold", DEFAULT_SOUL_THRESHOLD))
    commit_failures_before = report.commit_failures

    for item in items:
        # DEDUPE
        if is_duplicate(item.text, store=store, threshold=dedup_threshold):
            report.deduped += 1
            continue

        # COMMIT
        mem_id = commit_item(
            item, session_id=session_id, store=store, hebbian=hebbian, extra_metadata=extra_metadata, created_at=created_at
        )
        if mem_id is None:
            report.errors += 1
            report.commit_failures += 1
            continue

        # #167: route_write() (called inside commit_item) returns a non-None id
        # from BOTH its durable (store.create) and queue-only (PendingQueue.enqueue)
        # branches, so mem_id alone can't tell them apart. Re-derive the same
        # classification route_write itself used: it branches on
        # mem.memory_type in GATE_BYPASS_TYPES, and commit_item always sets
        # memory_type=item.label, so item.label in GATE_BYPASS_TYPES is the
        # identical predicate. Real conversation-extraction labels are always
        # in VALID_LABELS, which is disjoint from GATE_BYPASS_TYPES, so this
        # branch is a correctness/consistency guarantee rather than something
        # conversation-ingest can trigger today — see IngestReport.committed's
        # docstring.
        if item.label in GATE_BYPASS_TYPES:
            report.committed += 1
        else:
            report.enqueued += 1
        report.memory_ids.append(mem_id)

        # SOUL
        if queue_soul and item.importance >= crystallize_threshold:
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

    return ChunkResult.COMMIT_FAILED if report.commit_failures > commit_failures_before else ChunkResult.OK


def _run_chunks(
    persona_dir: Path,
    session_id: str,
    turns: list[dict],
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    cfg: dict,
    report: IngestReport,
    record_failures: bool,
    max_chunks: int | None,
) -> None:
    """Extract ``turns`` (the non-summary rows after the cursor) chunk by chunk, oldest first, under the caller's extraction lock.

    After each chunk whose commits all succeed the cursor moves to that chunk's end (``chunk.cursor_ts``), so a pass that stops
    resumes at the next chunk with no row skipped or repeated. Stops at the first chunk that is deferred (``report.deferred``,
    nothing counted, #473), fails or has a commit failure (the cursor stays before that chunk), or when ``max_chunks`` chunks
    were run with chunks left. Sets ``report.reached_end`` only when every chunk committed.

    ``record_failures`` (the snapshot): a failed chunk or a commit failure is recorded through ``record_extraction_failure``
    (F-011), and a committed chunk clears the backoff sidecar. ``close_session`` passes False: it keeps its rule of no
    backoff accounting.
    """
    user_name = load_user_name(persona_dir)
    assistant_name = persona_dir.name
    valid_emotions, emotion_vocab = load_emotion_vocab()
    user_pronouns = load_user_pronouns(persona_dir)
    max_chars = extraction_chunk_chars()
    chunks = chunk_turns(turns, max_chars=max_chars, user_name=user_name, assistant_name=assistant_name)
    for index, chunk in enumerate(chunks):
        if max_chunks is not None and index >= max_chunks:
            return
        if chunk.oversize:
            count_throttled(
                logger,
                "conversation_extract_oversize_unit",
                logging.WARNING,
                "conversation_extract_oversize_unit session=%s chars=%d max_chars=%d: one exchange longer than the limit, sent whole",
                session_id,
                len(chunk.text),
                max_chars,
            )
        result = run_chunk_stages(
            persona_dir,
            session_id,
            chunk.text,
            store=store,
            hebbian=hebbian,
            provider=provider,
            cfg=cfg,
            report=report,
            user_name=user_name,
            assistant_name=assistant_name,
            valid_emotions=valid_emotions,
            emotion_vocab=emotion_vocab,
            user_pronouns=user_pronouns,
        )
        if result is ChunkResult.DEFERRED:
            # #473: "not now" is not a failure. No error, no backoff entry (three of those move the buffer to poison/),
            # the cursor stays after the last committed chunk, and the next pass tries again from there.
            report.deferred = True
            logger.info(
                "conversation_extract_deferred session=%s chunk=%d/%d; cursor unchanged from here", session_id, index + 1, len(chunks)
            )
            return
        if result is not ChunkResult.OK:
            # Extraction failed, or a commit failed: the cursor stays before this chunk, so the next pass retries it and
            # never skips its un-committed rows.
            if result is ChunkResult.COMMIT_FAILED:
                logger.warning(
                    "conversation_extract_commit_failed session=%s chunk=%d/%d commit_failures=%d; cursor unchanged from here",
                    session_id,
                    index + 1,
                    len(chunks),
                    report.commit_failures,
                )
            if record_failures:
                record_extraction_failure(persona_dir, session_id)
            return
        if record_failures:
            # F-011: success clears any prior backoff state so the next failure starts a fresh window.
            delete_backoff(persona_dir, session_id)
        if chunk.cursor_ts:
            try:
                write_cursor(persona_dir, session_id, chunk.cursor_ts)
            except (OSError, ValueError):
                logger.warning("conversation_extract_cursor_write_failed session=%s ts=%s", session_id, chunk.cursor_ts)
    report.reached_end = True
    if len(chunks) > 1:
        logger.info(
            "conversation_extract_chunked session=%s chunks=%d chars=%d",
            session_id,
            report.chunks,
            sum(len(c.text) for c in chunks),
        )


def close_session(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
) -> IngestReport:
    """Close one session under its extraction lock (#477): a close that finds another extraction running skips (``busy``).

    Without the lock a close extracts and deletes the buffer while the stale sweep, a rollover or the snapshot route is
    mid-extraction of the same turns: both commit them, and the snapshot then writes a cursor for a deleted session.
    """
    with file_lock(extract_lock_path(persona_dir, session_id), blocking=False) as acquired:
        if not acquired:
            logger.info("conversation_close_busy session=%s: another extraction is running", session_id)
            return IngestReport(session_id=session_id, busy=True)
        return _close_session_locked(
            persona_dir, session_id, store=store, hebbian=hebbian, provider=provider, config=config
        )


def _row_key(turn: dict) -> tuple:
    return (turn.get("ts"), turn.get("speaker"), turn.get("text"))


def _close_session_locked(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
) -> IngestReport:
    """Extract every row after the session's cursor, chunk by chunk, then delete its buffer, cursor and backoff.

    Reads only the rows after the cursor (summary rows dropped), like the snapshot, so a buffer a snapshot already extracted
    in part is not extracted twice, and runs the chunk loop to the end with a cursor write per committed chunk: a close that
    stops (a deferral #473, a failed chunk, a commit failure) keeps the buffer, and the next close resumes after the last
    committed chunk. No backoff accounting here (unchanged rule): retries of a held buffer are the snapshot sweep's.

    Before deleting, the rows after the cursor are read again: a row that arrived during the pass (it was never extracted)
    keeps the buffer, logged ``conversation_close_late_rows``, and the report does not reach the end. Nothing after the
    cursor at the start: the buffer is deleted and the report reaches the end.

    ``config`` knobs: see ``run_chunk_stages``.
    """
    cfg = config or {}
    report = IngestReport(session_id=session_id)

    cursor = read_cursor(persona_dir, session_id)
    turns = [t for t in read_session_after(persona_dir, session_id, cursor) if t.get("speaker") != "summary"]
    if turns:
        _run_chunks(
            persona_dir,
            session_id,
            turns,
            store=store,
            hebbian=hebbian,
            provider=provider,
            cfg=cfg,
            report=report,
            record_failures=False,
            max_chunks=None,
        )
        logger.info(
            "conversation_ingested session=%s turns=%d chunks=%d extracted=%d committed=%d enqueued=%d "
            "deduped=%d soul_candidates=%d soul_queue_errors=%d errors=%d",
            session_id,
            len(turns),
            report.chunks,
            report.extracted,
            report.committed,
            report.enqueued,
            report.deduped,
            report.soul_candidates,
            report.soul_queue_errors,
            report.errors,
        )
    else:
        report.reached_end = True

    if not report.reached_end:
        logger.warning(
            "conversation_close_held session=%s deferred=%s errors=%d commit_failures=%d; buffer retained, cursor after the "
            "last committed chunk",
            session_id,
            report.deferred,
            report.errors,
            report.commit_failures,
        )
        return report

    # S2 (#523): the pass took one or more model calls; a row persisted meanwhile is after the cursor and was never
    # extracted. The rows the pass itself extracted are excluded by identity, so a failed cursor write cannot hold a buffer.
    seen = {_row_key(t) for t in turns}
    late = [
        t
        for t in read_session_after(persona_dir, session_id, read_cursor(persona_dir, session_id))
        if t.get("speaker") != "summary" and _row_key(t) not in seen
    ]
    if late:
        report.reached_end = False
        logger.warning(
            "conversation_close_late_rows session=%s rows=%d: rows arrived during the close; buffer retained", session_id, len(late)
        )
        return report

    delete_session_buffer(persona_dir, session_id)
    delete_cursor(persona_dir, session_id)
    delete_backoff(persona_dir, session_id)
    return report


def extract_session_snapshot(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
    max_chunks: int | None = None,
) -> IngestReport:
    """Run one snapshot extraction of ``session_id``; one extraction per session at a time (#477).

    The periodic sweep, the session rollover, the bridge's snapshot route and the CLI can all
    reach the same session together. Two concurrent passes extract the same turns twice and race
    on the cursor write, so a pass that finds another one running skips: the running pass
    advances the cursor, and the next sweep picks up anything it did not cover. The skip is not
    an error and counts for nothing, and the report says so (``busy``), so a caller that acts on the
    result (the rollover) can tell it from a pass that had nothing to do.

    ``max_chunks`` (#523): at most this many chunks (model calls) in this pass; None runs to the end. Callers that do not
    need the whole span extracted (the sweep, finalize, the snapshot route) pass 1, so they make no more calls per run than
    before chunking; the rest is taken by the next pass. The extraction lock is held across the whole pass.
    """
    with file_lock(extract_lock_path(persona_dir, session_id), blocking=False) as acquired:
        if not acquired:
            logger.info("conversation_snapshot_busy session=%s: another extraction is running", session_id)
            return IngestReport(session_id=session_id, busy=True)
        return _extract_session_snapshot_locked(
            persona_dir,
            session_id,
            store=store,
            hebbian=hebbian,
            provider=provider,
            config=config,
            max_chunks=max_chunks,
        )


def _extract_session_snapshot_locked(
    persona_dir: Path,
    session_id: str,
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
    max_chunks: int | None = None,
) -> IngestReport:
    """Extract the rows after the session's cursor, chunk by chunk, without deleting the buffer.

    The cursor sidecar records how far extraction got: after each chunk whose commits all succeed it moves to that chunk's
    end. A deferral stops the pass with nothing counted (#473); a failed chunk or a commit failure stops it with one failure
    recorded in the backoff sidecar (F-011) and the cursor before that chunk, so the next pass retries it. ``max_chunks``:
    see ``extract_session_snapshot``. The report reaches the end only when no row is left after the cursor.

    Returns an IngestReport with the same shape as close_session.
    """
    cfg = config or {}
    report = IngestReport(session_id=session_id)

    # F-011 — backoff guard: if we've failed N consecutive times within the
    # window, skip extraction entirely until the window expires. Stops the
    # snapshot sweep from burning LLM calls on a wedged buffer.
    backoff_state = read_backoff(persona_dir, session_id)
    if backoff_state is not None and backoff_state["failures"] >= _BACKOFF_FAILURE_THRESHOLD:
        first_failure_dt = datetime.fromisoformat(
            backoff_state["first_failure_at"].replace("Z", "+00:00")
        )
        if first_failure_dt.tzinfo is None:
            first_failure_dt = first_failure_dt.replace(tzinfo=UTC)
        elapsed_min = (datetime.now(UTC) - first_failure_dt).total_seconds() / 60.0
        if elapsed_min < _BACKOFF_WINDOW_MINUTES:
            logger.info(
                "conversation_snapshot_backoff session=%s failures=%d elapsed_min=%.2f",
                session_id,
                backoff_state["failures"],
                elapsed_min,
            )
            report.errors += 1
            report.backoff_skipped = True  # no provider call was made: not a failed attempt
            return report

    cursor = read_cursor(persona_dir, session_id)
    turns = read_session_after(persona_dir, session_id, cursor)
    # Drop compaction `summary` blocks before chunking AND before any cursor
    # advance — a faded summary must never be re-extracted as memory, and
    # its compaction-time ts must not move the ingest cursor past un-extracted
    # turns. Only the new summary type is excluded (arbitrary name-speakers
    # remain valid conversation turns).
    turns = [t for t in turns if t.get("speaker") != "summary"]

    if not turns:
        logger.info(
            "conversation_snapshot_empty session=%s cursor=%s",
            session_id,
            cursor,
        )
        report.reached_end = True
        return report

    _run_chunks(
        persona_dir,
        session_id,
        turns,
        store=store,
        hebbian=hebbian,
        provider=provider,
        cfg=cfg,
        report=report,
        record_failures=True,
        max_chunks=max_chunks,
    )

    logger.info(
        "conversation_snapshot session=%s turns=%d chunks=%d reached_end=%s extracted=%d committed=%d enqueued=%d "
        "deduped=%d soul_candidates=%d cursor=%s",
        session_id,
        len(turns),
        report.chunks,
        report.reached_end,
        report.extracted,
        report.committed,
        report.enqueued,
        report.deduped,
        report.soul_candidates,
        read_cursor(persona_dir, session_id),
    )
    return report


def snapshot_stale_sessions(
    persona_dir: Path,
    *,
    silence_minutes: float = 10.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
    should_pause: Callable[[], bool] | None = None,
    paused_out: list[bool] | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; snapshot any whose last turn is past silence_minutes, ONE chunk per session per call.

    Bounded per pass (#523): each session's snapshot makes at most one extraction call (``max_chunks=1``), as before
    chunking. A longer backlog keeps the session-snapshot job's "has work" probe true (rows left after the cursor), so the
    next idle pass of the central cadence takes the next chunk, and the other gated jobs run in between.

    Non-destructive: buffer files and the in-memory _SESSIONS registry are
    left intact. The caller (supervisor) should NOT call remove_session()
    for sessions returned here — finalize_stale_sessions handles that on
    its own 24h cadence.

    Returns reports for sessions where extract_session_snapshot ran. Skips
    fresh sessions; silently cleans up ghost buffers (files with no
    readable turns).

    Per-session try/except so a single bad session (corrupt buffer,
    transient FS error) can't abort the whole sweep — mirrors the
    isolation in finalize_stale_sessions.

    ``should_pause`` (INC-10, S14/S32/S41/S65): checked only after a session
    was actually snapshotted (a skipped/fresh/ghost session isn't the S32
    table's "one session" item). Progress is each session's own extraction
    cursor (persisted per chunk by ``extract_session_snapshot``), so resuming
    just re-lists active sessions: a session already snapshotted takes its
    next chunk, if any, and nothing is extracted twice.
    """
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        try:
            turns = read_session(persona_dir, sid)
            if not turns:
                # Ghost file — clean up without generating a report.
                delete_session_buffer(persona_dir, sid)
                continue
            age = session_silence_minutes(turns)
            if age >= silence_minutes:
                report = extract_session_snapshot(
                    persona_dir,
                    sid,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                    config=config,
                    max_chunks=1,
                )
                reports.append(report)
                if should_pause is not None and should_pause():
                    logger.info(
                        "snapshot_stale_sessions: pausing between sessions for chat (INC-10)"
                    )
                    if paused_out is not None:
                        paused_out.append(True)
                    return reports
        except ThrottleDeferred:
            # #456/#473: the session stays as it is (cursor, backoff); the job is skipped, not failed.
            raise
        except Exception:
            logger.exception(
                "snapshot_stale_sessions: per-session failure session=%s; "
                "skipping and continuing sweep",
                sid,
            )
    return reports


def finalize_stale_sessions(
    persona_dir: Path,
    *,
    finalize_after_hours: float = 24.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
    should_pause: Callable[[], bool] | None = None,
    paused_out: list[bool] | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; finalize any whose last turn is past
    ``finalize_after_hours``.

    "Finalize" = run one snapshot extraction of ONE chunk per session (#523: ``max_chunks=1``, so a run makes no more model
    calls per session than before chunking and has no long uninterrupted stretch; the snapshot sweep takes the rest at the
    next lulls). It is now **extraction-only**
    (cascade-compaction 1c, stage-3 L2): it NO LONGER deletes conversation buffers —
    the rollover path (brain/chat/rollover.perform_rollover) owns buffer deletion, so
    a rolled-over buffer is reaped exactly once and a still-live session's buffer is
    left intact (bounded by the daily cascade + 1d segmentation, and becomes
    rollover-eligible at the 24h-idle or weekly boundary). The empty-buffer branch's
    delete is removed too. Only the poison-move path (corrupt input) still quarantines
    a buffer out of the active set. Caller (supervisor) follows up with
    remove_session() for each returned report.session_id (registry-cache eviction; a
    still-live session re-hydrates from disk on next access).

    Per-session try/except so one bad session can't abort the loop.
    """
    threshold_minutes = finalize_after_hours * 60.0
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        turns = read_session(persona_dir, sid)
        if not turns:
            # Extraction-only: an empty live buffer is bounded and rollover-eligible;
            # do NOT delete it here (the rollover / ghost-cleanup sweep reaps it).
            continue
        age = session_silence_minutes(turns)
        if age < threshold_minutes:
            continue
        try:
            report = extract_session_snapshot(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=provider,
                config=config,
                max_chunks=1,
            )
        except ThrottleDeferred:
            # #456/#473: not a failure: no backoff counted, the session is left for a later pass.
            raise
        except Exception:
            logger.exception("finalize_stale_sessions: snapshot failed session=%s", sid)
            report = IngestReport(session_id=sid)
            report.errors += 1
            report.commit_failures += 1
        # Check backoff sidecar for dead-letter threshold regardless of whether
        # commit_failures was set via the report (the snapshot backoff-guard
        # returns early with errors>0 but commit_failures=0 when the sidecar
        # already shows >= threshold — finalize must still dead-letter in that case).
        backoff = read_backoff(persona_dir, sid)
        at_threshold = (
            backoff is not None and backoff["failures"] >= _BACKOFF_FAILURE_THRESHOLD
        )
        if at_threshold:
            # Buffer has failed >= threshold times — move to poison queue so it
            # doesn't block the finalize loop indefinitely. This is the ONLY
            # buffer-lifecycle action finalize still performs (quarantine of
            # corrupt input); the successful path no longer deletes.
            # Under the session's extraction lock (#477), non-blocking: a pass that holds it (the busy
            # snapshot above, a rollover's poison_unextracted_span) must not have the buffer renamed and
            # the cursor deleted under it; the next finalize tick retries.
            with file_lock(extract_lock_path(persona_dir, sid), blocking=False) as acquired:
                if not acquired:
                    logger.info("conversation_finalize_deadletter_busy session=%s: another extraction holds it", sid)
                else:
                    src = persona_dir / "active_conversations" / f"{sid}.jsonl"
                    poison_dir = persona_dir / "active_conversations" / "poison"
                    dst = poison_dir / f"{sid}.jsonl"
                    # The backoff above was read BEFORE this lock: a rollover that succeeded in between has reaped the
                    # buffer, and there is then nothing to poison. Checked under the lock, and the rename is guarded too
                    # (a reap in that instant), so a vanished buffer can never abort the sweep for the other sessions.
                    moved = False
                    if src.exists():
                        try:
                            poison_dir.mkdir(parents=True, exist_ok=True)
                            src.rename(dst)
                            moved = True
                        except FileNotFoundError:
                            pass
                    if moved:
                        delete_cursor(persona_dir, sid)
                        delete_backoff(persona_dir, sid)
                        logger.warning(
                            "conversation_finalize_deadletter session=%s failures=%d — moved to poison/",
                            sid,
                            backoff["failures"],  # type: ignore[index]
                        )
                    else:
                        delete_backoff(persona_dir, sid)  # nothing left to poison: the failure record is stale
                        logger.info(
                            "conversation_finalize_deadletter_gone session=%s: the buffer is already gone "
                            "(reaped by a rollover); nothing to poison, backoff cleared",
                            sid,
                        )
        reports.append(report)
        logger.info(
            "conversation_finalized session=%s silence_hours=%.2f "
            "committed=%d enqueued=%d deduped=%d errors=%d",
            sid,
            age / 60.0,
            report.committed,
            report.enqueued,
            report.deduped,
            report.errors,
        )
        # INC-10 (S14/S32/S41/S65): item unit = one finalized session; checked
        # only after a session was actually finalized (a skipped/too-fresh
        # session isn't an "item"). Resuming re-lists active sessions and
        # re-applies the same age gate; a session already finalized takes its
        # next chunk, if any (its cursor advanced per chunk, so nothing repeats).
        if should_pause is not None and should_pause():
            logger.info("finalize_stale_sessions: pausing between sessions for chat (INC-10)")
            if paused_out is not None:
                paused_out.append(True)
            return reports
    return reports


def close_stale_sessions(
    persona_dir: Path,
    *,
    silence_minutes: float = 5.0,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    config: dict | None = None,
) -> list[IngestReport]:
    """Iterate active sessions; close any whose last turn is older than silence_minutes.

    Destructive — runs the full close_session pipeline (extract every chunk after the
    cursor + delete buffer + delete cursor sidecar) for every match. Retained for
    explicit close callers only; bridge shutdown and recovery do not call it.

    For the periodic non-destructive sweep, use ``snapshot_stale_sessions`` instead.
    For the 24h extraction-only cadence, use ``finalize_stale_sessions``.

    Empty sessions (no turns) are cleaned up silently without running the
    full pipeline.

    Returns reports for closed sessions only (skips fresh sessions).
    """
    reports: list[IngestReport] = []
    for sid in list_active_sessions(persona_dir):
        turns = read_session(persona_dir, sid)
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
                config=config,
            )
            reports.append(report)
    return reports
