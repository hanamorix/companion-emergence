"""One-time backwards-compatibility migration for timed conversation compaction.

An install that predates compaction (or any upgrade into it) has an
``active_conversations/<session_id>.jsonl`` buffer that has never been compacted —
potentially hundreds of aged, already-extracted turns. The live path
(``apply_budget`` backstop) and the daily cadence both call ``compact_conversation``,
which folds *all* removable turns in *one* provider call — on a large backlog that is
a single enormous summary call ("hit cold").

``run_backlog_migration`` drains that backlog ONCE, GRACEFULLY, at startup, before the
live path touches it, by REPLAYING the daily cadence: it folds in 24h time-increments,
oldest cohort first — fold everything ≥N days old, then ≥(N-1) days old + the running
summary, … down to the live 24h cutoff. Each provider call therefore sees at most ~one
day of new messages plus the re-compressed summary, never the whole backlog. It is a
thin *driver* over the shared compaction core (``compact_conversation`` with a stepped
``older_than``) — no second compaction implementation, so every core invariant
(lossless-before-lossy, cursor guard, ``min_keep_tail``, idempotency, lock) is reused.

Run-once is gated by the marker ``archived_conversations/.compat_migrated``, written
**only** when every active session reached a *drained* end-state. A transient miss
(``locked`` — a live compaction holds the lock; ``archive_failed`` — I/O hiccup) leaves
the session undrained and withholds the marker, so a later restart retries cleanly. The
whole entry point is fault-isolated: it never raises, so a migration failure can never
break bridge startup.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.chat.compaction import compact_conversation
from brain.ingest.buffer import list_active_sessions, read_cursor, read_session

logger = logging.getLogger(__name__)

_MARKER_NAME = ".compat_migrated"

# Reasons (from CompactionResult.reason) that mean "nothing left to fold now" — a
# genuine drained end-state. Everything else that returns compacted=False is a
# transient miss (locked / archive_failed) that must NOT be treated as done.
_DRAINED_REASONS = frozenset({"nothing_aged", "cursor_none"})


@dataclass
class MigrationResult:
    """Outcome of one run_backlog_migration call (for logging + tests)."""

    already_migrated: bool = False     # marker present at entry → no-op
    marker_written: bool = False       # all sessions drained → marker written this run
    sessions_seen: int = 0
    sessions_drained: int = 0
    total_compacted: int = 0           # raw turns folded across all sessions
    total_passes: int = 0             # compact_conversation calls made
    undrained_sessions: list[str] = field(default_factory=list)  # transient / errored / ceiling


def _marker_path(persona_dir: Path) -> Path:
    # Resolve directly — do NOT route through buffer._archived_conversations_dir,
    # which mkdirs the dir as a side effect (we must read existence first).
    return Path(persona_dir) / "archived_conversations" / _MARKER_NAME


def _write_marker(persona_dir: Path, payload: dict) -> None:
    """Atomically write the run-once marker (mkdir parent first, tmp + os.replace)."""
    path = _marker_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _parse_ts(raw: object) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _drain_session(
    persona_dir: Path,
    session_id: str,
    *,
    provider: LLMProvider,
    step: timedelta,
    older_than_floor: timedelta,
    now: datetime,
    max_passes: int,
    result: MigrationResult,
) -> bool:
    """Fold one session's backlog in time increments — OLDEST cohort first.

    This historically REPLAYS the daily cadence: fold everything older than the
    oldest whole-``step`` boundary, then lower the age threshold by one ``step``
    each pass, folding the next cohort into the running summary, down to
    ``older_than_floor`` (the live cadence's own 24h cutoff). With ``step`` =
    ``older_than_floor`` = 24h that is exactly "fold all messages ≥3 days old, then
    ≥2 days old + the summary, then ≥1 day old + the summary". Each provider call
    therefore sees at most ~one ``step`` of new messages plus the re-compressed
    summary — never the whole backlog. Returns True iff drained (no transient miss
    / ceiling / error).
    """
    cursor = read_cursor(persona_dir, session_id)
    if cursor is None:
        # Nothing extracted yet → genuinely nothing foldable; a drained no-op.
        return True
    turns = read_session(persona_dir, session_id)
    aged = [
        now - ts
        for ts in (
            _parse_ts(t.get("ts")) for t in turns if t.get("speaker") != "summary"
        )
        if ts is not None and (now - ts) >= older_than_floor
    ]
    if not aged:
        return True  # nothing older than the floor → nothing to fold
    # Highest step that still covers the oldest aged turn; descend to the floor so
    # the OLDEST messages fold first (the user's "3 days old, then 2 days old +
    # summary" semantics). int() floors, so step*k ≤ oldest_age — the oldest cohort
    # is always captured at k = n_steps.
    n_steps = max(1, int(max(aged) / step))
    if n_steps > max_passes:
        logger.warning(
            "backlog migration: session=%s needs %d steps > max_passes=%d; capping "
            "(some of the oldest backlog will fold next start)",
            session_id, n_steps, max_passes,
        )
        n_steps = max_passes
    for k in range(n_steps, 0, -1):
        res = compact_conversation(
            persona_dir,
            session_id,
            older_than=max(older_than_floor, step * k),
            fold_existing_summary=True,
            provider=provider,
            now=now,
        )
        result.total_passes += 1
        if res.compacted:
            # Progress (reason=="ok"; a provider failure still returns compacted=
            # True with a deterministic soft note — still progress, never spins).
            result.total_compacted += res.compacted_n
            continue
        if res.reason not in _DRAINED_REASONS:
            # locked / archive_failed → transient: stop, withhold marker, retry.
            return False
        # nothing_aged for this cohort (an empty step) → continue to the next.
    return True


def run_backlog_migration(
    persona_dir: Path,
    *,
    provider: LLMProvider,
    step: timedelta = timedelta(hours=24),
    older_than_floor: timedelta = timedelta(hours=24),
    now: datetime | None = None,
    max_passes_per_session: int = 1000,
) -> MigrationResult:
    """Drain every active conversation's pre-compaction backlog, once, by REPLAYING
    the daily cadence in ``step`` (default 24h) time-increments, oldest cohort first.

    Fault-isolated: never raises. The run-once marker is written only when EVERY
    active session reaches a drained end-state; any transient miss / error / ceiling
    withholds it so a later restart retries. See module docstring + _drain_session.
    """
    result = MigrationResult()
    persona_dir = Path(persona_dir)
    now = now or datetime.now(UTC)
    try:
        if _marker_path(persona_dir).exists():
            result.already_migrated = True
            return result

        # Create the archive dir up-front (before any provider call) as a visible
        # "migration started" indicator, and log a start line — so a running-but-slow
        # migration is observable from the filesystem and the log. The MARKER (not
        # the dir) is the run-once gate, so creating the dir early is safe for
        # idempotency: an incomplete run leaves the dir present but the marker absent.
        (persona_dir / "archived_conversations").mkdir(parents=True, exist_ok=True)
        sessions = list_active_sessions(persona_dir)
        result.sessions_seen = len(sessions)
        logger.info(
            "backlog migration: starting — %d active session(s): %s",
            len(sessions), sessions,
        )
        all_drained = True
        for sid in sessions:
            try:
                drained = _drain_session(
                    persona_dir, sid,
                    provider=provider, step=step, older_than_floor=older_than_floor,
                    now=now, max_passes=max_passes_per_session, result=result,
                )
            except Exception:
                logger.exception("backlog migration: session=%s raised; left undrained", sid)
                drained = False
            if drained:
                result.sessions_drained += 1
            else:
                all_drained = False
                result.undrained_sessions.append(sid)

        # Write the run-once marker ONLY if nothing was left undrained.
        if all_drained:
            _write_marker(persona_dir, {
                "migrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_sessions": result.sessions_seen,
                "total_compacted": result.total_compacted,
            })
            result.marker_written = True

        logger.info(
            "backlog migration: sessions=%d drained=%d turns=%d passes=%d "
            "undrained=%d marker=%s",
            result.sessions_seen, result.sessions_drained, result.total_compacted,
            result.total_passes, len(result.undrained_sessions), result.marker_written,
        )
    except Exception:
        # Belt-and-suspenders: the entry point must never break bridge startup.
        logger.exception("backlog migration: top-level failure; marker NOT written")
    return result
