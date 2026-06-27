"""One-time backwards-compatibility migration for timed conversation compaction.

An install that predates compaction (or any upgrade into it) has an
``active_conversations/<session_id>.jsonl`` buffer that has never been compacted —
potentially hundreds of aged, already-extracted turns. The live path
(``apply_budget`` backstop) and the daily cadence both call ``compact_conversation``,
which folds *all* removable turns in *one* provider call — on a large backlog that is
a single enormous summary call ("hit cold").

``run_backlog_migration`` drains that backlog ONCE, GRACEFULLY, in bounded batches, at
startup, before the live path touches it. It is a thin *driver* over the shared
compaction core (``compact_conversation`` with ``max_compact_turns``) — no second
compaction implementation, so every core invariant (lossless-before-lossy, cursor
guard, ``min_keep_tail``, idempotency, lock) is reused, not re-derived.

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
from brain.ingest.buffer import list_active_sessions

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


def _drain_session(
    persona_dir: Path,
    session_id: str,
    *,
    provider: LLMProvider,
    batch: int,
    older_than: timedelta,
    max_passes: int,
    result: MigrationResult,
) -> bool:
    """Fold one session's backlog in bounded batches. Returns True iff drained.

    Each pass folds the oldest ``batch`` removable turns into the (re-compressed)
    head summary, so no single provider call ever sees the whole backlog. A pass
    that returns compacted=True made progress and the loop continues; the first
    compacted=False ends it — drained only if the reason says so.
    """
    for _ in range(max_passes):
        res = compact_conversation(
            persona_dir,
            session_id,
            older_than=older_than,
            fold_existing_summary=True,
            provider=provider,
            max_compact_turns=batch,
        )
        result.total_passes += 1
        if res.compacted:
            # Progress (reason=="ok"). Note: a provider failure also returns
            # compacted=True with a deterministic soft note (fell_soft) — still
            # progress, so we keep draining, never spin on it.
            result.total_compacted += res.compacted_n
            continue
        # compacted is False → a no-op. Drained only for a genuine end-state;
        # locked / archive_failed are transient → leave undrained, retry later.
        return res.reason in _DRAINED_REASONS
    # Hit the pass ceiling without draining (pathological) — withhold the marker.
    logger.warning(
        "backlog migration: session=%s hit max_passes=%d without draining",
        session_id, max_passes,
    )
    return False


def run_backlog_migration(
    persona_dir: Path,
    *,
    provider: LLMProvider,
    batch: int = 40,
    older_than: timedelta = timedelta(hours=24),
    max_passes_per_session: int = 1000,
) -> MigrationResult:
    """Drain every active conversation's pre-compaction backlog, once, in batches.

    Fault-isolated: never raises. The run-once marker is written only when EVERY
    active session reaches a drained end-state; any transient miss / error / ceiling
    withholds it so a later restart retries. See module docstring.
    """
    result = MigrationResult()
    persona_dir = Path(persona_dir)
    try:
        if _marker_path(persona_dir).exists():
            result.already_migrated = True
            return result

        sessions = list_active_sessions(persona_dir)
        result.sessions_seen = len(sessions)
        all_drained = True
        for sid in sessions:
            try:
                drained = _drain_session(
                    persona_dir, sid,
                    provider=provider, batch=batch, older_than=older_than,
                    max_passes=max_passes_per_session, result=result,
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
