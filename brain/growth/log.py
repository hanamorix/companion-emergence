# brain/growth/log.py
"""Append-only biography of brain growth events.

Each line is one complete JSON object. Never edited, never deleted —
the brain's biography is preserved. Atomic append via the standard
`.new + os.replace` rotation so a crash mid-write leaves either the
old log or the old log + the new line, never a partial line.

Per principle audit 2026-04-25 (Phase 2a §6): the growth log is the
record of who-they-became — not telemetry an owner consults, but
biography future engineers (and the user, via GUI) can read.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrowthLogEvent:
    """One event in the brain's growth biography.

    `type` is a discriminator allowing the same log to record events from
    any future engine — Phase 2a only emits "emotion_added"; Phase 2a-extension
    PRs add "arc_added", "interest_added", "soul_crystallized".
    """

    timestamp: datetime  # tz-aware UTC
    type: str
    name: str
    description: str
    decay_half_life_days: float | None
    reason: str
    evidence_memory_ids: tuple[str, ...]
    score: float
    relational_context: str | None


def append_growth_event(path: Path, event: GrowthLogEvent) -> None:
    """Append one event line through a cross-process file lock.

    Audit 2026-05-07 P2-1: the previous read-rewrite-replace shape
    could lose events when two writers (e.g. heartbeat-driven growth
    + autonomous reflex emergence) overlapped — both read the same
    old file, both replaced with old + their one new line, the later
    rename clobbered the earlier writer's event.

    Cross-platform note (2026-05-08): POSIX guarantees writes ≤
    PIPE_BUF (~4096 bytes) in ``O_APPEND`` mode are atomic at the
    filesystem layer, so on macOS / Linux the explicit lock is
    technically redundant. Windows does NOT have that guarantee —
    concurrent appends from different file handles can interleave
    mid-line and drop events under contention (the
    ``test_growth_log_concurrent_appends_dont_clobber_each_other``
    test reproduces this with 8×5 = 40 threaded appends). Always
    taking the lock costs ~µs on the happy path and gives us
    correct semantics on every platform.
    """
    from brain.utils.file_lock import file_lock

    line = (json.dumps(_event_to_dict(event)) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        with open(path, "ab") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def read_growth_log(path: Path, *, limit: int | None = None) -> list[GrowthLogEvent]:
    """Read events oldest-first. `limit=N` returns the N most-recent events.

    Corrupt lines (partial write, hand-edit) are skipped with a warning via
    `read_jsonl_skipping_corrupt`; well-formed lines around them still parse.
    """
    events: list[GrowthLogEvent] = []
    for data in read_jsonl_skipping_corrupt(path):
        try:
            events.append(_event_from_dict(data))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("skipping growth log entry with schema error in %s: %s", path, exc)
            continue
    if limit is not None:
        events = events[-limit:]
    return events


def _event_to_dict(event: GrowthLogEvent) -> dict:
    return {
        "timestamp": iso_utc(event.timestamp),
        "type": event.type,
        "name": event.name,
        "description": event.description,
        "decay_half_life_days": event.decay_half_life_days,
        "reason": event.reason,
        "evidence_memory_ids": list(event.evidence_memory_ids),
        "score": event.score,
        "relational_context": event.relational_context,
    }


def _event_from_dict(data: dict) -> GrowthLogEvent:
    return GrowthLogEvent(
        timestamp=parse_iso_utc(data["timestamp"]),
        type=str(data["type"]),
        name=str(data["name"]),
        description=str(data["description"]),
        decay_half_life_days=(
            None if data["decay_half_life_days"] is None else float(data["decay_half_life_days"])
        ),
        reason=str(data["reason"]),
        evidence_memory_ids=tuple(str(x) for x in data["evidence_memory_ids"]),
        score=float(data["score"]),
        relational_context=(
            None if data["relational_context"] is None else str(data["relational_context"])
        ),
    )


# ---------------------------------------------------------------------------
# Arc lifecycle event constructors (Phase 2 reflex emergence)
# ---------------------------------------------------------------------------


def arc_added_event(
    *,
    timestamp: datetime,
    name: str,
    description: str,
    reasoning: str,
    created_by: str,  # "brain_emergence" | "user_authored" | "og_migration"
) -> GrowthLogEvent:
    """Constructor for arc_added events. Stashes created_by in relational_context
    so the brain can read provenance back from its own growth log."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_added",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=created_by,
    )


def arc_pruned_by_brain_event(
    *,
    timestamp: datetime,
    name: str,
    description: str,
    reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_pruned_by_brain events."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_pruned_by_brain",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_removed_by_user_event(
    *,
    timestamp: datetime,
    name: str,
    description: str,
) -> GrowthLogEvent:
    """Constructor for arc_removed_by_user events. Reason is hardcoded
    because user file-edit removals don't carry explicit reasoning."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_removed_by_user",
        name=name,
        description=description,
        decay_half_life_days=None,
        reason="user edited reflex_arcs.json",
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_rejected_user_removed_event(
    *,
    timestamp: datetime,
    name: str,
    reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_rejected_user_removed events — fired when the
    brain proposes an arc whose name is in the 15-day graveyard window."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_rejected_user_removed",
        name=name,
        description="",
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )


def arc_proposal_dropped_event(
    *,
    timestamp: datetime,
    name: str,
    reasoning: str,
) -> GrowthLogEvent:
    """Constructor for arc_proposal_dropped events — generic gate-rejection."""
    return GrowthLogEvent(
        timestamp=timestamp,
        type="arc_proposal_dropped",
        name=name,
        description="",
        decay_half_life_days=None,
        reason=reasoning,
        evidence_memory_ids=(),
        score=0.0,
        relational_context=None,
    )
