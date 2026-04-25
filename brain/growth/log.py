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
    """Atomic append: write `path + ".new"` containing existing-content + new-line, then os.replace.

    A crash between write and rename leaves the previous valid file intact.
    A crash after rename leaves the new line in the file. No partial-line
    state is ever observable to readers.
    """
    line = json.dumps(_event_to_dict(event)) + "\n"
    existing = path.read_bytes() if path.exists() else b""
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_bytes(existing + line.encode("utf-8"))
    os.replace(tmp, path)


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
