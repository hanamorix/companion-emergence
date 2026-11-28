"""Live session-age helper.

`compute_active_session_hours` reads the earliest timestamp from any open
conversation buffer under ``<persona_dir>/active_conversations/`` and
returns the elapsed wall-clock hours since then. Returns 0.0 when no
buffer is present or no parseable timestamps were found.

This used to live as ``_active_session_hours`` inside
``brain/bridge/persona_state.py`` (where the UI's ``/persona/state``
body-block builder calls it). The MCP-tools dispatcher needs the same
signal so the brain's ``get_body_state`` self-read matches what the UI
shows. Moving it here gives both callers a layer-neutral import:
``brain.bridge`` and ``brain.tools`` can both depend on ``brain.body``
without crossing each other.

See `docs/source-spec/2026-04-29-body-state-design.md` §8 (the 2026-05-16 divergence-fix note was never committed, #797)
and the 2026-05-17 commit that wired this into ``brain.tools.dispatch``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import Recovery, iter_jsonl

# No activity in the last 5 minutes → the session isn't currently live (0.0).
# Guards the wholly-idle / orphan-buffer case.
_IDLE_THRESHOLD_MINUTES = 5.0
# A gap of this size or larger BETWEEN turns is a session boundary: the turns
# before it belong to an earlier sitting, not the current one. Without this, a
# buffer that sat idle for hours/days and then got one fresh turn would report
# its entire wall-clock span as session age (the 69.7h energy-collapse bug).
_SESSION_GAP_MINUTES = 30.0


def _parse_ts(raw: object) -> datetime | None:
    """Parse an ISO-8601 ts (with optional ``Z``) into a tz-aware datetime."""
    if not raw or not isinstance(raw, str):
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _entry_timestamps(buffer: Path) -> list[datetime]:
    """Parsed, tz-aware timestamps for every turn in a buffer, in file (turn)
    order. Unparseable / malformed lines are skipped."""
    stamps: list[datetime] = []
    for entry in iter_jsonl(buffer, recover=Recovery.RAISE):
        # A compaction `summary` row carries a compaction-time ts, not a turn
        # timestamp — exclude it from live session-age.
        if entry.get("speaker") == "summary":
            continue
        dt = _parse_ts(entry.get("ts") or entry.get("timestamp"))
        if dt is not None:
            stamps.append(dt)
    return stamps


def split_into_runs(stamps: list[datetime]) -> list[list[datetime]]:
    """Split ``stamps`` (in the order given) into runs: a gap of ``_SESSION_GAP_MINUTES`` or more between two neighbours
    starts a new run. The one place the 30-minute rule is applied: the live session's start and the stats line's
    sittings and stretches both call it. An empty list gives no runs."""
    runs: list[list[datetime]] = []
    for stamp in stamps:
        if runs and (stamp - runs[-1][-1]).total_seconds() / 60.0 < _SESSION_GAP_MINUTES:
            runs[-1].append(stamp)
        else:
            runs.append([stamp])
    return runs


def compute_active_session_hours(persona_dir: Path, *, now: datetime) -> float:
    """How long the *current continuous* chat session has been live, in hours.

    For each active conversation buffer in
    ``<persona_dir>/active_conversations/*.jsonl``:

    - If the last turn was more than ``_IDLE_THRESHOLD_MINUTES`` (5 min) ago,
      the session isn't currently live — the buffer contributes 0.0 (stale /
      orphan guard).
    - Otherwise the session started at the head of the *latest contiguous run*
      of turns: walking back from the most recent turn, the run ends at the
      first gap >= ``_SESSION_GAP_MINUTES`` (30 min). A buffer that spans an
      idle gap (e.g. a 3-day-old turn plus a fresh reply) therefore counts only
      the current sitting, never the whole wall-clock span.

    Across multiple buffers (rare; concurrent sessions) the earliest current
    session start wins. Returns 0.0 when no buffer is open or live.
    """
    conv_dir = persona_dir / "active_conversations"
    if not conv_dir.exists():
        return 0.0
    earliest_start: datetime | None = None
    try:
        for buffer in conv_dir.glob("*.jsonl"):
            try:
                stamps = _entry_timestamps(buffer)
            except OSError:
                continue
            if not stamps:
                continue

            # Liveness gate: no activity in the last 5 min → not currently live.
            last_ts = stamps[-1]
            if (now - last_ts).total_seconds() / 60.0 >= _IDLE_THRESHOLD_MINUTES:
                continue

            # Session start = head of the latest contiguous run.
            session_start = split_into_runs(stamps)[-1][0]

            if earliest_start is None or session_start < earliest_start:
                earliest_start = session_start
    except OSError:
        return 0.0
    if earliest_start is None:
        return 0.0
    elapsed = (now - earliest_start).total_seconds() / 3600.0
    return max(0.0, elapsed)
