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

See `docs/superpowers/specs/2026-05-16-body-state-divergence-fix.md`
(if/when one exists) and the 2026-05-17 commit that wired this into
``brain.tools.dispatch``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

_IDLE_THRESHOLD_MINUTES = 5.0


def compute_active_session_hours(persona_dir: Path, *, now: datetime) -> float:
    """How long the current chat session has been live, in hours.

    Reads the earliest entry timestamp from any active conversation
    buffer in ``<persona_dir>/active_conversations/*.jsonl``. If multiple
    buffers exist (rare; concurrent sessions), takes the earliest.
    Returns 0.0 when no buffer is open, no timestamp could be parsed,
    or the buffer's last activity was more than 5 minutes ago (stale/orphan).
    """
    conv_dir = persona_dir / "active_conversations"
    if not conv_dir.exists():
        return 0.0
    earliest_ts: datetime | None = None
    try:
        for buffer in conv_dir.glob("*.jsonl"):
            try:
                first_line = ""
                last_line = ""
                with buffer.open("r", encoding="utf-8") as fh:
                    for raw in fh:
                        stripped = raw.strip()
                        if not stripped:
                            continue
                        if not first_line:
                            first_line = stripped
                        last_line = stripped
                if not first_line:
                    continue

                # Idle threshold: skip buffers with no activity in the last 5 min.
                last_entry = json.loads(last_line)
                last_ts_raw = last_entry.get("ts") or last_entry.get("timestamp")
                if not last_ts_raw or not isinstance(last_ts_raw, str):
                    continue
                if last_ts_raw.endswith("Z"):
                    last_ts_raw = last_ts_raw[:-1] + "+00:00"
                last_ts = datetime.fromisoformat(last_ts_raw)
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=UTC)
                idle_minutes = (now - last_ts).total_seconds() / 60.0
                if idle_minutes >= _IDLE_THRESHOLD_MINUTES:
                    continue

                entry = json.loads(first_line)
                ts_raw = entry.get("timestamp") or entry.get("ts")
                if not ts_raw or not isinstance(ts_raw, str):
                    continue
                if ts_raw.endswith("Z"):
                    ts_raw = ts_raw[:-1] + "+00:00"
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if earliest_ts is None or ts < earliest_ts:
                    earliest_ts = ts
            except (json.JSONDecodeError, OSError, ValueError):
                continue
    except OSError:
        return 0.0
    if earliest_ts is None:
        return 0.0
    elapsed = (now - earliest_ts).total_seconds() / 3600.0
    return max(0.0, elapsed)
