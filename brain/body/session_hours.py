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


def compute_active_session_hours(persona_dir: Path, *, now: datetime) -> float:
    """How long the current chat session has been live, in hours.

    Reads the earliest entry timestamp from any active conversation
    buffer in ``<persona_dir>/active_conversations/*.jsonl``. If multiple
    buffers exist (rare; concurrent sessions), takes the earliest.
    Returns 0.0 when no buffer is open or no timestamp could be parsed,
    matching the panel's "fresh session" expectation.
    """
    conv_dir = persona_dir / "active_conversations"
    if not conv_dir.exists():
        return 0.0
    earliest_ts: datetime | None = None
    try:
        for buffer in conv_dir.glob("*.jsonl"):
            try:
                with buffer.open("r", encoding="utf-8") as fh:
                    first = fh.readline().strip()
                if not first:
                    continue
                entry = json.loads(first)
                ts_raw = entry.get("timestamp") or entry.get("ts")
                if not ts_raw:
                    continue
                if isinstance(ts_raw, str):
                    if ts_raw.endswith("Z"):
                        ts_raw = ts_raw[:-1] + "+00:00"
                    ts = datetime.fromisoformat(ts_raw)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                else:
                    continue
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
