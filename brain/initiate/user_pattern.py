"""user_pattern.py — infer user availability and responsiveness from audit + buffer.

Produces a UserPresence snapshot each initiate-review tick, consumed by
check_send_allowed to adjust gate thresholds in real time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

log = logging.getLogger(__name__)

_ACTIVE_CONVERSATIONS_DIR = "active_conversations"
_SCHEDULE_LOOKBACK_DAYS = 30
_SCHEDULE_MIN_TURNS = 50
_SCHEDULE_ACTIVE_PERCENTILE = 0.20
_COLD_START_LAG_MIN = 3
_COLD_START_STREAK_AUDIT_FILENAME = "initiate_audit.jsonl"
_SEND_DECISIONS = frozenset({"send_notify", "send_quiet"})
_STREAK_STATES = frozenset({"unanswered", "dismissed"})
_RESET_STATES = frozenset({"replied_explicit", "acknowledged_unclear"})


@dataclass(frozen=True)
class UserPresence:
    silence_days: float           # days since last inbound chat turn; 0.0 when uncertain
    ignore_streak: int            # consecutive unanswered/dismissed proactive sends
    likely_active: bool           # within inferred active window; True when unknown
    response_lag_p50: float | None  # median response lag in seconds; None = cold start


def _compute_silence_days(persona_dir: Path, *, _now: datetime | None = None) -> float:
    """Return days since the most recent inbound (non-companion) chat turn.

    Returns 0.0 when no buffer files exist or no inbound turns are found —
    uncertainty stays permissive.
    """
    conversations_dir = persona_dir / _ACTIVE_CONVERSATIONS_DIR
    if not conversations_dir.exists():
        return 0.0

    now = _now or datetime.now(UTC)
    companion_name = persona_dir.name.lower()
    latest_ts: datetime | None = None

    for jsonl_file in conversations_dir.glob("*.jsonl"):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            if str(row.get("speaker", "")).lower() == companion_name:
                continue
            ts_str = row.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
            except (ValueError, TypeError):
                continue

    if latest_ts is None:
        return 0.0
    return max(0.0, (now - latest_ts).total_seconds() / 86400.0)
