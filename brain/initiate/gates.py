"""Cost-cap + cooldown enforcement for the initiate pipeline.

Hard floor circuit breakers (NOT advisory) prevent runaway:

    notify: 3 / rolling 24h, min 4h gap, blackout 23:00-07:00 user-local
    quiet:  8 / rolling 24h, min 1h gap, no blackout

User-local time comes from `datetime.now().astimezone()` - the OS is
the source of truth. No PersonaConfig knob for timezone.

The decision prompt sees the same numbers as text context for adaptive
self-restraint; this module is the gate that fires regardless of what
the prompt does.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from brain.initiate.audit import read_recent_audit

logger = logging.getLogger(__name__)


# Defaults per spec.
DEFAULT_NOTIFY_CAP = 3
DEFAULT_QUIET_CAP = 8
DEFAULT_NOTIFY_MIN_GAP_HOURS = 4.0
DEFAULT_QUIET_MIN_GAP_HOURS = 1.0
DEFAULT_BLACKOUT_START_HOUR = 23
DEFAULT_BLACKOUT_END_HOUR = 7


UrgencyShort = Literal["notify", "quiet"]


def in_blackout_window(
    now_local: datetime,
    *,
    start_hour: int = DEFAULT_BLACKOUT_START_HOUR,
    end_hour: int = DEFAULT_BLACKOUT_END_HOUR,
) -> bool:
    """Return True if `now_local`'s hour is in [start_hour, end_hour).

    The window wraps around midnight when start_hour > end_hour
    (e.g. 23-07 means 23:xx OR 0:xx through 6:59).
    """
    h = now_local.hour
    if start_hour <= end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour


def count_recent_sends(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    window_hours: float,
    now: datetime | None = None,
) -> int:
    """Count audit rows in the last `window_hours` where decision matches urgency."""
    target_decision = f"send_{urgency}"
    return sum(
        1
        for row in read_recent_audit(persona_dir, window_hours=window_hours, now=now)
        if row.decision == target_decision
    )


def _latest_send_time(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    now: datetime | None = None,
) -> datetime | None:
    target_decision = f"send_{urgency}"
    latest: datetime | None = None
    for row in read_recent_audit(persona_dir, window_hours=72, now=now):
        if row.decision != target_decision:
            continue
        try:
            ts = datetime.fromisoformat(row.ts)
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def check_send_allowed(
    persona_dir: Path,
    *,
    urgency: UrgencyShort,
    now: datetime | None = None,
    notify_cap: int = DEFAULT_NOTIFY_CAP,
    quiet_cap: int = DEFAULT_QUIET_CAP,
    notify_min_gap_hours: float = DEFAULT_NOTIFY_MIN_GAP_HOURS,
    quiet_min_gap_hours: float = DEFAULT_QUIET_MIN_GAP_HOURS,
    blackout_start_hour: int = DEFAULT_BLACKOUT_START_HOUR,
    blackout_end_hour: int = DEFAULT_BLACKOUT_END_HOUR,
) -> tuple[bool, str | None]:
    """Return (allowed, reason_if_denied). Reason is structured tag for audit."""
    now = now or datetime.now(UTC)
    # If caller supplied a tz-aware datetime in a non-UTC zone, treat its
    # wall-clock as user-local (tests inject LA-zoned datetimes). For UTC
    # or naive inputs, convert to the OS's local zone — the production path.
    if now.tzinfo is not None and now.utcoffset() != timedelta(0):
        now_local = now
    else:
        now_local = now.astimezone()

    if urgency == "notify" and in_blackout_window(
        now_local,
        start_hour=blackout_start_hour,
        end_hour=blackout_end_hour,
    ):
        return False, "blackout_window"

    cap = notify_cap if urgency == "notify" else quiet_cap
    sent = count_recent_sends(persona_dir, urgency=urgency, window_hours=24, now=now)
    if sent >= cap:
        return False, f"{urgency}_cap_24h_reached"

    min_gap = notify_min_gap_hours if urgency == "notify" else quiet_min_gap_hours
    last = _latest_send_time(persona_dir, urgency=urgency, now=now)
    if last is not None:
        delta = now - last
        if delta < timedelta(hours=min_gap):
            return False, f"{urgency}_min_gap_not_met"

    return True, None
