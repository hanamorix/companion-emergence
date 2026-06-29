"""user_pattern.py — infer user availability and responsiveness from audit + buffer.

Produces a UserPresence snapshot each initiate-review tick, consumed by
check_send_allowed to adjust gate thresholds in real time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
            speaker = str(row.get("speaker", "")).lower()
            if speaker == companion_name:
                continue
            # A compaction `summary` row is not a real user turn — its
            # compaction-time ts must not corrupt the "last user message" signal.
            if speaker == "summary":
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


def _compute_ignore_streak(persona_dir: Path) -> int:
    """Count consecutive unanswered/dismissed proactive sends, most recent first.

    Stops and returns the current count when it hits a replied_explicit or
    acknowledged_unclear row. Only send_notify/send_quiet rows count.
    Returns 0 when no audit file exists.
    """
    audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    if not audit_path.exists():
        return 0

    rows = [r for r in read_jsonl_skipping_corrupt(audit_path) if r.get("ts")]
    rows_desc = sorted(rows, key=lambda r: r["ts"], reverse=True)

    streak = 0
    for row in rows_desc:
        if row.get("decision") not in _SEND_DECISIONS:
            continue
        state = (row.get("delivery") or {}).get("current_state", "")
        if state in _STREAK_STATES:
            streak += 1
        elif state in _RESET_STATES:
            break
    return streak


def _compute_likely_active(  # noqa: PLR0912
    persona_dir: Path, *, _now: datetime | None = None
) -> bool:
    """Return True if current hour is within the user's inferred active window."""
    conversations_dir = persona_dir / _ACTIVE_CONVERSATIONS_DIR
    if not conversations_dir.exists():
        return True
    now = _now or datetime.now(UTC)
    cutoff = now - timedelta(days=_SCHEDULE_LOOKBACK_DAYS)
    hour_counts: list[int] = [0] * 24
    total = 0
    for jsonl_file in conversations_dir.glob("*.jsonl"):
        for row in read_jsonl_skipping_corrupt(jsonl_file):
            speaker = str(row.get("speaker", "")).lower()
            if speaker == persona_dir.name.lower():
                continue
            # A compaction `summary` row is not a real user turn — exclude from
            # hour-distribution so the active-window inference isn't skewed by
            # compaction-time timestamps.
            if speaker == "summary":
                continue
            ts_str = row.get("ts")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts < cutoff:
                    continue
                hour_counts[ts.astimezone().hour] += 1
                total += 1
            except (ValueError, TypeError):
                continue
    if total < _SCHEDULE_MIN_TURNS:
        return True
    sorted_counts = sorted(hour_counts)
    threshold = max(sorted_counts[int(len(sorted_counts) * _SCHEDULE_ACTIVE_PERCENTILE)], 1)
    return hour_counts[now.astimezone().hour] >= threshold


def _compute_response_lag_p50(persona_dir: Path) -> float | None:
    """Return median response lag in seconds from replied_explicit audit rows.

    Returns None when fewer than _COLD_START_LAG_MIN rows with confirmed
    replies exist (cold-start guard).
    """
    audit_path = persona_dir / _COLD_START_STREAK_AUDIT_FILENAME
    if not audit_path.exists():
        return None

    lags: list[float] = []
    for row in read_jsonl_skipping_corrupt(audit_path):
        if row.get("decision") not in _SEND_DECISIONS:
            continue
        delivery = row.get("delivery") or {}
        if delivery.get("current_state") != "replied_explicit":
            continue
        send_ts_str = row.get("ts", "")
        transitions = delivery.get("state_transitions", [])
        reply_ts_str = next(
            (t["at"] for t in reversed(transitions) if t.get("to") == "replied_explicit"),
            None,
        )
        if not reply_ts_str or not send_ts_str:
            continue
        try:
            send_ts = datetime.fromisoformat(send_ts_str)
            reply_ts = datetime.fromisoformat(reply_ts_str)
            if send_ts.tzinfo is None:
                send_ts = send_ts.replace(tzinfo=UTC)
            if reply_ts.tzinfo is None:
                reply_ts = reply_ts.replace(tzinfo=UTC)
            lag = (reply_ts - send_ts).total_seconds()
            if lag >= 0:
                lags.append(lag)
        except (ValueError, TypeError):
            continue

    if len(lags) < _COLD_START_LAG_MIN:
        return None

    lags.sort()
    mid = len(lags) // 2
    return lags[mid] if len(lags) % 2 == 1 else (lags[mid - 1] + lags[mid]) / 2.0


def compute_user_presence(persona_dir: Path, *, _now: datetime | None = None) -> UserPresence:
    """Compute current UserPresence from audit log and chat buffer.

    All four signals default to permissive values on failure — uncertainty
    never tightens gates.
    """
    try:
        silence_days = _compute_silence_days(persona_dir, _now=_now)
    except Exception:
        log.debug("user_pattern: _compute_silence_days failed", exc_info=True)
        silence_days = 0.0

    try:
        ignore_streak = _compute_ignore_streak(persona_dir)
    except Exception:
        log.debug("user_pattern: _compute_ignore_streak failed", exc_info=True)
        ignore_streak = 0

    try:
        likely_active = _compute_likely_active(persona_dir, _now=_now)
    except Exception:
        log.debug("user_pattern: _compute_likely_active failed", exc_info=True)
        likely_active = True

    try:
        response_lag_p50 = _compute_response_lag_p50(persona_dir)
    except Exception:
        log.debug("user_pattern: _compute_response_lag_p50 failed", exc_info=True)
        response_lag_p50 = None

    return UserPresence(
        silence_days=silence_days,
        ignore_streak=ignore_streak,
        likely_active=likely_active,
        response_lag_p50=response_lag_p50,
    )
