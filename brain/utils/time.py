"""Shared time helpers — ISO-8601 Z-suffix conversion.

Previously triplicated across dream/heartbeat/reflex engines;
consolidated here before the fourth engine (research) lands.
"""

from __future__ import annotations

from datetime import UTC, datetime


def iso_utc(dt: datetime) -> str:
    """ISO-8601 with Z suffix (matches Week 3.5 manifest format).

    Requires a tz-aware UTC datetime — a naive datetime would silently
    write a malformed stamp (no Z suffix, no offset) that doesn't parse
    back cleanly.
    """
    if dt.tzinfo is None:
        raise ValueError("iso_utc requires a tz-aware datetime")
    return dt.isoformat().replace("+00:00", "Z")


def parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 Z-suffix timestamp back to tz-aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
