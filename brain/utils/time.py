"""Shared time helpers — ISO-8601 Z-suffix conversion, local-display conversion.

Previously triplicated across dream/heartbeat/reflex engines;
consolidated here before the fourth engine (research) lands.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


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


def to_local(dt: datetime) -> datetime:
    """Convert a datetime to local wall-clock time, for anything shown to the
    companion as an absolute "when did/does this happen" reading (issue #217).

    Storage stays UTC everywhere; this is a display-time-only conversion.

    Naive and UTC-aware datetimes (the only kinds this codebase produces —
    everything is stored via ``datetime.now(UTC)``/``iso_utc``) are converted
    to the OS's local zone via ``astimezone()``. A tz-aware datetime that
    already carries a non-UTC offset (a test double built in a specific zone,
    e.g. an LA-zoned fixture) is trusted as already representing the intended
    wall-clock time and is returned as-is — mirrors the guard in
    ``brain/initiate/gates.py:146-149``.

    Never call ``.astimezone()`` on a naive datetime without this guard:
    Python treats a naive datetime passed to ``astimezone()`` as already
    being in the local zone, which would silently no-op on a UTC-naive value
    instead of converting it. This codebase's naive datetimes are always
    UTC-in-spirit (matches ``parse_iso_utc``'s own naive-means-UTC
    convention), so they're stamped UTC first, then converted.
    """
    if dt.tzinfo is not None and dt.utcoffset() != timedelta(0):
        return dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone()


def format_local(dt: datetime) -> str:
    """Render a datetime as local-zone ISO-8601 (seconds precision, explicit
    UTC offset instead of a 'Z' suffix — 'Z' means UTC, and a converted local
    time still labeled 'Z' would misstate its own zone)."""
    return to_local(dt).isoformat(timespec="seconds")


def local_display(ts: str) -> str:
    """Convert a stored UTC ISO-8601 string (``iso_utc()``'s format, or any
    ISO string with a 'Z' suffix or explicit offset) to a local-zone display
    string. Fails soft to the raw input on a malformed timestamp — a display
    seam must never break composition over a bad ts."""
    try:
        return format_local(parse_iso_utc(ts))
    except (ValueError, TypeError):
        return ts
