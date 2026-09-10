"""Shared time helpers — ISO-8601 Z-suffix conversion, local-display conversion.

Previously triplicated across dream/heartbeat/reflex engines;
consolidated here before the fourth engine (research) lands.

Persona-facing local-timezone display (issue #217)
----------------------------------------------------
Every site below applies a local-timezone offset to a timestamp before it is
shown to the companion. Storage stays UTC everywhere; these are display-only
conversions. Grep the tag ``tz-local-display`` across the repo to find every
call site currently applying the local offset — that grep is the
authoritative list, kept current by convention (every site below carries the
tag); this comment is a human-readable index into it, not a substitute for
running the grep.

Routes through this shared formatter (``format_local()`` / ``local_display()``
below) — switch the presentation style ONE TIME, here, and all of these
follow automatically:
  - brain/bridge/provider.py — block-level "Current time:" anchor
  - brain/bridge/provider.py — per-message `ts` in the JSONL chat context
  - brain/chat/prompt.py — ambient "[current time: ...]" tail anchor
  - brain/initiate/ambient.py — outbound-recall "Recent outbound" row ts
  - brain/initiate/ambient.py — outbound-recall "Pending uncertainty" row ts
  - brain/initiate/ambient.py — recent-conversation-excerpt bracket ts
  - brain/monologue/recall.py — monologue recall snippet ts
  - brain/tools/impls/_common.py — search/read_full_memory result created_at
  - brain/tools/impls/list_works.py — list_works summary created_at
  - brain/tools/impls/read_work.py — read_work full content created_at

Formats independently (exceptions) — these convert via ``to_local()`` but
build their OWN final string (a coarse date/part-of-day bucket, a bare
date), so they do NOT follow a ``format_local()``/``local_display()`` edit
automatically. To switch the presentation style, these two also need their
own edit:
  - brain/chat/compaction.py — ``_coarse_stamp()``: "Aug 10 evening" compaction
    marker (date + part-of-day; not a full ISO datetime)
  - brain/chat/prompt.py — ``_build_recent_journal_block()``: journal digest
    date, "%Y-%m-%d" only (no time-of-day)

To switch to a fully-localized presentation later (e.g. locale-aware /
human-phrase instead of a naive-local ISO string): change ``format_local()``
(and/or ``local_display()``, which calls it) here in this file, and every
site in the first list follows automatically. Then separately edit each
exception site named above, since those do not route through either helper.
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
    """Render a datetime as local-zone ISO-8601, seconds precision, with NO
    offset suffix (tz-local-display, issue #218): the wall-clock value is
    still the OS-local time (e.g. 21:21:30, not UTC), but the trailing
    '-04:00'/'+00:00' is dropped before rendering. A 'Z' suffix is also
    wrong here — 'Z' means UTC, and a converted local time still labeled
    'Z' would misstate its own zone — so the fix is to go naive, not to
    relabel. This is display-only: strip tzinfo on the local-converted
    value right before isoformat(), never on a stored/ordering timestamp.
    An explicit offset was tried first and dropped: the substrate echoed
    the offset token back verbatim, which naive-local avoids."""
    return to_local(dt).replace(tzinfo=None).isoformat(timespec="seconds")


def local_display(ts: str) -> str:
    """Convert a stored UTC ISO-8601 string (``iso_utc()``'s format, or any
    ISO string with a 'Z' suffix or explicit offset) to a local-zone display
    string. Fails soft to the raw input on a malformed timestamp — a display
    seam must never break composition over a bad ts."""
    try:
        return format_local(parse_iso_utc(ts))
    except (ValueError, TypeError):
        return ts
