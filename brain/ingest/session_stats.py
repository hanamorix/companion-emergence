"""The stats line at the end of a finished conversation file (#650 spec section 5).

When the weekly rollover, a stale-resume rollover or a session close has finished a conversation file, one row is appended
to the end of that session's archive:

    {"type": "session_stats", "avg_sitting_minutes": <float>, "busiest": {"<local day>": ["HH:MM-HH:MM", ...], ...}}

It is a record. Nothing reads it for any rule. It has no ``speaker`` and no ``ts``, so every reader that filters on a
speaker or needs a time skips it. ``type`` is the first field so the row is tagged at its front.

``append_session_stats`` is the one writer for all three callers. It is computed from every row of the session: the archive
plus the live rows read just before the buffer is removed, each turn once (``turn_identity``). A sitting is a run of rows
split by the 30-minute gap rule of ``brain.body.session_hours.split_into_runs``; a stretch is the same split over the user's
rows. Days and clock times are local time. A stretch that runs past local midnight is listed under the day it started.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from brain import dev_constants
from brain.body.session_hours import split_into_runs
from brain.ingest.buffer import append_archive, read_archive, turn_identity
from brain.utils.time import lenient_utc, to_local

logger = logging.getLogger(__name__)

STATS_ROW_TYPE = "session_stats"


def _dated_rows(rows: list[dict]) -> list[tuple[datetime, dict]]:
    """The rows that count, each turn once, as (time, row) in time order.

    A summary row, a stats row and any row whose ``ts`` does not parse are not turns. The same turn in the archive and in the
    live buffer (the carried tail, a repeated close) counts once."""
    seen: set[tuple] = set()
    out: list[tuple[datetime, dict]] = []
    for row in rows:
        if row.get("speaker") == "summary" or row.get("type") == STATS_ROW_TYPE:
            continue
        ts = lenient_utc(row.get("ts"))
        if ts is None:
            continue
        identity = turn_identity(row)
        if identity in seen:
            continue
        seen.add(identity)
        out.append((ts, row))
    out.sort(key=lambda pair: pair[0])
    return out


def build_session_stats(rows: list[dict]) -> dict | None:
    """The stats row for these rows, or None when none of them is a dated turn."""
    dated = _dated_rows(rows)
    if not dated:
        return None
    sittings = split_into_runs([ts for ts, _ in dated])
    # A one-message sitting is 0 minutes and stays in the average.
    minutes = [(run[-1] - run[0]).total_seconds() / 60.0 for run in sittings]
    stretches = split_into_runs([ts for ts, row in dated if row.get("speaker") == "user"])
    by_day: dict[str, list[tuple[int, datetime, datetime]]] = {}
    for run in stretches:
        start, end = to_local(run[0]), to_local(run[-1])
        by_day.setdefault(start.date().isoformat(), []).append((len(run), start, end))
    busiest: dict[str, list[str]] = {}
    for day in sorted(by_day):
        top = sorted(by_day[day], key=lambda s: (-s[0], s[1]))[: dev_constants.BODY_STATS_BUSIEST_COUNT]
        busiest[day] = [f"{start:%H:%M}-{end:%H:%M}" for _, start, end in top]
    return {
        "type": STATS_ROW_TYPE,
        "avg_sitting_minutes": round(sum(minutes) / len(minutes), 1),
        "busiest": busiest,
    }


def append_session_stats(persona_dir: Path, session_id: str, live_rows: list[dict]) -> bool:
    """Append the stats row to ``session_id``'s archive; True when one was written.

    ``live_rows`` are the rows of the buffer about to be removed. An empty buffer gets no row, and neither does a file with
    no dated turn. If the row cannot be computed or written, the error is logged with the session and the caller's rollover or
    close still finishes (a record that is missing is a smaller loss than a conversation file left unfinished); nothing is
    retried, so the file simply has no stats row.

    A file whose archive already ends with a stats row gets no second one: that is the retry of a close or rollover whose
    buffer removal failed after the row was written (a turn archived after that row makes the row no longer last, and a new
    one follows it). Accepted edge: the callers hold the buffer's append lock while this reads the whole archive (a record
    written as the buffer finishes; appenders wait briefly).
    """
    if not live_rows:
        return False
    try:
        archived = read_archive(persona_dir, session_id)
        if archived and archived[-1].get("type") == STATS_ROW_TYPE:
            return False
        row = build_session_stats(archived + list(live_rows))
        if row is None:
            return False
        append_archive(persona_dir, session_id, [row])
    except Exception:
        logger.exception("session stats: the stats row of session %s could not be written; the file finishes without it", session_id)
        return False
    return True
