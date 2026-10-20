"""Evidence for model replies that failed to parse (#364).

Sites that parse JSON out of a model reply keep their tolerance (fence strip,
brace scrape, a safe default), but the raw reply used to vanish with the
failure. ``record_parse_failure`` keeps it: one row per failure in
``parse_failures.jsonl`` (site, error, the reply truncated to RAW_CHARS, its
full length). The file is size-capped through the shared rotation, so it never
grows without bound.

Reader: ``count_recent`` feeds ``/health``'s ``parse_failures_24h``, so a site
that fails quietly shows up as a count per site.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.json_io import append_jsonl, iter_jsonl
from brain.health.log_rotation import rotate_rolling_size
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

PARSE_FAILURE_LOG = "parse_failures.jsonl"
RAW_CHARS = 2_000
# `error` can carry reply text (pydantic's input_value=..., a model's own value), so it is
# capped, and reduced to its class when the raw text is not kept (#427 review).
ERROR_CHARS = 200
# What a keep_raw=False class may look like: "unknown decision", "JSONDecodeError".
_ERROR_CLASS = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]{0,39}")
MAX_BYTES = 1024 * 1024
ARCHIVE_KEEP = 2
# Pass-2 worker and supervisor ticks write from different threads; rotation
# then append must not interleave, or a rotation drops a concurrent row.
_WRITE_LOCK = threading.Lock()


def record_parse_failure(
    persona_dir: Path | None, *, site: str, raw: str, error: str, keep_raw: bool = True
) -> None:
    """Append the failed reply to the persona's parse-failure log. Never raises:
    a failure to record evidence must not fail the caller's fallback.

    ``keep_raw=False`` stores only the length, and only the error's class (the
    text before its first colon, kept only if it looks like a bare class name,
    else ``"error"``): for replies about another Kindled's words,
    which stay inside ``kindled_link/``. The runtime log line carries neither
    reply nor error text: the row in the file is the evidence."""
    logger.warning(
        "%s: model reply did not parse (%d chars); see %s", site, len(raw), PARSE_FAILURE_LOG
    )
    if persona_dir is None:
        return
    path = Path(persona_dir) / PARSE_FAILURE_LOG
    try:
        error = str(error)
        if not keep_raw:
            cls = error.split(":", 1)[0].strip()
            error = cls if _ERROR_CLASS.fullmatch(cls) else "error"
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "site": site,
            "error": error[:ERROR_CHARS],
            "raw": raw[:RAW_CHARS] if keep_raw else "",
            "raw_len": len(raw),
        }
        # The thread lock serialises this process; the file lock makes rotate-then-append one step across
        # processes too (a rotation truncates the log under the appender otherwise).
        with _WRITE_LOCK, file_lock(path):
            rotate_rolling_size(path, MAX_BYTES, ARCHIVE_KEEP, lock_held=True)
            append_jsonl(path, row, lock="held", nonfinite="null")
    except Exception:  # noqa: BLE001 -- evidence is best-effort by contract
        logger.warning("could not record a %s parse failure to %s", site, path, exc_info=True)


def count_recent(persona_dir: Path, *, hours: float = 24.0, now: datetime | None = None) -> dict[str, int]:
    """Failures per site within the last ``hours`` (active log only). Never
    raises: ``/health`` reads it on the restart poll."""
    path = Path(persona_dir) / PARSE_FAILURE_LOG
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=hours)
    counts: Counter[str] = Counter()
    # One bad line costs only itself (the module skips it); an unreadable file reads as empty with a warning.
    for row in iter_jsonl(path):
        try:
            if row["ts"] and datetime.fromisoformat(row["ts"]) >= cutoff:
                counts[str(row.get("site", "unknown"))] += 1
        except (KeyError, TypeError, ValueError, AttributeError):
            continue  # tz-less or otherwise unusable row
    return dict(counts)
