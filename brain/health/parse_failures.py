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

import json
import logging
import threading
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.log_rotation import rotate_rolling_size

logger = logging.getLogger(__name__)

PARSE_FAILURE_LOG = "parse_failures.jsonl"
RAW_CHARS = 2_000
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

    ``keep_raw=False`` stores only the length: for replies about another
    Kindled's words, which stay inside ``kindled_link/``. The runtime log line
    never carries reply text."""
    logger.warning("%s: model reply did not parse (%s; %d chars)", site, error, len(raw))
    if persona_dir is None:
        return
    path = Path(persona_dir) / PARSE_FAILURE_LOG
    row = {
        "ts": datetime.now(UTC).isoformat(),
        "site": site,
        "error": error,
        "raw": raw[:RAW_CHARS] if keep_raw else "",
        "raw_len": len(raw),
    }
    try:
        with _WRITE_LOCK:
            rotate_rolling_size(path, MAX_BYTES, ARCHIVE_KEEP)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 -- evidence is best-effort by contract
        logger.warning("could not record a %s parse failure to %s", site, path, exc_info=True)


def count_recent(persona_dir: Path, *, hours: float = 24.0, now: datetime | None = None) -> dict[str, int]:
    """Failures per site within the last ``hours`` (active log only). Never
    raises: ``/health`` reads it on the restart poll."""
    path = Path(persona_dir) / PARSE_FAILURE_LOG
    if not path.exists():
        return {}
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=hours)
    counts: Counter[str] = Counter()
    try:
        # Bytes, decoded per line: one bad byte costs its line, not the count.
        lines = path.read_bytes().splitlines()
    except OSError:
        logger.warning("could not read %s", path, exc_info=True)
        return {}
    for line in lines:
        try:
            row = json.loads(line.decode("utf-8"))
            if row["ts"] and datetime.fromisoformat(row["ts"]) >= cutoff:
                counts[str(row.get("site", "unknown"))] += 1
        except (KeyError, TypeError, ValueError, AttributeError):
            continue  # corrupt, non-UTF-8, tz-less or non-dict row
    return dict(counts)
