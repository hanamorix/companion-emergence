"""Age-based reaping of stale sidecar files in the persona directory (#176).

Two families accumulate at the persona root and nothing reaped them:

* ``<file>.lock.stale-<stamp>`` — a dead bridge lock archived as evidence by
  ``brain.bridge.daemon._archive_stale_lock`` (never deleted at the time, on
  purpose: it is the record of a crash).
* ``<file>.corrupt-<stamp>`` / ``<file>.bakN.corrupt-<stamp>`` — a corrupt
  state file quarantined by ``brain.health.attempt_heal`` before the heal.

Both are forensic residue with a shelf life. Live ``<file>.lock`` sidecars
are NOT touched: ``brain.utils.file_lock`` documents that removing them
between operations races the lock itself. ``.bak`` rotation is bounded on
its own and is not this module's concern.

Runs on the supervisor's maintenance cadence (with forgetting, narrative,
and the pending-write sweep). Pure function over the directory; no LLM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ponytail: one fixed shelf life; make it an ops tunable if anyone asks.
MAX_AGE_DAYS = 14
_PATTERNS = ("*.lock.stale-*", "*.corrupt-*")


def sweep_stale_sidecars(
    persona_dir: Path, *, now: datetime, max_age_days: float = MAX_AGE_DAYS
) -> list[Path]:
    """Delete stale-lock archives and corruption quarantines older than
    ``max_age_days`` (by mtime). Returns the paths removed. Never raises."""
    if not persona_dir.is_dir():
        return []
    cutoff = now.timestamp() - timedelta(days=max_age_days).total_seconds()
    removed: list[Path] = []
    for pattern in _PATTERNS:
        for path in persona_dir.glob(pattern):
            try:
                if not path.is_file() or path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed.append(path)
            except OSError as exc:
                logger.warning("sidecar sweep: could not remove %s: %s", path, exc)
    if removed:
        logger.info("sidecar sweep: removed %d stale sidecar(s)", len(removed))
    return removed
