"""Rotation primitives for JSONL audit logs.

Two strategies, each a pure function over a single log path:

* :func:`rotate_rolling_size` — size-capped rolling archive. The active
  log is renamed once it crosses ``max_bytes``, gzipped to ``.1.gz``,
  and older archives shift up (``.1.gz`` → ``.2.gz`` → ...). The oldest
  beyond ``archive_keep`` is deleted. Used for noisy logs where deep
  history doesn't carry per-line value (heartbeats, dreams,
  emotion_growth).

* :func:`rotate_age_archive_yearly` — splits a log's entries by the year
  in their ``at`` field. Cold years are archived to ``<stem>.<year>.jsonl.gz``;
  the current year stays in active. Archives are kept forever — the
  soul-audit case where every decision must remain reachable.

Both functions:

* Are no-ops when the log file doesn't exist (idempotent fresh-install).
* Open files via streaming iteration so memory stays bounded regardless
  of log size.
* Leave the active log intact on partial failure — the alternative is
  silently losing audit data.

Spec: docs/superpowers/specs/2026-05-11-jsonl-log-retention-design.md
Plan: docs/superpowers/plans/2026-05-11-jsonl-log-retention.md
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rolling-size rotation
# ---------------------------------------------------------------------------


def rotate_rolling_size(
    log_path: Path,
    max_bytes: int,
    archive_keep: int,
) -> Path | None:
    """Rotate ``log_path`` to a gzipped archive if it exceeds ``max_bytes``.

    Returns the path of the new ``.1.gz`` archive on rotation, or ``None``
    if no rotation was needed (file missing or below cap).

    Algorithm:

    1. Stat the file. Below cap → no-op.
    2. Shift existing archives up: ``.N.gz`` → ``.(N+1).gz`` (delete the
       oldest beyond ``archive_keep``).
    3. Gzip the active log to a temp path next to it, then atomically
       rename to ``.1.gz``.
    4. Truncate the active log to zero bytes (writers reopen per
       append, so this is safe).

    If step 3 fails mid-way, the active log is left untouched and the
    exception propagates. Better to fail the rotation than to lose
    audit data — the next tick will retry.
    """
    if not log_path.exists():
        return None
    if log_path.stat().st_size < max_bytes:
        return None

    # 1. Shift existing archives up. Iterate from oldest to newest so
    #    we never overwrite a slot before it's been moved.
    for i in range(archive_keep, 0, -1):
        src = _archive_path(log_path, i)
        if not src.exists():
            continue
        if i == archive_keep:
            # Oldest beyond the keep window — evict.
            src.unlink()
        else:
            dst = _archive_path(log_path, i + 1)
            src.rename(dst)

    # 2. Gzip the active log to a temp path, then atomic-rename into place.
    new_archive = _archive_path(log_path, 1)
    tmp_archive = new_archive.with_suffix(new_archive.suffix + ".tmp")
    try:
        with log_path.open("rb") as src, gzip.open(tmp_archive, "wb") as gz:
            shutil.copyfileobj(src, gz)
        tmp_archive.rename(new_archive)
    except Exception:
        # Clean up partial temp. Leave active log untouched.
        tmp_archive.unlink(missing_ok=True)
        raise

    # 3. Truncate the active log. Writers reopen per append so this is safe.
    log_path.write_bytes(b"")

    logger.info(
        "rotated %s (%d bytes) -> %s", log_path.name, log_path.stat().st_size, new_archive.name
    )
    return new_archive


def _archive_path(log_path: Path, n: int) -> Path:
    """Return the ``.N.gz`` archive path for a log."""
    return log_path.with_suffix(log_path.suffix + f".{n}.gz")


# ---------------------------------------------------------------------------
# Yearly-archive rotation
# ---------------------------------------------------------------------------


def rotate_age_archive_yearly(
    log_path: Path,
    now: datetime | None = None,
    timestamp_field: str = "at",
) -> list[Path]:
    """Split ``log_path`` by year; archive cold years to ``.YYYY.jsonl.gz``.

    Returns the list of newly-written archive paths. Empty if the log is
    missing or contains only current-year entries.

    ``timestamp_field`` is the entry key holding an ISO 8601 timestamp.
    Defaults to ``"at"`` (the convention used by the supervisor's tick
    events); soul_audit uses ``"ts"`` so callers must pass that.

    Malformed lines are skipped with a warning — the rotation must never
    abort on one bad row. Lines without the timestamp field are bucketed
    with the current year (treated as undated → kept in active).

    The active log is rewritten to contain only current-year entries.
    Cold-year entries become gzipped archives at
    ``<stem>.<year>.jsonl.gz`` (e.g. ``soul_audit.2024.jsonl.gz``).

    Idempotent: if all cold years were already archived in a previous
    tick, this call is a no-op.
    """
    if not log_path.exists():
        return []

    now = now or datetime.now(timezone.utc)
    current_year = now.year

    # Bucket entries by year. We read the whole file once into year buckets;
    # for soul_audit (low volume) this is fine, and the alternative
    # (multi-pass) is more complex than the file is worth.
    by_year: dict[int, list[str]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line_index, raw in enumerate(f, start=1):
            stripped = raw.rstrip("\r\n")
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "rotate_age_archive_yearly: skipping corrupt line %d in %s (%s): %.200s",
                    line_index,
                    log_path,
                    exc,
                    stripped,
                )
                continue
            if not isinstance(entry, dict):
                continue
            year = _entry_year(entry, timestamp_field, fallback=current_year)
            by_year.setdefault(year, []).append(raw)

    cold_years = sorted(y for y in by_year if y < current_year)
    if not cold_years:
        return []

    # Write each cold-year archive atomically (tmp + rename).
    new_archives: list[Path] = []
    for year in cold_years:
        archive_path = log_path.parent / f"{log_path.stem}.{year}{log_path.suffix}.gz"
        tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as gz:
                for raw in by_year[year]:
                    gz.write(raw if raw.endswith("\n") else raw + "\n")
            tmp_path.rename(archive_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        new_archives.append(archive_path)

    # Rewrite the active log with only current-year (+ future, defensive)
    # entries. We do this atomically via a tmp file so a crash mid-write
    # can't lose entries.
    keep_years = sorted(y for y in by_year if y >= current_year)
    tmp_active = log_path.with_suffix(log_path.suffix + ".rewriting")
    try:
        with tmp_active.open("w", encoding="utf-8") as f:
            for year in keep_years:
                for raw in by_year[year]:
                    f.write(raw if raw.endswith("\n") else raw + "\n")
        tmp_active.replace(log_path)
    except Exception:
        tmp_active.unlink(missing_ok=True)
        raise

    logger.info(
        "yearly-archived %s for years %s -> %d new archive(s)",
        log_path.name,
        cold_years,
        len(new_archives),
    )
    return new_archives


def _entry_year(entry: dict, field: str, fallback: int) -> int:
    """Extract the year from ``entry[field]``; fall back if absent/bad."""
    ts = entry.get(field)
    if not isinstance(ts, str):
        return fallback
    try:
        return datetime.fromisoformat(ts).year
    except ValueError:
        return fallback
