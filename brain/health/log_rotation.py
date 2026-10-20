"""Rotation primitives for JSONL audit logs.

Four policies, each a function over a single log path. Every one of them runs under
``file_lock(log_path)``, the lock every appender of these logs takes (``append_jsonl``), so a row
appended while a rotation runs waits for it instead of being truncated or rewritten away (#476).
The size is re-checked under the lock, since a peer may have rotated while this one waited:

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

* :func:`rotate_single_backup` — size-capped, one ``.1`` backup (the MCP tool audit log).

* :func:`trim_jsonl_by_age` — drops rows older than a cutoff (the chat tick log).

The rolling-size and yearly functions:

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
import logging
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

from brain.health.attempt_heal import _replace_with_retry
from brain.health.json_io import Recovery, iter_jsonl_raw, write_jsonl
from brain.utils.file_lock import file_lock
from brain.utils.time import lenient_utc

logger = logging.getLogger(__name__)


@contextmanager
def _append_lock(log_path: Path) -> Iterator[None]:
    """Hold the appenders' lock for ``log_path``; a filesystem that refuses the lock degrades to unlocked.

    Skipping rotation because the lock call raised (some network mounts) would let the log grow
    without bound, and the worst case without the lock is the pre-#357 race, so proceed with a
    warning, as ``append_jsonl`` does.
    """
    with ExitStack() as held:
        try:
            held.enter_context(file_lock(log_path))
        except OSError as exc:
            logger.warning("log lock unavailable for %s, rotating without a lock: %s", log_path.name, exc)
        yield


# ---------------------------------------------------------------------------
# Rolling-size rotation
# ---------------------------------------------------------------------------


def rotate_rolling_size(
    log_path: Path,
    max_bytes: int,
    archive_keep: int,
    *,
    lock_held: bool = False,
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

    ``lock_held=True`` is for a caller that already holds ``file_lock(log_path)`` for its own
    rotate-then-append step (``parse_failures``): the lock is not reentrant, so taking it again
    would block forever. The default takes the lock itself.

    If step 3 fails mid-way, the active log is left untouched and the
    exception propagates. Better to fail the rotation than to lose
    audit data — the next tick will retry.
    """
    if not _over(log_path, max_bytes):
        return None
    with nullcontext() if lock_held else _append_lock(log_path):
        # A peer rotator, or the same tick on another process, may have rotated while this one waited.
        if not _over(log_path, max_bytes):
            return None
        return _rotate_rolling_locked(log_path, archive_keep)


def _over(log_path: Path, max_bytes: int) -> bool:
    try:
        return log_path.stat().st_size >= max_bytes
    except FileNotFoundError:
        return False


def _rotate_rolling_locked(log_path: Path, archive_keep: int) -> Path:
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

    # 3. Truncate the active log. Appenders wait on the lock held here, so nothing is truncated away.
    rotated_bytes = log_path.stat().st_size  # before the truncation: the size that went into the archive
    log_path.write_bytes(b"")

    logger.info("rotated %s (%d bytes) -> %s", log_path.name, rotated_bytes, new_archive.name)
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

    now = now or datetime.now(UTC)
    # Read, archive and rewrite under the appenders' lock: a row appended between the read and
    # the rewrite would otherwise be replaced away (#476).
    with _append_lock(log_path):
        return _rotate_yearly_locked(log_path, now, timestamp_field)


def _rotate_yearly_locked(log_path: Path, now: datetime, timestamp_field: str) -> list[Path]:
    current_year = now.year

    # Bucket entries by year. We read the whole file once into year buckets;
    # for soul_audit (low volume) this is fine, and the alternative
    # (multi-pass) is more complex than the file is worth.
    by_year: dict[int, list[str]] = {}
    # RAISE: the active log is rewritten below from what was read, so a failed read must stop the
    # rotation (and leave the log alone), never rewrite it with only the prefix. Corrupt lines are
    # skipped with a warning by the module; the line text is kept exactly as read.
    for line, entry in iter_jsonl_raw(log_path, recover=Recovery.RAISE):
        year = _entry_year(entry, timestamp_field, fallback=current_year)
        by_year.setdefault(year, []).append(line)

    cold_years = sorted(y for y in by_year if y < current_year)
    if not cold_years:
        return []

    # Write each cold-year archive atomically; one that already exists is merged into, never replaced.
    new_archives: list[Path] = []
    for year in cold_years:
        archive_path = log_path.parent / f"{log_path.stem}.{year}{log_path.suffix}.gz"
        _merge_into_year_archive(archive_path, by_year[year])
        new_archives.append(archive_path)

    # Rewrite the active log with only current-year (+ future, defensive)
    # entries. We do this atomically via a tmp file so a crash mid-write
    # can't lose entries.
    keep_years = sorted(y for y in by_year if y >= current_year)
    kept = "".join(raw + "\n" for year in keep_years for raw in by_year[year])
    write_jsonl(log_path, kept.encode("utf-8"), fsync=False)

    logger.info(
        "yearly-archived %s for years %s -> %d new archive(s)",
        log_path.name,
        cold_years,
        len(new_archives),
    )
    return new_archives


def _merge_into_year_archive(archive_path: Path, new_lines: list[str]) -> None:
    """Add ``new_lines`` to the year archive, keeping every row already in it, and adding only rows it does not already hold.

    The archive is kept forever, so an existing one is read (its lines verbatim), the lines it does not
    already hold are appended, and the whole is written to a temp file and put in place with a retried
    replace. A failure at any step removes the temp and leaves the existing archive as it was.

    Idempotent: the active log is rewritten AFTER this merge, so a failed or interrupted rewrite leaves
    the cold rows in the active log and the next tick offers them again; a row the archive already holds
    is skipped. The skip is count-aware: a line offered twice is added twice unless the archive already
    holds two copies, so genuinely identical rows survive.
    """
    existing: list[str] = []
    if archive_path.exists():
        with gzip.open(archive_path, "rt", encoding="utf-8") as gz:  # json-io: ok -- raw-line gz archive: existing lines pass through unchanged, json_io has no gz reader that keeps damaged lines
            existing = [line.rstrip("\n") for line in gz if line.strip()]
    held = Counter(existing)
    to_add: list[str] = []
    for raw in new_lines:
        if held[raw] > 0:
            held[raw] -= 1
        else:
            to_add.append(raw)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as gz:  # json-io: ok -- raw-line gz archive: lines pass through unchanged, json_io has no gz writer
            for raw in (*existing, *to_add):
                gz.write(raw + "\n")
        _replace_with_retry(tmp_path, archive_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Single-backup rotation (size-capped, one .1 backup)
# ---------------------------------------------------------------------------


def rotate_single_backup(log_path: Path, max_bytes: int) -> None:
    """Move ``log_path`` to ``<name>.1`` (replacing any older backup) once it exceeds ``max_bytes``.

    Every appender may run this, so the check-then-rename is serialised on the appenders' lock and
    re-checked under it (#357): a writer that saw an oversize log must not rename the fresh one a
    concurrent rotator left behind over ``.1``. The size check first runs unlocked so the common
    no-rotation append takes no lock. Raises OSError if the rename fails.
    """
    if not _exceeds(log_path, max_bytes):
        return
    with _append_lock(log_path):
        if _exceeds(log_path, max_bytes):
            # replace() overwrites an existing .1 atomically on every OS.
            log_path.replace(log_path.with_name(f"{log_path.name}.1"))


def _exceeds(log_path: Path, max_bytes: int) -> bool:
    try:
        return log_path.stat().st_size > max_bytes
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# Age trim
# ---------------------------------------------------------------------------


def trim_jsonl_by_age(
    log_path: Path,
    cutoff: datetime,
    *,
    timestamp_field: str = "ts",
    keep_undated: bool = False,
) -> int:
    """Drop the rows of ``log_path`` older than ``cutoff``; return how many were dropped.

    The file is read again under the appenders' lock, so a row appended since the caller last read
    it is kept, and the rest is written back with the lines exactly as read. A row whose timestamp
    is missing or unreadable is dropped unless ``keep_undated`` (a naive timestamp is read as UTC).
    Nothing is rewritten when nothing is dropped. Raises OSError on a failed read or write, leaving
    the log as it was.
    """
    if not log_path.exists():
        return 0
    with _append_lock(log_path):
        kept: list[str] = []
        dropped = 0
        for line, entry in iter_jsonl_raw(log_path, recover=Recovery.RAISE):
            when = _entry_time(entry, timestamp_field)
            if (when is None and keep_undated) or (when is not None and when >= cutoff):
                kept.append(line)
            else:
                dropped += 1
        if dropped:
            write_jsonl(log_path, "".join(line + "\n" for line in kept).encode("utf-8"), fsync=False)
        return dropped


def _entry_time(entry: dict | None, field: str) -> datetime | None:
    return lenient_utc(entry.get(field)) if entry else None


def _entry_year(entry: dict, field: str, fallback: int) -> int:
    """Extract the year from ``entry[field]``; fall back if absent/bad."""
    ts = entry.get(field)
    if not isinstance(ts, str):
        return fallback
    try:
        return datetime.fromisoformat(ts).year
    except ValueError:
        return fallback
