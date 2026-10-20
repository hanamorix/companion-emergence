"""Rotation primitives for JSONL audit logs.

Four policies, each a function over a single log path. Every one of them runs under
``file_lock(log_path)``, the lock every appender of these logs takes (``append_jsonl``), so a row
appended while a rotation runs waits for it instead of being truncated or rewritten away (#476).
The size is re-checked under the lock, since a peer may have rotated while this one waited:

* :func:`rotate_rolling_size` — size-capped rolling archive. Once the active
  log crosses ``max_bytes`` it is gzipped to ``.1.gz`` (written first, to a staging
  file), and older archives shift up (``.1.gz`` → ``.2.gz`` → ...). The oldest
  beyond ``archive_keep`` is dropped. Used for noisy logs where deep
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

import errno
import gzip
import io
import logging
import os
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import Recovery, iter_jsonl_raw, write_jsonl
from brain.utils.atomic_write import _replace_with_retry, fsync_dir, save_bytes_atomic
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

    Algorithm (#499: the new archive is written before anything is moved or evicted):

    1. Stat the file. Below cap → no-op.
    2. Gzip the active log into memory and write it to the staging file ``<log>.next.gz`` through the shared atomic
       writer. If that fails, nothing else has been touched.
    3. Move the existing archives up into the first empty slot (``.N.gz`` → ``.(N+1).gz``; the move into the last
       slot overwrites the oldest beyond ``archive_keep``, which is the normal eviction).
    4. Publish the staging file as ``.1.gz``.
    5. fsync the directory so the archive is crash-durable, then truncate the active log to zero bytes (writers
       reopen per append, so this is safe).

    ``lock_held=True`` is for a caller that already holds ``file_lock(log_path)`` for its own
    rotate-then-append step (``parse_failures``): the lock is not reentrant, so taking it again
    would block forever. The default takes the lock itself.

    If any of steps 2 to 4 fails (or the process is killed there), the active log is left untouched, no retained
    archive is lost beyond the one normal eviction, and the exception propagates. Better to fail the rotation than
    to lose audit data; the next tick will retry.
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
    """Archive the active log as ``.1.gz``, losing no retained archive and no log row if anything goes wrong (#499).

    The new archive is written FIRST, to the staging file ``<log>.next.gz``; only when that write succeeded are the
    old archives moved up, so a full disk or a permission error can no longer cost an archive. The move is
    gap-aware and re-runnable: it fills the first empty slot instead of assuming ``.1`` is the one to free, and the
    oldest archive is evicted only by being overwritten by the move that replaces it (never by an explicit delete),
    so the archives plus the staging file never fall below ``archive_keep`` after the one normal eviction. A failed
    or killed publish leaves ``.1`` empty and the next rotation just publishes again; a stale staging file is
    rewritten (its rows are a prefix of the still-untruncated log). The active log is truncated last. The one window
    left (a kill between the publish and the truncation archives those rows twice on the next rotation) is the one
    this code has always had.
    """
    new_archive = _archive_path(log_path, 1)
    staging = log_path.with_name(log_path.name + ".next.gz")

    # 1. Compress the active log into memory (read as a stream; the buffer holds the compressed result only).
    buf = io.BytesIO()
    with log_path.open("rb") as src, gzip.open(buf, "wb") as gz:
        shutil.copyfileobj(src, gz)
    data = buf.getvalue()

    # 2. Write the staging file through the shared atomic writer. Nothing else is touched unless this succeeds.
    try:
        save_bytes_atomic(staging, data)
    except OSError as exc:
        if exc.errno != errno.ENOSPC or not all(_archive_path(log_path, n).exists() for n in range(1, archive_keep + 1)):
            raise
        # A full disk with every slot occupied: free the ONE archive a successful rotation would evict anyway, then
        # try once more. Never more than this per rotation; after it a slot is empty, so a later failing rotation
        # evicts nothing.
        logger.warning("rotating %s: disk full; evicting the oldest archive to make room", log_path.name)
        _archive_path(log_path, archive_keep).unlink(missing_ok=True)
        save_bytes_atomic(staging, data)

    # 3. Move the archives up into the first empty slot (all slots occupied: the move into the last slot overwrites,
    #    and so evicts, the oldest).
    first_empty = next(
        (n for n in range(1, archive_keep + 1) if not _archive_path(log_path, n).exists()), archive_keep + 1
    )
    for i in range(min(first_empty, archive_keep) - 1, 0, -1):
        os.replace(_archive_path(log_path, i), _archive_path(log_path, i + 1))  # atomic-write: ok -- the rotation's shift is a plain rename of a finished archive up one slot, not a write

    # 4. Publish the staged archive as .1.gz.
    _replace_with_retry(staging, new_archive)  # atomic-write: ok -- publishes the finished, fsynced staging file by rename (the write above went through the shared writer)

    # 5. Make the archive crash-durable (the shift and the publish are renames), THEN truncate the active log, so a
    #    power cut cannot leave the log emptied and the archive's directory entry not yet on disk.
    fsync_dir(log_path.parent)
    # 6. Truncate the active log. Appenders wait on the lock held here, so nothing is truncated away.
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
    already hold are appended, and the whole is compressed in memory and written through the shared atomic
    writer (a unique temp, fsync, one retried replace). A failure at any step removes the temp and leaves the
    existing archive as it was.

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
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8", newline="\n") as gz:  # json-io: ok -- raw-line gz archive: lines pass through unchanged, json_io has no gz writer
        for raw in (*existing, *to_add):
            gz.write(raw + "\n")
    # The shared atomic writer: a unique temp, fsync, one retried replace; a failure at any step removes the temp
    # and leaves the existing archive as it was.
    save_bytes_atomic(archive_path, buf.getvalue())


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
            log_path.replace(log_path.with_name(f"{log_path.name}.1"))  # atomic-write: ok -- moves the finished log aside to .1 (a rename of a finished file, not a write)


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
