"""Recall diagnostics log: one JSONL record per passive recall or
``search_memories`` call (name-recall fix, spec §6, S15/S30/S48/S53/S56/S59).

The file is ``<persona_dir>/recall_diagnostics.log.jsonl`` (S59). It is
diagnostic data only: nothing here touches memories.db or the calibration log
(S30). Each record carries message-level fields (paragraph count, whether the
whole-message fallback was used, total rerank width, time budget) and a
per-paragraph list of (path, width, pass mark, scale).

Record shape (one JSON object per line)::

    {"ts": "<UTC ISO>", "source": "passive" | "tool", "mode": "<tool only>",
     "paragraph_count": int, "whole_message_fallback": bool,
     "total_width": int, "budget": float | null,       # budget in seconds
     "paragraphs": [{"path": "reranked" | "cosine", "width": int,
                     "pass_mark": float | null, "scale": str | null}, ...]}

Concurrency: every writer (append, prune) holds the same OS-level
``file_lock`` on the log path, so a prune's read -> rewrite window can never
swallow a concurrent append (plan §5, CONC-3). The prune rewrites through a
temp file and ``os.replace`` while holding the lock; the data file is never
held open across the replace (Windows refuses to replace an open file).

The append never waits on the reply path (S189, I6): it tries the lock once
without blocking. If the lock is held (a prune in progress, or another append)
the record goes into an in-memory buffer; the buffer is written, oldest first,
by the next append that gets the lock, by the supervisor's lull job
(``recall_diagnostics_flush``, due while ``has_pending``), at the end of a prune,
and at a clean stop (the bridge's stop, ``nell chat --no-bridge``'s exit), the
same shape as the cost-series buffer (S164). A buffered record older than the retention window is dropped when the
buffer is next touched. Records still in memory when the process dies are lost
(accepted, the same shape as S164/S176: this is diagnostic data).

Fail-soft (INV-I13a): a diagnostics failure must never reach the recall
caller. ``log_recall``, ``append_record`` and ``prune`` swallow every
``Exception``, log it, and return a falsy / zero result. ``build_record`` is
the one strict piece (it validates), and ``log_recall`` catches its errors.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain import dev_constants
from brain.health.json_io import Recovery, append_jsonl, encode_jsonl, iter_jsonl_raw, write_jsonl
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

SOURCE_PASSIVE = "passive"
SOURCE_TOOL = "tool"
PATH_RERANKED = "reranked"
PATH_COSINE = "cosine"

_VALID_SOURCES = frozenset({SOURCE_PASSIVE, SOURCE_TOOL})
_VALID_PATHS = frozenset({PATH_RERANKED, PATH_COSINE})


@dataclass(frozen=True)
class ParagraphDiagnostic:
    """One paragraph's outcome: which path served it, how many real
    candidates it was given (``width``), the pass mark applied and the score
    scale that pass mark is in (e.g. ``"normalized"`` for the reranked path,
    a cosine scale name for the cosine path)."""

    path: str
    width: int
    pass_mark: float | None
    scale: str | None


def diagnostics_path(persona_dir: Path) -> Path:
    """``<persona_dir>/recall_diagnostics.log.jsonl`` (S59)."""
    return Path(persona_dir) / dev_constants.RECALL_DIAGNOSTICS_LOG_FILENAME


def _utc(when: datetime | None) -> datetime:
    if when is None:
        return datetime.now(UTC)
    if when.tzinfo is None:
        return when.replace(tzinfo=UTC)
    return when.astimezone(UTC)


def _finite_or_none(value: Any) -> float | None:
    """A JSON-safe float: NaN/inf become null (json.dumps would emit the
    non-standard tokens NaN/Infinity, which strict readers reject)."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def build_record(
    *,
    source: str,
    paragraph_count: int,
    whole_message_fallback: bool,
    total_width: int,
    budget: float | None,
    paragraphs: Iterable[ParagraphDiagnostic],
    mode: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build one diagnostics record. Strict: raises ``ValueError`` on an
    unknown ``source`` or paragraph ``path`` (a programming error; the
    fail-soft entry point ``log_recall`` catches it). ``mode`` is the tool's
    search mode and is omitted for a passive recall. ``now`` is injectable."""
    if source not in _VALID_SOURCES:
        raise ValueError(f"recall diagnostics: unknown source {source!r}")
    rows: list[dict[str, Any]] = []
    for para in paragraphs:
        if para.path not in _VALID_PATHS:
            raise ValueError(f"recall diagnostics: unknown paragraph path {para.path!r}")
        rows.append(
            {
                "path": para.path,
                "width": int(para.width),
                "pass_mark": _finite_or_none(para.pass_mark),
                "scale": para.scale,
            }
        )
    record: dict[str, Any] = {
        "ts": _utc(now).isoformat(),
        "source": source,
    }
    if mode is not None:
        record["mode"] = mode
    record.update(
        {
            "paragraph_count": int(paragraph_count),
            "whole_message_fallback": bool(whole_message_fallback),
            "total_width": int(total_width),
            "budget": _finite_or_none(budget),
            "paragraphs": rows,
        }
    )
    return record


# Records that could not take the lock (S189): path -> [(record ts, encoded line)],
# oldest first. In memory only; a crash loses them (accepted).
_pending_lock = threading.Lock()
_pending: dict[str, list[tuple[datetime, bytes]]] = {}


def _entry_ts(record: dict[str, Any]) -> datetime:
    """The record's own ``ts`` (for the buffer's age check), or now if it has
    none that parses."""
    try:
        return _utc(datetime.fromisoformat(record["ts"]))
    except Exception:  # noqa: BLE001 — an unparseable ts ages from the moment of buffering
        return datetime.now(UTC)


def _drop_stale(entries: list[tuple[datetime, bytes]]) -> list[tuple[datetime, bytes]]:
    """Entries within the retention window (the window the prune uses, read now).
    If the window cannot be read, nothing is dropped."""
    try:
        cutoff = datetime.now(UTC) - timedelta(days=_default_window_days())
    except Exception:  # noqa: BLE001 — diagnostics must never reach recall
        return entries
    return [entry for entry in entries if entry[0] >= cutoff]


def _buffer(key: str, entry: tuple[datetime, bytes]) -> None:
    with _pending_lock:
        _pending[key] = _drop_stale([*_pending.get(key, []), entry])


def _take_pending(key: str) -> list[bytes]:
    """Remove and return the buffered lines for ``key`` that are still within the
    retention window, oldest first."""
    with _pending_lock:
        entries = _pending.pop(key, [])
        if not entries:
            return []
        return [line for _, line in _drop_stale(entries)]


def _write_lines(path: Path, lines: list[bytes]) -> None:
    # No fsync: this runs on the recall path (I6) and the data is diagnostic. The caller holds the
    # non-blocking file lock (never-wait contract, S189), so the append takes none of its own.
    append_jsonl(path, b"".join(lines), lock="held", fsync=False)


def append_record(path: Path, record: dict[str, Any]) -> bool:
    """Write ``record`` as one JSON line, or hold it in memory if the lock is
    busy (S189: never waits). Returns True if the record was written or buffered,
    False if it was dropped; never raises (a failure is logged)."""
    try:
        # encode_jsonl keeps every line pure ASCII, so no exotic separator (U+2028) can split a line for a
        # `str.splitlines()` reader; a NaN is written as null (it used to drop the whole record).
        line = encode_jsonl(record, encode_default=str, nonfinite="null")
        key = str(path)
        with file_lock(path, blocking=False) as acquired:
            if not acquired:
                _buffer(key, (_entry_ts(record), line))
            else:
                _write_lines(path, [*_take_pending(key), line])
                return True
        # Lock was held: the record is buffered. One more non-blocking try, in case the
        # holder released between our failed attempt and the buffering (then nothing
        # would flush a record that has no later append).
        flush_pending(path)
        return True
    except Exception as exc:  # noqa: BLE001 — diagnostics must never reach recall
        # A warning without a traceback: a persistent failure (unwritable dir, full disk)
        # repeats on every recall and must not flood the log with stack traces.
        logger.warning("recall diagnostics: append to %s failed (%r); record dropped", path, exc)
        return False


def has_pending(path: Path) -> bool:
    """True while records are buffered in this process for ``path`` (the lull
    job's due test; opens nothing, takes no lock)."""
    with _pending_lock:
        return bool(_pending.get(str(path)))


def flush_pending(path: Path) -> int:
    """Write the records buffered for ``path`` if the lock is free now (the
    lull flush, S189); returns how many lines were written. Does nothing, and
    creates no lock file, when nothing is buffered. Never waits, never raises."""
    key = str(path)
    try:
        with _pending_lock:
            if not _pending.get(key):
                return 0
        with file_lock(path, blocking=False) as acquired:
            if not acquired:
                return 0
            lines = _take_pending(key)
            if lines:
                _write_lines(path, lines)
            return len(lines)
    except Exception as exc:  # noqa: BLE001 — diagnostics must never reach recall
        logger.warning("recall diagnostics: flush to %s failed (%r); records dropped", path, exc)
        return 0


def log_recall(
    path: Path,
    *,
    source: str,
    paragraph_count: int,
    whole_message_fallback: bool,
    total_width: int,
    budget: float | None,
    paragraphs: Sequence[ParagraphDiagnostic],
    mode: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Build and append one record. The call sites use this (in a ``finally``,
    P-22). Never raises: any failure (bad input included) is logged and the
    record dropped; returns whether the record was written or buffered (S189)."""
    try:
        record = build_record(
            source=source,
            paragraph_count=paragraph_count,
            whole_message_fallback=whole_message_fallback,
            total_width=total_width,
            budget=budget,
            paragraphs=paragraphs,
            mode=mode,
            now=now,
        )
    except Exception:  # noqa: BLE001 — diagnostics must never reach recall
        logger.exception("recall diagnostics: could not build a record; dropped")
        return False
    return append_record(path, record)


def _record_ts(record: dict | None) -> datetime | None:
    """The record's ``ts`` as an aware UTC datetime, or None if the line did not parse to a
    JSON object or its ``ts`` is not parseable."""
    if record is None:
        return None
    try:
        return _utc(datetime.fromisoformat(record["ts"]))
    except Exception:  # noqa: BLE001 — any ts that cannot be aged (missing, wrong type, an out-of-range ts's OverflowError) is un-agable
        return None


def _default_window_days() -> float:
    """The calibration log's live retention window (S48: pruned "on the
    calibration log's retention window"): the same tunable, read at call time,
    that ``MemoryStore.prune_calibration_log`` reads."""
    from brain import tunables
    from brain.memory.store import CALIBRATION_LOG_RETENTION_WINDOW_DAYS

    return float(
        tunables.get_tunable(
            "calibration.retention_window_days", CALIBRATION_LOG_RETENTION_WINDOW_DAYS
        )
    )


def prune(path: Path, *, window_days: float | None = None, now: datetime | None = None) -> int:
    """Prune, then write any records that were buffered while the lock was held
    (S189). See ``_prune`` for the pruning rules; this wrapper never raises."""
    try:
        return _prune(path, window_days=window_days, now=now)
    finally:
        flush_pending(path)


def _prune(path: Path, *, window_days: float | None, now: datetime | None) -> int:
    """Delete records older than the retention window; return how many were
    removed (0 if none, or on any failure). Never raises.

    The whole read -> filter -> atomic rewrite runs under the
    same ``file_lock`` the appends take, so an append attempted in the window
    finds the lock held, is buffered and lands after the rewrite (CONC-3, S189). A record is kept iff its
    ``ts`` is at or after ``now - window_days``. A line that is not a JSON
    object with a parseable ``ts`` (a torn write from a crash, corruption)
    cannot be aged and is removed, so it cannot live forever. Nothing is
    written when nothing is removed. If the temp write or ``os.replace``
    fails (e.g. a Windows reader holds the file), the original file is left
    intact, the temp file is removed, and the failure is logged (INV-I13a).
    """
    try:
        if window_days is None:
            window_days = _default_window_days()
        if not path.exists():
            return 0  # nothing to prune; do not create a lock sidecar for a log that does not exist
        cutoff = _utc(now) - timedelta(days=window_days)
        with file_lock(path):
            kept: list[str] = []
            removed = 0
            unagable = 0
            # RAISE (a failed read must not shrink the log) + bad_lines (a line that does not parse is still
            # counted as un-agable and removed, so it cannot live forever).
            # Read the whole file first (as the byte read did), so the filter below sees one fixed snapshot.
            rows = list(iter_jsonl_raw(path, recover=Recovery.RAISE, bad_lines=True))
            for line, record in rows:
                ts = _record_ts(record)
                if ts is not None and ts >= cutoff:
                    kept.append(line)
                else:
                    removed += 1
                    if ts is None:
                        unagable += 1
            if removed == 0:
                return 0
            if unagable:
                logger.warning(
                    "recall diagnostics: removing %d line(s) of %s with no readable ts",
                    unagable,
                    path,
                )
            write_jsonl(path, "".join(line + "\n" for line in kept).encode("utf-8"))
        return removed
    except Exception:  # noqa: BLE001 — prune runs in the calibration tick; must not break it
        logger.exception("recall diagnostics: prune of %s failed; file left as it was", path)
        return 0
