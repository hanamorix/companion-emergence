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

Fail-soft (INV-I13a): a diagnostics failure must never reach the recall
caller. ``log_recall``, ``append_record`` and ``prune`` swallow every
``Exception``, log it, and return a falsy / zero result. ``build_record`` is
the one strict piece (it validates), and ``log_recall`` catches its errors.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from brain import dev_constants
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


def append_record(path: Path, record: dict[str, Any]) -> bool:
    """Append ``record`` as one JSON line under the OS file lock. Returns True
    on success; never raises (a failure is logged and returns False)."""
    try:
        line = (json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n").encode(
            "utf-8"
        )
        with file_lock(path):
            with open(path, "ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
        return True
    except Exception:  # noqa: BLE001 — diagnostics must never reach recall
        logger.exception("recall diagnostics: append to %s failed; record dropped", path)
        return False


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
    record dropped; returns whether a record was written."""
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


def _record_ts(raw_line: bytes) -> datetime | None:
    """The record's ``ts`` as an aware UTC datetime, or None if the line is not
    a JSON object with a parseable ``ts``."""
    try:
        obj = json.loads(raw_line)
        if not isinstance(obj, dict):
            return None
        ts = datetime.fromisoformat(obj["ts"])
    except (ValueError, KeyError, TypeError):
        return None
    return _utc(ts)


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
    """Delete records older than the retention window; return how many were
    removed (0 if none, or on any failure). Never raises.

    The whole read -> filter -> temp write -> ``os.replace`` runs under the
    same ``file_lock`` the appends take, so an append attempted in the window
    blocks and lands after the rewrite (CONC-3). A record is kept iff its
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        with file_lock(path):
            try:
                with open(path, "rb") as fh:
                    raw = fh.read()
            except FileNotFoundError:
                return 0
            kept: list[bytes] = []
            removed = 0
            for line in raw.split(b"\n"):
                line = line.rstrip(b"\r")
                if not line.strip():
                    continue
                ts = _record_ts(line)
                if ts is not None and ts >= cutoff:
                    kept.append(line)
                else:
                    removed += 1
            if removed == 0:
                return 0
            try:
                with open(tmp, "wb") as out:
                    for line in kept:
                        out.write(line + b"\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(tmp, path)
            except BaseException:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    logger.warning("recall diagnostics: could not remove temp file %s", tmp)
                raise
        return removed
    except Exception:  # noqa: BLE001 — prune runs in the calibration tick; must not break it
        logger.exception("recall diagnostics: prune of %s failed; file left as it was", path)
        return 0
