"""chat_log.py — per-tick chat-turn log for rolling baseline computation.

Written on every heartbeat tick (including zero-turn ticks) so the
7-day rolling mean reflects true cadence, not just active periods.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

log = logging.getLogger(__name__)

CHAT_TURNS_LOG_FILENAME = "chat_turns.log.jsonl"
_WINDOW_DAYS = 7
_RETAIN_DAYS = 30
_COLD_START_MIN = 3


def append_chat_tick(persona_dir: Path, *, ts: datetime, turns: int) -> None:
    """Append one heartbeat-tick row to chat_turns.log.jsonl.

    Best-effort: OSError is logged, not raised, so a write failure cannot
    crash the felt-time tick that calls this.
    """
    persona_dir.mkdir(parents=True, exist_ok=True)
    log_path = persona_dir / CHAT_TURNS_LOG_FILENAME
    entry = {"ts": ts.astimezone(UTC).isoformat(), "turns": turns}
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        log.debug("chat_log: append failed", exc_info=True)


def _trim_old_entries(entries: list[dict], *, cutoff: datetime) -> list[dict]:
    """Pure helper: return only entries with ts >= cutoff."""
    return [e for e in entries if e["ts"] >= cutoff]


def load_recent_samples(
    persona_dir: Path, *, window_days: int = _WINDOW_DAYS
) -> list[tuple[datetime, float]] | None:
    """Return (datetime, turns) pairs within window_days, sorted ascending.

    Returns None when fewer than _COLD_START_MIN entries exist in the window
    (cold-start guard — caller falls back to fixed baseline).
    Lazily rewrites the log when more than 10 old entries were trimmed
    (bounds file size to ~30 days x 96 ticks/day = ~2880 rows max).
    """
    log_path = persona_dir / CHAT_TURNS_LOG_FILENAME
    if not log_path.exists():
        return None

    now = datetime.now(UTC)
    retain_cutoff = now - timedelta(days=_RETAIN_DAYS)
    window_cutoff = now - timedelta(days=window_days)

    all_entries: list[dict] = []
    try:
        for raw in iter_jsonl_skipping_corrupt(log_path):
            try:
                ts = datetime.fromisoformat(raw["ts"])
                all_entries.append({"ts": ts, "turns": float(raw.get("turns", 0))})
            except (KeyError, ValueError, TypeError):
                pass
    except OSError:
        return None

    kept = _trim_old_entries(all_entries, cutoff=retain_cutoff)
    if len(all_entries) - len(kept) > 10:
        _rewrite_log(log_path, kept)

    window = sorted(
        [(e["ts"], e["turns"]) for e in kept if e["ts"] >= window_cutoff],
        key=lambda x: x[0],
    )
    return window if len(window) >= _COLD_START_MIN else None


def _rewrite_log(log_path: Path, entries: list[dict]) -> None:
    """Atomically rewrite log with trimmed entries (lazy GC)."""
    try:
        tmp = log_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in entries:
                row = {"ts": e["ts"].isoformat(), "turns": int(e["turns"])}
                f.write(json.dumps(row) + "\n")
        tmp.replace(log_path)
    except OSError:
        log.debug("chat_log: rewrite failed", exc_info=True)
