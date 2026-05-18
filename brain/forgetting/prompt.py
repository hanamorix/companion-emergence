"""prompt.py — ambient "fading/lost this week" block per spec §5.

Compact ≤120-token aggregate so Nell can talk about the lived
experience of recent loss, not just individual lost things.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.forgetting import graveyard
from brain.memory.store import MemoryStore

_LOST_WINDOW_DAYS = 7


def render_fading_summary_block(persona_dir: Path, store: MemoryStore) -> str:
    """Return a compact ≤120-token ambient block about fading + recent lost memories.

    Counts:
      - fading_count: memories currently in state='fading'
      - recent_lost: graveyard entries with forgotten_at_iso within the last
        _LOST_WINDOW_DAYS days

    Returns:
      - "memory: nothing has softened lately."  when both counts == 0
      - "memory: {fading_count} softened (fading), {recent_lost} lost in the last 7 days."
        otherwise
    """
    fading_count = _count_fading(store)
    recent_lost = _count_recent_lost(persona_dir)

    if fading_count == 0 and recent_lost == 0:
        return "memory: nothing has softened lately."

    return (
        f"memory: {fading_count} softened (fading), "
        f"{recent_lost} lost in the last {_LOST_WINDOW_DAYS} days."
    )


def _count_fading(store: MemoryStore) -> int:
    """Count memories in state='fading' in the store."""
    row = store._conn.execute("SELECT COUNT(*) FROM memories WHERE state = 'fading'").fetchone()
    return int(row[0]) if row else 0


def _count_recent_lost(persona_dir: Path) -> int:
    """Count graveyard entries with forgotten_at_iso within the last _LOST_WINDOW_DAYS days."""
    cutoff = datetime.now(UTC) - timedelta(days=_LOST_WINDOW_DAYS)
    cutoff_iso = cutoff.isoformat()
    count = 0
    for entry in graveyard.read_all(persona_dir):
        forgotten_at = entry.get("forgotten_at_iso") or ""
        if forgotten_at >= cutoff_iso:
            count += 1
    return count
