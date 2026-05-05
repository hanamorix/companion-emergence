"""Word-count helper for body state energy calculation.

Sums words from recent assistant turns within the session window.
Reads the JSONL session-buffer files directly — they are the canonical
source of "what the brain said when." Memory store has no rows tagged
as conversation turns; the energy formula needs the raw turn-level data.

Fails-safe to 0 on any exception so chat composition never breaks.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from brain.health.jsonl_reader import read_jsonl_skipping_corrupt
from brain.memory.store import MemoryStore


def count_words_in_session(
    store: MemoryStore,  # noqa: ARG001 — unused; kept for signature symmetry across body helpers
    *,
    persona_dir: Path,
    session_hours: float,
    now: datetime,
) -> int:
    """Sum word counts of recent assistant turns within the session window.

    Reads <persona_dir>/active_conversations/*.jsonl, the actual source of
    truth for chat-turn data (one line per turn, ``{session_id, speaker,
    text, ts}``). Memory rows are extracted observations / facts / etc.
    and have no ``memory_type="conversation"`` rows for this helper to
    query — that's the historical bug this function used to silently
    return 0 from.

    Window is ``session_hours`` clamped to a 1.0-hour minimum: when called
    from CLI (no bridge), session_hours=0.0 — falling back to "last hour"
    gives a reasonable energy signal without zero-window edge cases.

    Returns 0 on any exception. The energy formula treats 0 as "no
    creative drain" — preferable to crashing chat composition.
    """
    cutoff = now - timedelta(hours=max(session_hours, 1.0))
    total = 0
    try:
        active_dir = persona_dir / "active_conversations"
        if not active_dir.exists():
            return 0
        for jsonl_path in active_dir.glob("*.jsonl"):
            for turn in read_jsonl_skipping_corrupt(jsonl_path):
                if turn.get("speaker") != "assistant":
                    continue
                ts_raw = turn.get("ts")
                if not ts_raw:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_raw)
                except ValueError:
                    continue
                if ts < cutoff:
                    continue
                text = turn.get("text") or ""
                total += len(text.split())
    except Exception:  # noqa: BLE001
        return 0
    return total
