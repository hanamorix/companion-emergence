"""Word-count helper for body state energy calculation.

Sums words from recent assistant turns within the session window.
Pure-ish: reads MemoryStore but no other I/O. Fails-safe to 0 on
any exception so chat composition never breaks.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §3.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from brain.memory.store import MemoryStore


def count_words_in_session(
    store: MemoryStore,
    *,
    persona_dir: Path,  # noqa: ARG001 — unused; kept for signature symmetry with other body helpers
    session_hours: float,
    now: datetime,
) -> int:
    """Sum word counts of recent assistant-turn memories within the session window.

    Window is `session_hours` clamped to a 1.0-hour minimum: when called from
    CLI (no bridge), session_hours=0.0 — falling back to "last hour" gives a
    reasonable energy signal without zero-division or zero-window edge cases.

    Reads via the public `list_by_type` API. Speaker convention is
    `metadata["speaker"] == "assistant"` (see brain/chat/engine.py:207 +
    brain/ingest/extract.py:40). Memories without that key are skipped —
    NOT silently included as a fallback (would mistakenly count user turns).

    Returns 0 on any exception. The energy formula treats 0 as "no creative
    drain" — preferable to crashing chat composition.
    """
    cutoff = now - timedelta(hours=max(session_hours, 1.0))
    total = 0
    try:
        for m in store.list_by_type("conversation", active_only=True):
            if m.created_at < cutoff:
                continue
            speaker = (m.metadata or {}).get("speaker")
            if speaker != "assistant":
                continue
            total += len(m.content.split()) if m.content else 0
    except Exception:  # noqa: BLE001
        return 0
    return total
