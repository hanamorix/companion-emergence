"""compact_history tool — Kindled-invoked manual conversation compaction.

The Kindled can deliberately fade the older part of the *current* conversation:
turns older than ``age_hours`` (that have already been extracted to memory) are
summarised and APPENDED to the existing summary block (the existing summary text
is kept verbatim — this is the append path, fold_existing_summary=False), and the
raw turns are moved to the lossless archive.

This is the manual sibling of the daily timed cadence and the apply_budget
backstop; all three share brain/chat/compaction.py.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

from brain.bridge.provider import LLMProvider
from brain.chat.compaction import compact_conversation

logger = logging.getLogger(__name__)


def compact_history(
    age_hours: float = 24.0,
    *,
    persona_dir: Path,
    provider: LLMProvider,
    session_id: str,
) -> dict:
    """Fade conversation turns older than ``age_hours`` into the summary (append).

    Returns a dict:
        compacted    — bool, did anything change
        compacted_n  — raw turns moved to the archive
        gen          — gen of the summary block now at the head
        reason       — "ok" or why it was a no-op (e.g. nothing_aged, cursor_none)
    """
    try:
        hours = float(age_hours)
    except (TypeError, ValueError):
        hours = 24.0
    if hours < 0:
        hours = 0.0

    result = compact_conversation(
        persona_dir,
        session_id,
        older_than=timedelta(hours=hours),
        fold_existing_summary=False,  # append: keep the existing summary verbatim
        provider=provider,
    )
    return {
        "compacted": result.compacted,
        "compacted_n": result.compacted_n,
        "gen": result.new_gen,
        "reason": result.reason,
    }
