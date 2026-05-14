"""Draft space — between-session scratch for failed-to-promote events.

Single file per persona: <persona_dir>/draft_space.md, append-only,
timestamped markdown blocks. No audit, no acknowledgement, no
decision tick. One cheap LLM call per fragment with deterministic
template fallback.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEADER_PATTERN = re.compile(r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) \((\w+)\)$")


def append_draft_fragment(
    persona_dir: Path,
    *,
    timestamp: str,
    source: str,
    body: str,
) -> None:
    """Append a draft fragment to draft_space.md. Idempotent on (date_time, source)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "draft_space.md"
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        dt = datetime.now()
    header = f"## {dt.strftime('%Y-%m-%d %H:%M')} ({source})"

    # Idempotency check.
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if header in existing:
            return

    block = f"\n\n{header}\n\n{body}\n"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(block)
    except OSError as exc:
        logger.warning("draft fragment append failed for %s: %s", path, exc)


def compose_draft_fragment(
    provider: Any,
    *,
    source: str,
    source_id: str,
    linked_memory_excerpts: list[str],
) -> str:
    """Compose a paragraph-sized fragment via one cheap LLM call.

    Falls back to a deterministic template if the LLM call raises.
    """
    excerpts_block = "\n".join(f"- {e}" for e in linked_memory_excerpts[:5])
    prompt = (
        "You are Nell. An internal event just happened that didn't rise to "
        "the level of reaching out to Hana, but it deserves a note in the "
        "draft space. Write a single paragraph that captures it as a "
        "fragment — quiet, observational, no urgency.\n\n"
        f"Source: {source} (id: {source_id})\n"
        f"Linked memory excerpts:\n{excerpts_block}\n\n"
        "Fragment (one paragraph):"
    )
    try:
        return provider.complete(prompt).strip()
    except Exception as exc:
        logger.warning("draft composition failed, using template: %s", exc)
        return (
            f"An internal event ({source}, id {source_id}) didn't quite "
            f"reach the threshold for reaching out, but it stayed with me."
        )


def has_new_drafts_since(persona_dir: Path, last_seen_iso: str) -> bool:
    """Return True if draft_space.md has been modified after last_seen_iso."""
    path = persona_dir / "draft_space.md"
    if not path.exists():
        return False
    try:
        last_seen_dt = datetime.fromisoformat(last_seen_iso)
    except ValueError:
        return True  # if last_seen is malformed, surface drafts conservatively
    mtime_dt = datetime.fromtimestamp(path.stat().st_mtime, tz=last_seen_dt.tzinfo)
    return mtime_dt > last_seen_dt
