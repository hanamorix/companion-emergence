"""Draft space — between-session scratch for failed-to-promote events.

Single file per persona: <persona_dir>/draft_space.md, append-only,
timestamped markdown blocks. No audit, no acknowledgement, no
decision tick. One cheap LLM call per fragment with deterministic
template fallback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
    user_name: str = "my user",
    companion_name: str = "Nell",
) -> str:
    """Compose a paragraph-sized fragment via one cheap LLM call.

    Falls back to a deterministic template if the LLM call raises.
    """
    excerpts_block = "\n".join(f"- {e}" for e in linked_memory_excerpts[:5])
    prompt = (
        f"You are {companion_name}. An internal event just happened that didn't rise to "
        f"the level of reaching out to {user_name}, but it deserves a note in the "
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


@dataclass(frozen=True)
class DraftFragment:
    """One parsed draft-space entry."""

    ts: str   # "YYYY-MM-DD HH:MM" — as written in the header
    source: str
    body: str


def read_drafts_since(persona_dir: Path, last_seen_iso: str) -> list[DraftFragment]:
    """Return all fragments whose header timestamp is strictly after *last_seen_iso*.

    - Missing file → ``[]``.
    - Malformed cutoff → treat as ``datetime.min`` (surface all).
    - Tz-aware cutoff is coerced to naive so the comparison never raises.
    - Multi-line bodies are captured in full (lines between two headers, stripped
      of leading/trailing blank lines).
    """
    path = persona_dir / "draft_space.md"
    if not path.exists():
        return []

    # Parse the cutoff datetime.
    try:
        cutoff = datetime.fromisoformat(last_seen_iso)
    except (ValueError, TypeError):
        cutoff = datetime.min
    if cutoff.tzinfo is not None:
        cutoff = cutoff.replace(tzinfo=None)

    # Walk the file collecting header-delimited blocks.
    fragments: list[DraftFragment] = []
    current_ts: str | None = None
    current_source: str | None = None
    current_body_lines: list[str] = []

    def _flush() -> None:
        if current_ts is None:
            return
        body = "\n".join(current_body_lines).strip()
        try:
            hdr_dt = datetime.strptime(current_ts, "%Y-%m-%d %H:%M")
        except ValueError:
            return
        if hdr_dt > cutoff:
            fragments.append(DraftFragment(ts=current_ts, source=current_source or "", body=body))

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEADER_PATTERN.match(line)
        if m:
            _flush()
            current_ts = m.group(1)
            current_source = m.group(2)
            current_body_lines = []
        elif current_ts is not None:
            current_body_lines.append(line)

    _flush()
    return fragments


_DRAFT_CURSOR_FILE = "draft_space_review_cursor.json"


def load_draft_review_cursor(persona_dir: Path) -> str:
    """Return the ISO timestamp of the last draft reviewed by the soul-review tick.

    Returns empty string if the cursor file does not exist or is unreadable.
    """
    path = persona_dir / _DRAFT_CURSOR_FILE
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("last_seen", ""))
    except (OSError, json.JSONDecodeError, ValueError):
        return ""


def save_draft_review_cursor(persona_dir: Path, iso: str) -> None:
    """Persist *iso* as the last-seen timestamp for the soul-review draft cursor."""
    path = persona_dir / _DRAFT_CURSOR_FILE
    try:
        persona_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"last_seen": iso}, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("draft review cursor save failed: %s", exc)


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
