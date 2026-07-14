"""Per-interest research notes store — free-form markdown, session-appended.

Spec: docs/source-spec/2026-07-13-research-engine-redesign-design.md §4.2.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

NOTES_DIR = "research"
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def notes_path(persona_dir: Path, interest_id: str) -> Path:
    safe = _SAFE.sub("_", interest_id) or "_"
    return persona_dir / NOTES_DIR / f"{safe}.md"


def append_session_notes(persona_dir: Path, interest_id: str, notes: str, *, now: datetime) -> None:
    path = notes_path(persona_dir, interest_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    block = f"## Session {now:%Y-%m-%d}\n\n{notes.strip()}\n\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(existing + block)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_notes_tail(persona_dir: Path, interest_id: str, *, max_chars: int = 4000) -> str:
    path = notes_path(persona_dir, interest_id)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    tail = text[-max_chars:]
    idx = tail.find("## Session")
    if idx > 0:
        tail = tail[idx:]
    return tail.strip()
