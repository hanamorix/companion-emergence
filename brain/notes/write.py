"""brain.notes.write — the autonomous, folder-bounded note write (create-only)."""
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from brain.files.write_guard import check_write_target, is_within_authorized
from brain.health.json_io import append_jsonl
from brain.notes.compose import Note

logger = logging.getLogger(__name__)


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:60] or "note"


def _audit(persona_dir: Path, *, path: str, ok: bool, error: str | None = None) -> None:
    append_jsonl(
        persona_dir / "notes_audit.jsonl",
        {"ts": datetime.now(UTC).isoformat(), "path": path, "ok": ok, "error": error},
        on_error="log",
        nonfinite="null",
    )


def commit_note(persona_dir: Path, folder: Path, note: Note, *, now: datetime | None = None) -> str | None:
    """Write the note inside `folder` (create-only, collision-suffixed). Returns
    the path, or None if the guard refuses (escape / deny-list / disabled)."""
    now = now or datetime.now(UTC)
    base = f"{now.date().isoformat()} — {_slug(note.subject)}"
    content = f"# {note.subject}\n\n_{now.date().isoformat()}_\n\n{note.body}\n"
    i = 1
    while True:
        candidate = folder / (f"{base}.md" if i == 1 else f"{base}-{i}.md")
        i += 1
        if candidate.exists():
            continue
        resolved = candidate.resolve()
        g = check_write_target(str(resolved), op="create", persona_dir=persona_dir)
        if not g.ok or not is_within_authorized(resolved, folder):
            _audit(persona_dir, path=str(resolved), ok=False, error=(g.error or "outside authorized folder"))
            return None
        try:
            # Exclusive creation (#484): "create-only" has to hold even when another tick took this name after the check above;
            # a taken name moves on to the next suffix instead of truncating the other note.
            with open(resolved, "x", encoding="utf-8") as fh:
                fh.write(content)
        except FileExistsError:
            continue
        except OSError as exc:
            _audit(persona_dir, path=str(resolved), ok=False, error=str(exc))
            return None
        break
    _audit(persona_dir, path=str(resolved), ok=True)
    return str(resolved)
