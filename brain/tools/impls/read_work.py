"""read_work tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.works.storage import read_markdown
from brain.works.store import WorksStore


def read_work(id: str, *, persona_dir: Path) -> dict:
    """Return one work's full content + metadata.

    Returns {"error": "..."} if the work is unknown or the file is missing.
    """
    db_path = persona_dir / "data" / "works.db"
    if not db_path.exists():
        return {"error": f"no works.db for persona at {persona_dir}"}
    work = WorksStore(db_path).get(id)
    if work is None:
        return {"error": f"unknown work id {id!r}"}
    try:
        _, content = read_markdown(persona_dir, id)
    except FileNotFoundError:
        return {"error": f"work {id!r} indexed but file missing"}
    except ValueError as exc:
        return {"error": f"work {id!r} file corrupt: {exc}"}
    return {
        "id": work.id,
        "title": work.title,
        "type": work.type,
        "created_at": work.created_at.isoformat(),
        "summary": work.summary,
        "word_count": work.word_count,
        "content": content,
    }
