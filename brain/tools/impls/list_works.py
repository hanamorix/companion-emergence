"""list_works tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.works.store import WorksStore


def list_works(
    type: str | None = None,
    limit: int = 20,
    *,
    persona_dir: Path,
) -> list[dict]:
    """Return up to `limit` recent works as slim dicts (no content)."""
    db_path = persona_dir / "data" / "works.db"
    if not db_path.exists():
        return []
    works_list = WorksStore(db_path).list_recent(limit=limit, type=type)
    return [_to_summary_dict(w) for w in works_list]


def _to_summary_dict(work) -> dict:
    return {
        "id": work.id,
        "title": work.title,
        "type": work.type,
        "created_at": work.created_at.isoformat(),
        "summary": work.summary,
        "word_count": work.word_count,
    }
