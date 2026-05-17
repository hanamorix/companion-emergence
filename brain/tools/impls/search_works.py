"""search_works tool implementation."""

from __future__ import annotations

from pathlib import Path

from brain.tools.impls.list_works import _to_summary_dict
from brain.works.store import WorksStore


def search_works(
    query: str,
    type: str | None = None,
    limit: int = 20,
    *,
    persona_dir: Path,
) -> list[dict]:
    """Full-text search over title + summary + content. Slim dicts (no content)."""
    db_path = persona_dir / "data" / "works.db"
    if not db_path.exists():
        return []
    matches = WorksStore(db_path).search(query, limit=limit, type=type)
    return [_to_summary_dict(w) for w in matches]
