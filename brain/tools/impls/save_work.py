"""save_work tool implementation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.works import WORK_TYPES, Work, make_work_id
from brain.works.storage import write_markdown
from brain.works.store import WorksStore

logger = logging.getLogger(__name__)


_TITLE_MAX = 200
_SUMMARY_MAX = 500


def save_work(
    title: str,
    type: str,
    content: str,
    summary: str | None = None,
    *,
    persona_dir: Path,
    session_id: str | None = None,
) -> dict:
    """Save a brain-authored work — story, code, planning, idea, role-play, letter.

    Returns {"id": "<12-char>", "path": "data/works/<id>.md"} on success.
    Returns {"error": "..."} on validation failure.
    """
    if not title or not title.strip():
        return {"error": "title cannot be empty"}
    if len(title) > _TITLE_MAX:
        return {"error": f"title exceeds {_TITLE_MAX} chars"}
    if type not in WORK_TYPES:
        return {
            "error": (f"invalid type {type!r} — must be one of: {', '.join(sorted(WORK_TYPES))}")
        }
    if not content or not content.strip():
        return {"error": "content cannot be empty"}
    if summary is not None and len(summary) > _SUMMARY_MAX:
        return {"error": f"summary exceeds {_SUMMARY_MAX} chars"}

    work = Work(
        id=make_work_id(content),
        title=title.strip(),
        type=type,
        created_at=datetime.now(UTC),
        session_id=session_id,
        word_count=len(content.split()),
        summary=summary.strip() if summary else None,
    )
    # Insert into store FIRST. If that fails (OperationalError, disk full,
    # permissions race), no markdown file is written. If the markdown sidecar
    # write fails after a new insert, roll back the index row so /works cannot
    # advertise missing content. Duplicate ids are true idempotent no-ops and
    # must not rewrite existing markdown with different metadata.
    store = WorksStore(persona_dir / "data" / "works.db")
    inserted = store.insert(work, content=content)
    if not inserted:
        return {"id": work.id, "path": f"data/works/{work.id}.md", "deduped": True}
    try:
        write_markdown(persona_dir, work, content=content)
    except Exception:
        try:
            store.delete(work.id)
        except Exception:  # noqa: BLE001
            logger.warning("failed to roll back works index row %s", work.id, exc_info=True)
        raise
    return {"id": work.id, "path": f"data/works/{work.id}.md"}
