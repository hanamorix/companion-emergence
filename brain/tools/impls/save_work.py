"""save_work tool implementation."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.works import WORK_TYPES, Work, make_work_id
from brain.works.storage import write_markdown
from brain.works.store import WorksStore


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
            "error": (
                f"invalid type {type!r} — must be one of: "
                f"{', '.join(sorted(WORK_TYPES))}"
            )
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
    write_markdown(persona_dir, work, content=content)
    WorksStore(persona_dir / "data" / "works.db").insert(work, content=content)
    return {"id": work.id, "path": f"data/works/{work.id}.md"}
