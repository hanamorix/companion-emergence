"""brain.maker.persist — write a making to the portfolio.

private / eventual_share → artifact markdown + full row.
discard → thin row (no markdown content), so she recalls having made + released.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.maker.maker import Making
from brain.works import Work, make_work_id
from brain.works.storage import write_markdown
from brain.works.store import WorksStore

logger = logging.getLogger(__name__)


def persist_making(persona_dir: Path, store: WorksStore, m: Making, *,
                   charge_sources: list[str], now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    wid = make_work_id(m.content + m.title)
    work = Work(
        id=wid, title=m.title, type=m.type, created_at=now, session_id=None,
        word_count=len(m.content.split()), summary=None,
        disposition=m.disposition, private_reason=m.private_reason,
        origin="maker", charge_sources=json.dumps(charge_sources), shared_at=None,
    )
    if m.disposition == "discard":
        # thin row, no content markdown
        store.insert(work, content="")  # content not written to disk for discard
        # ensure no markdown artifact lingers
        from brain.works.storage import _work_path
        p = _work_path(persona_dir, wid)
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
        return wid
    store.insert(work, content=m.content)
    try:
        write_markdown(persona_dir, work, content=m.content)
    except Exception:
        logger.exception("maker: writing artifact markdown failed for %s", wid)
    return wid
