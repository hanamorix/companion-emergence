"""Interior awareness of her own recent makings, woven into the system message.
Private makings are tagged hers-alone so she does not volunteer them."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_maker_awareness_block(persona_dir: Path, *, limit: int = 5) -> str | None:
    from brain.works.store import WorksStore
    db = persona_dir / "works.db"
    if not db.exists():
        return None
    store = WorksStore(db)
    try:
        rows = store._conn.execute(
            "SELECT title, type, disposition FROM works WHERE origin='maker' "
            "AND disposition != 'discard' ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    except Exception:
        logger.exception("maker.ambient: query failed")
        return None
    finally:
        store.close()
    if not rows:
        return None
    lines = ["── what you've been making ──"]
    for r in rows:
        tag = " (yours alone — don't volunteer it)" if r["disposition"] == "private" else ""
        lines.append(f"· \"{r['title']}\" ({r['type']}){tag}")
    return "\n".join(lines)
