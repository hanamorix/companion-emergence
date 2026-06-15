"""brain.maker.disclosure — the conversational disclosure surface (#8).

Returns her makings TO HER (the model) so she composes the disclosure in her
reply: share-eligible with content, private ones flagged with their reason so
she chooses per-item to share or decline-with-reason. discard makings (no
content) are omitted. This is her DELIBERATE channel — distinct from the
automatic-surface gate (privacy.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from brain.maker.privacy import is_disclosable_on_request

logger = logging.getLogger(__name__)


def surface_makings(*, persona_dir: Path, limit: int = 20) -> dict:
    from brain.works.storage import read_markdown
    from brain.works.store import WorksStore

    db = persona_dir / "works.db"
    if not db.exists():
        return {"makings": []}
    store = WorksStore(db)
    try:
        ids = [
            r["id"]
            for r in store._conn.execute(
                "SELECT id FROM works WHERE origin='maker' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        ]
        works = [store.get(i) for i in ids]
    except Exception:
        logger.exception("maker.disclosure: query failed")
        return {"makings": []}
    finally:
        store.close()
    out = []
    for w in works:
        if w is None or not is_disclosable_on_request(w):
            continue  # discard: nothing to surface
        item = {
            "id": w.id,
            "title": w.title,
            "type": w.type,
            "private": w.disposition == "private",
            "private_reason": w.private_reason,
        }
        try:
            _, content = read_markdown(persona_dir, w.id)
            item["content"] = content
        except Exception:
            item["content"] = w.summary or ""
        out.append(item)
    return {"makings": out}
