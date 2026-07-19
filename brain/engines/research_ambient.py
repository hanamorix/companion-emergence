"""Interior awareness of her own recent research, woven into the system message.

Wire-back: the research engine files memories and leaves cards, but nothing
surfaced them into her chat context — she only found research by actively
searching a matching keyword. This mirrors brain.maker.ambient so recent
research is present in her interior the way her makings are.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_EXCERPT_CHARS = 200


def build_research_awareness_block(store, *, limit: int = 3) -> str | None:
    try:
        rows = store.list_by_type("research", active_only=True, limit=limit)
    except Exception:
        logger.exception("research.ambient: query failed")
        return None
    if not rows:
        return None
    lines = ["── what you've been turning over on your own ──"]
    for m in rows:
        excerpt = " ".join(m.content.split())[:_EXCERPT_CHARS].strip()
        lines.append(f"· {excerpt}")
    return "\n".join(lines)
