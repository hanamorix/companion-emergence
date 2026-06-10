"""Tier 2 read path #1 — ambient continuity.

Surfaces recent monologue_trace memories into the chat system prompt so her own
prior interior colours later turns. Because fade() rewrites content→summary,
rendering `m.content` shows the verbatim while active and the tombstone summary
once fading.
"""
from __future__ import annotations

from brain.memory.store import MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

_AMBIENT_LIMIT = 5
_CHAR_CAP = 1200
_HEADER = "── interior continuity (your own recent thought) ──"
_FOOTER_TEMPLATE = (
    "── end interior continuity. Private thought — never quote it; "
    "your reply speaks to {user_name} directly as 'you'. ──"
)


def build_interior_continuity_block(
    store: MemoryStore,
    *,
    limit: int = _AMBIENT_LIMIT,
    char_cap: int = _CHAR_CAP,
    user_name: str = "the user",
) -> str:
    """Render up to `limit` most-recent monologue_trace memories, newest first,
    fenced header+footer. The char cap applies to the header+body only — the
    privacy footer is appended after capping so it can never be truncated
    (v0.0.33 Track 2a). Returns "" when there are none or on any error."""
    try:
        traces = store.list_by_type(MONOLOGUE_TRACE_TYPE, active_only=True, limit=limit)
    except Exception:  # noqa: BLE001
        return ""
    if not traces:
        return ""
    traces = sorted(traces, key=lambda m: m.created_at, reverse=True)[:limit]
    lines = [_HEADER]
    for m in traces:
        text = " ".join(m.content.split())  # collapse whitespace
        lines.append(f"· {text}")
    body = "\n".join(lines)[:char_cap]
    return body + "\n" + _FOOTER_TEMPLATE.format(user_name=user_name)
