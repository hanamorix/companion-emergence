"""Tier 2 read path #1 — ambient continuity.

Surfaces recent monologue_trace memories into the chat system prompt so her own
prior interior colours later turns. Because fade() rewrites content→summary,
rendering `m.content` shows the trace as stored while active (the verbatim
thought with the switch off, its feed digest once the emotion engine is live)
and the tombstone summary once fading.

Once the emotion engine is live each stored trace is a short note, so the block becomes a trail of whole notes under
its own header: newest first, each note kept whole or dropped, never cut mid-note (only the newest, when it alone
passes the cap, is cut). With the switch off, or when the switch cannot be read, the block is today's.
"""
from __future__ import annotations

from brain import prompt_strings
from brain.memory.store import MemoryStore
from brain.monologue.trace import MONOLOGUE_TRACE_TYPE

_AMBIENT_LIMIT = 5
_CHAR_CAP = 1200

# Text externalized to prompt_strings.toml [monologue.ambient] (issue #129 stage 2c).
_HEADER = prompt_strings.register("monologue.ambient.header")
_HEADER_LIVE = prompt_strings.register("monologue.ambient.header_live")
_FOOTER_TEMPLATE = prompt_strings.register("monologue.ambient.footer_template")


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
        # monologue_trace is now GATED — traces live in the pending-candidate
        # queue (not memories.db), so interior-continuity reads them there. The
        # limit (_AMBIENT_LIMIT=5) and rendering semantics are unchanged; only
        # the read source moves. TEMP (Root 2 stopgap; ≤2-thought rework → Phase 4).
        from brain.memory.pending import PendingQueue

        traces = PendingQueue(store.persona_dir).read_recent(
            MONOLOGUE_TRACE_TYPE, limit=limit
        )
    except Exception:  # noqa: BLE001
        return ""
    if not traces:
        return ""
    traces = sorted(traces, key=lambda m: m.created_at, reverse=True)[:limit]
    try:
        eng = store.emotion_engine()
        live = eng is not None and eng.is_live()
    except Exception:  # noqa: BLE001
        live = False
    texts = [" ".join(m.content.split()) for m in traces]  # collapse whitespace
    if live:
        lines = [f"· {t}" for t in texts]  # newest first
        body = (_HEADER_LIVE + "\n" + lines[0])[:char_cap]
        for line in lines[1:]:
            candidate = body + "\n" + line
            if len(candidate) > char_cap:
                break
            body = candidate
    else:
        body = "\n".join([_HEADER] + [f"· {t}" for t in texts])[:char_cap]
    return body + "\n" + _FOOTER_TEMPLATE.format(user_name=user_name)
