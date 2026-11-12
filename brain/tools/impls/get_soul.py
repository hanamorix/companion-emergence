"""get_soul tool implementation — one bounded page of the soul (#596).

A long-running kindled holds thousands of crystallizations (2,091, ~0.9 MB, in
the #596 report). The Claude CLI does not hand a tool result that size to the
model: it saves it to a file and passes a short preview (#595), so an unbounded
get_soul showed her only fragments of her own soul. Each call therefore returns
one page, bounded in items (``limit``) and in serialized bytes
(``MAX_RESULT_BYTES``), newest first, and says how to fetch the next page.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.naming import TOOL_PREFIX_SLOT

# Newest first: ``resonance`` cannot rank the soul — it defaults to 8 and the
# crystallize_soul schema asks for "at least 8", so nearly every item ties.
# Recency is the one field that discriminates, and it tracks who she is now.
DEFAULT_LIMIT = 20
MAX_LIMIT = 50

# Byte bound on the whole serialized result (what the MCP handler sends). The
# CLI inlines an MCP tool result of up to 50,000 characters and persists anything
# longer to a file with a 2 KB preview, which chat cannot open (Read is not in its
# --tools); measured on claude 2.1.321, see docs/cli-provider-capabilities.md. 8 KB
# sits far under that hard ceiling (bytes >= characters, so a byte bound is the
# stricter one) and is sized for context cost instead: about 2k tokens and ~15
# typical crystallizations (~430 bytes each in the #596 soul) per call.
MAX_RESULT_BYTES = 8_000

_CLIP_MARK = "…"
# Every free-text field of an active crystallization (revoked_reason is empty
# on active ones); crystallize_soul caps none of them.
_TEXT_FIELDS = ("moment", "why_it_matters", "who_or_what")


def result_bytes(obj: Any) -> int:
    """Bytes as the MCP handler serialises a result."""
    return len(json.dumps(obj, default=str, ensure_ascii=False).encode("utf-8"))


def _as_int(value: Any) -> int | None:
    """A whole number from what the model sent, else None.

    The same whole numbers are accepted as a number or as text: ``20``, ``20.0``,
    ``"20"``, ``" 20 "``, ``"20.0"``, ``"2e1"``. Fractions, NaN/inf, booleans and
    Python-only spellings such as ``"1_0"`` are refused.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if "_" in value:  # int()/float() accept "1_0"; JSON and people don't mean it
            return None
        try:
            return int(value.strip())
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                return None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _clip_to_fit(item: dict, budget: int) -> dict:
    """Shorten the item's text fields until it serialises within ``budget`` bytes.

    Only an item too large to share a page with anything else reaches here; it
    is marked ``clipped`` so the model knows the text was cut.
    """
    clipped = {**item, "clipped": True}
    lo, hi = 0, max(len(item[f]) for f in _TEXT_FIELDS)
    best = {**clipped, **dict.fromkeys(_TEXT_FIELDS, _CLIP_MARK)}
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = dict(clipped)
        for f in _TEXT_FIELDS:
            if len(item[f]) > mid:
                trial[f] = item[f][:mid] + _CLIP_MARK
        if result_bytes(trial) <= budget:
            best, lo = trial, mid + 1
        else:
            hi = mid - 1
    return best


def get_soul(
    *,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    love_type: str | None = None,
    max_bytes: int | None = None,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Return one page of the persona's active crystallizations, newest first.

    ``count`` is the number of items in this result; ``total`` is how many
    active crystallizations exist. When more remain, ``truncated`` is true,
    ``next_offset`` is the offset of the next page, and ``message`` says how to
    call for it. ``next_offset`` counts the items actually returned, so a page
    cut short by the byte bound never skips anything.

    ``max_bytes`` is not in the tool schema: ``boot`` passes the room left
    beside its other parts. It can only lower the bound, never raise it.
    """
    from brain.soul.love_types import LOVE_TYPES
    from brain.soul.store import SoulStore

    if love_type is not None and love_type not in LOVE_TYPES:
        return {
            "loaded": False,
            "error": (
                f"unknown love_type {love_type!r}. Use one of: {', '.join(sorted(LOVE_TYPES))}; "
                "or leave it out for the whole soul."
            ),
        }
    parsed_limit = _as_int(limit)
    if parsed_limit is None:
        return {
            "loaded": False,
            "error": f"limit must be a whole number from 1 to {MAX_LIMIT}, got {limit!r}.",
        }
    parsed_offset = _as_int(offset)
    if parsed_offset is None:
        return {
            "loaded": False,
            "error": f"offset must be a whole number (0 for the newest), got {offset!r}.",
        }
    # Out-of-range values are clamped, not refused: the result reports the
    # limit and offset it actually used.
    limit = min(max(parsed_limit, 1), MAX_LIMIT)
    offset = max(parsed_offset, 0)

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        total = soul_store.count(love_type)
        # Past the end is past the end: capping at total also keeps a huge
        # offset (10**30, 1e300) inside SQLite's 64-bit INTEGER.
        offset = min(offset, total)
        rows = soul_store.list_active_page(offset=offset, limit=limit, love_type=love_type)
        by_love_type = soul_store.active_love_type_counts()
    finally:
        soul_store.close()

    result: dict[str, Any] = {
        "loaded": True,
        "count": 0,
        "total": total,
        "love_type": love_type,
        # The whole soul by type (~25 short entries), so a filter can be chosen
        # without walking every page first.
        "by_love_type": by_love_type,
        "offset": offset,
        "limit": limit,
        "truncated": False,
        "next_offset": None,
        "crystallizations": [],
    }

    def finish(items: list[dict]) -> dict:
        out = {**result, "count": len(items), "crystallizations": items}
        if offset + len(items) < total:
            nxt = offset + len(items)
            args = f'love_type="{love_type}" and offset={nxt}' if love_type else f"offset={nxt}"
            out["truncated"] = True
            out["next_offset"] = nxt
            out["message"] = (
                f"Showing {len(items)} of {total} crystallizations, newest first, "
                f"from offset {offset}. More remain: call {TOOL_PREFIX_SLOT}get_soul "
                f"with {args} for the next page."
            )
        return out

    # The envelope with the longest hint this page could carry, plus room for
    # the real tool prefix that later replaces the slot.
    reserve = result_bytes(finish([])) + 64
    # A caller embedding this page in a larger result (boot) passes what is left
    # of the bound; never more than the bound itself.
    parsed_budget = _as_int(max_bytes)
    budget = MAX_RESULT_BYTES if parsed_budget is None else min(parsed_budget, MAX_RESULT_BYTES)
    items: list[dict] = []
    for row in rows:
        item = row.to_dict()
        if reserve + result_bytes(items + [item]) <= budget:
            items.append(item)
            continue
        if not items:
            # Alone it is still too big: cut its text so the page holds it. If
            # even a clipped item cannot fit (a tiny budget), return none: the
            # hint still points at this offset.
            clipped = _clip_to_fit(item, budget - reserve - 2)  # 2: the list's brackets
            if reserve + result_bytes([clipped]) <= budget:
                items.append(clipped)
        break

    return finish(items)
