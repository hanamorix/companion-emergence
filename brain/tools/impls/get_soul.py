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
    is marked ``clipped`` so the model knows the text was cut, and ``clipped_at``
    gives, for each field it cut, how many characters are shown: the
    ``char_offset`` that reads on (#734).
    """
    clipped = {**item, "clipped": True}
    lo, hi = 0, max(len(item[f]) for f in _TEXT_FIELDS)
    best = {
        **clipped,
        **dict.fromkeys(_TEXT_FIELDS, _CLIP_MARK),
        "clipped_at": {f: 0 for f in _TEXT_FIELDS if item[f]},
    }
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = dict(clipped)
        trial["clipped_at"] = {f: mid for f in _TEXT_FIELDS if len(item[f]) > mid}
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
    after: str | None = None,
    id: str | None = None,  # noqa: A002 -- the tool's argument name
    field: str | None = None,
    char_offset: int = 0,
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

    ``after`` (#734) is the id of the last item of the previous page, given as
    ``next_after``. It continues strictly after that item, so a crystallization
    added or revoked between calls can't make the walk skip or repeat one, as
    offset paging can. With ``after`` the result's ``offset`` is null.

    ``id`` with ``field`` and ``char_offset`` (#734) reads one text field of one
    crystallization from that character, as much as fits the bound, for an item
    a page showed clipped (its ``clipped_at`` gives the offset to start from).

    ``max_bytes`` is not in the tool schema: ``boot`` passes the room left
    beside its other parts. It can only lower the bound, never raise it.
    """
    from brain.soul.love_types import LOVE_TYPES
    from brain.soul.store import SoulStore

    parsed_budget = _as_int(max_bytes)
    budget = MAX_RESULT_BYTES if parsed_budget is None else min(parsed_budget, MAX_RESULT_BYTES)
    if id is not None or field is not None:
        if after is not None or love_type is not None or _as_int(offset) not in (0, None):
            return {
                "loaded": False,
                "error": "id and field read one item's text: leave out after, offset and love_type with them.",
            }
        return _read_field(persona_dir, id, field, char_offset, budget)

    if love_type is not None and (
        not isinstance(love_type, str) or (love_type not in LOVE_TYPES and love_type not in _stored_love_types(persona_dir))
    ):
        # #734: a type the soul already holds is a filter too, even outside the list (the emergence-kit
        # migrator stores love_type unchecked), so every type by_love_type shows can be passed back.
        return {
            "loaded": False,
            "error": (
                f"unknown love_type {love_type!r}. Use one of: {', '.join(sorted(LOVE_TYPES))}, or a type "
                "by_love_type shows; or leave it out for the whole soul."
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
    if after is not None and (not isinstance(after, str) or not after.strip()):
        return {"loaded": False, "error": f"after must be the id a previous result gave as next_after, got {after!r}."}
    if after is not None and offset:
        return {
            "loaded": False,
            "error": "pass after (from next_after) or offset, not both: after already says where the page starts.",
        }

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        if after is not None and soul_store.get(after) is None:
            return {
                "loaded": False,
                "error": f"no crystallization has id {after!r}: pass the next_after a previous result gave.",
            }
        total = soul_store.count(love_type)
        # Past the end is past the end: capping at total also keeps a huge
        # offset (10**30, 1e300) inside SQLite's 64-bit INTEGER.
        offset = min(offset, total)
        # What is left from where this page starts: the rest after the cursor, or after the offset.
        remaining = soul_store.count(love_type, after=after) if after is not None else total - offset
        rows = soul_store.list_active_page(offset=offset, limit=limit, love_type=love_type, after=after)
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
        "offset": None if after is not None else offset,
        "limit": limit,
        "truncated": False,
        "next_offset": None,
        "next_after": None,
        "crystallizations": [],
    }

    def finish(items: list[dict], *, longest_id: str | None = None) -> dict:
        out = {**result, "count": len(items), "crystallizations": items}
        if len(items) < remaining:
            # The cursor for the next page; with nothing shown, the page starts where this one did.
            nxt_after = longest_id or (items[-1]["id"] if items else after)
            out["truncated"] = True
            if after is None:
                out["next_offset"] = offset + len(items)
            out["next_after"] = nxt_after
            where = f"after {after}" if after is not None else f"from offset {offset}"
            how = f'after="{nxt_after}"' if nxt_after is not None else "offset=0"
            args = f'love_type="{love_type}" and {how}' if love_type else how
            out["message"] = (
                f"Showing {len(items)} of {total} crystallizations, newest first, "
                f"{where}. More remain: call {TOOL_PREFIX_SLOT}get_soul "
                f"with {args} for the next page."
            )
        return out

    # The envelope with the longest hint this page could carry (the longest id
    # it could name), plus room for the real tool prefix that later replaces the slot.
    longest = max((r.id for r in rows), key=len, default=after)
    reserve = result_bytes(finish([], longest_id=longest)) + 64
    # A caller embedding this page in a larger result (boot) passes what is left
    # of the bound (``budget`` above); never more than the bound itself.
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


def _read_field(persona_dir: Path, cid: str | None, field: str | None, char_offset: Any, budget: int) -> dict:
    """One text field of one active crystallization, from ``char_offset``, as much as fits ``budget`` (#734)."""
    from brain.soul.store import SoulStore

    if not cid or not isinstance(cid, str):
        return {"loaded": False, "error": "pass the id of the crystallization whose text you want to read."}
    if field not in _TEXT_FIELDS:
        return {"loaded": False, "error": f"field must be one of {', '.join(_TEXT_FIELDS)}, got {field!r}."}
    start = _as_int(char_offset)
    if start is None:
        return {"loaded": False, "error": f"char_offset must be a whole number (0 for the start), got {char_offset!r}."}
    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        crystal = soul_store.get(cid)
    finally:
        soul_store.close()
    if crystal is None or crystal.revoked_at is not None:
        return {"loaded": False, "error": f"no active crystallization has id {cid!r}."}
    text = crystal.to_dict()[field]
    start = min(max(start, 0), len(text))
    out: dict[str, Any] = {
        "loaded": True,
        "id": cid,
        "field": field,
        "char_offset": start,
        "length": len(text),
        "text": "",
        "next_char_offset": None,
    }

    def with_text(end: int) -> dict:
        return {**out, "text": text[start:end], "next_char_offset": end if end < len(text) else None}

    # The longest slice from `start` whose result fits the bound.
    lo, hi = start, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if result_bytes(with_text(mid)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return with_text(lo)


def _stored_love_types(persona_dir: Path) -> set[str]:
    """The love types the persona's active crystallizations hold (#734)."""
    from brain.soul.store import SoulStore

    soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        return set(soul_store.active_love_type_counts())
    finally:
        soul_store.close()
