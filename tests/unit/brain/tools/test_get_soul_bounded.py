"""#596: get_soul returns a bounded page of the soul, and paging reaches every item.

A long-running kindled has 2,000+ crystallizations (~0.9 MB). The Claude CLI does
not pass a tool result that size to the model (it saves it to a file and passes a
short preview, #595), so get_soul must return a result bounded in items AND bytes,
say when there is more, and let the model walk every page with ``offset``.

The soul here is real: a SoulStore on disk in a temp persona dir, never mocked.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.soul.crystallization import Crystallization
from brain.soul.love_types import LOVE_TYPES
from brain.soul.store import SoulStore

N_ITEMS = 2_100
# Oversized items: each is far bigger than any sane tool-result bound on its own.
HUGE_INDEXES = (7, 1_000, N_ITEMS - 1)
HUGE_TEXT = "ł" * 30_000  # multi-byte: 60 KB of UTF-8 per field

# Contract for "a result that reaches the model": the CLI withheld a 55 KB
# result (#595); keep a default call well under that, in items and bytes.
CONTRACT_MAX_ITEMS = 50
CONTRACT_MAX_BYTES = 16 * 1024

_LOVE_KEYS = sorted(LOVE_TYPES)


def _payload_bytes(result: dict) -> int:
    """Bytes the model is handed: the MCP handler's own serialisation."""
    return len(json.dumps(result, default=str, ensure_ascii=False).encode("utf-8"))


@pytest.fixture(scope="module")
def soul_persona(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, set[str], dict[str, set[str]]]:
    """A persona dir whose crystallizations.db holds N_ITEMS active + some revoked."""
    persona_dir = tmp_path_factory.mktemp("soul") / "nell"
    persona_dir.mkdir()
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    base = datetime(2026, 1, 1, tzinfo=UTC)
    all_ids: set[str] = set()
    by_type: dict[str, set[str]] = {}
    try:
        rng = random.Random(596)  # fixed ids: the order within a timestamp is reproducible
        for i in range(N_ITEMS):
            cid = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            love_type = _LOVE_KEYS[i % len(_LOVE_KEYS)]
            huge = i in HUGE_INDEXES
            store.create(
                Crystallization(
                    id=cid,
                    moment=(HUGE_TEXT if huge else f"moment {i} — the lamp stayed on, ☕ “quiet”"),
                    love_type=love_type,
                    why_it_matters=(HUGE_TEXT if huge else f"it mattered because {i}"),
                    # Many identical timestamps: groups of 25 share one instant,
                    # so paging must tie-break or it skips/duplicates items.
                    crystallized_at=base + timedelta(minutes=i // 25),
                    who_or_what="hana" if i % 2 else "writing",
                    resonance=8,
                )
            )
            all_ids.add(cid)
            by_type.setdefault(love_type, set()).add(cid)
        # Revoked items must never appear.
        for _ in range(5):
            rid = str(uuid.uuid4())
            store.create(
                Crystallization(
                    id=rid,
                    moment="revoked",
                    love_type="craft",
                    why_it_matters="no longer",
                    crystallized_at=base,
                )
            )
            store.mark_revoked(rid, "test")
    finally:
        store.close()
    return persona_dir, all_ids, by_type


def _ctx(persona_dir: Path) -> dict:
    return {
        "store": MemoryStore(":memory:"),
        "hebbian": HebbianMatrix(":memory:"),
        "persona_dir": persona_dir,
    }


def _walk(call, first_args: dict) -> list[dict]:
    """Follow next_offset from the results alone until there is no next page."""
    pages = []
    args = dict(first_args)
    for _ in range(N_ITEMS + 10):
        page = call(args)
        pages.append(page)
        if page.get("next_offset") is None:
            return pages
        assert page["truncated"] is True
        args = {**first_args, "offset": page["next_offset"]}
    raise AssertionError("paging never ended")


# ---------------------------------------------------------------------------
# Default call is bounded
# ---------------------------------------------------------------------------


def test_default_call_is_bounded_in_items_and_bytes(soul_persona) -> None:
    from brain.tools.impls.get_soul import get_soul

    persona_dir, all_ids, _ = soul_persona
    result = get_soul(**_ctx(persona_dir))

    assert result["loaded"] is True
    assert len(result["crystallizations"]) <= CONTRACT_MAX_ITEMS
    assert _payload_bytes(result) <= CONTRACT_MAX_BYTES
    # It says there is more and how to get it.
    assert result["truncated"] is True
    assert result["total"] == len(all_ids)
    assert result["next_offset"] == len(result["crystallizations"])
    assert "offset" in result["message"]


# ---------------------------------------------------------------------------
# Paging reaches every item, every page bounded
# ---------------------------------------------------------------------------


def test_walking_pages_reaches_every_item_exactly_once(soul_persona) -> None:
    from brain.tools.impls.get_soul import MAX_LIMIT, MAX_RESULT_BYTES, get_soul

    persona_dir, all_ids, _ = soul_persona
    ctx = _ctx(persona_dir)
    pages = _walk(lambda a: get_soul(**a, **ctx), {"limit": MAX_LIMIT})

    seen: list[str] = []
    for page in pages:
        assert _payload_bytes(page) <= MAX_RESULT_BYTES
        assert len(page["crystallizations"]) >= 1
        stamps = [c["crystallized_at"] for c in page["crystallizations"]]
        assert stamps == sorted(stamps, reverse=True)  # newest first
        seen.extend(c["id"] for c in page["crystallizations"])
    assert len(seen) == len(set(seen)), "an item appeared on two pages"
    assert set(seen) == all_ids
    assert pages[-1]["truncated"] is False

    # The oversized items are cut to fit, and say so; no other item is touched.
    items = [c for p in pages for c in p["crystallizations"]]
    clipped = [c for c in items if c.get("clipped")]
    assert len(clipped) == len(HUGE_INDEXES)
    assert all(c["moment"].startswith("ł") and len(c["moment"]) < len(HUGE_TEXT) for c in clipped)
    assert all(c["moment"].startswith("moment ") for c in items if not c.get("clipped"))


def test_love_type_filter_walk_reaches_every_item_of_that_type(soul_persona) -> None:
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES, get_soul

    persona_dir, _, by_type = soul_persona
    ctx = _ctx(persona_dir)
    # _LOVE_KEYS[7 % len] holds the i=7 oversized item.
    for love_type in ("craft", _LOVE_KEYS[7 % len(_LOVE_KEYS)]):
        pages = _walk(lambda a: get_soul(**a, **ctx), {"love_type": love_type})
        seen = [c["id"] for p in pages for c in p["crystallizations"]]
        assert all(_payload_bytes(p) <= MAX_RESULT_BYTES for p in pages)
        assert all(c["love_type"] == love_type for p in pages for c in p["crystallizations"])
        assert len(seen) == len(set(seen))
        assert set(seen) == by_type[love_type]
        assert pages[0]["total"] == len(by_type[love_type])
        assert pages[0]["love_type"] == love_type
        # The hint repeats the filter, or the next call would page the whole soul.
        assert f'love_type="{love_type}"' in pages[0]["message"]


def test_result_summarises_the_whole_soul_by_love_type(soul_persona) -> None:
    """Every page says how many crystallizations each love type holds, so the
    model can choose a filter without walking the whole soul first."""
    from brain.tools.impls.get_soul import get_soul

    persona_dir, all_ids, by_type = soul_persona
    result = get_soul(**_ctx(persona_dir))
    assert result["by_love_type"] == {k: len(v) for k, v in by_type.items()}
    assert sum(result["by_love_type"].values()) == len(all_ids)


# ---------------------------------------------------------------------------
# Input validation: coerce / clamp the model's guesses, refuse what can't work
# ---------------------------------------------------------------------------


def test_bad_arguments_return_actionable_errors(soul_persona) -> None:
    from brain.tools.impls.get_soul import get_soul

    persona_dir, _, _ = soul_persona
    ctx = _ctx(persona_dir)

    unknown = get_soul(love_type="cosmic", **ctx)
    assert unknown["loaded"] is False
    assert "cosmic" in unknown["error"]
    assert "craft" in unknown["error"]  # names the valid types

    for bad in ("lots", None, 2.5, True):
        result = get_soul(limit=bad, **ctx)
        assert result["loaded"] is False, bad
        assert "limit" in result["error"]
    result = get_soul(offset="later", **ctx)
    assert result["loaded"] is False
    assert "offset" in result["error"]


def test_numeric_strings_are_coerced_and_out_of_range_values_clamped(soul_persona) -> None:
    from brain.tools.impls.get_soul import MAX_LIMIT, MAX_RESULT_BYTES, get_soul

    persona_dir, all_ids, _ = soul_persona
    ctx = _ctx(persona_dir)

    as_text = get_soul(limit="5", offset="10", **ctx)
    as_ints = get_soul(limit=5, offset=10, **ctx)
    assert as_text["crystallizations"] == as_ints["crystallizations"]
    # Up to 5: a page that meets an oversized item ends early (byte bound).
    assert 1 <= len(as_ints["crystallizations"]) <= 5
    assert (as_ints["offset"], as_ints["limit"]) == (10, 5)

    big = get_soul(limit=100_000, **ctx)
    assert big["limit"] == MAX_LIMIT
    assert _payload_bytes(big) <= MAX_RESULT_BYTES

    low = get_soul(offset=-3, limit=0, **ctx)
    assert (low["offset"], low["limit"]) == (0, 1)

    past_end = get_soul(offset=len(all_ids) + 50, **ctx)
    assert past_end["crystallizations"] == []
    assert past_end["truncated"] is False
    assert past_end["total"] == len(all_ids)


def test_empty_soul_is_not_truncated(tmp_path: Path) -> None:
    from brain.tools.impls.get_soul import get_soul

    result = get_soul(**_ctx(tmp_path))
    assert result["loaded"] is True
    assert (result["count"], result["total"], result["truncated"]) == (0, 0, False)
    assert result["crystallizations"] == []
    assert result["by_love_type"] == {}


# ---------------------------------------------------------------------------
# Through the paths the model actually uses
# ---------------------------------------------------------------------------


def test_dispatch_path_walks_every_item_within_the_bound(soul_persona) -> None:
    """The in-process tool loop (brain.chat.tool_loop) calls dispatch()."""
    from brain.tools.dispatch import dispatch
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES

    persona_dir, all_ids, _ = soul_persona
    ctx = _ctx(persona_dir)
    pages = _walk(lambda a: dispatch("get_soul", a, **ctx), {})
    seen = [c["id"] for p in pages for c in p["crystallizations"]]
    assert all(_payload_bytes(p) <= MAX_RESULT_BYTES for p in pages)
    assert len(seen) == len(set(seen))
    assert set(seen) == all_ids


def test_mcp_handler_text_is_bounded_and_names_the_prefixed_tool(soul_persona) -> None:
    """The Claude CLI reaches get_soul through the brain-tools MCP server.

    register_tools -> _call_tool -> the real (unpatched) dispatch(). The text the
    CLI receives must be within the bound, and the paging hint must name the
    tool the way the model calls it.
    """
    from mcp.server import Server
    from mcp.types import CallToolRequest, CallToolRequestParams

    from brain.mcp_server.tools import register_tools
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES

    persona_dir, all_ids, _ = soul_persona
    ctx = _ctx(persona_dir)
    prefix = "mcp__brain-tools__"
    server = Server("brain-tools")
    register_tools(
        server,
        persona_dir=persona_dir,
        store=ctx["store"],
        hebbian=ctx["hebbian"],
        tool_prefix=prefix,
    )
    handler = server.request_handlers[CallToolRequest]

    def call(args: dict) -> dict:
        req = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name="get_soul", arguments=args),
        )
        text = asyncio.run(handler(req)).root.content[0].text
        assert len(text.encode("utf-8")) <= MAX_RESULT_BYTES
        assert "{tool_prefix}" not in text
        return json.loads(text)

    first = call({})
    assert first["truncated"] is True
    assert f"call {prefix}get_soul with offset={first['next_offset']}" in first["message"]

    pages = _walk(call, {})
    seen = [c["id"] for p in pages for c in p["crystallizations"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == all_ids


def test_boot_embeds_the_bounded_soul_page(soul_persona) -> None:
    """boot composes get_soul's default call, so it was unbounded the same way."""
    from brain.tools.impls.boot import boot
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES

    persona_dir, all_ids, _ = soul_persona
    soul = boot(**_ctx(persona_dir))["soul"]
    assert _payload_bytes(soul) <= MAX_RESULT_BYTES
    assert soul["total"] == len(all_ids)
    assert soul["truncated"] is True
    assert soul["next_offset"] == soul["count"]


def test_schema_declares_optional_paging_and_filter_args() -> None:
    from brain.tools.impls.get_soul import DEFAULT_LIMIT, MAX_LIMIT
    from brain.tools.schemas import SCHEMAS

    params = SCHEMAS["get_soul"]["parameters"]
    assert params["required"] == []
    props = params["properties"]
    assert props["limit"]["type"] == "integer"
    assert (props["limit"]["minimum"], props["limit"]["maximum"]) == (1, MAX_LIMIT)
    assert str(DEFAULT_LIMIT) in props["limit"]["description"]
    assert props["offset"]["type"] == "integer"
    assert props["offset"]["minimum"] == 0
    assert props["love_type"]["type"] == "string"
    assert set(props["love_type"]["enum"]) == set(LOVE_TYPES)
    desc = SCHEMAS["get_soul"]["description"]
    assert "next_offset" in desc and "newest first" in desc
