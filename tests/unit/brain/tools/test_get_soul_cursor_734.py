"""#734: get_soul pages by cursor, reads a clipped field, and filters by any stored love type.

Offset paging (#596) shifts when the soul changes mid-walk: an item added between two calls makes one item come back
twice, a revoked one makes one never come back. A cursor (`after`, the last item's id) continues strictly after an
item, whatever was added or revoked in between.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore
from brain.tools.impls.get_soul import get_soul

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx(persona_dir: Path) -> dict:
    return {"store": MemoryStore(":memory:"), "hebbian": HebbianMatrix(":memory:"), "persona_dir": persona_dir}


def _add(persona_dir: Path, cid: str, minutes: int, **kw) -> None:
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        store.create(
            Crystallization(
                id=cid,
                moment=kw.get("moment", f"moment {cid}"),
                love_type=kw.get("love_type", "craft"),
                why_it_matters=kw.get("why_it_matters", "it mattered"),
                who_or_what=kw.get("who_or_what", "writing"),
                crystallized_at=_BASE + timedelta(minutes=minutes),
                resonance=8,
            )
        )
    finally:
        store.close()


def _revoke(persona_dir: Path, cid: str) -> None:
    store = SoulStore(str(persona_dir / "crystallizations.db"))
    try:
        store.mark_revoked(cid, "test")
    finally:
        store.close()


def _ids(page: dict) -> list[str]:
    return [c["id"] for c in page["crystallizations"]]


def test_a_cursor_walk_returns_every_item_active_throughout_exactly_once(tmp_path):
    """The issue's acceptance case: add and revoke between pages; nothing is skipped or repeated."""
    for i in range(30):
        _add(tmp_path, f"c{i:02d}", minutes=i // 3)  # groups of three share a timestamp
    throughout = {f"c{i:02d}" for i in range(30)}

    seen: list[str] = []
    page = get_soul(limit=4, **_ctx(tmp_path))
    pages = 0
    while True:
        seen += _ids(page)
        pages += 1
        if page["next_after"] is None:
            break
        # Between calls the soul changes: a newer item arrives, a returned one is revoked, an unseen one is revoked.
        _add(tmp_path, f"new{pages}", minutes=1000 + pages)
        _revoke(tmp_path, seen[-1])
        unseen = sorted(throughout - set(seen))
        if unseen:
            _revoke(tmp_path, unseen[0])
            throughout.discard(unseen[0])
        page = get_soul(limit=4, after=page["next_after"], **_ctx(tmp_path))
        assert pages < 40

    assert len(seen) == len(set(seen)), "an item came back twice"
    assert throughout <= set(seen), f"never returned: {sorted(throughout - set(seen))}"
    assert not [s for s in seen if s.startswith("new")], "an item added mid-walk is newer than the cursor"


def test_an_unknown_or_combined_cursor_is_an_error(tmp_path):
    _add(tmp_path, "c1", minutes=0)
    unknown = get_soul(after="nope", **_ctx(tmp_path))
    assert unknown["loaded"] is False and "next_after" in unknown["error"]
    both = get_soul(after="c1", offset=3, **_ctx(tmp_path))
    assert both["loaded"] is False and "not both" in both["error"]


def test_a_cursor_walk_stays_inside_its_love_type(tmp_path):
    for i in range(12):
        _add(tmp_path, f"c{i:02d}", minutes=i, love_type="craft" if i % 2 else "romantic")
    seen, page = [], get_soul(limit=2, love_type="craft", **_ctx(tmp_path))
    while True:
        seen += _ids(page)
        if page["next_after"] is None:
            break
        assert f'love_type="craft" and after="{page["next_after"]}"' in page["message"]
        page = get_soul(limit=2, love_type="craft", after=page["next_after"], **_ctx(tmp_path))
    assert seen == [f"c{i:02d}" for i in (11, 9, 7, 5, 3, 1)]
    assert page["offset"] is None and page["truncated"] is False


def test_the_schema_offers_after():
    from brain.tools.schemas import SCHEMAS

    params = SCHEMAS["get_soul"]["parameters"]
    assert params["properties"]["after"]["type"] == "string"
    assert "next_after" in SCHEMAS["get_soul"]["description"]


# ---- #734 part 2: a clipped item's text can be read in full ----
def test_a_clipped_item_can_be_read_field_by_field(tmp_path):
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES, result_bytes

    big = "ł" * 20_000  # 40 KB of UTF-8 per field
    _add(tmp_path, "huge", minutes=0, moment=big + "m", why_it_matters=big + "w", who_or_what="someone")

    page = get_soul(**_ctx(tmp_path))
    (item,) = page["crystallizations"]
    assert item["clipped"] is True
    assert set(item["clipped_at"]) == {"moment", "why_it_matters"}  # who_or_what fitted whole

    for field, want in (("moment", big + "m"), ("why_it_matters", big + "w")):
        got, at, reads = "", item["clipped_at"][field], 0
        assert item[field].removesuffix("…") == want[:at]
        got = want[:at]
        while at is not None:
            part = get_soul(id="huge", field=field, char_offset=at, **_ctx(tmp_path))
            assert part["loaded"] is True and result_bytes(part) <= MAX_RESULT_BYTES
            assert part["char_offset"] == at and part["length"] == len(want)
            got += part["text"]
            at = part["next_char_offset"]
            reads += 1
            assert reads < 20
        assert got == want, field


def test_a_field_read_names_each_mistake(tmp_path):
    _add(tmp_path, "c1", minutes=0)
    _add(tmp_path, "gone", minutes=1)
    _revoke(tmp_path, "gone")
    cases = {
        "unknown id": {"id": "nope", "field": "moment"},
        "revoked id": {"id": "gone", "field": "moment"},
        "no field": {"id": "c1"},
        "bad field": {"id": "c1", "field": "love_type"},
        "with after": {"id": "c1", "field": "moment", "after": "c1"},
        "with love_type": {"id": "c1", "field": "moment", "love_type": "craft"},
    }
    for name, args in cases.items():
        out = get_soul(**args, **_ctx(tmp_path))
        assert out["loaded"] is False and out["error"], name


def test_the_schema_offers_the_field_read():
    from brain.tools.schemas import SCHEMAS

    props = SCHEMAS["get_soul"]["parameters"]["properties"]
    assert props["id"]["type"] == "string"
    assert set(props["field"]["enum"]) == {"moment", "why_it_matters", "who_or_what"}
    assert props["char_offset"]["type"] == "integer"
    assert "clipped_at" in SCHEMAS["get_soul"]["description"]


# ---- #734 part 3: every love type the soul holds can be used as a filter ----
def test_a_stored_love_type_outside_the_list_is_a_usable_filter(tmp_path):
    from brain.soul.love_types import LOVE_TYPES

    assert "found-family-from-kit" not in LOVE_TYPES  # as the emergence-kit migrator can store
    _add(tmp_path, "kit1", minutes=0, love_type="found-family-from-kit")
    _add(tmp_path, "c1", minutes=1)

    page = get_soul(love_type="found-family-from-kit", **_ctx(tmp_path))
    assert page["loaded"] is True and _ids(page) == ["kit1"]
    assert "found-family-from-kit" in page["by_love_type"]
    unknown = get_soul(love_type="never-stored", **_ctx(tmp_path))
    assert unknown["loaded"] is False and "by_love_type" in unknown["error"]


def test_the_schema_lets_any_love_type_through():
    from brain.tools.schemas import SCHEMAS

    prop = SCHEMAS["get_soul"]["parameters"]["properties"]["love_type"]
    assert "enum" not in prop and "by_love_type" in prop["description"]


def test_a_page_with_a_clipped_item_names_the_call_that_reads_the_rest(tmp_path):
    """Review of #734: the spec says the message names the follow-up read; a one-page soul had no message at all."""
    from brain.tools.impls.get_soul import MAX_RESULT_BYTES, result_bytes

    _add(tmp_path, "huge", minutes=0, moment="ł" * 20_000)
    page = get_soul(**_ctx(tmp_path))
    (item,) = page["crystallizations"]
    n = item["clipped_at"]["moment"]
    assert page["truncated"] is False
    assert f'get_soul with id="huge", field="moment", char_offset={n}' in page["message"]
    assert result_bytes(page) + 64 <= MAX_RESULT_BYTES  # room left for the real tool prefix

    _add(tmp_path, "older", minutes=-1)  # now a second page follows: both hints are given
    page = get_soul(**_ctx(tmp_path))
    assert 'id="huge"' in page["message"] and f'after="{page["next_after"]}"' in page["message"]
    assert result_bytes(page) + 64 <= MAX_RESULT_BYTES
