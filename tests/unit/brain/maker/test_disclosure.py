from datetime import UTC, datetime

from brain.maker.disclosure import surface_makings
from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def _w(tmp_path, disp, title, content):
    w = Work(
        id=make_work_id(content),
        title=title,
        type="poem",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=1,
        summary=content,
        disposition=disp,
        private_reason="mine" if disp == "private" else None,
        origin="maker",
        charge_sources=None,
        shared_at=datetime.now(UTC).isoformat() if disp == "eventual_share" else None,
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content=content)
    s.close()


def test_surface_returns_shareable_with_content_and_flags_private(tmp_path):
    _w(tmp_path, "eventual_share", "Open", "open content")
    _w(tmp_path, "private", "Closed", "closed content")
    result = surface_makings(persona_dir=tmp_path)
    titles = {m["title"]: m for m in result["makings"]}
    assert titles["Open"]["content"] == "open content"  # she may share it
    assert titles["Closed"]["private"] is True  # flagged so SHE decides
    assert titles["Closed"]["private_reason"] == "mine"
    # discard makings have no content to surface
    assert all(m["title"] != "discarded" for m in result["makings"])


def test_discard_making_is_omitted(tmp_path):
    _w(tmp_path, "eventual_share", "Kept", "kept content")
    _w(tmp_path, "discard", "discarded", "thrown away")
    result = surface_makings(persona_dir=tmp_path)
    titles = {m["title"] for m in result["makings"]}
    assert "Kept" in titles
    assert "discarded" not in titles


def test_no_works_db_returns_empty(tmp_path):
    result = surface_makings(persona_dir=tmp_path)
    assert result == {"makings": []}


def test_surface_makings_registered_at_all_contract_sites():
    from brain.chat.voice import DEFAULT_VOICE_TEMPLATE
    from brain.tools import NELL_TOOL_NAMES
    from brain.tools.dispatch import _DISPATCH
    from brain.tools.schemas import SCHEMAS

    assert "surface_makings" in SCHEMAS
    assert "surface_makings" in _DISPATCH
    assert "surface_makings" in NELL_TOOL_NAMES
    assert "surface_makings" in DEFAULT_VOICE_TEMPLATE
