from datetime import UTC, datetime

from brain.tools.impls.propose_write import propose_write
from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def _making(tmp_path, content="A poem about dusk."):
    w = Work(
        id=make_work_id(content),
        title="Dusk",
        type="poem",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=4,
        summary=content,
        disposition="eventual_share",
        private_reason=None,
        origin="maker",
        charge_sources=None,
        shared_at=None,
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content=content)
    s.close()
    return w.id


def test_propose_write_pulls_content_from_making(tmp_path):
    # persona_dir holds the portfolio; the export target lives OUTSIDE it
    # (the write-guard denies any path inside persona_dir — her own substrate).
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    export_dir = tmp_path / "user_docs"

    wid = _making(persona_dir)
    # writing markdown so read_markdown can resolve the making content
    from brain.works.storage import write_markdown
    from brain.works.store import WorksStore

    s = WorksStore(persona_dir / "works.db")
    w = s.get(wid)
    s.close()
    write_markdown(persona_dir, w, content="A poem about dusk.")

    out = propose_write(
        path=str(export_dir / "dusk.md"),
        content=None,
        op="create",
        making_id=wid,
        persona_dir=persona_dir,
    )
    assert out["status"] == "proposed"
    # the pending request carries the making's content, not empty
    from brain.files.pending import get as get_pending

    req = get_pending(persona_dir, out["id"])
    assert "dusk" in req["content"].lower()
    # nothing written yet (still needs user approval)
    assert not (export_dir / "dusk.md").exists()
