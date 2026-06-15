"""LOAD-BEARING: a private (or discard) making escapes NO automatic surface."""
from datetime import UTC, datetime

from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def _make(tmp_path, disposition, title, content):
    w = Work(
        id=make_work_id(content),
        title=title,
        type="letter",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=1,
        summary=content,
        disposition=disposition,
        private_reason="mine",
        origin="maker",
        charge_sources=None,
        shared_at=datetime.now(UTC).isoformat() if disposition == "eventual_share" else None,
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content=content)
    s.close()


def test_private_making_in_no_automatic_surface(tmp_path):
    _make(tmp_path, "private", "SECRET_TITLE", "SECRET_BODY_XYZ")
    _make(tmp_path, "eventual_share", "ShareTitle", "share body")

    # 1. feed
    from brain.bridge.feed import build_maker_entries
    feed = build_maker_entries(tmp_path, limit=50)
    assert all("SECRET" not in (e.body or "") and "SECRET" not in (e.opener or "") for e in feed)

    # 2. ambient awareness block — private may appear to HER, but tagged hers-alone,
    #    and its CONTENT (body) must never be in the block (only title + type).
    from brain.maker.ambient import build_maker_awareness_block
    block = build_maker_awareness_block(tmp_path, limit=50) or ""
    assert "SECRET_BODY_XYZ" not in block

    # 3. the gate itself
    from brain.maker.privacy import is_auto_surfaceable
    from brain.works.store import WorksStore
    s = WorksStore(tmp_path / "works.db")
    priv = s.get(make_work_id("SECRET_BODY_XYZ"))
    shared = s.get(make_work_id("share body"))
    s.close()
    assert is_auto_surfaceable(priv) is False
    assert is_auto_surfaceable(shared) is True
