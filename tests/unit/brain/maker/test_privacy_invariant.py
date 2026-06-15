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


def test_feed_builder_filters_through_the_gate(tmp_path, monkeypatch):
    """Even if the SQL widened to return private rows, the gate drops them.

    Two halves prove defence-in-depth, not SQL-reliance:
    1. The builder must CONSULT is_auto_surfaceable() on every candidate row
       (a spy fails RED if the builder relies on the WHERE clause alone).
    2. With the store's recent-listing widened to return ALL rows (a simulated
       future query-widening bug), only the gate keeps the private making out.
    """
    _make(tmp_path, "private", "LEAK_TITLE", "LEAK_BODY")
    _make(tmp_path, "eventual_share", "ShareTitle", "share body")

    from brain.works.store import WorksStore, _row_to_work

    # widen: every list_recent call returns every row regardless of disposition
    def _all_rows(self, *, limit=20, type=None):
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM works ORDER BY created_at DESC").fetchall()
        return [_row_to_work(r) for r in rows]

    monkeypatch.setattr(WorksStore, "list_recent", _all_rows)

    # spy on the gate: it MUST be the thing the builder consults
    import brain.bridge.feed as feed
    import brain.maker.privacy as privacy
    seen_dispositions = []
    real_gate = privacy.is_auto_surfaceable

    def _spy(work):
        seen_dispositions.append(work.disposition)
        return real_gate(work)

    monkeypatch.setattr(feed, "is_auto_surfaceable", _spy, raising=False)

    entries = feed.build_maker_entries(tmp_path, limit=50)

    # the builder consulted the gate on the widened (private-included) rows
    assert "private" in seen_dispositions
    # the private making is dropped by the gate, not by SQL
    assert all("LEAK" not in (e.body or "") for e in entries)
    # ...but the legitimately-shared making still surfaces (gate isn't over-filtering)
    assert any("share body" in (e.body or "") for e in entries)
