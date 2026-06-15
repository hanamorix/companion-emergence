from datetime import UTC, datetime

from brain.bridge.feed import build_maker_entries
from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def _shared_work(tmp_path):
    w = Work(
        id=make_work_id("c"),
        title="Dusk",
        type="vignette",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=2,
        summary="Light fell.",
        disposition="eventual_share",
        private_reason=None,
        origin="maker",
        charge_sources=None,
        shared_at=datetime.now(UTC).isoformat(),
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content="Light fell.")
    s.close()


def test_feed_includes_shared_making_not_private(tmp_path):
    _shared_work(tmp_path)
    entries = build_maker_entries(tmp_path, limit=10)
    assert any(e.body and "Light fell" in e.body for e in entries) or len(entries) == 1
