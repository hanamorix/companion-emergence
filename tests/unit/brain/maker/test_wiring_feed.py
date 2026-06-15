from datetime import UTC, datetime, timedelta

from brain.maker.wiring import flip_ready_shares
from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def test_eventual_share_flips_to_shared_after_delay(tmp_path):
    old = datetime.now(UTC) - timedelta(hours=24)
    w = Work(
        id=make_work_id("x"),
        title="t",
        type="poem",
        created_at=old,
        session_id=None,
        word_count=1,
        summary=None,
        disposition="eventual_share",
        private_reason=None,
        origin="maker",
        charge_sources=None,
        shared_at=None,
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content="x")
    s.close()
    n = flip_ready_shares(tmp_path, now=datetime.now(UTC), delay_hours=12.0)
    assert n == 1
    s = WorksStore(tmp_path / "works.db")
    assert s.get(make_work_id("x")).shared_at is not None
    s.close()
