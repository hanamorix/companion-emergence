from datetime import UTC, datetime

from brain.maker.ambient import build_maker_awareness_block
from brain.works import Work, make_work_id
from brain.works.store import WorksStore


def test_block_tags_private_as_hers_alone(tmp_path):
    w = Work(
        id=make_work_id("p"),
        title="Secret",
        type="letter",
        created_at=datetime.now(UTC),
        session_id=None,
        word_count=1,
        summary="s",
        disposition="private",
        private_reason="raw",
        origin="maker",
        charge_sources=None,
        shared_at=None,
    )
    s = WorksStore(tmp_path / "works.db")
    s.insert(w, content="p")
    s.close()
    block = build_maker_awareness_block(tmp_path, limit=5)
    assert block is not None
    assert "Secret" in block
    assert "yours alone" in block.lower() or "private" in block.lower()
