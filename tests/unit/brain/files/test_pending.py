# tests/unit/brain/files/test_pending.py
from datetime import UTC, datetime, timedelta

from brain.files.pending import create, get, list_pending, mark, sweep_expired


def test_create_list_get(tmp_path):
    rid = create(tmp_path, op="create", resolved_path="/x/y.md", content="hi",
                 now=datetime(2026, 6, 14, tzinfo=UTC))
    assert get(tmp_path, rid)["content"] == "hi"
    assert any(r["id"] == rid for r in list_pending(tmp_path, now=datetime(2026, 6, 14, tzinfo=UTC)))


def test_expiry_after_24h(tmp_path):
    t0 = datetime(2026, 6, 14, 0, 0, tzinfo=UTC)
    rid = create(tmp_path, op="create", resolved_path="/x", content="c", now=t0)
    later = t0 + timedelta(hours=25)
    n = sweep_expired(tmp_path, now=later)
    assert n == 1
    assert get(tmp_path, rid)["status"] == "expired"
    assert list_pending(tmp_path, now=later) == []  # expired excluded


def test_mark_committed(tmp_path):
    rid = create(tmp_path, op="create", resolved_path="/x", content="c",
                 now=datetime(2026, 6, 14, tzinfo=UTC))
    mark(tmp_path, rid, status="committed")
    assert get(tmp_path, rid)["status"] == "committed"
