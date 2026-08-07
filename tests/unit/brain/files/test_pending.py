# tests/unit/brain/files/test_pending.py
import hashlib
from datetime import UTC, datetime, timedelta

from brain.files.pending import (
    _TTL_HOURS,
    create,
    find_duplicate,
    get,
    list_pending,
    mark,
    sweep_expired,
)


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


def test_find_duplicate_matches_identical_pending_record(tmp_path):
    now = datetime.now(UTC)
    sha = hashlib.sha256(b"body").hexdigest()
    rid = create(tmp_path, op="append", resolved_path="/x/n.md",
                 content="body", now=now)
    found = find_duplicate(tmp_path, op="append", resolved_path="/x/n.md",
                            content_sha=sha, now=now)
    assert found == rid


def test_find_duplicate_ignores_different_content(tmp_path):
    now = datetime.now(UTC)
    create(tmp_path, op="append", resolved_path="/x/n.md", content="body", now=now)
    other = hashlib.sha256(b"different").hexdigest()
    assert find_duplicate(tmp_path, op="append", resolved_path="/x/n.md",
                           content_sha=other, now=now) is None


def test_find_duplicate_ignores_different_path_and_op(tmp_path):
    now = datetime.now(UTC)
    sha = hashlib.sha256(b"body").hexdigest()
    create(tmp_path, op="append", resolved_path="/x/n.md", content="body", now=now)
    assert find_duplicate(tmp_path, op="append", resolved_path="/x/OTHER.md",
                           content_sha=sha, now=now) is None
    assert find_duplicate(tmp_path, op="create", resolved_path="/x/n.md",
                           content_sha=sha, now=now) is None


def test_find_duplicate_ignores_resolved_records(tmp_path):
    """Once the user acts, an identical later proposal is legitimately new."""
    now = datetime.now(UTC)
    sha = hashlib.sha256(b"body").hexdigest()
    rid = create(tmp_path, op="append", resolved_path="/x/n.md",
                 content="body", now=now)
    for status in ("committed", "declined", "refused", "expired"):
        mark(tmp_path, rid, status=status)
        assert find_duplicate(tmp_path, op="append", resolved_path="/x/n.md",
                               content_sha=sha, now=now) is None, status


def test_find_duplicate_ignores_stale_pending_past_ttl(tmp_path):
    """A record still marked pending but past the TTL has just not been swept —
    it must not suppress a legitimate new proposal."""
    old = datetime.now(UTC) - timedelta(hours=_TTL_HOURS + 1)
    sha = hashlib.sha256(b"body").hexdigest()
    create(tmp_path, op="append", resolved_path="/x/n.md", content="body", now=old)
    assert find_duplicate(tmp_path, op="append", resolved_path="/x/n.md",
                           content_sha=sha, now=datetime.now(UTC)) is None
