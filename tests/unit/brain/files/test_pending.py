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


# ---- #344: compare-and-set mark + extras + list_by_status ------------------


def _rec(tmp_path, **kw):
    return create(tmp_path, op="create", resolved_path="/x", content="c",
                  now=kw.pop("now", datetime(2026, 6, 14, tzinfo=UTC)), **kw)


def test_mark_merges_extra_fields(tmp_path):
    from brain.files import pending

    rid = _rec(tmp_path)
    assert pending.mark(tmp_path, rid, status="committing", claimed_at="2026-06-14T00:01:00+00:00")
    rec = get(tmp_path, rid)
    assert rec["status"] == "committing"
    assert rec["claimed_at"] == "2026-06-14T00:01:00+00:00"


def test_mark_expect_is_a_compare_and_set(tmp_path):
    from brain.files import pending

    rid = _rec(tmp_path)
    assert pending.mark(tmp_path, rid, status="committing", expect="pending") is True
    # A second claimer finds it already claimed: refused, record untouched.
    assert pending.mark(tmp_path, rid, status="committing", expect="pending", claimed_at="x") is False
    assert "claimed_at" not in get(tmp_path, rid)
    # Without expect the old unconditional behaviour is unchanged.
    assert pending.mark(tmp_path, rid, status="committed") is True


def test_list_by_status(tmp_path):
    from brain.files import pending

    a, b = _rec(tmp_path, making_id="1"), create(
        tmp_path, op="create", resolved_path="/y", content="d", now=datetime(2026, 6, 14, 1, tzinfo=UTC))
    pending.mark(tmp_path, a, status="committing")
    assert [r["id"] for r in pending.list_by_status(tmp_path, "committing")] == [a]
    assert [r["id"] for r in pending.list_by_status(tmp_path, "pending")] == [b]


def test_readers_tolerate_records_with_and_without_the_new_fields(tmp_path):
    """#344 added claimed_at / resolved_by to the record. Both readers and the dedupe must work
    on a record that has neither (written by an older brain) and on one that has them."""
    from brain.files import pending

    now = datetime(2026, 6, 14, tzinfo=UTC)
    old = create(tmp_path, op="create", resolved_path="/old", content="o", now=now)
    new = create(tmp_path, op="create", resolved_path="/new", content="n", now=now)
    pending.mark(tmp_path, new, status="pending", claimed_at="2026-06-14T00:00:00+00:00",
                 resolved_by="reconcile")
    assert {r["id"] for r in list_pending(tmp_path, now=now)} == {old, new}
    sha = hashlib.sha256(b"n").hexdigest()
    assert find_duplicate(tmp_path, op="create", resolved_path="/new", content_sha=sha,
                          now=now) == new


def test_two_simultaneous_claims_of_one_record_cannot_both_win(tmp_path, monkeypatch):
    """#346: expect= is a re-read; without a lock two claimers that both read 'pending' both
    write. A barrier inside get() forces exactly that overlap (it times out if the lock
    serialises them, which is the point)."""
    import threading

    from brain.files import pending

    rid = _rec(tmp_path)
    real_get, barrier = pending.get, threading.Barrier(2)

    def _overlapping_get(*a, **k):
        rec = real_get(*a, **k)
        try:
            barrier.wait(timeout=0.4)
        except threading.BrokenBarrierError:
            pass
        return rec

    monkeypatch.setattr(pending, "get", _overlapping_get)
    outcomes: list[object] = []

    def _claim():
        try:
            outcomes.append(pending.mark(tmp_path, rid, status="committing", expect="pending"))
        except Exception as exc:  # a crash on the shared .tmp is a failure too, not a lost race
            outcomes.append(exc)

    threads = [threading.Thread(target=_claim) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes, key=repr) == [False, True]


def test_create_and_mark_both_take_the_store_lock(tmp_path, monkeypatch):
    """create shares the `.tmp` naming with mark, so it must serialise with it too."""
    import contextlib

    from brain.files import pending

    held: list[str] = []
    real = pending.file_lock

    @contextlib.contextmanager
    def _spy(path, **kw):
        held.append(path.name)
        with real(path, **kw) as ok:
            yield ok

    monkeypatch.setattr(pending, "file_lock", _spy)
    rid = _rec(tmp_path)
    assert held == [".records"]
    pending.mark(tmp_path, rid, status="committing")
    assert held == [".records", ".records"]


def test_sweep_does_not_overwrite_a_record_claimed_after_its_read(tmp_path, monkeypatch):
    """#344 review F-d: sweep read 'pending', a claim landed, sweep's unconditional
    mark then overwrote the live claim with 'expired'. With expect='pending' it must not."""
    from brain.files import pending

    t0 = datetime(2026, 6, 14, tzinfo=UTC)
    rid = create(tmp_path, op="create", resolved_path="/x", content="c", now=t0)
    real_all = pending._all

    def _all_then_claim(persona_dir):
        rows = real_all(persona_dir)
        pending.mark(persona_dir, rid, status="committing")  # the competing claim
        return rows

    monkeypatch.setattr(pending, "_all", _all_then_claim)
    n = sweep_expired(tmp_path, now=t0 + timedelta(hours=25))
    monkeypatch.undo()
    assert get(tmp_path, rid)["status"] == "committing"
    assert n == 0, "a sweep whose mark was refused must not count the record"
