"""PendingQueue.claim: the crash-safe replacement for `drain` inside the consolidation gate (#459).

`drain` empties the queue BEFORE any item is processed, so a kill or an exception during the slow per-item work loses every
undone item. `claim` first writes the batch to a processing file (atomically), then empties the queue; each item is recorded in a
settled file once it is dealt with; the next `claim` recovers whatever was left. `drain` itself is unchanged.
"""

from __future__ import annotations

import threading

import pytest

from brain.health.json_io import read_jsonl, write_jsonl
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory

_PROCESSING = "pending_candidates.processing.jsonl"
_SETTLED = "pending_candidates.settled.jsonl"


def _mem(content: str) -> Memory:
    return Memory.create_new(content=content, memory_type="dream", domain="us", emotions={})


def _contents(entries) -> list[str]:
    return [e["content"] for e in entries]


def test_claim_of_an_empty_queue_with_no_files_creates_no_files(tmp_path):
    batch = PendingQueue(tmp_path).claim()

    assert batch.entries == []
    batch.close()
    assert not (tmp_path / _PROCESSING).exists() and not (tmp_path / _SETTLED).exists()


def test_claim_takes_the_queue_into_the_processing_file_and_empties_the_queue(tmp_path):
    q = PendingQueue(tmp_path)
    for c in ("a", "b"):
        q.enqueue(_mem(c), source="dream")

    batch = q.claim()

    assert _contents(batch.entries) == ["a", "b"]
    assert all(e["_claim_id"] for e in batch.entries) and all(e["_source"] == "dream" for e in batch.entries)
    assert _contents(read_jsonl(tmp_path / _PROCESSING)) == ["a", "b"]
    assert read_jsonl(q.path) == []


def test_close_with_everything_settled_removes_both_files(tmp_path):
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("a"), source="dream")
    batch = q.claim()
    for e in batch.entries:
        batch.settle(e)

    batch.close()

    assert not (tmp_path / _PROCESSING).exists() and not (tmp_path / _SETTLED).exists()


def test_close_with_an_unsettled_entry_leaves_both_files_and_the_next_claim_returns_only_that_entry(tmp_path):
    q = PendingQueue(tmp_path)
    for c in ("a", "b"):
        q.enqueue(_mem(c), source="dream")
    batch = q.claim()
    batch.settle(batch.entries[0])
    batch.close()

    assert (tmp_path / _PROCESSING).exists() and (tmp_path / _SETTLED).exists()
    again = q.claim()
    assert _contents(again.entries) == ["b"]


def test_a_killed_run_is_recovered_before_the_queue_and_new_entries_follow(tmp_path):
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("old"), source="dream")
    q.claim()  # "killed": never settled, never closed
    q.enqueue(_mem("new"), source="dream")

    batch = q.claim()

    assert _contents(batch.entries) == ["old", "new"]


def test_claim_recovers_when_the_processing_file_exists_and_the_queue_file_is_absent(tmp_path):
    # C459-q(4): `drain()` returns [] when the queue file is missing; `claim()` must not copy that.
    entry = {**_mem("left over").to_dict(), "_source": "dream", "_enqueued_at": "t", "_claim_id": "old1"}
    write_jsonl(tmp_path / _PROCESSING, [entry])
    q = PendingQueue(tmp_path)
    assert not q.path.exists()

    batch = q.claim()

    assert _contents(batch.entries) == ["left over"]


def test_stale_settled_ids_never_skip_an_entry_of_a_new_claim(tmp_path):
    # C459-q(1): the settled file is keyed by claim id and claim ids are restamped, so ids left over from an older
    # processing file cannot match the new batch.
    leftover = {**_mem("p1").to_dict(), "_source": "dream", "_enqueued_at": "t", "_claim_id": "n1"}
    write_jsonl(tmp_path / _PROCESSING, [leftover])
    write_jsonl(tmp_path / _SETTLED, [{"id": "zzz-stale"}])
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("q1"), source="dream")

    batch = q.claim()

    assert _contents(batch.entries) == ["p1", "q1"]


def test_an_unreadable_processing_file_makes_claim_raise_before_changing_anything(tmp_path):
    # C459-r: an I/O fault on the processing file (a directory in its place).
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("a"), source="dream")
    before = q.path.read_bytes()
    (tmp_path / _PROCESSING).mkdir()

    with pytest.raises(OSError):
        q.claim()

    assert q.path.read_bytes() == before


@pytest.mark.parametrize("failing_write", [1, 2, 3], ids=["processing", "settled-reset", "queue-empty"])
def test_claim_is_crash_ordered_at_each_of_its_three_writes(tmp_path, monkeypatch, failing_write):
    # C459-r: inject a failure at each write of `claim`. The processing file is written FIRST, so nothing is ever lost:
    # at (1) the queue is byte-identical and there is no processing file; at (2) and (3) the entries are in BOTH files.
    import brain.memory.pending as pending_mod

    q = PendingQueue(tmp_path)
    for c in ("a", "b"):
        q.enqueue(_mem(c), source="dream")
    before = q.path.read_bytes()
    real = pending_mod.write_jsonl
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == failing_write:
            raise OSError("injected")
        return real(*args, **kwargs)

    monkeypatch.setattr(pending_mod, "write_jsonl", flaky)
    with pytest.raises(OSError):
        q.claim()
    monkeypatch.setattr(pending_mod, "write_jsonl", real)

    if failing_write == 1:
        assert q.path.read_bytes() == before and not (tmp_path / _PROCESSING).exists()
    else:
        assert _contents(read_jsonl(tmp_path / _PROCESSING)) == ["a", "b"]
        assert _contents(read_jsonl(q.path)) == ["a", "b"]
    # Recovery: everything is still there (duplicates are expected after a mid-claim crash; the gate drops exact repeats).
    got = _contents(q.claim().entries)
    assert sorted(set(got)) == ["a", "b"]


def test_enqueue_during_a_claimed_run_lands_in_the_queue_not_the_processing_file(tmp_path):
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("a"), source="dream")
    batch = q.claim()

    q.enqueue(_mem("late"), source="dream")

    assert _contents(read_jsonl(q.path)) == ["late"]
    assert _contents(read_jsonl(tmp_path / _PROCESSING)) == ["a"]
    batch.close()


def test_drain_is_unchanged(tmp_path):
    q = PendingQueue(tmp_path)
    q.enqueue(_mem("a"), source="dream")

    assert _contents(q.drain()) == ["a"]
    assert q.drain() == []


def test_no_entry_is_lost_or_claimed_twice_under_an_enqueue_claim_race(tmp_path):
    # C459-s2: like the existing C10 test for `drain` (tests/test_consolidation_gate.py), for `claim`.
    q = PendingQueue(tmp_path)
    n = 60
    claimed: list[str] = []
    lock = threading.Lock()
    start = threading.Event()

    def enqueue_worker():
        start.wait()
        for i in range(n):
            q.enqueue(_mem(f"race {i}"), source="dream")

    def claim_worker():
        start.wait()
        for _ in range(30):
            batch = q.claim()
            for e in batch.entries:
                batch.settle(e)
            batch.close()
            with lock:
                claimed.extend(_contents(batch.entries))

    threads = [threading.Thread(target=enqueue_worker), threading.Thread(target=claim_worker)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()
    final = q.claim()
    claimed.extend(_contents(final.entries))

    assert len(claimed) == n and len(set(claimed)) == n
