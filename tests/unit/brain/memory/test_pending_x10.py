"""Engine 6, Q10c: the x10 fix of small feelings reaches the pending-candidate queue (`PendingQueue.scale_up_small_making_candidates`).

A `making` candidate that waits in `pending_candidates.jsonl`, or in the claimed batch `pending_candidates.processing.jsonl`, when a
persona's engine first starts is not a `memories.db` row yet, so the database half of the fix cannot reach it, and the consolidation
gate would promote it later at the old small size. Owner (Planning B21, Q10c): "They're memories so they should get treated as such".

The queue is driven through `nudge_scale_fix.scale_up_small_feelings` (the one entry the first start calls) wherever the behaviour is
the point, and through the method itself for counts and failure paths. Entries are written the way the product writes them (`enqueue`)
or as raw bytes where a line has to be damaged on purpose. Nothing here touches a real persona.
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, nudge_scale_fix
from brain.health.json_io import read_jsonl
from brain.memory import pending
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
QUEUE = "pending_candidates.jsonl"
PROCESSING = "pending_candidates.processing.jsonl"


def _key() -> str:
    """The metadata key of the marker, looked up late so a test of the behaviour fails on its own assertion first."""
    return dev_constants.NUDGE_SCALE_FIX_MARKER_KEY


def _memory(memory_id: str, emotions: dict[str, float], *, memory_type: str = "making", metadata: dict | None = None) -> Memory:
    numbers = [float(v) for v in emotions.values()]
    return Memory(
        id=memory_id,
        content=f"content of {memory_id}",
        memory_type=memory_type,
        domain="interior",
        created_at=NOW,
        emotions=dict(emotions),
        tags=["making"],
        importance=5.0,
        score=sum(numbers),
        metadata=dict(metadata or {}),
        peak_emotion_intensity=max(numbers, default=0.0),
    )


def _enqueue(persona: Path, *memories: Memory) -> None:
    queue = PendingQueue(persona)
    for memory in memories:
        queue.enqueue(memory, source="maker")


def _entries(persona: Path, name: str = QUEUE) -> dict[str, dict]:
    return {entry["id"]: entry for entry in read_jsonl(persona / name)}


def _raw_line(memory_id: str, emotions: object, **fields: object) -> bytes:
    """One queue line written by hand (spacing and key order are NOT the encoder's), so an unchanged line can be compared byte for byte."""
    entry = {
        "id": memory_id,
        "content": f"content of {memory_id}",
        "memory_type": "making",
        "domain": "interior",
        "created_at": NOW.isoformat(),
        "emotions": emotions,
        "tags": [],
        "importance": 5.0,
        "score": 0.2,
        "metadata": {},
        "_source": "maker",
        "_enqueued_at": NOW.isoformat(),
        **fields,
    }
    return json.dumps(entry, indent=None, separators=(",  ", ":  ")).encode("utf-8")


def _write_lines(path: Path, *lines: bytes) -> None:
    path.write_bytes(b"\n".join(lines) + b"\n")


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


# --- the selector and the multiply ---------------------------------------------------------------


def test_a_small_making_candidate_is_multiplied_whole_and_marked_and_nothing_else_is_touched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    _enqueue(
        tmp_path,
        _memory("plain", {"tenderness": 0.12, "satisfaction": 0.08}, metadata={"keep": "me"}),
        _memory("mixed", {"warmth": 0.4, "fear": -0.3}),
        _memory("empty", {}),
        _memory("big", {"pride": 4.0, "joy": 0.3}),  # largest value is 4: not a small entry
        _memory("one", {"warmth": 1.0}),  # exactly 1 is not below the threshold
        _memory("talk", {"joy": 0.2}, memory_type="conversation"),
    )
    before = _entries(tmp_path)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    after = _entries(tmp_path)
    # hand-worked: 0.12 x 10 = 1.2 and 0.08 x 10 = 0.8, score 0.20 x 10 = 2.0
    assert after["plain"]["emotions"] == pytest.approx({"tenderness": 1.2, "satisfaction": 0.8})
    assert after["plain"]["score"] == pytest.approx(2.0)
    # 0.4 x 10 = 4.0 and -0.3 x 10 = -3.0 (negatives included, the whole entry), score (0.4 - 0.3) x 10 = 1.0
    assert after["mixed"]["emotions"] == pytest.approx({"warmth": 4.0, "fear": -3.0})
    assert after["mixed"]["score"] == pytest.approx(1.0)
    assert after["empty"]["emotions"] == {} and after["empty"]["score"] == 0.0
    for kept in ("big", "one", "talk"):  # not selected: identical entries
        assert after[kept] == before[kept]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # an entry on the 0-10 scale is not a damaged one
    assert after["plain"]["metadata"][_key()] is True and after["plain"]["metadata"]["keep"] == "me"  # the marker joins the metadata
    assert after["mixed"]["metadata"][_key()] is True
    assert after["empty"]["metadata"][_key()] is True  # an empty dict counts as peak 0: selected
    assert not any(_key() in after[kept]["metadata"] for kept in ("big", "one", "talk"))
    # every other field of a multiplied entry is as it was
    for multiplied in ("plain", "mixed", "empty"):
        assert {k: v for k, v in after[multiplied].items() if k not in ("emotions", "score", "metadata")} == {
            k: v for k, v in before[multiplied].items() if k not in ("emotions", "score", "metadata")
        }


def test_an_entry_already_marked_is_not_multiplied_again_even_though_it_is_small(tmp_path: Path) -> None:
    _enqueue(tmp_path, _memory("small", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    marked = _entries(tmp_path)["small"]
    assert marked["emotions"] == pytest.approx({"joy": 0.5})  # 0.05 x 10, once
    _enqueue(tmp_path, _memory("hand-marked", {"joy": 0.3}, metadata={_key(): True}))
    first = QueuedBytes.of(tmp_path)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _entries(tmp_path)["small"]["emotions"] == pytest.approx({"joy": 0.5})  # not 5.0
    assert _entries(tmp_path)["hand-marked"]["emotions"] == {"joy": 0.3}
    assert QueuedBytes.of(tmp_path) == first  # nothing left to change: the file is not rewritten


class QueuedBytes:
    """The bytes of both queue files, to compare "not rewritten at all"."""

    @staticmethod
    def of(persona: Path) -> tuple[bytes | None, bytes | None]:
        return tuple((persona / name).read_bytes() if (persona / name).exists() else None for name in (QUEUE, PROCESSING))  # type: ignore[return-value]


def test_an_entry_queued_between_two_runs_is_caught_and_the_first_is_not_multiplied_twice(tmp_path: Path) -> None:
    """The retry case: a switch-over that fails after the queue step is run again at the next start."""
    _enqueue(tmp_path, _memory("first", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    _enqueue(tmp_path, _memory("second", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    entries = _entries(tmp_path)
    assert entries["first"]["emotions"] == pytest.approx({"joy": 0.5})  # 0.05 -> 0.5, never 5.0
    assert entries["second"]["emotions"] == pytest.approx({"joy": 0.5})


def test_the_marker_survives_promotion_and_the_database_half_then_leaves_the_promoted_row_alone(tmp_path: Path) -> None:
    _enqueue(tmp_path, _memory("small", {"joy": 0.05}, metadata={"keep": "me"}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    (entry,) = read_jsonl(tmp_path / QUEUE)
    promoted = Memory.from_dict(entry)  # what the gate does at consolidation.py (`Memory.from_dict(entry)`) before `store.create`
    assert promoted.emotions == pytest.approx({"joy": 0.5}) and promoted.score == pytest.approx(0.5)
    assert promoted.metadata == {"keep": "me", _key(): True}
    assert [m.metadata for m in PendingQueue(tmp_path).read_recent("making", limit=5)] == [{"keep": "me", _key(): True}]
    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(promoted, fold=False)  # the gate's `store.create(cand)`
    finally:
        store.close()
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 0  # a retried switch-over: 0.5 is still below 1, but the row is marked
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        row = store.get("small", bump=False)
        assert row is not None and row.emotions == pytest.approx({"joy": 0.5})  # not 5.0
    finally:
        store.close()


def test_a_file_with_nothing_to_multiply_is_not_rewritten_at_all_whatever_it_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOM, CRLF endings, a blank line, garbage, a torn last line: kept exactly as they are when no entry is selected."""
    content = (
        b"\xef\xbb\xbf"
        + _raw_line("big", {"joy": 6.0})
        + b"\r\n"
        + b"\n"
        + b"   \n"
        + b"garbage \xff\xfe\n"
        + _raw_line("done", {"joy": 0.3}, metadata={_key(): True})
        + b"\n"
        + _raw_line("talk", {"joy": 0.2}, memory_type="conversation")
        + b"\n"
        + b'{"id": "torn", "emotions": {"joy": 0.1}'
    )
    (tmp_path / QUEUE).write_bytes(content)
    writes: list[Path] = []
    real_write = pending.write_jsonl_bytes
    monkeypatch.setattr(pending, "write_jsonl_bytes", lambda path, data, **kw: writes.append(path) or real_write(path, data, **kw))
    assert PendingQueue(tmp_path).scale_up_small_making_candidates() == 0
    assert (tmp_path / QUEUE).read_bytes() == content and writes == []


def test_an_entry_with_no_score_is_multiplied_in_its_feelings_and_gets_no_score(tmp_path: Path) -> None:
    line = json.loads(_raw_line("noscore", {"joy": 0.05}))
    del line["score"]
    _write_lines(tmp_path / QUEUE, json.dumps(line).encode("utf-8"), _raw_line("nullscore", {"joy": 0.05}, score=None))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    entries = _entries(tmp_path)
    assert entries["noscore"]["emotions"] == pytest.approx({"joy": 0.5}) and "score" not in entries["noscore"]
    assert entries["nullscore"]["emotions"] == pytest.approx({"joy": 0.5}) and entries["nullscore"]["score"] is None  # null reads as missing


# --- both files ------------------------------------------------------------------------------------


def test_the_claimed_batch_in_the_processing_file_is_multiplied_too_and_so_is_the_queue(tmp_path: Path) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("claimed", {"joy": 0.05}))
    batch = queue.claim()  # the gate's claim: the entry now waits in the processing file, the queue is empty
    assert [e["id"] for e in batch.entries] == ["claimed"]
    _enqueue(tmp_path, _memory("queued", {"joy": 0.07}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _entries(tmp_path, PROCESSING)["claimed"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _entries(tmp_path, PROCESSING)["claimed"]["_claim_id"]  # the claim stays as it was
    assert _entries(tmp_path)["queued"]["emotions"] == pytest.approx({"joy": 0.7})
    assert _entries(tmp_path, PROCESSING)["claimed"]["metadata"][_key()] is True


def test_the_method_returns_how_many_entries_it_multiplied_across_both_files_and_a_second_call_returns_zero(tmp_path: Path) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}), _memory("b", {"joy": 0.06}))
    queue.claim()
    _enqueue(tmp_path, _memory("c", {"joy": 0.07}), _memory("big", {"joy": 5.0}))
    assert queue.scale_up_small_making_candidates() == 3
    assert queue.scale_up_small_making_candidates() == 0


def test_a_persona_with_no_queue_files_creates_nothing_and_a_missing_processing_file_is_fine(tmp_path: Path) -> None:
    assert PendingQueue(tmp_path).scale_up_small_making_candidates() == 0
    assert list(tmp_path.iterdir()) == []  # not even a lock sidecar
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    assert PendingQueue(tmp_path).scale_up_small_making_candidates() == 1
    assert not (tmp_path / PROCESSING).exists()
    (tmp_path / QUEUE).unlink()
    _write_lines(tmp_path / PROCESSING, _raw_line("only-claimed", {"joy": 0.05}))
    assert PendingQueue(tmp_path).scale_up_small_making_candidates() == 1  # the processing file alone is reached too
    assert not (tmp_path / QUEUE).exists()


# --- a line that cannot be used ---------------------------------------------------------------------


def test_an_unparsable_line_is_kept_byte_for_byte_in_its_place_and_the_others_are_still_multiplied(tmp_path: Path) -> None:
    spaced_big = _raw_line("spaced", {"joy": 6.0})  # a valid, unselected line with the hand spacing: not re-encoded
    bad_text = b"this is not json {"
    not_an_object = b'["a", "list"]'
    bad_utf8 = b'{"id": "x", "emotions": {"joy": 0.1}, \xff\xfe broken \xc3'
    _write_lines(
        tmp_path / QUEUE,
        _raw_line("one", {"joy": 0.05}),
        bad_text,
        spaced_big,
        not_an_object,
        bad_utf8,
        _raw_line("two", {"joy": 0.06}),
    )
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) is None  # no database: that half did not run
    lines = _lines(tmp_path / QUEUE)
    assert len(lines) == 6
    assert lines[1] == bad_text and lines[2] == spaced_big and lines[3] == not_an_object and lines[4] == bad_utf8
    assert json.loads(lines[0])["emotions"] == pytest.approx({"joy": 0.5})  # in its place, first
    assert json.loads(lines[5])["emotions"] == pytest.approx({"joy": 0.6})


def test_a_damaged_entry_is_skipped_with_a_warning_and_left_byte_identical_and_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    damaged = {
        "emotions-list": _raw_line("emotions-list", ["joy", 0.1]),
        "text-value": _raw_line("text-value", {"joy": "0.1"}),
        "bool-value": _raw_line("bool-value", {"joy": True}),
        "null-value": _raw_line("null-value", {"joy": None}),
        "nan-value": _raw_line("nan-value", {"joy": float("nan")}),
        "huge-negative": _raw_line("huge-negative", {"joy": -1e308}),  # ten times it is not a finite number
        "huge-int": _raw_line("huge-int", {"joy": -int("1" + "0" * 400)}),
        "text-score": _raw_line("text-score", {"joy": 0.1}, score="lots"),
        "metadata-list": _raw_line("metadata-list", {"joy": 0.1}, metadata=["a"]),
        "emotions-null": _raw_line("emotions-null", None),
    }
    _write_lines(tmp_path / QUEUE, *damaged.values(), _raw_line("good", {"joy": 0.05}))
    caplog.set_level(logging.WARNING)
    assert PendingQueue(tmp_path).scale_up_small_making_candidates() == 1  # only the good one is counted
    lines = _lines(tmp_path / QUEUE)
    assert lines[:-1] == list(damaged.values())  # every damaged line byte-identical, in place
    assert json.loads(lines[-1])["emotions"] == pytest.approx({"joy": 0.5})
    warned = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    for memory_id in damaged:
        assert memory_id in warned, memory_id
    assert "content of" not in warned  # an id is named, never content


# --- the lock, the write, a failure --------------------------------------------------------------------


def test_the_rewrite_runs_under_the_queues_own_lock_and_replaces_each_file_through_the_shared_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("b", {"joy": 0.05}))
    held: list[Path] = []
    events: list[tuple[str, Path | None]] = []
    real_lock, real_write = pending.file_lock, pending.write_jsonl_bytes

    @contextlib.contextmanager
    def lock_spy(path: Path, *args: object, **kwargs: object):
        events.append(("lock", Path(path)))
        held.append(Path(path))
        with real_lock(path, *args, **kwargs) as acquired:
            yield acquired
        held.pop()
        events.append(("unlock", Path(path)))

    def write_spy(path: Path, records: object, **kwargs: object):
        assert held == [queue.path]  # written with the queue's lock held
        events.append(("write", Path(path)))
        return real_write(path, records, **kwargs)

    monkeypatch.setattr(pending, "file_lock", lock_spy)
    monkeypatch.setattr(pending, "write_jsonl_bytes", write_spy)
    assert queue.scale_up_small_making_candidates() == 2
    assert events == [("lock", queue.path), ("write", queue.path), ("write", queue.processing_path), ("unlock", queue.path)]


def test_an_entry_enqueued_by_another_thread_while_the_pass_runs_is_not_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`enqueue` takes the same lock, so an enqueue that arrives mid-pass lands after the replaced file, not into the one being rewritten."""
    import threading

    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    real_write = pending.write_jsonl_bytes
    thread = threading.Thread(target=lambda: queue.enqueue(_memory("late", {"joy": 0.05}), source="maker"))

    def write_while_another_thread_enqueues(path: Path, records: object, **kwargs: object):
        thread.start()  # it needs the lock this pass holds; that it waits for it is the lock test's proof
        return real_write(path, records, **kwargs)

    monkeypatch.setattr(pending, "write_jsonl_bytes", write_while_another_thread_enqueues)
    assert queue.scale_up_small_making_candidates() == 1
    monkeypatch.undo()
    thread.join(10)
    assert not thread.is_alive()
    entries = _entries(tmp_path)
    assert set(entries) == {"a", "late"}
    assert entries["a"]["emotions"] == pytest.approx({"joy": 0.5}) and entries["late"]["emotions"] == {"joy": 0.05}  # caught at the next start


def test_a_failed_write_raises_and_leaves_the_file_as_it_was_and_the_retry_completes_without_a_double_multiply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("claimed", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("queued", {"joy": 0.05}))
    before = QueuedBytes.of(tmp_path)
    caplog.set_level(logging.INFO, logger="brain.memory.pending")
    real_write = pending.write_jsonl_bytes
    calls: list[Path] = []

    def fail_on_the_second_file(path: Path, records: object, **kwargs: object):
        calls.append(Path(path))
        if len(calls) == 2:
            raise OSError("disk full")
        return real_write(path, records, **kwargs)

    monkeypatch.setattr(pending, "write_jsonl_bytes", fail_on_the_second_file)
    with pytest.raises(OSError, match="disk full"):
        queue.scale_up_small_making_candidates()
    after = QueuedBytes.of(tmp_path)
    assert after[1] == before[1]  # the file whose write failed is as it was (the replace is atomic)
    # the entries multiplied in the first file are logged although the second file then failed
    logged = [r.getMessage() for r in caplog.records if "multiplied the feelings" in r.getMessage()]
    assert len(logged) == 1 and "queued" in logged[0] and "claimed" not in logged[0]
    assert _entries(tmp_path)["queued"]["emotions"] == pytest.approx({"joy": 0.5})  # the first file was replaced and is marked
    monkeypatch.undo()
    assert queue.scale_up_small_making_candidates() == 1  # only the claimed entry is left to do
    assert _entries(tmp_path, PROCESSING)["claimed"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _entries(tmp_path)["queued"]["emotions"] == pytest.approx({"joy": 0.5})  # not 5.0


def test_a_read_fault_raises_and_changes_nothing(tmp_path: Path) -> None:
    (tmp_path / QUEUE).mkdir()  # a folder where the queue file should be: the read cannot succeed
    with pytest.raises(OSError):
        PendingQueue(tmp_path).scale_up_small_making_candidates()
    assert (tmp_path / QUEUE).is_dir()


def test_the_first_start_runs_the_queue_half_even_with_no_database_and_only_when_the_switch_is_known_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    assert engine_files.write_switch(tmp_path, NOW)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) is None  # switched: the fix does not run again
    assert _entries(tmp_path)["a"]["emotions"] == {"joy": 0.05}
    from tests.unit.brain.emotion._engine_support import refuse_the_switch_stat

    other = tmp_path / "other"
    other.mkdir()
    _enqueue(other, _memory("b", {"joy": 0.05}))
    with monkeypatch.context() as patch:
        refuse_the_switch_stat(patch)
        assert nudge_scale_fix.scale_up_small_feelings(other) is None  # an unknown counts as not switched: nothing irreversible runs
    assert _entries(other)["b"]["emotions"] == {"joy": 0.05}
    engine_files.switch_path(tmp_path).unlink()
    nudge_scale_fix.scale_up_small_feelings(tmp_path)  # not switched, no database
    assert _entries(tmp_path)["a"]["emotions"] == pytest.approx({"joy": 0.5})
