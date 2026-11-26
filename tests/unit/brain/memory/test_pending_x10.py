"""Engine 6, Q10c and C2: the x10 fix of small feelings reaches the pending-candidate queue (`PendingQueue.scale_up_small_making_candidates`).

A `making` candidate that waits in `pending_candidates.jsonl`, or in the claimed batch `pending_candidates.processing.jsonl`, when a
persona's engine first starts is not a `memories.db` row yet, so the database half of the fix cannot reach it, and the consolidation
gate would promote it later at the old small size. Owner (Planning B21, Q10c): "They're memories so they should get treated as such".

C2 (Planning B29 items 1, 3, 4; B27 Q15): negatives are held to 0.0 and then multiplied; an entry with no feelings has its score
multiplied; no entry carries a marker; the fix's progress is `PRAGMA user_version` of `memories.db` (0 -> 1 with the rows, -> 2 under
the queue lock before any queue file is rewritten, so 2 means "queue step started"), read and written only through `MemoryStore`
methods.

X10S (owner S84, Planning B32): the queue step is the seven steps in order: no queue file sets 2 without a lock; the lock; the number
read inside it; both files read into memory (a read fault raises, nothing written); the number set; each file with a selected entry
rewritten once with the shared atomic writer (a failed write is logged and start-up goes on); release. There are no temporary
`.x10tmp` copies, no check of a copy and no roll-forward; `_copies` below asserts none is ever left.

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
from brain.emotion import nudge_scale_fix
from brain.health.json_io import read_jsonl
from brain.memory import pending
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
QUEUE = "pending_candidates.jsonl"
PROCESSING = "pending_candidates.processing.jsonl"
COPY_SUFFIX = ".x10tmp"


def _memory(memory_id: str, emotions: dict[str, float], *, memory_type: str = "making", metadata: dict | None = None) -> Memory:
    numbers = [float(v) for v in emotions.values()]
    return Memory.from_stored_row(
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


def _database(persona: Path) -> None:
    """A persona's `memories.db`, which every real queue has beside it (the progress note lives in it)."""
    MemoryStore(persona / "memories.db").close()


def _enqueue(persona: Path, *memories: Memory) -> None:
    _database(persona)
    queue = PendingQueue(persona)
    for memory in memories:
        queue.enqueue(memory, source="maker")


def _scale(persona: Path) -> int:
    """The queue half alone, as the first start runs it: the persona's store open, its progress note read and set through it."""
    store = MemoryStore(persona / "memories.db")
    try:
        return PendingQueue(persona).scale_up_small_making_candidates(store)
    finally:
        store.close()


def _progress(persona: Path) -> int:
    store = MemoryStore(persona / "memories.db")
    try:
        return store.x10_progress()
    finally:
        store.close()


def _set_progress(persona: Path, number: int) -> None:
    store = MemoryStore(persona / "memories.db")
    try:
        store.set_x10_progress(number)
    finally:
        store.close()


def _copies(persona: Path) -> list[str]:
    return sorted(p.name for p in persona.iterdir() if p.name.endswith(COPY_SUFFIX))


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


class QueuedBytes:
    """The bytes of both queue files, to compare "not rewritten at all"."""

    @staticmethod
    def of(persona: Path) -> tuple[bytes | None, bytes | None]:
        return tuple((persona / name).read_bytes() if (persona / name).exists() else None for name in (QUEUE, PROCESSING))  # type: ignore[return-value]


# --- the selector and the multiply ---------------------------------------------------------------


def test_a_small_making_candidate_is_multiplied_whole_with_no_marker_and_nothing_else_is_touched(
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
    # 0.4 x 10 = 4.0 and the negative held to 0.0 then x10 = 0.0; the stored score (0.4 - 0.3 = 0.1) x 10 = 1.0
    assert after["mixed"]["emotions"] == pytest.approx({"warmth": 4.0, "fear": 0.0})
    assert after["mixed"]["score"] == pytest.approx(1.0)
    assert after["empty"]["emotions"] == {} and after["empty"]["score"] == 0.0
    for kept in ("big", "one", "talk"):  # not selected: identical entries
        assert after[kept] == before[kept]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]  # an entry on the 0-10 scale is not a damaged one
    for multiplied in ("plain", "mixed", "empty"):  # no marker: every other field, metadata included, is as it was
        assert {k: v for k, v in after[multiplied].items() if k not in ("emotions", "score")} == {
            k: v for k, v in before[multiplied].items() if k not in ("emotions", "score")
        }
    assert after["plain"]["metadata"] == {"keep": "me"}


@pytest.mark.parametrize("stored", [-1e308, -float("inf"), -(10**400)], ids=["huge negative float", "negative infinity", "huge negative int"])
def test_a_huge_negative_feeling_in_an_entry_is_held_to_zero_and_the_entry_is_multiplied(tmp_path: Path, stored: float) -> None:
    _database(tmp_path)
    _write_lines(tmp_path / QUEUE, _raw_line("neg", {"joy": stored, "calm": 0.05}))
    assert _scale(tmp_path) == 1
    assert _entries(tmp_path)["neg"]["emotions"] == pytest.approx({"joy": 0.0, "calm": 0.5})


def test_a_decayed_entry_with_no_feelings_still_has_its_score_multiplied(tmp_path: Path) -> None:
    """Planning B27 Q15: no skip for no feelings, through the shared multiply."""
    _database(tmp_path)
    _write_lines(tmp_path / QUEUE, _raw_line("decayed", {}, score=0.2), _raw_line("absent", None, score=0.3))
    lines = _lines(tmp_path / QUEUE)
    entry = json.loads(lines[1])
    del entry["emotions"]
    _write_lines(tmp_path / QUEUE, lines[0], json.dumps(entry).encode("utf-8"))
    assert _scale(tmp_path) == 2
    entries = _entries(tmp_path)
    assert entries["decayed"]["emotions"] == {} and entries["decayed"]["score"] == pytest.approx(2.0)
    assert entries["absent"]["emotions"] == {} and entries["absent"]["score"] == pytest.approx(3.0)


def test_a_second_run_multiplies_nothing_already_multiplied_and_leaves_the_files_alone(tmp_path: Path) -> None:
    _enqueue(tmp_path, _memory("small", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _entries(tmp_path)["small"]["emotions"] == pytest.approx({"joy": 0.5})  # 0.05 x 10, once
    first = QueuedBytes.of(tmp_path)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _entries(tmp_path)["small"]["emotions"] == pytest.approx({"joy": 0.5})  # not 5.0
    assert QueuedBytes.of(tmp_path) == first and _copies(tmp_path) == []
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED


def test_an_entry_queued_after_the_fix_is_not_multiplied_the_accepted_edge_of_the_progress_note(tmp_path: Path) -> None:
    """Owner S77 (P4): an entry written by something else while the app is down between steps is missed."""
    _enqueue(tmp_path, _memory("first", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    _enqueue(tmp_path, _memory("second", {"joy": 0.05}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    entries = _entries(tmp_path)
    assert entries["first"]["emotions"] == pytest.approx({"joy": 0.5})  # 0.05 -> 0.5, never 5.0
    assert entries["second"]["emotions"] == {"joy": 0.05}


def test_a_promoted_row_built_from_a_multiplied_entry_carries_no_marker_and_a_retried_switch_over_leaves_it_alone(tmp_path: Path) -> None:
    _enqueue(tmp_path, _memory("small", {"joy": 0.05}, metadata={"keep": "me"}))
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    (entry,) = read_jsonl(tmp_path / QUEUE)
    promoted = Memory.from_dict(entry)  # what the gate does at consolidation.py (`Memory.from_dict(entry)`) before `store.create`
    assert promoted._stored_emotions == pytest.approx({"joy": 0.5}) and promoted._stored_score == pytest.approx(0.5)
    assert promoted.metadata == {"keep": "me"}
    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(promoted, fold=False)  # the gate's `store.create(cand)`
    finally:
        store.close()
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 0  # a retried switch-over: 0.5 is still below 1, but the number says done
    store = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    try:
        row = store.get("small", bump=False)
        assert row is not None and row._stored_emotions == pytest.approx({"joy": 0.5})  # not 5.0
    finally:
        store.close()


def test_a_file_with_nothing_to_multiply_is_not_rewritten_at_all_whatever_it_holds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOM, CRLF endings, a blank line, garbage, a torn last line: kept exactly as they are when no entry is selected."""
    _database(tmp_path)
    content = (
        b"\xef\xbb\xbf"
        + _raw_line("big", {"joy": 6.0})
        + b"\r\n"
        + b"\n"
        + b"   \n"
        + b"garbage \xff\xfe\n"
        + _raw_line("talk", {"joy": 0.2}, memory_type="conversation")
        + b"\n"
        + b'{"id": "torn", "emotions": {"joy": 0.1}'
    )
    (tmp_path / QUEUE).write_bytes(content)
    writes: list[Path] = []
    real_write = pending.write_jsonl_bytes
    monkeypatch.setattr(pending, "write_jsonl_bytes", lambda path, data, **kw: writes.append(path) or real_write(path, data, **kw))
    assert _scale(tmp_path) == 0
    assert (tmp_path / QUEUE).read_bytes() == content and writes == [] and _copies(tmp_path) == []
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # the step ran: nothing selected, nothing rewritten


def test_an_entry_with_no_score_is_multiplied_in_its_feelings_and_gets_no_score(tmp_path: Path) -> None:
    _database(tmp_path)
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
    assert _copies(tmp_path) == []


def test_the_method_returns_how_many_entries_it_multiplied_across_both_files_and_a_second_call_returns_zero(tmp_path: Path) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}), _memory("b", {"joy": 0.06}))
    queue.claim()
    _enqueue(tmp_path, _memory("c", {"joy": 0.07}), _memory("big", {"joy": 5.0}))
    assert _scale(tmp_path) == 3
    assert _scale(tmp_path) == 0


def test_a_persona_with_no_queue_files_creates_nothing_and_a_missing_processing_file_is_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _database(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    real_lock = pending.file_lock
    locks: list[Path] = []
    monkeypatch.setattr(pending, "file_lock", lambda path, *a, **kw: locks.append(Path(path)) or real_lock(path, *a, **kw))
    assert _scale(tmp_path) == 0
    assert locks == []  # X10S step (1): no queue file sets the number and stops, taking no lock
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # not even a lock sidecar
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # the queue step is recorded: a later re-run multiplies nothing
    with_queue = tmp_path / "with-queue"
    with_queue.mkdir()
    _enqueue(with_queue, _memory("a", {"joy": 0.05}))
    assert _scale(with_queue) == 1
    assert not (with_queue / PROCESSING).exists() and _copies(with_queue) == []
    other = tmp_path / "other"
    other.mkdir()
    _database(other)
    _write_lines(other / PROCESSING, _raw_line("only-claimed", {"joy": 0.05}))
    assert _scale(other) == 1  # the processing file alone is reached too
    assert not (other / QUEUE).exists()  # a missing file is not created
    empty = tmp_path / "empty"
    empty.mkdir()
    _database(empty)
    (empty / QUEUE).write_bytes(b"")
    assert _scale(empty) == 0
    assert (empty / QUEUE).read_bytes() == b"" and not (empty / PROCESSING).exists()  # an empty file is not rewritten, a missing one not created


# --- a line that cannot be used ---------------------------------------------------------------------


def test_an_unparsable_line_is_kept_byte_for_byte_in_its_place_and_the_others_are_still_multiplied(tmp_path: Path) -> None:
    _database(tmp_path)
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
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 0  # the database holds no rows
    lines = _lines(tmp_path / QUEUE)
    assert len(lines) == 6
    assert lines[1] == bad_text and lines[2] == spaced_big and lines[3] == not_an_object and lines[4] == bad_utf8
    assert json.loads(lines[0])["emotions"] == pytest.approx({"joy": 0.5})  # in its place, first
    assert json.loads(lines[5])["emotions"] == pytest.approx({"joy": 0.6})


def test_a_damaged_entry_is_left_byte_identical_and_logged_and_does_not_stop_the_others(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Planning B29 C2 item 3 (P4, P9b): feelings or metadata that are not a mapping, or hold a value that is not a number, are left unchanged."""
    _database(tmp_path)
    damaged = {
        "emotions-list": _raw_line("emotions-list", ["joy", 0.1]),
        "text-value": _raw_line("text-value", {"joy": "0.1"}),
        "bool-value": _raw_line("bool-value", {"joy": True}),
        "null-value": _raw_line("null-value", {"joy": None}),
        "nan-value": _raw_line("nan-value", {"joy": float("nan")}),
        "text-score": _raw_line("text-score", {"joy": 0.1}, score="lots"),
        "metadata-list": _raw_line("metadata-list", {"joy": 0.1}, metadata=["a"]),
        "emotions-null": _raw_line("emotions-null", None),
        # a field the fix does not touch holds a NaN: writing the entry back would change it, so the entry is left as it is
        "nan-importance": _raw_line("nan-importance", {"joy": 0.1}, importance=float("nan")),
    }
    _write_lines(tmp_path / QUEUE, *damaged.values(), _raw_line("good", {"joy": 0.05}))
    caplog.set_level(logging.WARNING)
    assert _scale(tmp_path) == 1  # only the good one is counted
    lines = _lines(tmp_path / QUEUE)
    assert lines[:-1] == list(damaged.values())  # every damaged line byte-identical, in place
    assert json.loads(lines[-1])["emotions"] == pytest.approx({"joy": 0.5})
    warned = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    for memory_id in damaged:
        assert memory_id in warned, memory_id
    assert "content of" not in warned  # an id is named, never content


# --- the order, the lock, a failure ----------------------------------------------------------------------


def test_the_steps_run_in_the_order_lock_then_record_2_then_each_file_rewritten_once_then_unlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X10S: (2) lock, (5) set 2, (6) one `write_jsonl_bytes` per file with a selected entry, (7) release; the record comes before any rewrite."""
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("b", {"joy": 0.05}))
    held: list[Path] = []
    events: list[tuple[str, Path | int | None]] = []
    real_lock, real_write, real_set = pending.file_lock, pending.write_jsonl_bytes, MemoryStore.set_x10_progress

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

    def set_spy(self: MemoryStore, number: int) -> None:
        assert held == [queue.path]  # recorded with the queue's lock held
        events.append(("set", number))
        real_set(self, number)

    monkeypatch.setattr(pending, "file_lock", lock_spy)
    monkeypatch.setattr(pending, "write_jsonl_bytes", write_spy)
    monkeypatch.setattr(MemoryStore, "set_x10_progress", set_spy)
    assert _scale(tmp_path) == 2
    assert events == [
        ("lock", queue.path),
        ("set", dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED),
        ("write", queue.path),
        ("write", queue.processing_path),
        ("unlock", queue.path),
    ]


@pytest.mark.no_base_failure(
    reason="retained behaviour: the base also read the number inside the lock and skipped; this pins X10S step (3) (the second starter does nothing)"
)
def test_a_second_starter_that_waited_for_the_lock_sees_2_inside_it_and_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """X10S step (3): a bridge and a command starting together cannot both multiply: the number is read inside the lock."""
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    before = QueuedBytes.of(tmp_path)
    real_lock = pending.file_lock

    @contextlib.contextmanager
    def lock_after_the_other_starter_finished(path: Path, *args: object, **kwargs: object):
        with real_lock(path, *args, **kwargs) as acquired:
            _set_progress(tmp_path, dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED)  # the starter that held the lock first has recorded 2
            yield acquired

    monkeypatch.setattr(pending, "file_lock", lock_after_the_other_starter_finished)
    writes: list[Path] = []
    real_write = pending.write_jsonl_bytes
    monkeypatch.setattr(pending, "write_jsonl_bytes", lambda path, data, **kw: writes.append(path) or real_write(path, data, **kw))
    store = MemoryStore(tmp_path / "memories.db")
    try:
        # this starter read 0 before it waited: it must read again inside the lock
        assert store.x10_progress() == 0
        assert queue.scale_up_small_making_candidates(store) == 0
    finally:
        store.close()
    assert QueuedBytes.of(tmp_path) == before and writes == []


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
    assert _scale(tmp_path) == 1
    monkeypatch.undo()
    thread.join(10)
    assert not thread.is_alive()
    entries = _entries(tmp_path)
    assert set(entries) == {"a", "late"}
    assert entries["a"]["emotions"] == pytest.approx({"joy": 0.5}) and entries["late"]["emotions"] == {"joy": 0.05}  # missed: the accepted edge (S77)


def test_every_progress_step_is_in_its_place_0_then_1_with_the_rows_then_2_after_both_files_are_read_and_before_any_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.brain.emotion.test_nudge_scale_fix import _memory as _row

    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("q", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("p", {"joy": 0.06}))
    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(_row("row", "making", {"pride": 0.2}), fold=False)
    finally:
        store.close()
    assert _progress(tmp_path) == 0
    seen: list[tuple[int, bool]] = []
    real_set = MemoryStore.set_x10_progress

    def spy(self: MemoryStore, number: int) -> None:
        # at the moment the number is set to 2: no file has been rewritten yet (the record comes before the rewrite), and no
        # temporary copy exists (the fix writes none)
        unchanged = (
            _entries(tmp_path)["p"]["emotions"] == {"joy": 0.06}
            and _entries(tmp_path, PROCESSING)["q"]["emotions"] == {"joy": 0.05}
            and not list(tmp_path.glob("*.x10tmp"))
        )
        seen.append((number, unchanged))
        real_set(self, number)

    monkeypatch.setattr(MemoryStore, "set_x10_progress", spy)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 1
    assert seen == [(2, True)]
    assert _progress(tmp_path) == 2 and not list(tmp_path.glob("*.x10tmp"))
    assert _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})
    assert _entries(tmp_path, PROCESSING)["q"]["emotions"] == pytest.approx({"joy": 0.5})


def test_the_rows_set_the_number_to_1_in_their_own_transaction_before_the_queue_step_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.brain.emotion.test_nudge_scale_fix import _memory as _row

    _enqueue(tmp_path, _memory("q", {"joy": 0.05}))
    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(_row("row", "making", {"pride": 0.2}), fold=False)
    finally:
        store.close()
    seen: list[int] = []
    real = PendingQueue.scale_up_small_making_candidates

    def spy(self: PendingQueue, store: MemoryStore) -> int:
        seen.append(store.x10_progress())
        return real(self, store)

    monkeypatch.setattr(PendingQueue, "scale_up_small_making_candidates", spy)
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert seen == [dev_constants.NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE]


def test_a_failure_to_set_the_number_raises_with_nothing_written_and_the_next_start_runs_the_queue_step_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """X10S step (5): the number is set before any file is written, so a failure to set it leaves every file as it was."""
    from tests.unit.brain.emotion.test_nudge_scale_fix import _memory as _row

    _enqueue(tmp_path, _memory("p", {"joy": 0.06}))
    store = MemoryStore(tmp_path / "memories.db")
    try:
        store.create(_row("row", "making", {"pride": 0.2}), fold=False)
    finally:
        store.close()
    original = QueuedBytes.of(tmp_path)
    real_set = MemoryStore.set_x10_progress

    def refuse(self: MemoryStore, number: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(MemoryStore, "set_x10_progress", refuse)
    with pytest.raises(OSError, match="disk full"):
        nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE  # the rows were done, the queue step was not
    assert QueuedBytes.of(tmp_path) == original  # nothing was written
    assert not list(tmp_path.glob("*.x10tmp"))  # and no temporary copy was made
    monkeypatch.setattr(MemoryStore, "set_x10_progress", real_set)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 0  # the rows are done
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED
    assert _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})  # from the originals, once


@pytest.mark.no_base_failure(
    reason="retained behaviour, renamed because the copy check its old name described is gone: the base also read a BOM and CRLF file through the shared reader and multiplied every entry"
)
def test_a_bom_and_crlf_file_is_read_by_the_shared_reader_and_every_entry_is_multiplied(tmp_path: Path) -> None:
    _database(tmp_path)
    (tmp_path / QUEUE).write_bytes(
        b"\xef\xbb\xbf" + _raw_line("one", {"joy": 0.05}) + b"\r\n" + _raw_line("big", {"joy": 6.0}) + b"\r\n" + b"\r\n" + _raw_line("two", {"joy": 0.06}) + b"\r\n"
    )
    assert _scale(tmp_path) == 2
    lines = _lines(tmp_path / QUEUE)
    assert len(lines) == 3 and all(not line.endswith(b"\r") and not line.startswith(b"\xef\xbb\xbf") for line in lines)
    assert lines[1] == _raw_line("big", {"joy": 6.0})  # an unselected line, as the reader yielded it
    entries = _entries(tmp_path)
    assert entries["one"]["emotions"] == pytest.approx({"joy": 0.5}) and entries["two"]["emotions"] == pytest.approx({"joy": 0.6})


def test_a_failed_write_after_the_number_is_2_is_logged_the_other_file_is_rewritten_start_up_goes_on_and_the_entries_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Owner S84, X10S step (6): a failed write is logged, start-up continues, that file's entries stay at the old scale and are not retried."""
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("claimed", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("queued", {"joy": 0.06}))
    caplog.set_level(logging.WARNING, logger="brain.memory.pending")
    real_write = pending.write_jsonl_bytes

    def fail_on_the_queue_file(path: Path, records: object, **kwargs: object):
        if Path(path) == queue.path:
            raise OSError("disk full")
        return real_write(path, records, **kwargs)

    monkeypatch.setattr(pending, "write_jsonl_bytes", fail_on_the_queue_file)
    assert _scale(tmp_path) == 1  # start-up goes on: only the file that was rewritten is counted
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED
    assert _entries(tmp_path, PROCESSING)["claimed"]["emotions"] == pytest.approx({"joy": 0.5})  # the other file was rewritten
    assert _entries(tmp_path)["queued"]["emotions"] == {"joy": 0.06}  # the failed file stays at the old scale
    assert not list(tmp_path.glob("*.x10tmp"))
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(QUEUE in line and "old scale" in line and "not be retried" in line and "queued" in line for line in warned)
    monkeypatch.undo()
    assert _scale(tmp_path) == 0  # not retried, and never multiplied twice
    assert _entries(tmp_path)["queued"]["emotions"] == {"joy": 0.06}
    assert _entries(tmp_path, PROCESSING)["claimed"]["emotions"] == pytest.approx({"joy": 0.5})


def test_a_read_fault_raises_and_changes_nothing(tmp_path: Path) -> None:
    _database(tmp_path)
    (tmp_path / QUEUE).mkdir()  # a folder where the queue file should be: the read cannot succeed
    with pytest.raises(OSError):
        _scale(tmp_path)
    assert (tmp_path / QUEUE).is_dir()


def test_a_read_fault_in_the_second_file_raises_with_nothing_written_and_the_number_not_set_and_the_next_start_completes_once(
    tmp_path: Path,
) -> None:
    """X10S step (4): both files are read before the number is set or any file is written, so a fault in the second leaves the first alone."""
    _enqueue(tmp_path, _memory("queued", {"joy": 0.05}))
    (tmp_path / PROCESSING).mkdir()  # the processing file cannot be read
    before = (tmp_path / QUEUE).read_bytes()
    with pytest.raises(OSError):
        _scale(tmp_path)
    assert (tmp_path / QUEUE).read_bytes() == before
    assert _progress(tmp_path) != dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED
    assert not list(tmp_path.glob("*.x10tmp"))
    (tmp_path / PROCESSING).rmdir()
    assert _scale(tmp_path) == 1  # with the fault gone, the next start runs the step once
    assert _entries(tmp_path)["queued"]["emotions"] == pytest.approx({"joy": 0.5})


def test_the_first_start_runs_the_queue_half_only_when_the_persona_is_known_not_switched_and_has_a_database(tmp_path: Path) -> None:
    from tests.unit.brain.emotion._engine_support import lose_the_state, make_live, ruin_the_state

    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    make_live(tmp_path, NOW)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) is None  # switched: the fix does not run again
    assert _entries(tmp_path)["a"]["emotions"] == {"joy": 0.05}
    other = tmp_path / "other"
    other.mkdir()
    _enqueue(other, _memory("b", {"joy": 0.05}))
    make_live(other, NOW)
    ruin_the_state(other)  # cannot tell whether it was switched over: the database's progress note decides (SW4)
    assert nudge_scale_fix.scale_up_small_feelings(other) == 0
    assert _entries(other)["b"]["emotions"] == pytest.approx({"joy": 0.5})
    lose_the_state(tmp_path)  # no state at all: not switched
    nudge_scale_fix.scale_up_small_feelings(tmp_path)
    assert _entries(tmp_path)["a"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED
    assert _progress(other) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # the one that could not be told ran too
