"""Engine 6, Q10c and C2: the x10 fix of small feelings reaches the pending-candidate queue (`PendingQueue.scale_up_small_making_candidates`).

A `making` candidate that waits in `pending_candidates.jsonl`, or in the claimed batch `pending_candidates.processing.jsonl`, when a
persona's engine first starts is not a `memories.db` row yet, so the database half of the fix cannot reach it, and the consolidation
gate would promote it later at the old small size. Owner (Planning B21, Q10c): "They're memories so they should get treated as such".

C2 (Planning B29 items 1, 3, 4; B27 Q15): negatives are held to 0.0 and then multiplied; an entry with no feelings has its score
multiplied; no entry carries a marker; each file's new copy is written to `<file>.x10tmp` and checked before any replace; the fix's
progress is `PRAGMA user_version` of `memories.db` (0 -> 1 with the rows, -> 2 under the queue lock between the checked copies and
the replace), read and written only through `MemoryStore` methods.

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
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED


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
    assert promoted.emotions == pytest.approx({"joy": 0.5}) and promoted.score == pytest.approx(0.5)
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
        assert row is not None and row.emotions == pytest.approx({"joy": 0.5})  # not 5.0
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
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED  # done: nothing to copy


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


def test_a_persona_with_no_queue_files_creates_nothing_and_a_missing_processing_file_is_fine(tmp_path: Path) -> None:
    _database(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    assert _scale(tmp_path) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # not even a lock sidecar
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED  # the queue step is done: a later re-run multiplies nothing
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
    assert not (other / QUEUE).exists()


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


# --- the copy, the check, the lock, a failure ------------------------------------------------------------


def test_the_work_runs_under_the_queues_own_lock_the_copies_are_written_then_each_original_is_replaced_by_its_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("a", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("b", {"joy": 0.05}))
    held: list[Path] = []
    events: list[tuple[str, Path | None]] = []
    real_lock, real_write, real_replace = pending.file_lock, pending.write_jsonl_bytes, pending.file_ops.replace_file

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

    def replace_spy(src: Path, dst: Path):
        assert held == [queue.path]
        events.append(("replace", Path(dst)))
        assert Path(src).name == Path(dst).name + COPY_SUFFIX  # the fixed sibling name, so recovery finds it
        return real_replace(src, dst)

    monkeypatch.setattr(pending, "file_lock", lock_spy)
    monkeypatch.setattr(pending, "write_jsonl_bytes", write_spy)
    monkeypatch.setattr(pending.file_ops, "replace_file", replace_spy)
    assert _scale(tmp_path) == 2
    copy_of = lambda path: path.with_name(path.name + COPY_SUFFIX)  # noqa: E731
    assert events == [
        ("lock", queue.path),
        ("write", copy_of(queue.path)),
        ("write", copy_of(queue.processing_path)),
        ("replace", queue.path),
        ("replace", queue.processing_path),
        ("unlock", queue.path),
    ]


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


def test_every_progress_step_is_in_its_place_0_then_1_with_the_rows_then_2_after_both_checked_copies_and_before_the_replace(
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
    seen: list[tuple[int, list[str], bool]] = []
    real_set = MemoryStore.set_x10_progress

    def spy(self: MemoryStore, number: int) -> None:
        # at the moment the number is set to 2: both copies exist and have passed, and no original is replaced yet
        unchanged = _entries(tmp_path)["p"]["emotions"] == {"joy": 0.06} and _entries(tmp_path, PROCESSING)["q"]["emotions"] == {"joy": 0.05}
        seen.append((number, _copies(tmp_path), unchanged))
        real_set(self, number)

    monkeypatch.setattr(MemoryStore, "set_x10_progress", spy)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 1
    assert seen == [(2, [QUEUE + COPY_SUFFIX, PROCESSING + COPY_SUFFIX], True)]
    assert _progress(tmp_path) == 2 and _copies(tmp_path) == []
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


def test_a_crash_after_the_number_is_2_with_a_checked_copy_still_there_rolls_forward_without_a_second_multiply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("q", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("p", {"joy": 0.06}))
    original = QueuedBytes.of(tmp_path)
    real_replace = pending.file_ops.replace_file
    calls: list[Path] = []

    def die_on_the_first_replace(src: Path, dst: Path) -> None:
        calls.append(Path(dst))
        raise OSError("power cut")

    monkeypatch.setattr(pending.file_ops, "replace_file", die_on_the_first_replace)
    with pytest.raises(OSError, match="power cut"):
        _scale(tmp_path)
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED
    assert QueuedBytes.of(tmp_path) == original and _copies(tmp_path) == [QUEUE + COPY_SUFFIX, PROCESSING + COPY_SUFFIX]
    monkeypatch.setattr(pending.file_ops, "replace_file", real_replace)
    assert _scale(tmp_path) == 0  # the roll-forward multiplies nothing itself: it finishes the replacing from the checked copies
    assert _copies(tmp_path) == []
    assert _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})
    assert _entries(tmp_path, PROCESSING)["q"]["emotions"] == pytest.approx({"joy": 0.5})  # once, not 5.0
    assert _scale(tmp_path) == 0 and _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})


def test_a_crash_between_the_two_replaces_finishes_the_second_from_its_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    queue = PendingQueue(tmp_path)
    _enqueue(tmp_path, _memory("q", {"joy": 0.05}))
    queue.claim()
    _enqueue(tmp_path, _memory("p", {"joy": 0.06}))
    real_replace = pending.file_ops.replace_file
    calls: list[Path] = []

    def die_on_the_second(src: Path, dst: Path) -> None:
        calls.append(Path(dst))
        if len(calls) == 2:
            raise OSError("power cut")
        real_replace(src, dst)

    monkeypatch.setattr(pending.file_ops, "replace_file", die_on_the_second)
    with pytest.raises(OSError, match="power cut"):
        _scale(tmp_path)
    assert _copies(tmp_path) == [PROCESSING + COPY_SUFFIX]
    monkeypatch.setattr(pending.file_ops, "replace_file", real_replace)
    _scale(tmp_path)
    assert _copies(tmp_path) == []
    assert _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})  # the first file was replaced once and stays
    assert _entries(tmp_path, PROCESSING)["q"]["emotions"] == pytest.approx({"joy": 0.5})


def test_a_failure_before_the_number_is_2_with_a_copy_left_deletes_it_and_starts_the_queue_step_again_from_the_originals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert QueuedBytes.of(tmp_path) == original  # no original was replaced
    assert _copies(tmp_path) == [QUEUE + COPY_SUFFIX]
    (tmp_path / (QUEUE + COPY_SUFFIX)).write_bytes(b"garbage left by a crash\n")  # a copy that cannot be trusted
    monkeypatch.setattr(MemoryStore, "set_x10_progress", real_set)
    assert nudge_scale_fix.scale_up_small_feelings(tmp_path) == 0  # the rows are done
    assert _copies(tmp_path) == [] and _progress(tmp_path) == 2
    assert _entries(tmp_path)["p"]["emotions"] == pytest.approx({"joy": 0.6})  # from the originals, once, not from the leftover


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda data, original: data.rsplit(b"\n", 2)[0] + b"\n", id="a line dropped"),
        pytest.param(lambda data, original: data + b'{"id": "extra"}\n', id="a line added"),
        pytest.param(lambda data, original: data.replace(b"content of big", b"content of BIG"), id="an unselected line altered"),
        pytest.param(lambda data, original: original, id="nothing multiplied"),
        pytest.param(lambda data, original: data.replace(b"0.5", b"0.6"), id="a feeling that is not ten times"),
        pytest.param(lambda data, original: data.replace(b'"score": 5.0', b'"score": 4.0'), id="a score that is not ten times"),
        pytest.param(lambda data, original: data.replace(b'"importance": 5.0', b'"importance": 6.0'), id="another field changed"),
    ],
)
def test_a_copy_that_fails_its_check_raises_before_any_original_is_replaced_and_the_number_is_not_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt
) -> None:
    _database(tmp_path)
    _write_lines(
        tmp_path / QUEUE,
        _raw_line("one", {"joy": 0.05}, score=0.5),
        b"bad line {",
        _raw_line("big", {"joy": 6.0}),
        _raw_line("last", {"calm": 0.05}),
    )
    original = (tmp_path / QUEUE).read_bytes()
    real_write = pending.write_jsonl_bytes
    monkeypatch.setattr(pending, "write_jsonl_bytes", lambda path, data, **kw: real_write(path, corrupt(data, original), **kw))
    with pytest.raises(RuntimeError, match="nothing was replaced"):
        _scale(tmp_path)
    assert (tmp_path / QUEUE).read_bytes() == original  # the original is kept
    assert _progress(tmp_path) != dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED
    monkeypatch.setattr(pending, "write_jsonl_bytes", real_write)
    assert _scale(tmp_path) == 2  # with the fault gone the next start completes, once
    assert _entries(tmp_path)["one"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _copies(tmp_path) == []


def test_a_bom_and_crlf_file_is_counted_the_same_way_on_both_sides_so_the_check_passes_and_every_entry_is_multiplied(tmp_path: Path) -> None:
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


def test_a_first_line_that_starts_with_a_second_byte_order_mark_does_not_fail_the_check(tmp_path: Path) -> None:
    """The reader drops one leading mark of a stream; a line that itself begins with another reads back one mark shorter (a mark is not content)."""
    _database(tmp_path)
    (tmp_path / QUEUE).write_bytes(b"\xef\xbb\xbf\xef\xbb\xbfnot json {\n" + _raw_line("one", {"joy": 0.05}) + b"\n")
    assert _scale(tmp_path) == 1
    assert _entries(tmp_path)["one"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _lines(tmp_path / QUEUE)[0] == b"not json {"[:0] + "﻿not json {".encode()  # one mark is left, the junk is kept


def test_a_failed_write_of_the_second_copy_raises_and_changes_no_original_and_the_next_start_completes_once(
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
        _scale(tmp_path)
    assert QueuedBytes.of(tmp_path) == before  # neither original was replaced
    assert _progress(tmp_path) != dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED
    monkeypatch.undo()
    assert _scale(tmp_path) == 2  # both entries, from the originals
    assert _entries(tmp_path, PROCESSING)["claimed"]["emotions"] == pytest.approx({"joy": 0.5})
    assert _entries(tmp_path)["queued"]["emotions"] == pytest.approx({"joy": 0.5})  # not 5.0
    assert _copies(tmp_path) == []
    logged = [r.getMessage() for r in caplog.records if "multiplied by" in r.getMessage()]
    assert any("queued" in line for line in logged)  # the ids are in the log, so the entries can be found by hand


def test_a_read_fault_raises_and_changes_nothing(tmp_path: Path) -> None:
    _database(tmp_path)
    (tmp_path / QUEUE).mkdir()  # a folder where the queue file should be: the read cannot succeed
    with pytest.raises(OSError):
        _scale(tmp_path)
    assert (tmp_path / QUEUE).is_dir()


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
    assert _progress(tmp_path) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED
    assert _progress(other) == dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_COPIES_CHECKED  # the one that could not be told ran too
