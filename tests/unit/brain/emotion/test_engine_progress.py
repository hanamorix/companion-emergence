"""Engine 6, HR5 parts (2), (3) and (6): the progress record a long start-up step writes, and the reader that says who is alive.

Hana's review of #633, point 5 (owner S101, S102): the start-up shows its progress while a rebuild runs. Two steps write
`engine_progress.json` in the persona directory: the x10 fix ("rescaling", no count) and the first rebuild ("rebuilding", memories
done of the total). `engine_files.read_engine_progress` returns it only while its writer is alive. These tests cover the writers,
the never-over-another-live-record rule, the removal rules and the reader; the display and the timers that read it are later parts.

A second process is real where liveness matters: a child that sleeps is alive, and a child that has exited is gone. Nothing here
touches a real persona.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from brain import dev_constants, process_identity
from brain.emotion import engine_files, engine_rebuild, nudge_scale_fix
from brain.emotion.engine import EngineStartupError, ensure_engine_ready
from brain.health.anomaly import BrainIntegrityError
from brain.memory.store import MemoryStore
from tests.unit.brain.emotion._engine_support import (
    NOW,
    change,
    install,
    local_day,
    local_noon,
    make_live,
    make_persona,
    row,
    state_of,
)

T = [local_noon(local_day(n)) for n in range(40)]


def _joy_rows(count: int, **kwargs: object):
    return [row(T[20 + n % 15], emotions={"joy": 3.0 + n % 5}, mid=f"r{n}", **kwargs) for n in range(count)]


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every record `engine_files` writes to the progress file, in order (the write itself still happens)."""
    seen: list[dict] = []
    real = engine_files.write_json

    def spy(path: Path, data: dict, **kwargs: object) -> bool:
        if Path(path).name == dev_constants.ENGINE_PROGRESS_FILENAME:
            seen.append(dict(data))
        return real(path, data, **kwargs)

    monkeypatch.setattr(engine_files, "write_json", spy)
    return seen


def _counts(writes: list[dict], phase: str = "rebuilding") -> list[tuple[int | None, int | None]]:
    """(done, total) of the writes of one phase; a first start runs the x10 step ("rescaling") before the rebuild."""
    return [(w["done"], w["total"]) for w in writes if w["phase"] == phase]


@pytest.fixture
def live_other() -> Iterator[subprocess.Popen]:
    """A second process that is alive for the test."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield child
    finally:
        child.kill()
        child.wait()


def _gone_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _foreign_record(persona: Path, pid: int, token: str | None, phase: str = "rescaling") -> bytes:
    record = engine_files.progress_record(
        pid=pid, pid_token=token, persona=persona.name, phase=phase, done=None, total=None, updated_at="2026-11-16T10:00:00.000000+00:00"
    )
    engine_files.progress_path(persona).write_text(json.dumps(record), encoding="utf-8")
    return engine_files.progress_path(persona).read_bytes()


# ---- the rebuild's writer (C3, C4) ------------------------------------------------------------------------------------------


def test_the_first_rebuild_writes_at_entry_with_the_total_and_on_each_percentage_change_and_removes_it(
    tmp_path: Path, writes: list[dict]
) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 6.0}, mid="a"), row(T[36], mid="b"), row(T[37], mid="c")])
    install(persona, NOW)
    ensure_engine_ready(persona)
    # The x10 step first, then the rebuild: its wait record (no count, written before the lock is tried), then the in-lock entry
    # write (no count), the total, and each percentage (Planning's HRV answer to Hana's 11-18 point 1).
    assert _counts(writes) == [(None, None), (None, None), (0, 3), (1, 3), (2, 3), (3, 3)]
    assert [w["phase"] for w in writes] == ["rescaling"] + ["rebuilding"] * 6
    assert all(w["pid"] == os.getpid() and w["persona"] == persona.name and w["version"] == 1 for w in writes)
    assert not engine_files.progress_path(persona).exists()
    assert state_of(persona).switch_over_at == NOW


def test_a_large_window_writes_at_most_one_record_per_whole_percent_and_one_at_entry(tmp_path: Path, writes: list[dict]) -> None:
    persona = make_persona(tmp_path, _joy_rows(150))
    install(persona, NOW)
    ensure_engine_ready(persona)
    rebuilding = [w for w in writes if w["phase"] == "rebuilding"]
    assert 60 <= len(rebuilding) <= 103  # 150 rows cross 101 percentages at most, plus the wait's record and the entry write
    assert _counts(writes)[:2] == [(None, None), (None, None)] and _counts(writes)[-1] == (150, 150)
    percents = [w["done"] * 100 // w["total"] for w in rebuilding[2:]]
    assert percents == sorted(set(percents))  # strictly rising: a write only when the percentage changed


def test_done_reaches_total_with_a_skipped_row_a_failed_fold_stamp_points_and_a_change_point(
    tmp_path: Path, writes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [row(T[30], emotions={"joy": 6.0}, mid="a"), row(T[31], emotions={}, mid="empty"), row(T[32], emotions={"joy": 7.7}, mid="bad"),
            row(T[33], emotions={"joy": 5.0}, mid="d")]
    persona = make_persona(tmp_path, rows)
    install(persona, NOW)
    assert engine_files.append_change(persona, change(T[34]))
    real = engine_rebuild.fold_entry

    def failing(state, name, value, *rest):  # noqa: ANN001, ANN002, ANN202 - a stand-in for the engine's own fold
        if value == 7.7:
            raise ValueError("a bad value")
        return real(state, name, value, *rest)

    monkeypatch.setattr(engine_rebuild, "fold_entry", failing)
    ensure_engine_ready(persona)
    # 4 rows: one skipped (no feelings) is added when the points are built; three row points then fold, the failing one counted too.
    assert _counts(writes) == [(None, None), (None, None), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]  # wait record, entry write, counts


def test_an_empty_window_writes_no_counted_record_and_still_removes_the_entry_record(tmp_path: Path, writes: list[dict]) -> None:
    persona = make_persona(tmp_path, [])
    install(persona, NOW)
    ensure_engine_ready(persona)
    assert _counts(writes) == [(None, None), (None, None)]  # the wait's record and the in-lock entry record
    assert not engine_files.progress_path(persona).exists()


@pytest.mark.no_base_failure(reason="pins that the catch-up stays callback-free; the base code never had a callback, so this passes there by design")
def test_a_catch_up_passes_no_progress_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writes: list[dict]) -> None:
    from tests.unit.brain.emotion._engine_support import simple_state

    persona = make_persona(tmp_path, [row(T[39], emotions={"joy": 6.0}, mid="p")])
    instance = install(persona, NOW)
    seen: dict[str, object] = {}
    real = engine_rebuild.replay

    def spy(*args: object, **kwargs: object):  # noqa: ANN202
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_rebuild, "replay", spy)
    engine_rebuild.catch_up(
        persona, simple_state(saved_at=T[38]), NOW, queued=[], emotions_of=instance._replay_emotions, modifier_of=instance._replay_modifier_with_memo()
    )
    assert seen.get("on_progress") is None
    assert writes == []


def test_the_rebuild_record_is_removed_when_the_rebuild_fails(tmp_path: Path, writes: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)

    def refuse(state: object) -> None:
        assert engine_files.progress_path(persona).exists()  # still there while the rebuild runs
        raise engine_rebuild.RebuildCheckError("a refused rebuild")

    monkeypatch.setattr(engine_rebuild, "_check_rebuilt", refuse)
    with pytest.raises(EngineStartupError):
        ensure_engine_ready(persona)
    assert writes and not engine_files.progress_path(persona).exists()


def test_the_rebuild_record_is_removed_inside_the_lock_hold_not_after_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    real_lock = engine_files.state_lock
    at_release: list[bool] = []

    @contextmanager
    def watching(persona_dir: Path) -> Iterator[None]:
        with real_lock(persona_dir):
            yield
            at_release.append(engine_files.progress_path(persona_dir).exists())  # the end of the hold, before the lock is let go

    monkeypatch.setattr(engine_files, "state_lock", watching)
    ensure_engine_ready(persona)
    assert at_release == [False]


def test_a_switched_persona_and_a_start_that_finds_nothing_to_rebuild_write_no_record(tmp_path: Path, writes: list[dict]) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    make_live(persona, T[30])
    ensure_engine_ready(persona)
    assert writes == []


# ---- the x10 step's writer (C5) ---------------------------------------------------------------------------------------------


class _StoreSpy:
    """Stands in for `nudge_scale_fix.MemoryStore`: records each open and the progress record at that moment, and opens the real one."""

    def __init__(self, persona: Path, monkeypatch: pytest.MonkeyPatch, *, raises: Exception | None = None) -> None:
        self.opens: list[tuple[tuple, dict, dict | None]] = []
        self._raises = raises

        def factory(*args: object, **kwargs: object) -> MemoryStore:
            self.opens.append((args, kwargs, engine_files.read_progress(persona)))
            if self._raises is not None:
                raise self._raises
            return MemoryStore(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(nudge_scale_fix, "MemoryStore", factory)


def _small_making_persona(tmp_path: Path) -> Path:
    return make_persona(tmp_path, [row(T[35], memory_type="making", emotions={"joy": 0.3}, mid="m")])


def test_the_x10_step_writes_a_rescaling_record_with_no_count_before_its_one_checked_open_and_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writes: list[dict]
) -> None:
    persona = _small_making_persona(tmp_path)
    spy = _StoreSpy(persona, monkeypatch)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 1
    assert len(spy.opens) == 1  # no store opened for the record: the one open is the checked one
    args, kwargs, record_at_open = spy.opens[0]
    assert kwargs == {} and len(args) == 1  # the default integrity check
    assert record_at_open is not None and record_at_open["phase"] == "rescaling"
    assert record_at_open["done"] is None and record_at_open["total"] is None and record_at_open["pid"] == os.getpid()
    assert _counts(writes, "rescaling") == [(None, None)]
    assert not engine_files.progress_path(persona).exists()


def test_the_x10_step_with_nothing_to_do_writes_and_removes_its_record_and_opens_no_store_before_the_checked_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writes: list[dict]
) -> None:
    persona = _small_making_persona(tmp_path)
    store = MemoryStore(persona / "memories.db")
    try:
        store.set_x10_progress(2)
    finally:
        store.close()
    spy = _StoreSpy(persona, monkeypatch)
    assert nudge_scale_fix.scale_up_small_feelings(persona) == 0
    assert len(spy.opens) == 1 and spy.opens[0][2] is not None  # the record was there at the checked open
    assert len(writes) == 1
    assert not engine_files.progress_path(persona).exists()


def test_the_x10_record_is_removed_when_the_checked_open_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _small_making_persona(tmp_path)
    spy = _StoreSpy(persona, monkeypatch, raises=BrainIntegrityError("memories.db", "page 3 is damaged"))
    with pytest.raises(BrainIntegrityError):
        nudge_scale_fix.scale_up_small_feelings(persona)
    assert spy.opens[0][2] is not None  # it was written before the open that failed
    assert not engine_files.progress_path(persona).exists()


def test_the_x10_step_writes_nothing_for_a_switched_persona_or_a_missing_database(tmp_path: Path, writes: list[dict]) -> None:
    switched = _small_making_persona(tmp_path / "a")
    make_live(switched, T[30])
    assert nudge_scale_fix.scale_up_small_feelings(switched) is None
    empty = tmp_path / "b"
    empty.mkdir()
    assert nudge_scale_fix.scale_up_small_feelings(empty) is None
    assert writes == [] and not (empty / "memories.db").exists()


# ---- never over another live process's record (C6, C7) ---------------------------------------------------------------------


def test_the_x10_step_and_the_rebuild_leave_a_live_foreign_record_alone(
    tmp_path: Path, live_other: subprocess.Popen, writes: list[dict]
) -> None:
    persona = make_persona(tmp_path, [row(T[35], memory_type="making", emotions={"joy": 0.3}, mid="m"), *_joy_rows(3)])
    install(persona, NOW)
    before = _foreign_record(persona, live_other.pid, process_identity.token_of(live_other.pid))
    ensure_engine_ready(persona)  # the x10 step, then the first rebuild
    assert writes == []
    assert engine_files.progress_path(persona).read_bytes() == before  # not replaced, not removed by the others
    assert state_of(persona).switch_over_at == NOW  # the start itself went through


@pytest.mark.parametrize("kind", ["gone", "bad pid", "unreadable"])
def test_a_record_of_a_gone_process_or_of_no_process_is_replaced(tmp_path: Path, writes: list[dict], kind: str) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    if kind == "gone":
        _foreign_record(persona, _gone_pid(), "linux:0b9c2f4e-1111-2222-3333-444455556666:1:1")
    elif kind == "bad pid":
        engine_files.progress_path(persona).write_text(json.dumps({"pid": "abc", "pid_token": None, "phase": "rescaling"}), encoding="utf-8")
    else:
        engine_files.progress_path(persona).write_bytes(b"\x00 not json")
    ensure_engine_ready(persona)
    assert writes and writes[0]["pid"] == os.getpid()
    assert not engine_files.progress_path(persona).exists()


def test_a_record_naming_a_pid_the_platform_cannot_hold_is_replaced(tmp_path: Path, writes: list[dict]) -> None:
    """Such a record names no process: the step writes over it (on base the failed liveness check made the write skip)."""
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    engine_files.progress_path(persona).write_text(
        json.dumps({"pid": 99999999999999999999, "pid_token": None, "phase": "rescaling"}), encoding="utf-8"
    )
    ensure_engine_ready(persona)
    assert writes and writes[0]["pid"] == os.getpid()
    assert not engine_files.progress_path(persona).exists()


def test_a_rebuild_that_skipped_its_write_over_a_live_x10_record_writes_again_once_that_record_is_removed(
    tmp_path: Path, live_other: subprocess.Popen, writes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    _foreign_record(persona, live_other.pid, process_identity.token_of(live_other.pid))
    real = engine_rebuild._read_window

    def window_then_the_other_step_ends(*args: object, **kwargs: object):  # noqa: ANN202
        out = real(*args, **kwargs)
        engine_files.progress_path(persona).unlink()  # the other process's x10 step ended and removed its record
        return out

    monkeypatch.setattr(engine_rebuild, "_read_window", window_then_the_other_step_ends)
    ensure_engine_ready(persona)
    assert _counts(writes) == [(0, 3), (1, 3), (2, 3), (3, 3)]  # the entry write was skipped, the next went through
    assert all(w["pid"] == os.getpid() for w in writes)
    assert not engine_files.progress_path(persona).exists()


def test_a_blocked_writer_reads_the_file_once_per_percentage_not_once_per_row(
    tmp_path: Path, live_other: subprocess.Popen, monkeypatch: pytest.MonkeyPatch
) -> None:
    _foreign_record(tmp_path, live_other.pid, process_identity.token_of(live_other.pid))
    reads: list[int] = []
    real = engine_files.read_progress
    monkeypatch.setattr(engine_files, "read_progress", lambda p: reads.append(1) or real(p))
    progress = engine_files.EngineProgress(tmp_path, "rebuilding")
    for done in range(8):
        progress.report(done, 1000)  # all inside the same whole percent
    assert len(reads) == 1


# ---- removal and failures (C8, C12) -----------------------------------------------------------------------------------------


def test_a_removal_leaves_another_processes_record(tmp_path: Path, live_other: subprocess.Popen) -> None:
    before = _foreign_record(tmp_path, live_other.pid, process_identity.token_of(live_other.pid))
    engine_files.EngineProgress(tmp_path, "rescaling").remove()
    assert engine_files.progress_path(tmp_path).read_bytes() == before


def test_a_writer_with_no_token_removes_its_own_record_and_the_reader_sees_it_only_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(process_identity, "own_token", lambda: None)
    progress = engine_files.EngineProgress(tmp_path, "rebuilding")
    progress.write()
    progress.report(1, 2)
    assert engine_files.read_progress(tmp_path)["pid_token"] is None
    running = engine_files.read_engine_progress(tmp_path)  # this process is alive: a pid check answers alive
    assert running is not None and running["pid"] == os.getpid() and running["done"] == 1
    progress.remove()
    assert not engine_files.progress_path(tmp_path).exists()
    assert engine_files.read_engine_progress(tmp_path) is None


@pytest.mark.no_base_failure(reason="a data-safety pin: the old removal also kept a record that names a token, and the new one must keep doing so")
def test_a_writer_with_no_token_leaves_a_record_that_names_a_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A record with a token is not this token-less writer's: keep it (data safety)."""
    own = process_identity.own_token()
    if own is None:
        pytest.skip("this test needs a platform where the token can be read")
    progress = engine_files.EngineProgress(tmp_path, "rebuilding")
    progress.write()
    before = engine_files.progress_path(tmp_path).read_bytes()
    monkeypatch.setattr(process_identity, "own_token", lambda: None)
    progress.remove()
    assert engine_files.progress_path(tmp_path).read_bytes() == before


def test_a_failed_write_does_not_fail_the_start_and_is_logged_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    real = engine_files.write_json

    def broken(path: Path, data: dict, **kwargs: object) -> bool:
        if Path(path).name == dev_constants.ENGINE_PROGRESS_FILENAME:
            raise OSError("disk full")
        return real(path, data, **kwargs)

    warned: list[tuple[str, dict]] = []
    real_warn = engine_files.warn_throttled
    monkeypatch.setattr(engine_files, "write_json", broken)
    monkeypatch.setattr(engine_files, "warn_throttled", lambda lg, key, msg, *a, **kw: (warned.append((key, kw)), real_warn(lg, key, msg, *a, **kw))[1])
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW
    assert warned and all(kw == {"exc_info": False} for _, kw in warned)


def test_a_failed_removal_does_not_fail_the_start_and_the_record_stays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)

    def refuse(path: Path) -> None:
        raise PermissionError("held by another handle")

    monkeypatch.setattr(engine_files.file_ops, "delete_file", refuse)
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW
    leftover = engine_files.read_engine_progress(persona)  # names this live process: the later timers would pause on it
    assert leftover is not None and leftover["pid"] == os.getpid()


def test_a_liveness_check_that_raises_does_not_fail_the_step(
    tmp_path: Path, live_other: subprocess.Popen, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = make_persona(tmp_path, _joy_rows(3))
    install(persona, NOW)
    _foreign_record(persona, live_other.pid, process_identity.token_of(live_other.pid))

    def broken(pid: int | None, recorded: str | None) -> bool:
        raise RuntimeError("cannot ask")

    monkeypatch.setattr(process_identity, "holder_alive", broken)
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW


# ---- the reader (C9) --------------------------------------------------------------------------------------------------------


def _own_record(persona: Path, **overrides: object) -> dict:
    record = engine_files.progress_record(
        pid=os.getpid(), pid_token=process_identity.own_token(), persona="nell", phase="rebuilding", done=3, total=9,
        updated_at="2001-01-01T00:00:00.000000+00:00",
    )
    record.update(overrides)
    engine_files.progress_path(persona).write_text(json.dumps(record), encoding="utf-8")
    return record


def test_the_reader_returns_the_record_of_a_live_writer_however_old_it_is(tmp_path: Path) -> None:
    record = _own_record(tmp_path)  # updated_at is decades old: no age test
    assert engine_files.read_engine_progress(tmp_path) == record


def test_the_reader_counts_an_unknown_liveness_answer_as_alive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _own_record(tmp_path)
    monkeypatch.setattr(process_identity, "token_of", lambda pid: process_identity.UNKNOWN)
    assert engine_files.read_engine_progress(tmp_path) == record


def test_the_reader_falls_back_to_a_pid_check_for_a_record_with_no_token(tmp_path: Path) -> None:
    record = _own_record(tmp_path, pid_token=None)
    assert engine_files.read_engine_progress(tmp_path) == record


def test_the_reader_returns_none_for_a_gone_writer_and_for_a_record_that_names_nothing_usable(tmp_path: Path) -> None:
    assert engine_files.read_engine_progress(tmp_path) is None  # no record
    _own_record(tmp_path, pid=_gone_pid(), pid_token="linux:0b9c2f4e-1111-2222-3333-444455556666:1:1")
    assert engine_files.read_engine_progress(tmp_path) is None  # a writer that is gone
    for pid in ("abc", True, -3, None):
        _own_record(tmp_path, pid=pid)
        assert engine_files.read_engine_progress(tmp_path) is None
    engine_files.progress_path(tmp_path).write_text(json.dumps([1, 2]), encoding="utf-8")
    assert engine_files.read_engine_progress(tmp_path) is None  # not a mapping


@pytest.mark.no_base_failure(reason="pins behaviour the shared liveness test must keep: on base the failed pid check read as no record")
def test_the_reader_returns_none_for_a_record_naming_a_pid_the_platform_cannot_hold(tmp_path: Path) -> None:
    _own_record(tmp_path, pid=99999999999999999999, pid_token=None)
    assert engine_files.read_engine_progress(tmp_path) is None


def test_a_record_cannot_name_another_bridge_as_its_own(tmp_path: Path) -> None:
    """`is_own_identity`, the own-record test the writer's removal uses, needs the pid AND the token to match."""
    _own_record(tmp_path, pid_token="linux:other:1:1")
    record = engine_files.read_progress(tmp_path)
    assert not process_identity.is_own_identity(record["pid"], record["pid_token"])
