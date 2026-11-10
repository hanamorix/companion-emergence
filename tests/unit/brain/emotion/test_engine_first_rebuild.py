"""Engine 6, G1: the first rebuild at start-up, the recovery rebuild, the live fade, the change log and peer trust.

Covers `EmotionEngine.ensure_ready` / `ensure_engine_ready`, `engine_rebuild.rebuild`, `needs_recovery`, `recovery_rebuild`,
`faded_value` (live), `fold_change`'s change-log line, and the pass-through of `engine_files.peer_trust`. Every expected number is
hand-worked in the comment beside it. Times are local noon of a civil day (`local_noon`), so a day boundary never falls inside a
case. Personas are tmp directories with synthetic rows; nothing here touches a real persona.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine, engine_files, engine_rebuild
from brain.emotion.engine import EmotionEngine, EngineStartupError, ensure_engine_ready
from brain.emotion.engine_rebuild import CatchUpError
from brain.emotion.engine_utils import EmotionState, EngineState, day_of
from brain.health.anomaly import BrainIntegrityError
from brain.kindled_link import relationship
from brain.kindled_link.relationship import PeerRelationshipState, persist_relationship_state
from brain.kindled_link.store import KindledLinkStore, kindled_db_path
from brain.memory.store import Memory, MemoryStore
from brain.utils import file_ops
from tests.unit.brain.emotion._engine_support import (
    LockSpy,
    change,
    install,
    local_day,
    local_noon,
    lose_the_state,
    make_live,
    make_persona,
    queue,
    row,
    ruin_the_state,
    save,
    save_as_backup,
    showing,
    state_of,
    tree,
)
from tests.unit.brain.emotion._engine_support import simple_state as _support_simple_state

T = [local_noon(local_day(n)) for n in range(60)]  # T[n] = local noon of day n
NOW = T[40] + timedelta(hours=3)  # local 3 PM of day 40: the clock of every case unless it says otherwise
LOGGER = "brain.emotion.engine"


def simple_state(*, saved_at: datetime, joy: float = 4.0, switch_over_at: datetime | None = None) -> EngineState:
    """The shared small state, with its open day set (a state with `last_update` and no `active_day` is not one the reader accepts)."""
    return _support_simple_state(saved_at=saved_at, joy=joy, day=day_of(saved_at), switch_over_at=switch_over_at)


def _standard_rows() -> list[Memory]:
    return [
        row(T[30], emotions={"joy": 6.0}, mid="old"),
        row(T[40], emotions={"joy": 2.0}, mid="recent"),
        row(T[35], memory_type="making", emotions={"pride": 0.4}, mid="made"),  # a small making: the x10 fix runs first
    ]


def _standard_persona(root: Path) -> Path:
    persona = make_persona(root, _standard_rows())
    queue(persona, showing("old", T[40]))
    return persona


def _ensure(persona: Path, now: datetime = NOW) -> EmotionEngine:
    instance = install(persona, now)
    ensure_engine_ready(persona)
    return instance


def _first_state_bytes(persona: Path) -> bytes:
    """The state file a first start leaves: it holds the switch-over time, so two clean first runs are byte for byte equal."""
    return engine_files.state_path(persona).read_bytes()


def _stored(memory: Memory):
    return engine_rebuild._stored_emotions(memory)


def _unit_modifier(_: Memory) -> float:
    return 1.0


def _rebuild(persona: Path, mode: str, now: datetime = NOW, *, faded=None, modifier=None, switch_over_at: datetime | None = None) -> None:
    engine_rebuild.rebuild(
        persona,
        mode,
        now,
        faded_emotions_of=faded or _stored,
        modifier_of=modifier or _unit_modifier,
        switch_over_at=switch_over_at,
    )


def _x10():
    """The x10 module, imported where a test needs it, so a test of an older behaviour does not fail at import on code that lacks it."""
    from brain.emotion import nudge_scale_fix

    return nudge_scale_fix


class _Spy:
    """Counts the constructions of `MemoryStore` made by the rebuild and by the x10 fix ("no transaction" is "no open")."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.opened: list[str] = []
        for module, label in ((engine_rebuild, "rebuild"), (_x10(), "x10")):
            real = module.MemoryStore

            def counted(*args: object, _real=real, _label=label, **kwargs: object) -> MemoryStore:
                self.opened.append(_label)
                return _real(*args, **kwargs)

            monkeypatch.setattr(module, "MemoryStore", counted)


# ================================================================================================
# C-1: the first start
# ================================================================================================


def test_a_first_start_runs_x10_takes_the_lock_once_empties_the_queue_and_writes_one_verified_state_that_records_the_switch_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _standard_persona(tmp_path)
    spy = LockSpy(monkeypatch)
    order: list[object] = []
    for module, name in ((_x10(), "scale_up_small_feelings"), (engine_files, "empty_queue"), (engine_files, "save_state")):
        real = getattr(module, name)

        def wrapped(*args: object, _real=real, _name=name, **kwargs: object):
            order.append((_name, kwargs.get("verify")))
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapped)
    _ensure(persona)
    # the verified state write is the last step: it carries the switch-over time, and no other write follows it
    assert order == [("scale_up_small_feelings", None), ("empty_queue", None), ("save_state", True)]
    assert spy.within == [(engine_files.state_path(persona), dev_constants.EMOTION_LOCK_WAIT_S)]  # the lock, once, with the wait
    # x10 ran before the rebuild read the rows: the small making row was multiplied, and the rebuild folded its 4.0
    assert state_of(persona).emotions["pride"].short.avg == pytest.approx(4.0)
    assert engine_files.queue_path(persona).read_text(encoding="utf-8") == ""  # emptied, not deleted
    assert state_of(persona).queue_line == 0
    assert state_of(persona).switch_over_at == NOW  # the rebuild's own clock, in the same write as saved_at and reset_at
    assert state_of(persona).saved_at == state_of(persona).reset_at == NOW


def test_a_second_start_returns_with_no_store_open_and_no_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    instance = _ensure(persona)
    before = tree(persona)
    spy = LockSpy(monkeypatch)
    opened = _Spy(monkeypatch)
    ensure_engine_ready(persona)  # the same engine, which has seen the switch
    EmotionEngine(persona.resolve(), now=lambda: NOW).ensure_ready()  # a new process: it asks the file
    assert spy.calls == 0 and opened.opened == []
    assert tree(persona) == before
    assert instance.is_live()


def test_a_fresh_persona_with_no_database_gets_an_empty_state_that_records_the_switch_over_and_no_database_is_created(tmp_path: Path) -> None:
    persona = tmp_path / "fresh"
    persona.mkdir()
    _ensure(persona)
    state = state_of(persona)
    assert state.emotions == {} and state.persona_start_at is None and state.saved_at == NOW
    assert state.switch_over_at == NOW
    assert not (persona / "memories.db").exists()


def test_a_first_start_writes_no_switch_file(tmp_path: Path) -> None:
    """There is no switch file: the folder of a switched persona holds the state, its backups, the queue and the lock sidecar."""
    persona = _standard_persona(tmp_path)
    _ensure(persona)
    assert state_of(persona).switch_over_at == NOW  # the switch-over happened, and its only record is the state's field
    assert not [path.name for path in persona.rglob("*") if "switch" in path.name]  # nowhere under the persona, at any depth


def test_the_first_start_replaces_a_shadow_state_with_the_rebuilt_one(tmp_path: Path) -> None:
    persona = _standard_persona(tmp_path)
    save(persona, simple_state(saved_at=T[39], joy=9.0))  # what the shadow engine wrote
    _ensure(persona)
    assert "pride" in state_of(persona).emotions  # the rebuilt state, built from empty
    assert state_of(persona).saved_at == NOW


def test_a_switch_over_recorded_by_another_process_while_this_one_waited_for_the_lock_makes_this_one_rebuild_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    real = _x10().scale_up_small_feelings

    def another_process_finishes_first(path: Path):
        result = real(path)
        make_live(path, T[39])
        return result

    monkeypatch.setattr(_x10(), "scale_up_small_feelings", another_process_finishes_first)
    opened = _Spy(monkeypatch)
    _ensure(persona)
    assert "rebuild" not in opened.opened  # no rebuild: its window read never happened (the x10 fix, which ran first, did open the store)
    after = state_of(persona)
    assert after.switch_over_at == T[39] and after.emotions == {}  # the other process's state, not a rebuilt one


# ================================================================================================
# C-2: what a failing or uninspectable start does
# ================================================================================================


def test_a_check_failure_raises_naming_the_step_leaves_the_queue_and_the_old_state_and_records_no_switch_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    save(persona, simple_state(saved_at=T[39]))
    queue_before = engine_files.queue_path(persona).read_bytes()
    state_before = engine_files.state_path(persona).read_bytes()
    install(persona, NOW)
    with monkeypatch.context() as patch:
        # not `validate_state`: the reader uses it too, and `ensure` reads the shadow state first
        patch.setattr(engine_files, "state_to_json", lambda state: (_ for _ in ()).throw(ValueError("forced")))
        with pytest.raises(EngineStartupError) as failure:
            ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild" and isinstance(failure.value.cause, engine_rebuild.RebuildCheckError)
    assert engine_files.queue_path(persona).read_bytes() == queue_before  # the check comes before the queue step
    assert engine_files.state_path(persona).read_bytes() == state_before
    assert state_of(persona).switch_over_at is None
    # with the fault gone the next start completes, equal to a clean first run
    ensure_engine_ready(persona)
    clean = _standard_persona(tmp_path / "clean")
    save(clean, simple_state(saved_at=T[39]))
    install(clean, NOW)
    ensure_engine_ready(clean)
    assert _first_state_bytes(persona) == _first_state_bytes(clean)


def test_an_average_outside_zero_to_ten_fails_the_check() -> None:
    state = EngineState(saved_at=NOW, emotions={"joy": EmotionState()})
    state.emotions["joy"].short.avg = 11.0
    with pytest.raises(engine_rebuild.RebuildCheckError, match="outside 0 to"):
        engine_rebuild._check_rebuilt(state)
    state.emotions["joy"].short.avg = float("nan")
    with pytest.raises(engine_rebuild.RebuildCheckError):
        engine_rebuild._check_rebuilt(state)


def test_a_state_write_failure_after_the_emptying_leaves_the_old_state_and_an_empty_queue_and_the_next_start_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The E7.18 class: the queued lines are gone, the old state stays (showings are in the rows, changes in the change log)."""
    persona = _standard_persona(tmp_path)
    save(persona, simple_state(saved_at=T[39]))
    state_before = engine_files.state_path(persona).read_bytes()
    install(persona, NOW)
    with monkeypatch.context() as patch:
        patch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(EngineStartupError) as failure:
            ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild" and isinstance(failure.value.cause, OSError)
    assert engine_files.state_path(persona).read_bytes() == state_before
    assert engine_files.queue_path(persona).read_text(encoding="utf-8") == ""
    assert state_of(persona).switch_over_at is None
    ensure_engine_ready(persona)
    clean = _standard_persona(tmp_path / "clean")
    save(clean, simple_state(saved_at=T[39]))
    install(clean, NOW)
    ensure_engine_ready(clean)
    assert _first_state_bytes(persona) == _first_state_bytes(clean)


def test_a_state_write_that_does_not_verify_is_a_failed_check_and_writes_no_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    install(persona, NOW)
    with monkeypatch.context() as patch:
        patch.setattr(engine_files, "save_state", lambda *a, **k: False)
        with pytest.raises(EngineStartupError) as failure:
            ensure_engine_ready(persona)
    assert isinstance(failure.value.cause, engine_rebuild.RebuildCheckError)
    assert engine_files.read_state(persona).state is None  # nothing was written, so the persona is not switched over


def test_a_lock_that_does_not_come_raises_naming_the_lock_and_writes_nothing_after_the_x10_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    install(persona, NOW)
    # A real second holder of the engine lock, with the wait cut to 0.05 s: patching the lock to raise would not prove that
    # the lock `ensure` takes is the one the other holder has (the only real wait in this file).
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.05)
    with engine_files.state_lock(persona), pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the engine lock"
    assert not engine_files.state_path(persona).exists()
    monkeypatch.undo()
    ensure_engine_ready(persona)  # the x10 marker is set, the rebuild runs: idempotent
    assert state_of(persona).switch_over_at == NOW


def test_an_x10_error_raises_naming_the_fix_takes_no_lock_and_writes_no_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    install(persona, NOW)
    spy = LockSpy(monkeypatch)
    monkeypatch.setattr(_x10(), "scale_up_small_feelings", lambda p: (_ for _ in ()).throw(RuntimeError("rolled back")))
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the x10 fix of small feelings" and isinstance(failure.value.cause, RuntimeError)
    assert spy.calls == 0 and not engine_files.state_path(persona).exists()


def test_an_integrity_failure_raises_naming_the_check_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    install(persona, NOW)

    def refuse(*args: object, **kwargs: object):
        raise BrainIntegrityError("memories.db", "page 3 is damaged")

    monkeypatch.setattr(_x10(), "MemoryStore", refuse)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the memories.db integrity check" and isinstance(failure.value.cause, BrainIntegrityError)
    assert not engine_files.state_path(persona).exists()


def _candidate(memory_id: str, emotions: dict[str, float]) -> Memory:
    return row(T[36], memory_type="making", emotions=emotions, mid=memory_id)


def test_the_queue_step_runs_after_the_database_transaction_and_before_the_lock_and_the_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q10c: a small `making` candidate waiting in the pending queue is multiplied once, marked, and promoted at its real size."""
    from brain.memory import pending

    persona = _standard_persona(tmp_path)
    pending.PendingQueue(persona).enqueue(_candidate("waiting", {"pride": 0.3}), source="maker")
    spy = LockSpy(monkeypatch)
    order: list[str] = []
    for owner, name, label in (
        (MemoryStore, "scale_up_small_nudge_rows", "database"),
        (pending.PendingQueue, "scale_up_small_making_candidates", "queue"),
        (engine_rebuild, "rebuild", "rebuild"),
    ):
        real = getattr(owner, name)

        def wrapped(*args: object, _real=real, _label=label, **kwargs: object):
            order.append(_label + (":locked" if spy.within else ""))
            return _real(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)
    _ensure(persona)
    assert order == ["database", "queue", "rebuild:locked"]  # the lock came after the queue step, for the rebuild only
    (entry,) = pending.read_jsonl(persona / "pending_candidates.jsonl")
    assert entry["emotions"] == pytest.approx({"pride": 3.0}) and entry["metadata"][dev_constants.NUDGE_SCALE_FIX_MARKER_KEY] is True
    ensure_engine_ready(persona)  # a second start, and a new process's start, multiply nothing
    EmotionEngine(persona.resolve(), now=lambda: NOW).ensure_ready()
    (entry,) = pending.read_jsonl(persona / "pending_candidates.jsonl")
    assert entry["emotions"] == pytest.approx({"pride": 3.0})


def test_a_failed_queue_rewrite_stops_start_up_and_the_next_start_completes_catching_what_was_written_in_between(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.memory import pending

    persona = _standard_persona(tmp_path)
    pending.PendingQueue(persona).enqueue(_candidate("waiting", {"pride": 0.3}), source="maker")
    install(persona, NOW)
    spy = LockSpy(monkeypatch)
    real_write = pending.write_jsonl_bytes

    def refuse(*args: object, **kwargs: object):
        raise OSError("disk full")

    monkeypatch.setattr(pending, "write_jsonl_bytes", refuse)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the x10 fix of small feelings" and isinstance(failure.value.cause, OSError)
    assert spy.calls == 0 and not engine_files.state_path(persona).exists()
    assert _marked(persona, "made")  # the database half had finished
    (entry,) = pending.read_jsonl(persona / "pending_candidates.jsonl")
    assert entry["emotions"] == {"pride": 0.3}  # the queue is as it was
    # between the failed start and the next, an old writer adds a small row and a candidate
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(_candidate("late-row", {"pride": 0.2}), fold=False)
    finally:
        store.close()
    pending.PendingQueue(persona).enqueue(_candidate("late-candidate", {"pride": 0.2}), source="maker")
    monkeypatch.setattr(pending, "write_jsonl_bytes", real_write)
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        assert store.get("made", bump=False).emotions == pytest.approx({"pride": 4.0})  # 0.4 x 10 once, not 40
        assert store.get("late-row", bump=False).emotions == pytest.approx({"pride": 2.0})  # caught by the retry
    finally:
        store.close()
    entries = {e["id"]: e for e in pending.read_jsonl(persona / "pending_candidates.jsonl")}
    assert entries["waiting"]["emotions"] == pytest.approx({"pride": 3.0})
    assert entries["late-candidate"]["emotions"] == pytest.approx({"pride": 2.0})


def test_a_state_that_exists_but_cannot_be_read_skips_the_switch_over_with_one_warning_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Owner 2026-11-05: "If the system can't tell if its switched, assume its not." Nothing irreversible runs on an unknown."""
    persona = _standard_persona(tmp_path)
    save(persona, simple_state(saved_at=T[39]))
    ruin_the_state(persona)  # the state file is there and no copy of it can be read
    instance = install(persona, NOW)
    before = tree(persona)
    spy = LockSpy(monkeypatch)
    opened = _Spy(monkeypatch)
    caplog.set_level(logging.WARNING)
    ensure_engine_ready(persona)  # returns normally
    warnings = [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and str(engine_files.state_path(persona)) in warnings[0] and "cannot be read" in warnings[0]
    assert spy.calls == 0 and opened.opened == []  # no x10, no lock, no rebuild
    assert instance.is_live() is False and instance.needs_recovery() is False
    assert tree(persona) == before  # no state, no queue change, no marker
    # nothing about the unknown is cached: the next call in the same process, with the state gone or fixed, completes
    lose_the_state(persona)
    ensure_engine_ready(persona)
    clean = _standard_persona(tmp_path / "clean")
    _ensure(clean)
    assert _first_state_bytes(persona) == _first_state_bytes(clean)
    assert instance.is_live()


def test_a_state_that_cannot_be_read_only_at_the_locked_recheck_runs_x10_but_no_rebuild_and_raises_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona = _standard_persona(tmp_path)
    install(persona, NOW)
    missing = engine_files.LoadedState(None, None, None, "missing")
    unreadable = engine_files.LoadedState(None, None, None, "unreadable")
    real = engine_files.read_state
    answers = [missing, missing, unreadable]  # the fast path, the x10 fix's own guard, the re-check under the lock

    def scripted(path: Path):
        return answers.pop(0) if answers else real(path)

    monkeypatch.setattr(engine_files, "read_state", scripted)
    caplog.set_level(logging.WARNING)
    ensure_engine_ready(persona)
    assert answers == []
    assert _marked(persona, "made")  # x10 ran and marked the row it multiplied
    assert not engine_files.state_path(persona).exists()
    assert len([r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]) == 1
    ensure_engine_ready(persona)  # the fault is gone
    clean = _standard_persona(tmp_path / "clean")
    _ensure(clean)
    assert _first_state_bytes(persona) == _first_state_bytes(clean)


def test_a_missing_state_means_not_switched_and_the_switch_over_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _standard_persona(tmp_path)
    assert not engine_files.state_path(persona).exists()
    opened = _Spy(monkeypatch)
    instance = _ensure(persona)
    assert opened.opened == ["x10", "rebuild"]
    assert instance.is_live() and state_of(persona).switch_over_at == NOW


def test_a_state_that_already_holds_the_switch_over_time_returns_at_once_with_no_store_open_and_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _standard_persona(tmp_path)
    make_live(persona, T[39])  # another process switched the persona over earlier
    before = tree(persona)
    spy = LockSpy(monkeypatch)
    opened = _Spy(monkeypatch)
    instance = install(persona, NOW)
    ensure_engine_ready(persona)
    assert spy.calls == 0 and opened.opened == [] and tree(persona) == before
    assert instance.is_live() and state_of(persona).switch_over_at == T[39]  # the time never moves


def _marked(persona: Path, memory_id: str) -> bool:
    """True when the row carries the x10 fix's marker in its metadata."""
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        found = store.get(memory_id, bump=False)
        return found is not None and found.metadata.get(dev_constants.NUDGE_SCALE_FIX_MARKER_KEY) is True
    finally:
        store.close()


# ================================================================================================
# C-3: the rebuild itself
# ================================================================================================


def test_a_numeric_golden_for_two_rows_two_days_apart_by_one_day(tmp_path: Path) -> None:
    # Row A: joy 2.0 at T[30]. Row B: joy 6.0 at T[31], 24 hours later, three hours before now (so B gets a repeat entry).
    # Fold A: short (avg 2, weight 1), long (avg 2, weight 1). Fold B 24 h later ages the weights: short 0.5^(24/24) = 0.5,
    # long 0.5^(24/720); then folds 6 at weight 1.
    #   short: weight 0.5 + 1 = 1.5, avg 2 + (6 - 2) x 1 / 1.5 = 14/3
    #   long: weight w + 1 with w = 0.5^(1/30), avg 2 + 4 / (w + 1)
    now = T[31] + timedelta(hours=3)
    persona = make_persona(tmp_path, [row(T[30], emotions={"joy": 2.0}, mid="a"), row(T[31], emotions={"joy": 6.0}, mid="b")])
    _rebuild(persona, "first", now)
    state = state_of(persona)
    joy = state.emotions["joy"]
    w = 0.5 ** (1 / 30)
    assert (joy.short.avg, joy.short.weight) == (pytest.approx(14 / 3), pytest.approx(1.5))
    assert (joy.long.avg, joy.long.weight) == (pytest.approx(2 + 4 / (w + 1)), pytest.approx(w + 1))
    assert state.persona_start_at == T[30]  # the oldest row
    assert set(state.repeats) == {"b"} and state.repeats["b"].first_at == T[31] and state.repeats["b"].count == 0  # within 24 h of now
    assert state.saved_at == state.reset_at == now and state.queue_line == 0
    assert [d.day for d in joy.snapshot] == [day_of(T[30])]  # day 30 closed when B arrived; day 31 is still open


def test_a_first_rebuild_folds_stored_values_and_a_recovery_rebuild_folds_the_faded_ones(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 8.0}, mid="a")])
    _rebuild(persona, "first")
    assert state_of(persona).emotions["joy"].short.avg == pytest.approx(8.0)
    half = lambda memory: {name: value / 2 for name, value in memory.emotions.items()}  # noqa: E731
    _rebuild(persona, "recovery", faded=half)
    assert state_of(persona).emotions["joy"].short.avg == pytest.approx(4.0)


def test_the_window_is_thirty_days_back_from_now_and_a_row_dated_after_now_folds_at_now(tmp_path: Path) -> None:
    after = NOW - timedelta(days=30)
    persona = make_persona(
        tmp_path,
        [
            row(after - timedelta(hours=1), emotions={"fear": 5.0}, mid="too-old"),
            row(after, emotions={"grief": 5.0}, mid="at-the-bound"),
            row(NOW + timedelta(days=2), emotions={"joy": 5.0}, mid="ahead"),
        ],
    )
    _rebuild(persona, "first")
    state = state_of(persona)
    assert set(state.emotions) == {"grief", "joy"}  # the row an hour before the window is not folded
    assert state.last_update == NOW  # the row dated ahead of now is folded at now, not dropped
    assert state.persona_start_at == after - timedelta(hours=1)  # but the oldest row of ALL rows still sets the start


def test_the_persona_start_is_the_earliest_of_the_oldest_row_the_replay_and_the_old_state(tmp_path: Path) -> None:
    ancient = NOW - timedelta(days=90)
    persona = make_persona(tmp_path, [row(ancient, memory_type="reflex_arc", mid="ancient"), row(T[35], emotions={"joy": 5.0}, mid="in")])
    _rebuild(persona, "first")
    assert state_of(persona).persona_start_at == ancient  # the query finds a row outside the window
    older = simple_state(saved_at=T[39])
    older.switch_over_at = T[30]
    older.persona_start_at = NOW - timedelta(days=200)
    save(persona, older)
    _rebuild(persona, "recovery")
    assert state_of(persona).persona_start_at == NOW - timedelta(days=200)  # a recovery keeps the old state's earlier start
    persona2 = make_persona(tmp_path / "second", [row(T[35], emotions={"joy": 5.0}, mid="in")])
    _rebuild(persona2, "first")
    assert state_of(persona2).persona_start_at == T[35]  # no row older than the replay's own


def test_every_row_of_any_type_marks_its_day_and_a_counted_row_with_no_feelings_marks_its_access_day(tmp_path: Path) -> None:
    emotionless_counted = dataclasses.replace(row(T[24], emotions={}, mid="quiet"), last_accessed_at=T[26])
    persona = make_persona(
        tmp_path,
        [
            row(T[20], emotions={"joy": 6.0}, mid="felt"),
            row(T[22], memory_type="reflex_arc", mid="uncounted"),  # any type marks its creation day
            emotionless_counted,  # counted, created on 24, no feelings, opened on 26
        ],
    )
    _rebuild(persona, "first")
    days = [d.day for d in state_of(persona).emotions["joy"].snapshot]
    # snapshot days are the days that closed after the emotion appeared: 20 (felt), 22, 24 (creation days) and 26 (the access
    # day of the emotion-less counted row, which `replay` would not mark by itself); no other day is an active one
    assert days == [day_of(T[n]) for n in (20, 22, 24, 26)]


def test_an_opened_counted_row_whose_stored_feelings_are_not_a_mapping_is_skipped_and_the_rebuild_completes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid JSON that is not a mapping (`"x"`, `[]`) loads as the row's feelings and breaks `emotions_of`; one such row must
    not stop every first start for good. The access-day scan and `replay` both meet it; `replay` logs the row and skips it."""
    persona = make_persona(
        tmp_path,
        # both on day 35, so no quiet day follows the good row's entry and its average stays 5.0
        [row(T[35], emotions={"joy": 5.0}, mid="good"), dataclasses.replace(row(T[35], emotions={"joy": 5.0}, mid="odd"), last_accessed_at=T[35] + timedelta(hours=1))],
    )
    connection = sqlite3.connect(persona / "memories.db")
    try:
        connection.execute("UPDATE memories SET emotions_json = ? WHERE id = ?", (json.dumps("not a mapping"), "odd"))
        connection.commit()
    finally:
        connection.close()
    caplog.set_level(logging.WARNING)
    _rebuild(persona, "first")
    assert state_of(persona).emotions["joy"].short.avg == pytest.approx(5.0)  # the good row folded
    assert any("odd" in r.getMessage() and "skipped" in r.getMessage() for r in caplog.records)


def test_the_queue_is_emptied_and_the_position_is_zero_or_the_end_line_when_the_emptying_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    queue(persona, showing("a", T[36]), showing("a", T[37]), showing("a", T[38]))
    with monkeypatch.context() as patch:
        patch.setattr(engine_files, "empty_queue", lambda p: False)
        _rebuild(persona, "first")
    assert state_of(persona).queue_line == 3  # the queue's end line: no queued line is folded on top of the rebuild
    assert len(engine_files.queue_path(persona).read_text(encoding="utf-8").splitlines()) == 3  # kept
    _rebuild(persona, "first")
    assert state_of(persona).queue_line == 0
    assert engine_files.queue_path(persona).read_text(encoding="utf-8") == ""


def test_a_first_write_with_no_backup_verifies_and_returns_true(tmp_path: Path) -> None:
    state = simple_state(saved_at=NOW)
    assert engine_files.save_state(tmp_path, state, verify=True) is True
    assert not engine_files.state_path(tmp_path).with_name("emotion_state.json.bak1").exists()
    assert engine_files.save_state(tmp_path, state) is True


def test_a_first_rebuild_with_no_database_creates_nothing_but_a_recovery_rebuild_raises_and_keeps_the_backup(tmp_path: Path) -> None:
    persona = tmp_path / "no-db"
    persona.mkdir()
    _rebuild(persona, "first")
    assert not (persona / "memories.db").exists() and state_of(persona).switch_over_at == NOW
    other = tmp_path / "other"
    other.mkdir()
    save_as_backup(other, simple_state(saved_at=T[39]))
    with pytest.raises(CatchUpError):
        _rebuild(other, "recovery")
    loaded = engine_files.read_state(other)
    assert loaded.state is not None and loaded.source == "bak1"  # a readable backup is still there


def test_the_change_log_is_replayed_with_missing_parts_and_a_torn_last_line(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [])
    # one file only, day 25 (no file before it or after it in the series), and it ends in a torn line
    engine_files.append_change(persona, change(T[25], {"fear": 4.0}))
    day_25 = engine_files.changes_dir(persona) / f"{day_of(T[25])}.jsonl"
    with day_25.open("ab") as handle:
        handle.write(b'{"kind": "change", "at": "2026-')  # torn: no newline, not JSON
    _rebuild(persona, "first")
    state = state_of(persona)
    # a change folds as an entry of (stored average 0 + 4) at weight 1: average 4, on its own day, which is the last day
    assert set(state.emotions) == {"fear"}
    assert state.emotions["fear"].short.avg == pytest.approx(4.0) and state.emotions["fear"].long.avg == pytest.approx(4.0)


def test_a_missing_change_log_folder_and_an_unlistable_one_do_not_stop_the_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    assert not engine_files.changes_dir(persona).exists()
    _rebuild(persona, "first")  # a missing folder is empty
    real = file_ops.list_dir

    def refuse(path):
        if Path(path) == engine_files.changes_dir(persona):
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(file_ops, "list_dir", refuse)
    _rebuild(persona, "first")  # logged, treated as missing
    assert state_of(persona).emotions["joy"].short.avg == pytest.approx(5.0)


def test_a_missing_change_log_file_in_the_middle_of_the_series_loses_only_its_own_lines(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [])
    for n, name in ((20, "joy"), (21, "grief"), (22, "fear")):
        engine_files.append_change(persona, change(T[n], {name: 4.0}))
    (engine_files.changes_dir(persona) / f"{day_of(T[21])}.jsonl").unlink()  # the file of day 21 is gone
    _rebuild(persona, "first")
    state = state_of(persona)
    assert set(state.emotions) == {"joy", "fear"}  # the lines of the files that exist are replayed, around the gap
    assert state.emotions["fear"].short.avg == pytest.approx(4.0)  # its own entry, on the last active day


def test_the_rebuild_is_called_with_the_lock_held_and_never_takes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    spy = LockSpy(monkeypatch)
    _rebuild(persona, "first")
    assert spy.calls == 0


# ================================================================================================
# C-7: peers
# ================================================================================================


def _peer_row(mid: str, created: datetime, peer: str = "peer-a", joy: float = 5.0) -> Memory:
    return row(created, memory_type="kindled_peer", emotions={"joy": joy}, mid=mid, metadata={"peer_id": peer})


def _trust(persona: Path, peer: str, score: float) -> None:
    kindled_db_path(persona).parent.mkdir(parents=True, exist_ok=True)
    store = KindledLinkStore(kindled_db_path(persona))
    try:
        persist_relationship_state(store, PeerRelationshipState(peer_id=peer, trust_score=score), NOW)
    finally:
        store.close()


def test_a_peer_at_trust_one_takes_modifier_one_through_the_real_kindled_link_accessor(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [_peer_row("p", T[35])])
    _trust(persona, "peer-a", 1.0)
    _ensure(persona)
    assert state_of(persona).emotions["joy"].short.weight == pytest.approx(1.0)  # 0.1 + 0.9 x 1.0
    other = make_persona(tmp_path / "no-link", [_peer_row("p", T[35])])
    _ensure(other)
    assert state_of(other).emotions["joy"].short.weight == pytest.approx(0.1)  # no kindled-link database: trust 0, the floor
    assert not kindled_db_path(other).exists()  # reading trust created nothing


def test_engine_files_peer_trust_is_a_pass_through_to_the_kindled_link_accessor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str]] = []

    def hers(persona_dir: Path, peer_id: str) -> float:
        calls.append((persona_dir, peer_id))
        return 0.77

    monkeypatch.setattr(relationship, "peer_trust", hers)
    assert engine_files.peer_trust(tmp_path, "someone") == 0.77
    assert calls == [(tmp_path, "someone")]


def test_peer_trust_at_the_engine_door_is_the_stored_score_and_zero_for_an_unknown_peer_or_no_database_and_creates_nothing(tmp_path: Path) -> None:
    persona = tmp_path / "with-link"
    persona.mkdir()
    _trust(persona, "peer-a", 0.8)
    assert engine_files.peer_trust(persona, "peer-a") == pytest.approx(0.8)  # the real accessor's raw trust score
    assert engine_files.peer_trust(persona, "stranger") == 0.0  # no row
    bare = tmp_path / "bare"
    bare.mkdir()
    assert engine_files.peer_trust(bare, "anyone") == 0.0
    assert engine_files.peer_trust(tmp_path / "no" / "such", "") == 0.0
    assert list(bare.iterdir()) == [] and not (tmp_path / "no").exists()  # reading trust created nothing


def _raising_trust(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    asked: list[str] = []

    def hers(persona_dir: Path, peer_id: str) -> float:
        asked.append(peer_id)
        raise OSError("permission denied")  # what her function does not catch

    monkeypatch.setattr(relationship, "peer_trust", hers)
    return asked


def test_an_exception_her_function_does_not_catch_gives_the_floor_modifier_for_a_hook_with_one_throttled_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona = make_persona(tmp_path, [])
    instance = install(persona, NOW)
    asked = _raising_trust(monkeypatch)
    caplog.set_level(logging.WARNING)
    memory = _peer_row("p", NOW - timedelta(hours=1))
    instance.on_memory_committed(memory)
    # the event was applied with the floor modifier (0.1 + 0.9 x 0), not dropped by the hook's guard
    assert state_of(persona).emotions["joy"].short.weight == pytest.approx(0.1)
    instance.on_memory_committed(dataclasses.replace(memory, id="q"))
    assert asked == ["peer-a", "peer-a"]  # a hook asks once per event, as before
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER and "trust in a peer" in r.getMessage()]
    assert len(lines) == 1  # throttled: one warning for the two events


def test_an_exception_her_function_does_not_catch_gives_the_floor_for_a_catch_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [_peer_row("p", T[39])])
    instance = install(persona, NOW)
    asked = _raising_trust(monkeypatch)
    state = simple_state(saved_at=T[38])
    folded = engine_rebuild.catch_up(
        persona, state, NOW, queued=[], emotions_of=instance._replay_emotions, modifier_of=instance._replay_modifier_with_memo()
    )
    assert asked == ["peer-a"]  # her function was asked, and raised
    assert folded == {"p"}  # the peer's row folded all the same (the replay's own guard would have skipped it)


def test_a_catch_up_asks_for_the_trust_in_a_peer_once_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [_peer_row("p1", T[39]), _peer_row("p2", T[39] + timedelta(hours=1))])
    instance = install(persona, NOW)
    asked: list[str] = []
    monkeypatch.setattr(relationship, "peer_trust", lambda persona_dir, peer_id: asked.append(peer_id) or 0.5)
    for _ in range(2):  # a second call makes its own memo
        engine_rebuild.catch_up(
            persona, simple_state(saved_at=T[38]), NOW, queued=[], emotions_of=instance._replay_emotions, modifier_of=instance._replay_modifier_with_memo()
        )
    assert asked == ["peer-a", "peer-a"]  # two rows of one peer, two calls: one ask each


def test_the_drain_catch_up_of_a_backup_state_asks_for_the_trust_in_a_peer_once_not_once_per_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`_run_apply` makes the catch_up's modifier with its own memo: a kindled-link open per peer row would run under the lock."""
    persona, instance = _live_persona(tmp_path, [_peer_row("p1", T[39]), _peer_row("p2", T[39] + timedelta(hours=1))])
    save_as_backup(persona, simple_state(saved_at=T[38]))  # the main file is unusable: the drain recovers from `.bak1` first
    asked: list[str] = []
    monkeypatch.setattr(relationship, "peer_trust", lambda persona_dir, peer_id: asked.append(peer_id) or 0.5)
    instance.note_active(NOW)
    assert asked == ["peer-a"]
    assert engine_files.read_state(persona).source == "main"  # the catch-up ran and its result was saved


def test_an_exception_her_function_does_not_catch_gives_the_floor_in_the_rebuild_and_the_row_still_folds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [_peer_row("p", T[35])])
    asked = _raising_trust(monkeypatch)
    _ensure(persona)
    assert state_of(persona).emotions["joy"].short.weight == pytest.approx(0.1)  # the floor modifier, 0.1 + 0.9 x 0
    assert asked == ["peer-a"]


def test_trust_is_asked_once_per_peer_per_rebuild_and_the_memo_is_dropped_after_the_call_success_or_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [_peer_row("p1", T[35]), _peer_row("p2", T[36]), _peer_row("p3", T[37], peer="peer-b")])
    asked: list[str] = []
    monkeypatch.setattr(relationship, "peer_trust", lambda persona_dir, peer_id: asked.append(peer_id) or 0.5)
    instance = install(persona, NOW)
    instance.ensure_ready()
    assert sorted(asked) == ["peer-a", "peer-b"]  # three peer rows, two peers, one ask each
    asked.clear()
    ruin_the_state(persona)
    instance.recovery_rebuild()
    assert sorted(asked) == ["peer-a", "peer-b"]  # a new call asks again: the memo was dropped
    asked.clear()
    ruin_the_state(persona)
    with monkeypatch.context() as patch:
        patch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
        with pytest.raises(EngineStartupError):
            instance.recovery_rebuild()
    asked.clear()
    instance.recovery_rebuild()
    assert sorted(asked) == ["peer-a", "peer-b"]  # and after a failed call too


# ================================================================================================
# C-8: the live fade and the change log
# ================================================================================================


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _live_engine(persona: Path, switch_at: datetime, now: datetime) -> tuple[EmotionEngine, _Clock]:
    clock = _Clock(now)
    make_live(persona, switch_at)
    return EmotionEngine(persona, now=clock), clock


def _memory(created: datetime, emotions: dict, importance: float = 0.0) -> Memory:
    return dataclasses.replace(row(created, emotions=emotions, mid="m"), importance=importance)


def test_faded_value_is_the_stored_value_before_the_switch_and_faded_after_it_on_the_same_instance(tmp_path: Path) -> None:
    clock = _Clock(T[30])
    instance = EmotionEngine(tmp_path, now=clock)
    memory = _memory(T[20], {"joy": 8.0})
    clock.now = T[50]
    assert instance.faded_value(memory, "joy") == 8.0  # not switched: the stored value, whatever the age
    make_live(tmp_path, T[44])
    # now it is live. Age counts from the switch (day 44 is later than the creation, day 20): 6 days. joy half-life 3 days,
    # importance 0 (stretch 1): 8 x 0.5^(6/3) = 2.0
    assert instance.faded_value(memory, "joy") == pytest.approx(2.0)
    # an emotion the memory lacks still raises, naming the memory and the emotion, live as before the switch (never a 0.0)
    with pytest.raises(KeyError, match=r"m.*'grief'"):
        instance.faded_value(memory, "grief")


def test_faded_value_numbers_for_importance_the_floor_a_vocabulary_half_life_and_a_future_row(tmp_path: Path) -> None:
    instance, clock = _live_engine(tmp_path, T[0], T[3])
    assert instance.faded_value(_memory(T[0], {"joy": 8.0}, 0.0), "joy") == pytest.approx(4.0)  # age 3 d = one half-life of joy, stretch 1
    # importance 10: stretch 1 + 52 = 53 (HORIZON_IMPORTANCE_GAIN 52): 8 x 0.5^(3 / (3 x 53))
    assert instance.faded_value(_memory(T[0], {"joy": 8.0}, 10.0), "joy") == pytest.approx(8.0 * 0.5 ** (1 / 53))
    # love: half-life 730 d; after 1460 d (two half-lives) the value would be 2.0 but the floor is a third of the stored value
    clock.now = T[0] + timedelta(days=1460)
    assert instance.faded_value(_memory(T[0], {"love": 8.0}, 0.0), "love") == pytest.approx(8.0 / 3)
    # a future-dated row never grows: age is held at 0
    clock.now = T[10]
    assert instance.faded_value(_memory(T[20], {"joy": 8.0}), "joy") == 8.0
    # a vocabulary half-life (6 days) replaces the table's 3 days: 12 days after the switch = two half-lives, 8 x 0.25
    (tmp_path / "emotion_vocabulary.json").write_text(json.dumps({"emotions": [{"name": "joy", "decay_half_life_days": 6.0}]}), encoding="utf-8")
    clock.now = T[12]
    assert instance.faded_value(_memory(T[0], {"joy": 8.0}), "joy") == pytest.approx(2.0)


def test_faded_value_needs_no_store_so_it_serves_a_row_that_hard_delete_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [_memory(T[0], {"joy": 8.0})])  # id "m", importance 0
    store = MemoryStore(persona / "memories.db")
    try:
        memory = store.get("m", bump=False)
        store.hard_delete("m")
        assert store.get("m", bump=False) is None
    finally:
        store.close()
    instance, _ = _live_engine(persona, T[0], T[3])

    def never(*args: object, **kwargs: object):
        raise AssertionError("faded_value read a store")

    monkeypatch.setattr(MemoryStore, "__init__", never)
    assert instance.faded_value(memory, "joy") == pytest.approx(4.0)


def test_an_unreadable_state_gives_the_stored_value_and_is_not_cached(tmp_path: Path) -> None:
    _live_engine(tmp_path, T[0], T[3])  # the persona is switched over, its state holds the time
    memory = _memory(T[0], {"joy": 8.0})
    fresh = EmotionEngine(tmp_path, now=lambda: T[3])  # a new instance has seen nothing
    ruin_the_state(tmp_path)
    assert fresh.faded_value(memory, "joy") == 8.0 and fresh.is_live() is False  # cannot tell: not switched, the stored value
    make_live(tmp_path, T[0])  # the state is readable again (the ruined copy goes to .bak1)
    assert fresh.faded_value(memory, "joy") == pytest.approx(4.0)  # nothing about the failure was kept


def test_a_state_whose_switch_over_time_is_not_a_time_is_unreadable_and_gives_the_stored_value(tmp_path: Path) -> None:
    state = engine_files.state_path(tmp_path)
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({**engine_files.state_to_json(simple_state(saved_at=T[1])), "switch_over_at": "never"}), encoding="utf-8")
    instance = EmotionEngine(tmp_path, now=lambda: T[3])
    memory = _memory(T[0], {"joy": 8.0})
    assert instance.is_live() is False  # the validator refuses the whole state, so it cannot be read
    assert instance.faded_value(memory, "joy") == 8.0
    make_live(tmp_path, T[0])
    assert instance.faded_value(memory, "joy") == pytest.approx(4.0)  # read again, now readable


def test_fold_change_writes_one_change_log_line_when_live_and_none_before_the_switch(tmp_path: Path) -> None:
    shadow = EmotionEngine(tmp_path / "shadow", now=lambda: NOW)
    shadow.fold_change({"joy": 1.0}, NOW, "chat")
    assert not engine_files.changes_dir(tmp_path / "shadow").exists()
    assert len(engine_files.read_queue(tmp_path / "shadow", from_line=0).events) == 1
    live, _ = _live_engine(tmp_path / "live", T[39], NOW)
    live.fold_change({"joy": 1.0}, NOW, "chat")
    (log,) = list(engine_files.changes_dir(tmp_path / "live").glob("*.jsonl"))
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1  # one line
    assert len(engine_files.read_queue(tmp_path / "live", from_line=0).events) == 1  # and the queue still has its own


# ================================================================================================
# C-10: recovery
# ================================================================================================


def _live_persona(tmp_path: Path, rows=()) -> tuple[Path, EmotionEngine]:
    """A switched-over persona and the running engine that has seen it (an instance keeps the `switch_over_at` it saw).

    The state file is gone afterwards: the running instance holds the only record, so a test starts from "live with a missing
    state" and saves whatever state it needs.
    """
    persona = make_persona(tmp_path, list(rows))
    make_live(persona, NOW - timedelta(days=6))
    instance = install(persona, NOW)
    assert instance.is_live()
    lose_the_state(persona)
    return persona, instance


def test_needs_recovery_truth_table(tmp_path: Path) -> None:
    # not switched: nothing to recover, even with no state
    not_live = make_persona(tmp_path / "a")
    assert install(not_live, NOW).needs_recovery() is False
    # live with a missing state
    persona, instance = _live_persona(tmp_path / "b")
    assert instance.needs_recovery() is True
    # live with a readable state
    save(persona, simple_state(saved_at=T[39]))
    assert instance.needs_recovery() is False
    # live with a state only a backup holds: the drain's own catch-up reads and heals that, not this job
    other, other_engine = _live_persona(tmp_path / "c")
    save_as_backup(other, simple_state(saved_at=T[39]))
    assert other_engine.needs_recovery() is False
    # live with an unreadable state (main and every backup)
    ruin_the_state(persona)
    assert instance.needs_recovery() is True
    # live with a readable state and the catch-up-failed flag set
    third, third_engine = _live_persona(tmp_path / "d")
    save(third, simple_state(saved_at=T[39]))
    third_engine._catch_up_failed = True
    assert third_engine.needs_recovery() is True
    # an instance that never saw the switch-over meets an unreadable state: cannot tell, so not switched and no recovery
    fourth, _ = _live_persona(tmp_path / "e")
    make_live(fourth, NOW - timedelta(days=6))
    ruin_the_state(fourth)
    assert EmotionEngine(fourth, now=lambda: NOW).needs_recovery() is False
    # and one that meets a missing state (accepted edge (a)): the first rebuild runs again at its next start, not a recovery
    fifth, _ = _live_persona(tmp_path / "f")
    assert EmotionEngine(fifth, now=lambda: NOW).needs_recovery() is False


def test_a_recovery_rebuild_folds_the_faded_values_empties_the_queue_keeps_the_switch_and_clears_the_failed_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # joy 8.0 created 9 days ago, switch 6 days ago, importance 0: faded from the switch, 6 d = 2 half-lives of joy (3 d): 8 x 0.25 = 2.0
    persona, instance = _live_persona(tmp_path, [dataclasses.replace(row(NOW - timedelta(days=9), emotions={"joy": 8.0}, mid="a"), importance=0.0)])
    queue(persona, showing("a", T[39]))
    instance._catch_up_failed = True
    spy = LockSpy(monkeypatch)
    instance.recovery_rebuild()
    assert spy.within == [(engine_files.state_path(persona), dev_constants.EMOTION_LOCK_WAIT_S)]  # the lock, once, with the wait
    assert state_of(persona).emotions["joy"].short.avg == pytest.approx(2.0)
    assert engine_files.queue_path(persona).read_text(encoding="utf-8") == "" and state_of(persona).queue_line == 0
    # a recovery never sets a new switch-over time: the state it wrote holds the one the engine saw (no readable copy had it)
    assert state_of(persona).switch_over_at == NOW - timedelta(days=6)
    assert instance._catch_up_failed is False and instance.needs_recovery() is False


def test_a_recovery_rebuild_also_repairs_a_state_that_only_a_backup_holds(tmp_path: Path) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    save_as_backup(persona, simple_state(saved_at=T[39]))
    instance.recovery_rebuild()
    loaded = engine_files.read_state(persona)
    assert loaded.source == "main" and loaded.state.saved_at == NOW


def test_the_recheck_under_the_lock_finds_a_state_another_process_repaired_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    save(persona, simple_state(saved_at=T[39], joy=7.0))  # readable from the main file, with history a rebuild would drop
    instance._catch_up_failed = True
    before = tree(persona)
    opened = _Spy(monkeypatch)
    instance.recovery_rebuild()
    assert tree(persona) == before and opened.opened == []
    assert instance._catch_up_failed is False  # the flag is cleared: the file is whole


def test_a_recovery_rebuild_on_a_persona_that_is_not_live_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    instance = install(persona, NOW)
    before = tree(persona)
    opened = _Spy(monkeypatch)
    instance.recovery_rebuild()
    assert tree(persona) == before and opened.opened == []


def test_a_failed_recovery_rebuild_raises_and_keeps_the_failed_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    ruin_the_state(persona)
    instance._catch_up_failed = True
    with monkeypatch.context() as patch:
        patch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
        with pytest.raises(EngineStartupError) as failure:
            instance.recovery_rebuild()
    assert failure.value.what == "the recovery rebuild"
    assert instance._catch_up_failed is True and instance.needs_recovery() is True


def test_a_recovery_rebuild_with_no_database_raises_and_leaves_a_readable_backup(tmp_path: Path) -> None:
    persona = tmp_path / "persona"
    persona.mkdir()
    save_as_backup(persona, simple_state(saved_at=T[39], switch_over_at=T[30]))  # the only readable copy, and it holds the switch-over time
    instance = install(persona, NOW)
    assert instance.is_live()
    with pytest.raises(EngineStartupError) as failure:
        instance.recovery_rebuild()
    assert isinstance(failure.value.cause, CatchUpError)
    assert engine_files.read_state(persona).state is not None


def test_the_recovery_job_rebuilds_a_ruined_state_and_reports_success(tmp_path: Path) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    ruin_the_state(persona)
    assert instance.needs_recovery() is True
    assert engine.recovery_job_run(persona) is True
    assert state_of(persona).saved_at == NOW and instance.needs_recovery() is False
    assert state_of(persona).switch_over_at == NOW - timedelta(days=6)  # the persona stays switched over, with its own time
    assert EmotionEngine(persona.resolve(), now=lambda: NOW).is_live()  # a new process reads it off the rebuilt state


def test_the_recovery_job_on_a_healthy_or_a_not_switched_persona_does_nothing_and_reports_success(tmp_path: Path) -> None:
    healthy, _ = _live_persona(tmp_path / "healthy", [row(T[35], emotions={"joy": 5.0}, mid="a")])
    save(healthy, simple_state(saved_at=T[39]))
    not_switched = make_persona(tmp_path / "not_switched", [row(T[35], emotions={"joy": 5.0}, mid="b")])
    install(not_switched, NOW)
    before = {"healthy": tree(healthy), "not_switched": tree(not_switched)}
    assert engine.recovery_job_run(healthy) is True and engine.recovery_job_run(not_switched) is True
    assert {"healthy": tree(healthy), "not_switched": tree(not_switched)} == before


def test_a_failing_recovery_job_reports_failure_logs_once_per_active_day_keeps_the_retry_and_recovers_when_the_fault_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    ruin_the_state(persona)
    with monkeypatch.context() as patch, caplog.at_level(logging.INFO, logger=LOGGER):
        patch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
        assert engine.recovery_job_run(persona) is False
        assert engine.recovery_job_run(persona) is False  # the next quiet pass, the same active day
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and r.name == LOGGER]
    assert len(errors) == 1  # once per active day, not once per pass, and not the start-up wording
    assert "start-up stops" not in errors[0].getMessage() and errors[0].exc_info is not None
    assert instance.needs_recovery() is True  # the failed flag is set: the next pass tries again
    assert engine.recovery_job_run(persona) is True  # the fault is gone
    assert state_of(persona).saved_at == NOW and instance.needs_recovery() is False


def test_a_failing_recovery_job_is_logged_even_when_the_drain_already_logged_the_missing_state_that_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona, instance = _live_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    ruin_the_state(persona)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        instance.note_active(NOW)  # the bridge's start-up drain says the state is missing
        with monkeypatch.context() as patch:
            patch.setattr(engine_files, "save_state", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
            assert engine.recovery_job_run(persona) is False
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and r.name == LOGGER]
    assert len(messages) == 2 and "recovery rebuild failed" in messages[1]  # why the rebuild failed is not hidden by the drain's line


# ================================================================================================
# SW2: the switch-over time is a field of the state (there is no switch file)
# ================================================================================================


def test_a_shadow_drain_and_a_direct_apply_never_set_the_switch_over_time(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    instance = install(persona, NOW)
    queue(persona, showing("a", T[39]))
    instance.note_active(NOW)  # a shadow drain: it folds the showing and writes the state
    assert state_of(persona).emotions and state_of(persona).switch_over_at is None  # it wrote, and set nothing
    instance.on_memory_committed(row(T[40], emotions={"joy": 3.0}, mid="b"))  # a direct apply
    assert state_of(persona).switch_over_at is None
    assert instance.is_live() is False


def test_every_save_of_a_switched_persona_keeps_the_switch_over_time(tmp_path: Path) -> None:
    persona = _standard_persona(tmp_path)
    instance = _ensure(persona)
    assert state_of(persona).switch_over_at == NOW
    queue(persona, showing("old", NOW + timedelta(hours=1)))
    instance.note_active(NOW + timedelta(hours=2))  # a drain that folds a showing
    assert state_of(persona).switch_over_at == NOW
    instance.note_active(NOW + timedelta(days=3))  # a drain that closes days
    assert state_of(persona).switch_over_at == NOW
    instance.on_memory_committed(row(T[40], emotions={"joy": 3.0}, mid="c"))  # a direct apply
    assert state_of(persona).switch_over_at == NOW


@pytest.mark.parametrize("copy", ["main", "backup"])
def test_a_recovery_rebuild_copies_the_switch_over_time_from_the_state_it_loaded(tmp_path: Path, copy: str) -> None:
    persona = make_persona(tmp_path, [row(T[35], emotions={"joy": 5.0}, mid="a")])
    old = simple_state(saved_at=T[39], switch_over_at=T[10])
    if copy == "main":
        save(persona, old)
    else:
        save_as_backup(persona, old)
    _rebuild(persona, "recovery", switch_over_at=T[12])  # the engine's own value is the fallback only
    assert state_of(persona).switch_over_at == T[10]


def test_a_recovery_rebuild_with_no_readable_copy_takes_the_time_the_engine_saw_and_with_none_at_all_refuses(tmp_path: Path) -> None:
    rows = [row(T[35], emotions={"joy": 5.0}, mid="a")]
    persona = make_persona(tmp_path / "engine-saw-it", rows)
    _rebuild(persona, "recovery", switch_over_at=T[12])
    assert state_of(persona).switch_over_at == T[12]
    other = make_persona(tmp_path / "nothing-to-carry", rows)
    save(other, simple_state(saved_at=T[39]))  # readable, but it holds no switch-over time
    queue(other, showing("a", T[40]))
    before = tree(other)
    with pytest.raises(engine_rebuild.RebuildCheckError, match="switch-over time"):
        _rebuild(other, "recovery")
    assert tree(other) == before  # nothing written, the queue kept: a state without the field would read as not switched


def test_a_drain_that_recovers_from_a_backup_copy_keeps_the_switch_over_time_the_copy_holds(tmp_path: Path) -> None:
    persona = make_persona(tmp_path, [row(T[39], emotions={"joy": 5.0}, mid="a")])
    save_as_backup(persona, simple_state(saved_at=T[38], switch_over_at=T[30]))
    instance = install(persona, NOW)  # a new process: it learns the switch-over from the backup copy
    assert instance.is_live()
    instance.note_active(NOW)
    loaded = engine_files.read_state(persona)
    assert loaded.source == "main" and loaded.state.switch_over_at == T[30]


def test_ensure_engine_ready_calls_ensure_ready_through_engine_for(tmp_path: Path, emotion_engine_stub) -> None:
    assert engine.engine_for(tmp_path) is emotion_engine_stub
    ensure_engine_ready(tmp_path)
    assert emotion_engine_stub.recorded_calls == [("ensure_ready",)]
