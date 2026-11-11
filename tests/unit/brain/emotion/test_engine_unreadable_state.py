"""Engine 6, SW4 and RD2 (Planning B31 owner S79 "option c"; B36 owner S88): a state that exists but cannot be read is decided from the database's x10 record, then simply replaced.

`ensure_ready` on such a state runs the x10 steps (they read the progress number in `memories.db` themselves), then the
first-mode rebuild, which writes a fresh state with `switch_over_at` = now through `save_state` and its normal backup rotation.
Nothing is renamed aside or deleted. When that write fails with an I/O error, this start is skipped and retried at the next
one. Personas are tmp directories with synthetic rows; nothing here touches a real persona.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine_files, engine_rebuild, nudge_scale_fix
from brain.emotion.engine import EngineStartupError, ensure_engine_ready
from brain.health.anomaly import BrainIntegrityError
from brain.memory import pending
from brain.memory.store import MemoryStore
from tests.unit.brain.emotion._engine_support import (
    NOW,
    LockSpy,
    install,
    local_day,
    local_noon,
    make_live,
    make_persona,
    row,
    save,
    simple_state,
    state_of,
    tree,
)

T35 = local_noon(local_day(35))
DONE = dev_constants.NUDGE_SCALE_FIX_PROGRESS_QUEUE_STARTED  # 2: the queue step started (the x10 fix's last number)
ROWS_DONE = dev_constants.NUDGE_SCALE_FIX_PROGRESS_ROWS_DONE
QUEUE = "pending_candidates.jsonl"


def _persona(root: Path) -> Path:
    """A small `making` row in the database (0.4) and a small `making` candidate waiting in the queue (0.3)."""
    persona = make_persona(
        root,
        [row(T35, emotions={"joy": 6.0}, mid="old"), row(T35, memory_type="making", emotions={"pride": 0.4}, mid="made")],
    )
    pending.PendingQueue(persona).enqueue(row(T35, memory_type="making", emotions={"pride": 0.3}, mid="waiting"), source="maker")
    return persona


def _copy(persona: Path, suffix: str) -> Path:
    base = engine_files.state_path(persona)
    return base.with_name(base.name + suffix)


def _damage(persona: Path, suffixes: tuple[str, ...] = ("", ".bak1", ".bak2")) -> dict[str, bytes]:
    """Leave the state file and its backups unreadable, each with its own bytes; return name -> bytes."""
    for _ in range(4):
        save(persona, simple_state(saved_at=T35))
    damaged: dict[str, bytes] = {}
    for suffix in suffixes:
        target = _copy(persona, suffix)
        assert target.exists()
        target.write_bytes(f"damaged{suffix}".encode())
        damaged[target.name] = target.read_bytes()
    return damaged


def _progress(persona: Path) -> int:
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        return store.x10_progress()
    finally:
        store.close()


def _pride(persona: Path, memory_id: str) -> float:
    store = MemoryStore(persona / "memories.db", integrity_check=False)
    try:
        found = store.get(memory_id, bump=False)
        assert found is not None
        return found.emotions["pride"]
    finally:
        store.close()


def _queue_pride(persona: Path) -> float:
    (entry,) = pending.read_jsonl(persona / QUEUE)
    return entry["emotions"]["pride"]


def _progress_world(persona: Path, how: str) -> None:
    """Leave the x10 fix at the stage `how` names (state still missing), the way an earlier start-up could have."""
    if how == "0":
        return
    if how == "1":
        store = MemoryStore(persona / "memories.db")
        try:
            store.scale_up_small_nudge_rows()
        finally:
            store.close()
        return
    nudge_scale_fix.scale_up_small_feelings(persona)  # runs to 2 (a crash after setting 2 has its own test)


@pytest.mark.parametrize("how", ["0", "1", "2"])
def test_an_unreadable_state_at_every_x10_progress_is_replaced_by_a_fresh_state_and_nothing_multiplied_twice(tmp_path: Path, how: str) -> None:
    persona = _persona(tmp_path)
    _progress_world(persona, how)
    damaged = _damage(persona)
    instance = install(persona, NOW)
    ensure_engine_ready(persona)
    state = state_of(persona)
    assert state.switch_over_at == NOW and instance.is_live()
    assert engine_files.read_state(persona).problem is None  # the new state is written and readable
    assert _copy(persona, ".bak1").read_bytes() == damaged[_copy(persona, "").name]  # the normal rotation moved the old main file
    assert _pride(persona, "made") == pytest.approx(4.0)  # 0.4 x 10 once, whatever the number said
    assert _queue_pride(persona) == pytest.approx(3.0)
    assert _progress(persona) == DONE
    ensure_engine_ready(persona)  # a second start multiplies nothing
    assert _pride(persona, "made") == pytest.approx(4.0) and _queue_pride(persona) == pytest.approx(3.0)


def test_nothing_is_renamed_aside_or_deleted_the_old_copies_only_age_out_through_the_normal_rotation(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    damaged = _damage(persona)
    folder = engine_files.state_path(persona).parent
    install(persona, NOW)
    ensure_engine_ready(persona)
    names = sorted(p.name for p in folder.iterdir() if p.name.startswith(engine_files.state_path(persona).name) and p.suffix != ".lock")
    state_name = engine_files.state_path(persona).name
    assert names == [state_name, f"{state_name}.bak1", f"{state_name}.bak2", f"{state_name}.bak3"]  # no `.corrupt-` name, nothing extra
    assert _copy(persona, ".bak2").read_bytes() == damaged[f"{state_name}.bak1"]  # each old copy moved one place down
    assert _copy(persona, ".bak3").read_bytes() == damaged[f"{state_name}.bak2"]


def test_a_failed_queue_write_after_the_number_was_set_to_2_leaves_the_entry_at_the_old_scale_and_is_never_multiplied_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner S84 accepted edge (X10S): 2 means "queue step started", so the entries of a file whose write failed are not retried."""
    persona = _persona(tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise OSError("crash before the write")

    monkeypatch.setattr(pending, "write_jsonl_bytes", refuse)
    nudge_scale_fix.scale_up_small_feelings(persona)  # the failed write is logged and the step goes on: progress is 2, the entry unmultiplied
    monkeypatch.undo()
    assert _progress(persona) == DONE and _queue_pride(persona) == pytest.approx(0.3)
    _damage(persona)
    install(persona, NOW)
    ensure_engine_ready(persona)
    assert _queue_pride(persona) == pytest.approx(0.3) and _pride(persona, "made") == pytest.approx(4.0)  # still at the old scale
    assert state_of(persona).switch_over_at == NOW


def _skip_world(persona: Path, how: str) -> Path:
    """Make the state write fail with an I/O error on an unreadable state; return what to remove to clear the fault."""
    if how == "directory at the live path":
        _damage(persona, suffixes=(".bak1", ".bak2", ".bak3"))
        live = _copy(persona, "")
        live.unlink()
        live.mkdir()
        return live
    _damage(persona, suffixes=("", ".bak1", ".bak2"))  # the rotation unlinks the oldest backup, `.bak3`
    oldest = _copy(persona, ".bak3")
    oldest.unlink()
    oldest.mkdir()
    return oldest


@pytest.mark.parametrize("how", ["directory at the live path", "directory at .bak3"])
def test_an_io_fault_on_the_rebuilds_state_write_skips_this_start_logs_one_warning_and_the_next_start_completes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    persona = _persona(tmp_path)
    fault = _skip_world(persona, how)
    assert engine_files.read_state(persona).problem == "unreadable"
    instance = install(persona, NOW)
    caplog.set_level(logging.WARNING)
    ensure_engine_ready(persona)  # returns: the start is skipped, not failed
    lines = [r.getMessage() for r in caplog.records if r.name == "brain.emotion.engine" and r.levelno == logging.WARNING]
    assert len(lines) == 1 and engine_files.state_path(persona).name in lines[0] and "next start" in lines[0]
    assert str(engine_files.state_path(persona)) in lines[0]
    assert not instance.is_live() and engine_files.read_state(persona).problem == "unreadable"  # not switched over until the fault clears
    fault.rmdir()
    ensure_engine_ready(persona)  # the next start tries again and completes
    assert state_of(persona).switch_over_at == NOW and instance.is_live()
    assert _pride(persona, "made") == pytest.approx(4.0) and _queue_pride(persona) == pytest.approx(3.0)  # the x10 steps ran once


@pytest.mark.no_base_failure(reason="a pin: the old code already stopped start-up on a failed check; this keeps the new I/O-fault skip from covering it")
def test_the_skip_covers_only_an_io_fault_of_the_state_write_not_a_failed_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _damage(persona)
    install(persona, NOW)
    monkeypatch.setattr(engine_files, "save_state", lambda *args, **kwargs: False)  # the write "did not verify": a failed check
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild" and isinstance(failure.value.cause, engine_rebuild.RebuildCheckError)


def test_an_io_fault_on_the_state_write_still_stops_start_up_when_the_state_was_not_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)  # no state at all: missing, not unreadable

    def crash(*args: object, **kwargs: object) -> bool:
        raise OSError("disk fault on the state write")

    install(persona, NOW)
    monkeypatch.setattr(engine_files, "save_state", crash)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild" and isinstance(failure.value.cause, engine_rebuild.RebuildStateWriteError)
    assert isinstance(failure.value.cause.cause, OSError)


def test_a_recovery_rebuild_of_an_unreadable_state_replaces_it_and_an_io_fault_there_still_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    instance = install(persona, NOW)
    make_live(persona, T35)
    assert instance.is_live()
    _damage_live(persona)
    live = _copy(persona, "")
    live.unlink()
    live.mkdir()  # an I/O fault at the live path: recovery mode is not skipped, it raises (the recovery job retries)
    with pytest.raises(EngineStartupError) as failure:
        instance.recovery_rebuild()
    assert failure.value.what == "the recovery rebuild"
    live.rmdir()
    _copy(persona, "").write_bytes(b"damaged again")
    instance.recovery_rebuild()
    assert state_of(persona).switch_over_at == T35 and engine_files.read_state(persona).problem is None


def _damage_live(persona: Path) -> dict[str, bytes]:
    base = engine_files.state_path(persona)
    state = state_of(persona)
    save(persona, state)
    damaged: dict[str, bytes] = {}
    for suffix in ("", ".bak1"):  # make_live wrote one copy and the save above rotated it into .bak1
        target = base.with_name(base.name + suffix)
        target.write_bytes(f"damaged{suffix}".encode())
        damaged[target.name] = target.read_bytes()
    return damaged


def test_a_database_that_fails_its_check_on_an_unreadable_state_skips_this_start_logs_why_and_the_next_start_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona = _persona(tmp_path)
    _damage(persona)
    install(persona, NOW)
    before = tree(persona)
    real = nudge_scale_fix.MemoryStore

    def broken(*args: object, **kwargs: object) -> MemoryStore:
        raise BrainIntegrityError(str(persona / "memories.db"), "page 7 is damaged")

    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", broken)
    spy = LockSpy(monkeypatch)
    caplog.set_level(logging.WARNING)
    ensure_engine_ready(persona)  # returns: the start is skipped, not failed
    lines = [r.getMessage() for r in caplog.records if r.name == "brain.emotion.engine" and r.levelno == logging.WARNING]
    assert len(lines) == 1 and "memories.db" in lines[0] and "next start" in lines[0]
    assert spy.calls == 0 and tree(persona) == before
    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", real)
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW


@pytest.mark.no_base_failure(reason="a pin: the old code already stopped start-up on an integrity failure; this keeps the new unreadable-state skip from swallowing it for a missing state")
def test_a_database_that_fails_its_check_on_a_state_that_is_not_unreadable_still_stops_start_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)  # no state at all: missing, not unreadable

    def broken(*args: object, **kwargs: object) -> MemoryStore:
        raise BrainIntegrityError(str(persona / "memories.db"), "page 7 is damaged")

    monkeypatch.setattr(nudge_scale_fix, "MemoryStore", broken)
    install(persona, NOW)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the memories.db integrity check"
