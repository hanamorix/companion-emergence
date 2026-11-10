"""Engine 6, SW4 (Planning B31, owner S79 "option c"): a state that exists but cannot be read is decided from the database's x10 record.

`ensure_ready` on such a state runs the x10 steps (they read the progress number in `memories.db` themselves), then the
first-mode rebuild, which writes a fresh state with `switch_over_at` = now. Inside `rebuild`, before the write, the unreadable
state file and each unreadable backup are renamed aside to `<name>.corrupt-<UTC stamp>`. Personas are tmp directories with
synthetic rows; nothing here touches a real persona.
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
STAMP = NOW.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
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


def _damage(persona: Path) -> dict[str, bytes]:
    """Leave the state file and its two backups unreadable, each with its own bytes; return name -> bytes."""
    base = engine_files.state_path(persona)
    for _ in range(3):
        save(persona, simple_state(saved_at=T35))
    damaged: dict[str, bytes] = {}
    for suffix in ("", ".bak1", ".bak2"):
        target = base.with_name(base.name + suffix)
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


def _asides(persona: Path) -> dict[str, bytes]:
    folder = engine_files.state_path(persona).parent
    return {p.name: p.read_bytes() for p in folder.iterdir() if ".corrupt-" in p.name}


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
def test_an_unreadable_state_at_every_x10_progress_ends_with_a_fresh_state_and_nothing_multiplied_twice(tmp_path: Path, how: str) -> None:
    persona = _persona(tmp_path)
    _progress_world(persona, how)
    damaged = _damage(persona)
    instance = install(persona, NOW)
    ensure_engine_ready(persona)
    state = state_of(persona)
    assert state.switch_over_at == NOW and instance.is_live()
    assert _pride(persona, "made") == pytest.approx(4.0)  # 0.4 x 10 once, whatever the number said
    assert _queue_pride(persona) == pytest.approx(3.0)
    assert _progress(persona) == DONE
    assert _asides(persona) == {f"{name}.corrupt-{STAMP}": data for name, data in damaged.items()}
    ensure_engine_ready(persona)  # a second start multiplies nothing
    assert _pride(persona, "made") == pytest.approx(4.0) and _queue_pride(persona) == pytest.approx(3.0)


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


def test_the_damaged_copies_are_renamed_aside_not_deleted_and_a_name_already_taken_is_not_overwritten(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    damaged = _damage(persona)
    folder = engine_files.state_path(persona).parent
    taken = folder / f"{engine_files.state_path(persona).name}.corrupt-{STAMP}"
    taken.write_bytes(b"an older damaged copy")
    install(persona, NOW)
    ensure_engine_ready(persona)
    assert taken.read_bytes() == b"an older damaged copy"
    asides = _asides(persona)
    assert asides[taken.name] == b"an older damaged copy"
    assert asides[taken.name + "-1"] == damaged[engine_files.state_path(persona).name]
    assert asides[f"{engine_files.state_path(persona).name}.bak1.corrupt-{STAMP}"] == damaged[f"{engine_files.state_path(persona).name}.bak1"]
    assert not (folder / f"{engine_files.state_path(persona).name}.bak1").exists()  # no unreadable copy is left as a backup
    assert engine_files.read_state(persona).problem is None


@pytest.mark.no_base_failure(reason="a pin: the old code never moved any copy aside either; this keeps the new move-aside step from touching a readable state")
def test_a_readable_state_is_never_moved_aside(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    save(persona, simple_state(saved_at=T35, day=local_day(35).isoformat()))  # a shadow state: readable, no switch-over time
    install(persona, NOW)
    ensure_engine_ready(persona)
    assert _asides(persona) == {}


def test_a_crash_between_the_renames_and_the_write_leaves_the_state_missing_and_the_next_start_completes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    damaged = _damage(persona)
    install(persona, NOW)

    def crash(*args: object, **kwargs: object) -> bool:
        raise OSError("crash before the state write")

    monkeypatch.setattr(engine_files, "save_state", crash)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild"
    assert engine_files.read_state(persona).problem == "missing"  # every copy moved aside, nothing written
    assert len(_asides(persona)) == len(damaged)
    monkeypatch.undo()
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW
    assert _pride(persona, "made") == pytest.approx(4.0) and _queue_pride(persona) == pytest.approx(3.0)  # the x10 steps skipped as done
    assert len(_asides(persona)) == len(damaged)  # nothing more to move, nothing lost


def test_a_failed_rename_stops_start_up_with_the_copies_in_place_and_the_next_start_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _damage(persona)
    folder = engine_files.state_path(persona).parent
    before = {k: v for k, v in tree(folder).items() if k.startswith(engine_files.state_path(persona).name)}
    install(persona, NOW)

    real = engine_rebuild.file_ops.replace_file

    def refuse(src: object, dst: object) -> None:
        if ".corrupt-" in str(dst):
            raise OSError("rename refused")
        real(src, dst)  # the x10 queue replace goes through the same accessor

    monkeypatch.setattr(engine_rebuild.file_ops, "replace_file", refuse)
    with pytest.raises(EngineStartupError) as failure:
        ensure_engine_ready(persona)
    assert failure.value.what == "the first rebuild"
    state_name = engine_files.state_path(persona).name
    assert {k: v for k, v in tree(folder).items() if k.startswith(state_name)} == before  # (the emptied queue file is not a state copy)
    monkeypatch.undo()
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW


def test_a_rename_that_fails_after_the_state_file_moved_leaves_the_rest_for_the_next_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    damaged = _damage(persona)
    install(persona, NOW)
    real = engine_rebuild.file_ops.replace_file

    def refuse_backups(src: object, dst: object) -> None:
        if ".bak1.corrupt-" in str(dst):
            raise OSError("rename refused")
        real(src, dst)

    monkeypatch.setattr(engine_rebuild.file_ops, "replace_file", refuse_backups)
    with pytest.raises(EngineStartupError):
        ensure_engine_ready(persona)
    assert engine_files.read_state(persona).problem == "unreadable"  # the backups are still there and still unreadable
    monkeypatch.undo()
    ensure_engine_ready(persona)
    assert state_of(persona).switch_over_at == NOW
    assert sorted(_asides(persona).values()) == sorted(damaged.values())  # every damaged copy kept, none twice


def test_a_recovery_rebuild_of_an_unreadable_state_keeps_the_damaged_copies_aside_too(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    instance = install(persona, NOW)
    make_live(persona, T35)
    assert instance.is_live()
    damaged = _damage_live(persona)
    instance.recovery_rebuild()
    assert state_of(persona).switch_over_at == T35
    assert _asides(persona) == {f"{name}.corrupt-{STAMP}": data for name, data in damaged.items()}


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
