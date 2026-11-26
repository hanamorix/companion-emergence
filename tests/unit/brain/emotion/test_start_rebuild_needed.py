"""Engine 6, S107 step (a), HS2: the need check a starter asks before it runs the rebuild process.

`start_rebuild_needed(persona_dir)` answers True when the persona is not switched over, and otherwise whether a roll-forward is due
(`engine_rebuild.rolled_forward` on a fresh read of the state, at the engine's own clock). It takes no lock and writes nothing.
`EmotionEngine.ensure_ready` keeps its own branch; these tests keep the two in step: whatever the check says, the real
`ensure_engine_ready` does or does not run a rebuild. Every database is a real `memories.db` in a tmp directory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from brain.emotion import engine, engine_files, engine_rebuild
from brain.emotion.engine import ensure_engine_ready
from brain.emotion.engine_utils import day_of
from brain.memory.store import Memory, MemoryStore
from tests.unit.brain.emotion._engine_support import (
    NOW,
    LockSpy,
    install,
    local_day,
    local_noon,
    make_persona,
    row,
    save,
    simple_state,
    tree,
)

T = [local_noon(local_day(n)) for n in range(40)]
SWITCHED = T[10]
SAVED = T[30]


def _unswitched(tmp_path: Path) -> Path:
    """A persona with one counted row and no state file: never switched over."""
    persona = make_persona(tmp_path, [row(T[20], mid="seed")])
    install(persona, NOW)
    return persona


def _switched(tmp_path: Path) -> Path:
    """A switched persona whose state has seen its one row (`rows_through` = 1), saved at `SAVED`."""
    persona = make_persona(tmp_path, [row(T[20], mid="seed")])
    state = simple_state(saved_at=SAVED, day=day_of(SAVED), switch_over_at=SWITCHED)
    state.rows_through = 1
    save(persona, state)
    install(persona, NOW)
    return persona


def _add_row_after_the_save(persona: Path, mid: str = "rolled") -> None:
    """A row an older brain wrote after the save: above `rows_through`, created after `SAVED`, with counted feelings."""
    store = MemoryStore(persona / "memories.db")
    try:
        memory: Memory = row(T[34], emotions={"joy": 6.0}, mid=mid)
        store.create(memory, fold=False)
    finally:
        store.close()


class _RebuildCalls:
    """Records the modes of the real rebuild's calls and calls through."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.modes: list[object] = []
        real = engine_rebuild.rebuild

        def counted(*args: object, **kwargs: object) -> object:
            self.modes.append(args[1])
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine_rebuild, "rebuild", counted)


def test_an_unswitched_persona_needs_the_rebuild_and_ensure_ready_runs_the_first_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _unswitched(tmp_path)
    assert engine.start_rebuild_needed(persona) is True
    calls = _RebuildCalls(monkeypatch)
    ensure_engine_ready(persona)
    assert calls.modes == ["first"]
    assert engine.start_rebuild_needed(persona) is False  # switched now, nothing newer


def test_a_switched_persona_with_no_newer_rows_needs_nothing_and_ensure_ready_runs_no_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _switched(tmp_path)
    assert engine.start_rebuild_needed(persona) is False
    calls = _RebuildCalls(monkeypatch)
    ensure_engine_ready(persona)
    assert calls.modes == []


def test_a_switched_persona_with_rows_newer_than_the_save_needs_the_rebuild_and_ensure_ready_runs_the_recovery_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _switched(tmp_path)
    _add_row_after_the_save(persona)
    assert engine.start_rebuild_needed(persona) is True
    calls = _RebuildCalls(monkeypatch)
    ensure_engine_ready(persona)
    assert calls.modes == ["recovery"]


def test_the_check_takes_no_lock_and_writes_nothing_in_each_of_the_three_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = LockSpy(monkeypatch)  # every lock of any kind, whether taken through state_lock or file_lock directly
    # (name, builder, rows newer than the save added, the answer)
    cases = (("unswitched", _unswitched, False, True), ("switched", _switched, False, False), ("rolled-forward", _switched, True, True))
    for name, build, newer_rows, expected in cases:
        persona = build(tmp_path / name)
        if newer_rows:
            _add_row_after_the_save(persona)
        before = tree(persona)
        calls_before = spy.calls
        assert engine.start_rebuild_needed(persona) is expected
        assert spy.calls == calls_before, name  # no lock taken
        assert tree(persona) == before  # no state, no queue, no progress record, no folder
        assert engine_files.read_progress(persona) is None


def test_the_roll_forward_check_is_asked_at_the_engines_own_clock_on_a_fresh_read_of_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _switched(tmp_path)  # the installed engine's clock is NOW, not the machine's
    asked: list[tuple[object, object]] = []
    real = engine_rebuild.rolled_forward

    def spy(persona_dir: Path, loaded: engine_files.LoadedState, now: object) -> bool:
        asked.append((loaded.source, now))
        return real(persona_dir, loaded, now)  # type: ignore[arg-type]

    monkeypatch.setattr(engine_rebuild, "rolled_forward", spy)
    assert engine.start_rebuild_needed(persona) is False
    assert asked == [("main", NOW)]


def test_a_failed_check_read_raises_out_of_the_need_check_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _switched(tmp_path)

    def broken(*args: object, **kwargs: object) -> object:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(engine_rebuild, "_read_window", broken)
    with pytest.raises(sqlite3.OperationalError):
        engine.start_rebuild_needed(persona)
