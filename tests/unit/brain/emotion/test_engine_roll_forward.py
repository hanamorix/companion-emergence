"""Engine 6, HR1 step (b): the start check for a roll-forward, and the rebuild it runs (Hana's point 1 on #633).

A persona rolled back to an older brain after the switch-over and then rolled forward again has memory rows the engine never
saw. At start-up a SWITCHED persona whose state came from the main file and holds `rows_through`, `saved_at` and
`switch_over_at` is checked: a row above `rows_through` whose `created_at` is after `saved_at` and whose stored feelings count
means a roll-forward, and the existing locked rebuild runs in its "recovery" mode (30-day window, faded values, the original
`switch_over_at` carried) with the first rebuild's lock wait, stop handling and progress record. Every database here is a real
`memories.db` in a tmp directory; the rebuild itself is the real one except where a test makes it fail.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.emotion import engine, engine_files, engine_rebuild
from brain.emotion.engine import EmotionEngine, EngineStartupError, ensure_engine_ready
from brain.emotion.engine_utils import EngineState, day_of
from brain.memory.store import Memory, MemoryStore
from brain.utils.file_lock import LockTimeoutError
from tests.unit.brain.emotion._engine_support import (
    NOW,
    install,
    local_day,
    local_noon,
    lose_the_state,
    make_persona,
    row,
    save,
    save_as_backup,
    simple_state,
    state_of,
)

T = [local_noon(local_day(n)) for n in range(40)]
SWITCHED = T[10]  # the persona's original switch-over time, which a roll-forward must keep
SAVED = T[30]  # the engine's last save before the rollback


@pytest.fixture(autouse=True)
def _short_lock_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.05)


def _persona(
    tmp_path: Path,
    *,
    rows_through: int | None = 1,
    saved_at: datetime = SAVED,
    switched: datetime | None = SWITCHED,
    backup: bool = False,
) -> Path:
    """A switched persona: `memories.db` holds one old counted row (rowid 1) and a state that has seen it."""
    persona = make_persona(tmp_path, [row(T[20], mid="seed")])
    state = simple_state(saved_at=saved_at, day=day_of(saved_at), switch_over_at=switched)
    state.rows_through = rows_through
    (save_as_backup if backup else save)(persona, state)
    install(persona, NOW)
    return persona


def _add(persona: Path, memory: Memory) -> int:
    """Write a row the engine is not told about (as an older brain would) and return its rowid."""
    store = MemoryStore(persona / "memories.db")
    try:
        store.create(memory, fold=False)
        return int(store._conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory.id,)).fetchone()[0])
    finally:
        store.close()


def _new_counted_row(persona: Path, mid: str = "rolled") -> int:
    """A row a roll-forward leaves: above `rows_through`, created after `SAVED`, with countable stored feelings."""
    return _add(persona, row(T[34], emotions={"joy": 6.0}, mid=mid))


def _a_real_roll_forward_runs(tmp_path: Path, spy: _RebuildSpy) -> None:
    """The positive control of a "does nothing" test: a sibling persona that really rolled forward runs the rebuild.

    A negative test alone passes on code that never runs the check; this makes it bite there.
    """
    control = _persona(tmp_path.parent / f"{tmp_path.name}-control")
    _new_counted_row(control)
    before = len(spy.calls)
    ensure_engine_ready(control)
    assert len(spy.calls) == before + 1


class _RebuildSpy:
    """Calls through to the real rebuild and records the call, and what the progress record held while it ran."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, persona: Path) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.records: list[dict | None] = []
        real = engine_rebuild.rebuild

        def counted(*args: object, **kwargs: object) -> object:
            self.calls.append((args, kwargs))
            self.records.append(engine_files.read_progress(persona))
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine_rebuild, "rebuild", counted)


class _WindowSpy:
    """Counts the window reads (the check's read is the only one a switched start makes) and can fail chosen ones."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail_on: set[int] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._fail_on = fail_on or set()
        real = engine_rebuild._read_window

        def counted(persona_dir: Path, since: datetime, **kwargs: object) -> object:
            self.calls.append(kwargs)
            if len(self.calls) in self._fail_on:
                raise sqlite3.OperationalError("disk I/O error")
            return real(persona_dir, since, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine_rebuild, "_read_window", counted)


class _Logged:
    """The `warn_throttled` calls of the engine, formatted (the throttle table is process-wide, so the level proves nothing)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.lines: list[str] = []
        real = engine.warn_throttled

        def spy(logger: logging.Logger, key: str, msg: str, *args: object, **kwargs: object) -> None:
            self.lines.append(msg % args)
            real(logger, key, msg, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine, "warn_throttled", spy)

    def roll_forward(self) -> list[str]:
        return [line for line in self.lines if "roll-forward" in line]


class _OtherHolder:
    """The engine lock held the way another process's rebuild holds it; a stop check that records every ask."""

    def __init__(self, persona: Path, *, answers: list[bool] | None = None, on_ask: dict[int, Callable[[], None]] | None = None) -> None:
        self.asks = 0
        self._answers = list(answers or [])
        self._on_ask = on_ask or {}
        self._stack = ExitStack()
        self._stack.enter_context(engine_files.state_lock(persona))

    def release(self) -> None:
        self._stack.close()

    def __call__(self) -> bool:
        self.asks += 1
        assert self.asks < 50, "the wait loop never ended"  # a bounded fallback: a broken loop fails instead of hanging
        action = self._on_ask.get(self.asks)
        if action is not None:
            action()
        return self._answers.pop(0) if self._answers else False


def _loaded(persona: Path) -> engine_files.LoadedState:
    return engine_files.read_state(persona)


# --- the start check and the rebuild it runs ------------------------------------------------------------------------


def test_a_roll_forward_runs_the_recovery_rebuild_at_start_up_keeps_the_switch_over_time_and_shows_the_progress_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    rowid = _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    ensure_engine_ready(persona)
    assert len(spy.calls) == 1
    (args, kwargs) = spy.calls[0]
    assert args[1] == "recovery"  # the existing mode, no new one
    assert kwargs["on_progress"] is not None  # the first rebuild's progress display
    assert spy.records[0] is not None and spy.records[0]["phase"] == "rebuilding"
    assert engine_files.read_progress(persona) is None  # removed when the rebuild ends
    rebuilt = state_of(persona)
    assert rebuilt.switch_over_at == SWITCHED  # the original switch-over time, so the fade clock is not restarted
    assert rebuilt.saved_at == NOW
    assert rebuilt.rows_through == rowid
    ensure_engine_ready(persona)  # the next start finds nothing newer
    assert len(spy.calls) == 1


def test_a_state_read_from_a_backup_copy_does_not_run_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, backup=True)
    _new_counted_row(persona)
    assert _loaded(persona).source != "main"
    spy = _RebuildSpy(monkeypatch, persona)
    window = _WindowSpy(monkeypatch)
    ensure_engine_ready(persona)
    assert spy.calls == [] and window.calls == []  # the drain's recovery-first catch-up owns a backup-sourced state
    _a_real_roll_forward_runs(tmp_path, spy)


def test_a_row_the_engine_has_seen_refreshed_to_a_newer_time_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path, rows_through=1)
    store = MemoryStore(persona / "memories.db")
    try:  # the temp gate's write-back: the seen row (rowid 1) now reads as created after the save
        store._conn.execute("UPDATE memories SET created_at = ? WHERE id = 'seed'", (T[35].isoformat(),))
        store._conn.commit()
    finally:
        store.close()
    spy = _RebuildSpy(monkeypatch, persona)
    ensure_engine_ready(persona)
    assert spy.calls == []
    _a_real_roll_forward_runs(tmp_path, spy)


def test_a_row_above_rows_through_that_was_created_before_the_last_save_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    # a restore or import keeps its own older time; it was opened after the save, so the window read returns it
    _add(persona, row(T[25], accessed=T[35], emotions={"joy": 6.0}, mid="old-dated"))
    spy = _RebuildSpy(monkeypatch, persona)
    window = _WindowSpy(monkeypatch)
    ensure_engine_ready(persona)
    assert spy.calls == []
    assert len(window.calls) == 1  # the check did read it, and said no
    _a_real_roll_forward_runs(tmp_path, spy)


@pytest.mark.parametrize(
    "newer",
    [
        pytest.param(lambda: row(T[34], emotions={}, mid="empty"), id="no feelings"),
        pytest.param(lambda: row(T[34], emotions={"joy": 6.0}, memory_type="dream", mid="uncounted"), id="a kind that does not count"),
    ],
)
def test_a_newer_row_with_no_countable_feelings_does_nothing(
    newer: Callable[[], Memory], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    _add(persona, newer())
    spy = _RebuildSpy(monkeypatch, persona)
    ensure_engine_ready(persona)
    assert spy.calls == []
    _a_real_roll_forward_runs(tmp_path, spy)


def test_the_read_is_the_windows_own_reader_with_the_rowid_filter_and_since_is_the_later_of_saved_at_and_30_days_ago(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path, rows_through=1)
    seen: list[tuple[datetime, int]] = []
    real = MemoryStore.list_engine_window

    def recording(self: MemoryStore, since: datetime, *, rowid_above: int | None = None) -> list[Memory]:
        seen.append((since, rowid_above))  # type: ignore[arg-type]
        return real(self, since, rowid_above=rowid_above)

    monkeypatch.setattr(MemoryStore, "list_engine_window", recording)
    ensure_engine_ready(persona)
    assert seen == [(SAVED, 1)]  # SAVED is later than NOW minus 30 days

    old = tmp_path / "old"
    persona_old = _persona(old, saved_at=T[0], switched=T[0])  # saved 40 days before NOW: the 30-day bound is later
    seen.clear()
    ensure_engine_ready(persona_old)
    assert len(seen) == 1 and seen[0][0] == NOW - timedelta(days=dev_constants.EMOTION_LONG_HALF_LIFE_D)


@pytest.mark.parametrize("missing", ["rows_through", "switch_over_at"])
def test_a_state_without_rows_through_or_switch_over_at_skips_the_check(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path, rows_through=None if missing == "rows_through" else 1)
    if missing == "switch_over_at":
        # a switched persona whose loaded state lost the field cannot be told apart from the engine's own seen value, so pin the helper
        loaded = _loaded(persona)
        assert loaded.state is not None
        loaded.state.switch_over_at = None
        _new_counted_row(persona)
        window = _WindowSpy(monkeypatch)
        assert engine_rebuild.rolled_forward(persona, loaded, NOW) is False
        assert window.calls == []
        return
    _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    window = _WindowSpy(monkeypatch)
    ensure_engine_ready(persona)
    assert spy.calls == [] and window.calls == []
    _a_real_roll_forward_runs(tmp_path, spy)


def test_a_state_without_saved_at_or_no_state_at_all_skips_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    window = _WindowSpy(monkeypatch)
    no_saved_at = engine_files.LoadedState(None, EngineState(rows_through=1, switch_over_at=SWITCHED), "main", None)
    no_state = engine_files.LoadedState(None, None, None, "missing")
    assert engine_rebuild.rolled_forward(persona, no_saved_at, NOW) is False
    assert engine_rebuild.rolled_forward(persona, no_state, NOW) is False
    assert window.calls == []
    # and the same state with every field is met, so the three Falses above are the skip and not the data
    assert engine_rebuild.rolled_forward(persona, _loaded(persona), NOW) is True


def test_a_missing_memories_db_skips_the_check_and_creates_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    (persona / "memories.db").unlink()
    spy = _RebuildSpy(monkeypatch, persona)
    ensure_engine_ready(persona)
    assert spy.calls == []
    assert not (persona / "memories.db").exists()  # opening a store on a missing path would have created it
    _a_real_roll_forward_runs(tmp_path, spy)


# --- failures: the check's read, the rebuild's own work ---------------------------------------------------------------


def test_a_failing_check_read_logs_and_start_up_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    before = engine_files.state_path(persona).read_bytes()
    spy = _RebuildSpy(monkeypatch, persona)
    window = _WindowSpy(monkeypatch, fail_on={1})
    logged = _Logged(monkeypatch)
    ensure_engine_ready(persona)  # does not raise
    assert len(window.calls) == 1 and spy.calls == []
    (line,) = logged.roll_forward()
    assert "failed" in line and "OperationalError" in line and "disk I/O error" in line and "next start tries again" in line
    assert engine_files.state_path(persona).read_bytes() == before
    ensure_engine_ready(persona)  # the next start tries again, and the read works now
    assert len(spy.calls) == 1


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(engine_rebuild.RebuildCheckError("an average is out of range"), id="check"),
        pytest.param(engine_rebuild.RebuildStateWriteError(Path("emotion_state.json"), OSError(28, "No space left")), id="state write"),
        pytest.param(OSError(5, "Input/output error"), id="os"),
        pytest.param(LockTimeoutError("a second lock inside the rebuild did not come"), id="a lock timeout inside the rebuild"),
    ],
)
def test_an_error_of_the_rebuilds_own_work_logs_and_start_up_goes_on_with_the_loaded_state(
    error: Exception, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    before = engine_files.state_path(persona).read_bytes()
    calls: list[int] = []

    def failing(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise error

    monkeypatch.setattr(engine_rebuild, "rebuild", failing)
    logged = _Logged(monkeypatch)
    asked: list[int] = []
    ensure_engine_ready(persona, stop_requested=lambda: asked.append(1) or True)  # never retried, never waited for
    assert calls == [1] and asked == []
    (line,) = logged.roll_forward()
    assert "failed" in line and "next start tries again" in line
    assert engine_files.state_path(persona).read_bytes() == before
    assert engine_files.read_progress(persona) is None  # the record is removed on a failure too


def test_a_failed_write_of_the_real_rebuild_leaves_the_loaded_state_as_it_was_and_start_up_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    before = engine_files.state_path(persona).read_bytes()
    saves: list[int] = []

    def full_disk(*_args: object, **_kwargs: object) -> bool:
        saves.append(1)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(engine_files, "save_state", full_disk)
    logged = _Logged(monkeypatch)
    ensure_engine_ready(persona)  # the real rebuild ran up to its write, which failed: does not raise
    assert saves == [1]
    (line,) = logged.roll_forward()
    assert "RebuildStateWriteError" in line and "No space left" in line
    assert engine_files.state_path(persona).read_bytes() == before
    assert engine_files.read_progress(persona) is None


def test_a_row_created_at_the_very_instant_of_the_last_save_is_not_newer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _add(persona, row(SAVED, emotions={"joy": 6.0}, mid="same-instant"))  # strictly after is the rule
    spy = _RebuildSpy(monkeypatch, persona)
    ensure_engine_ready(persona)
    assert spy.calls == []
    _a_real_roll_forward_runs(tmp_path, spy)


def test_an_error_that_is_not_the_rebuilds_own_work_stops_start_up_like_the_first_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)

    def failing(*_args: object, **_kwargs: object) -> None:
        raise engine_rebuild.CatchUpError("the database vanished between the check and the rebuild")

    monkeypatch.setattr(engine_rebuild, "rebuild", failing)
    with pytest.raises(EngineStartupError) as failed:
        ensure_engine_ready(persona)
    assert failed.value.what == "the roll-forward rebuild"


def test_the_recheck_under_the_lock_drops_a_row_applied_while_this_start_waited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    rowid = _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)

    def another_process_applied_it() -> None:
        later = simple_state(saved_at=NOW, day=day_of(NOW), switch_over_at=SWITCHED)
        later.rows_through = rowid
        save(persona, later)
        holder.release()

    holder = _OtherHolder(persona, on_ask={1: another_process_applied_it})
    ensure_engine_ready(persona, stop_requested=holder)
    assert holder.asks == 1
    assert spy.calls == []  # the pre-lock check said yes, the re-check under the lock said no
    assert state_of(persona).rows_through == rowid


def test_a_failing_recheck_read_under_the_lock_logs_and_runs_no_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    window = _WindowSpy(monkeypatch, fail_on={2})
    logged = _Logged(monkeypatch)
    ensure_engine_ready(persona)
    assert len(window.calls) == 2 and spy.calls == []
    assert len(logged.roll_forward()) == 1


# --- the lock wait and the stop --------------------------------------------------------------------------------------


def test_a_stop_asked_for_during_the_roll_forward_lock_wait_takes_the_normal_stop_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    holder = _OtherHolder(persona, answers=[False, True])
    try:
        with caplog.at_level(logging.INFO, logger="brain.emotion.engine"), pytest.raises(EngineStartupError) as stopped:
            ensure_engine_ready(persona, stop_requested=holder)
    finally:
        holder.release()
    assert stopped.value.what == "the engine lock"
    assert "stopped while waiting for the rebuild" in str(stopped.value)
    assert holder.asks == 2
    assert spy.calls == []
    engine_records = [r for r in caplog.records if r.name == "brain.emotion.engine"]
    assert not [r for r in engine_records if r.levelno >= logging.ERROR]  # a stop is not a failure


def test_a_lock_that_frees_during_the_wait_lets_the_roll_forward_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    holder = _OtherHolder(persona, on_ask={2: lambda: holder.release()})
    ensure_engine_ready(persona, stop_requested=holder)
    assert holder.asks == 2 and len(spy.calls) == 1


def test_a_command_line_caller_refuses_after_the_one_wait_when_the_lock_is_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persona = _persona(tmp_path)
    _new_counted_row(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    with engine_files.state_lock(persona), pytest.raises(EngineStartupError) as refused:
        ensure_engine_ready(persona)  # no stop_requested: the command line's one wait, then the refusal
    assert refused.value.what == "the engine lock"
    assert spy.calls == []


# --- what the roll-forward must not change ---------------------------------------------------------------------------


@pytest.mark.no_base_failure(reason="a retention pin: the supervisor job's recovery rebuild keeps writing no progress record and raising silently, as on the base code")
def test_a_plain_recovery_rebuild_still_writes_no_progress_record_and_raises_its_failure_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    persona = _persona(tmp_path)
    instance = install(persona, NOW)
    assert isinstance(instance, EmotionEngine) and instance.is_live()
    lose_the_state(persona)
    spy = _RebuildSpy(monkeypatch, persona)
    instance.recovery_rebuild()
    assert len(spy.calls) == 1
    (_args, kwargs) = spy.calls[0]
    assert kwargs["on_progress"] is None and spy.records == [None]

    def failing(*_args: object, **_kwargs: object) -> None:
        raise engine_rebuild.RebuildCheckError("an average is out of range")

    monkeypatch.setattr(engine_rebuild, "rebuild", failing)
    lose_the_state(persona)
    with pytest.raises(EngineStartupError) as failed:
        instance.recovery_rebuild()
    assert failed.value.what == "the recovery rebuild"


def test_a_roll_forward_is_only_ever_a_recovery_mode_rebuild(tmp_path: Path) -> None:
    persona = _persona(tmp_path)
    instance = install(persona, NOW)
    with pytest.raises(ValueError, match="roll_forward"):
        instance._locked_rebuild("first", lambda: True, roll_forward=True)
