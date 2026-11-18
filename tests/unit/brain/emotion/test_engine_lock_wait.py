"""Engine 6, HR5 and S107 (HS8): the rebuild process waits for the engine lock another process holds, and the commands' step.

Hana's review of #633, point 5: a rebuild holds the engine lock while another start-up begins, its 5 s lock wait runs out, and
`EngineStartupError("the engine lock", ...)` stops it. The owner's ruling (S101): the start-up waits until the rebuild is finished.
`ensure_engine_ready` is called only by the rebuild process (`nell supervisor rebuild`) now, so a lock that does not come within
`EMOTION_LOCK_WAIT_S` is tried again, one more step at a time, until it comes: no stop check, no limit. The bridge and the six
commands no longer call it: a command's `_engine_ready_or_refuse` registers the command and runs the rebuild step.

The second holder is real: the test holds `engine_files.state_lock` itself (flock locks belong to the open file description, so
a second open in the same thread conflicts) and the wait log releases it at a chosen wait step, so nothing sleeps beyond the wait
cut to 0.05 s and nothing runs in a thread.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import cli as brain_cli
from brain import dev_constants
from brain.bridge import daemon
from brain.emotion import engine_files, engine_rebuild
from brain.emotion.engine import EmotionEngine, EngineStartupError, ensure_engine_ready
from tests.unit.brain.emotion._engine_support import (
    NOW,
    OtherHolder,
    WaitLog,
    install,
    local_day,
    local_noon,
    lose_the_state,
    make_live,
    make_persona,
    row,
    state_of,
)

T = [local_noon(local_day(n)) for n in range(40)]
SWITCHED_BY_THE_HOLDER = T[30]
BUSY_STEP_LINE = "A start or an update of your memories is under way. Try again when it has finished."


@pytest.fixture
def persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A persona with one counted row, an engine with a fixed clock, and the lock wait cut from 5 s to 0.05 s."""
    path = make_persona(tmp_path, [row(T[35], emotions={"joy": 6.0}, mid="seed")])
    install(path, NOW)
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.05)
    return path


class _RebuildSpy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = engine_rebuild.rebuild

        def counted(*args: object, **kwargs: object) -> object:
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(engine_rebuild, "rebuild", counted)


def test_a_start_that_waits_on_a_held_lock_starts_when_it_frees_and_finds_the_holders_rebuild_done(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    holder = OtherHolder(persona)
    waits = WaitLog(monkeypatch, on_step={2: lambda: (make_live(persona, SWITCHED_BY_THE_HOLDER), holder.release())})
    try:
        ensure_engine_ready(persona)
    finally:
        holder.release()
    assert waits.steps == 2  # two steps ran out, the third took the lock
    assert len(waits.calls) == 2 and all(kwargs == {"exc_info": False} for _, kwargs in waits.calls)
    assert rebuilt.calls == 0  # the re-check under the lock saw the holder's switch-over
    assert state_of(persona).switch_over_at == SWITCHED_BY_THE_HOLDER


def test_a_start_that_waits_on_a_holder_that_failed_runs_the_rebuild_itself(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    holder = OtherHolder(persona)
    WaitLog(monkeypatch, on_step={1: holder.release})  # the holder lets go and has written nothing
    try:
        ensure_engine_ready(persona)
    finally:
        holder.release()
    assert rebuilt.calls == 1
    assert state_of(persona).switch_over_at == NOW


def test_a_lock_timeout_raised_inside_the_rebuild_is_not_waited_for(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.utils.file_lock import LockTimeoutError

    calls: list[int] = []

    def rebuild_that_times_out(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise LockTimeoutError("a second lock inside the rebuild did not come")

    monkeypatch.setattr(engine_rebuild, "rebuild", rebuild_that_times_out)
    WaitLog(monkeypatch, fail_on_wait=True)  # code that wrongly waited would loop for ever: the first wait step fails the test
    with pytest.raises(EngineStartupError) as failed:
        ensure_engine_ready(persona)
    assert failed.value.what == "the first rebuild"
    assert calls == [1]  # one rebuild, never retried


@pytest.mark.no_base_failure(reason="a retention pin: a lock file that cannot be opened (not a lock that did not come) stops start-up with the lock named, before and after the wait moved to a flag")
def test_a_lock_file_that_cannot_be_opened_is_not_waited_for_and_stops_start_up_naming_the_lock(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unopenable(persona_dir: Path) -> None:
        raise PermissionError("no access")

    monkeypatch.setattr(engine_files, "state_lock", unopenable)
    WaitLog(monkeypatch, fail_on_wait=True)  # only a lock that did not come is waited for, never a lock that cannot be opened
    with pytest.raises(EngineStartupError) as failed:
        ensure_engine_ready(persona)
    assert failed.value.what == "the engine lock"
    assert isinstance(failed.value.cause, PermissionError)


# --- the progress record while the rebuild waits for the lock (Hana's 11-18 review of #633, point 1; Planning's HRV answer) --


def _record(persona_dir: Path) -> dict | None:
    return engine_files.read_progress(persona_dir)


def _shows_this_process_rebuilding(record: dict | None) -> bool:
    return record is not None and record["phase"] == "rebuilding" and record["pid"] == os.getpid()


def test_a_first_start_shows_its_rebuilding_record_while_it_waits_for_the_lock_and_writes_it_again_at_the_next_step(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On an unswitched persona the x10 step removes the child's first record, and the lock is held by another process: the wait must show one."""
    rebuilt = _RebuildSpy(monkeypatch)
    holder = OtherHolder(persona)
    seen: dict[int, dict | None] = {}

    def first_step() -> None:
        seen[1] = _record(persona)
        engine_files.progress_path(persona).unlink()  # removed in the middle of the wait

    def second_step() -> None:
        seen[2] = _record(persona)
        make_live(persona, SWITCHED_BY_THE_HOLDER)
        holder.release()

    waits = WaitLog(monkeypatch, on_step={1: first_step, 2: second_step})
    try:
        ensure_engine_ready(persona)
    finally:
        holder.release()
    assert waits.steps == 2
    assert _shows_this_process_rebuilding(seen[1])
    assert _shows_this_process_rebuilding(seen[2])  # written again at the top of the next pass
    assert rebuilt.calls == 0  # needed() answered False under the lock
    assert _record(persona) is None  # and the record is gone with the function's exit


def test_a_first_start_that_waited_leaves_no_record_after_its_rebuild_ends(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    holder = OtherHolder(persona)
    seen: list[dict | None] = []
    WaitLog(monkeypatch, on_step={1: lambda: seen.append(_record(persona)), 2: holder.release})
    try:
        ensure_engine_ready(persona)
    finally:
        holder.release()
    assert len(seen) == 1 and _shows_this_process_rebuilding(seen[0])
    assert rebuilt.calls == 1 and state_of(persona).switch_over_at == NOW
    assert _record(persona) is None


def test_a_first_start_whose_needed_check_raises_after_the_wait_leaves_no_record(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit only the outer `finally` covers: `needed()` runs before the in-lock record is written."""
    holder = OtherHolder(persona)
    seen: list[dict | None] = []

    def needed_that_fails(self: object) -> bool:
        raise RuntimeError("the re-check failed")

    monkeypatch.setattr(EmotionEngine, "_first_rebuild_needed", needed_that_fails)
    WaitLog(monkeypatch, on_step={1: lambda: (seen.append(_record(persona)), holder.release())})
    try:
        with pytest.raises(EngineStartupError):
            ensure_engine_ready(persona)
    finally:
        holder.release()
    assert len(seen) == 1 and _shows_this_process_rebuilding(seen[0])
    assert _record(persona) is None


@pytest.mark.no_base_failure(reason="a retention pin: the supervisor job's recovery rebuild never writes a record, at its wait or otherwise; it passes on the base code by design")
def test_the_recovery_rebuild_shows_no_record_when_it_tries_the_lock(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_live(persona, SWITCHED_BY_THE_HOLDER)
    instance = install(persona, NOW)
    assert instance.is_live()
    lose_the_state(persona)
    real_lock = engine_files.state_lock
    seen: list[dict | None] = []

    def spying_lock(persona_dir: Path):  # noqa: ANN202
        seen.append(_record(persona_dir))
        return real_lock(persona_dir)

    holder = OtherHolder(persona)
    monkeypatch.setattr(engine_files, "state_lock", spying_lock)
    try:
        with pytest.raises(EngineStartupError):
            instance.recovery_rebuild()
    finally:
        holder.release()
    assert seen == [None]


@pytest.mark.no_base_failure(reason="a pin of the written condition: a recovery-mode rebuild that is not a roll-forward has no record even with wait_for_lock; reached only by a direct call of the private method, and it passes on the base code by design")
def test_a_recovery_mode_rebuild_that_is_not_a_roll_forward_writes_no_record_while_it_waits(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = install(persona, NOW)
    holder = OtherHolder(persona)
    seen: list[dict | None] = []
    WaitLog(monkeypatch, on_step={1: lambda: (seen.append(_record(persona)), holder.release())})
    try:
        instance._locked_rebuild("recovery", lambda: False, wait_for_lock=True)
    finally:
        holder.release()
    assert seen == [None]


# --- a command's rebuild step (HS7) ------------------------------------------------------------------------------------


def test_a_command_whose_persona_needs_no_rebuild_goes_on_while_the_engine_lock_is_held(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No rebuild is needed (the autouse fixture says so), so the command never touches the engine lock: a held lock is not a refusal."""
    rebuilt = _RebuildSpy(monkeypatch)
    with engine_files.state_lock(persona):
        assert brain_cli._engine_ready_or_refuse(persona, False) is None
    assert capsys.readouterr().err == ""
    assert rebuilt.calls == 0 and not engine_files.state_path(persona).exists()  # and it rebuilt nothing in-process


@pytest.mark.real_rebuild_step(reason="the helper's real step: it takes the start lock before it would spawn the child")
def test_a_command_is_refused_at_once_with_the_steps_line_when_a_start_holds_the_start_lock(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], rebuild_spawns
) -> None:
    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: None)  # no record left in the tmp persona
    holder = daemon.acquire_lock(persona)
    assert holder is not None
    try:
        refused = brain_cli._engine_ready_or_refuse(persona, False)
    finally:
        daemon.release_lock(persona, holder)
    assert refused == 2
    assert capsys.readouterr().err == BUSY_STEP_LINE + "\n"
    assert rebuild_spawns.calls == []  # nothing was run


@pytest.mark.real_rebuild_step(reason="nell chat's helper call reaches the real step, which a held start lock refuses")
def test_nell_chat_stops_on_a_refusal_and_goes_no_further(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(brain_cli, "get_persona_dir", lambda name: persona)
    monkeypatch.setattr(brain_cli, "_resolve_routing", lambda persona_dir, args: ("fake", "fake"))
    monkeypatch.setattr(brain_cli, "_start_floor_bootstrap_for_direct_chat", lambda persona_dir, *, blocking: None)
    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: None)
    reached: list[str] = []
    monkeypatch.setattr(brain_cli, "get_provider", lambda name: reached.append("provider") or pytest.fail("the provider was fetched"))
    args = SimpleNamespace(persona="p", message="hello", close=False, provider=None, model=None, dry_run=False)
    holder = daemon.acquire_lock(persona)
    assert holder is not None
    try:
        result = brain_cli._chat_direct_mode(args)
    finally:
        daemon.release_lock(persona, holder)
    assert result == 2
    assert capsys.readouterr().err == BUSY_STEP_LINE + "\n"
    assert reached == []
