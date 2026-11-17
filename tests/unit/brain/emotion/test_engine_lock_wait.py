"""Engine 6, HR5a: a bridge start-up waits for the engine lock another process holds for its rebuild, instead of stopping.

Hana's review of #633, point 5: a command-line rebuild holds the engine lock while the app starts, the bridge's 5 s lock wait
runs out, and `EngineStartupError("the engine lock", ...)` stops the bridge. The owner's ruling (S101): the start-up waits
until the rebuild is finished. The bridge lifespan passes `stop_requested`; with it a lock that does not come within
`EMOTION_LOCK_WAIT_S` is tried again, one more step at a time, until the lock comes or a stop is asked for. A command-line caller
passes nothing and keeps the one 5 s wait and the refusal (D2).

The second holder is real: the test holds `engine_files.state_lock` itself (flock locks belong to the open file description, so
a second open in the same thread conflicts, as in `test_a_lock_that_does_not_come_raises_naming_the_lock...`) and the stop check
releases it, so nothing sleeps beyond the wait cut to 0.05 s and nothing runs in a thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import cli as brain_cli
from brain import dev_constants
from brain.emotion import engine, engine_files, engine_rebuild
from brain.emotion.engine import EngineStartupError, ensure_engine_ready
from tests.unit.brain.emotion._engine_support import (
    NOW,
    install,
    local_day,
    local_noon,
    make_live,
    make_persona,
    row,
    state_of,
)

T = [local_noon(local_day(n)) for n in range(40)]
SWITCHED_BY_THE_HOLDER = T[30]


@pytest.fixture
def persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A persona with one counted row, an engine with a fixed clock, and the lock wait cut from 5 s to 0.05 s."""
    path = make_persona(tmp_path, [row(T[35], emotions={"joy": 6.0}, mid="seed")])
    install(path, NOW)
    monkeypatch.setattr(dev_constants, "EMOTION_LOCK_WAIT_S", 0.05)
    return path


class _OtherHolder:
    """The lock held the way another process's rebuild holds it, and a stop check that records every ask.

    `answers` is a list of what each ask returns, in order; after the list runs out every ask returns False. `on_ask` maps
    the number of an ask (1 is the first) to what the OTHER process does at that moment: the holder finishes or fails and lets go.
    """

    def __init__(self, persona: Path, *, answers: list[bool] | None = None, on_ask: dict[int, Callable[[], None]] | None = None) -> None:
        self.persona = persona
        self.asks = 0
        self._answers = list(answers or [])
        self._on_ask = on_ask or {}
        self._stack = ExitStack()
        self._stack.enter_context(engine_files.state_lock(persona))

    def release(self) -> None:
        self._stack.close()  # closing twice is a no-op

    def __call__(self) -> bool:
        self.asks += 1
        action = self._on_ask.get(self.asks)
        if action is not None:
            action()
        return self._answers.pop(0) if self._answers else False


class _RebuildSpy:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls = 0
        real = engine_rebuild.rebuild

        def counted(*args: object, **kwargs: object) -> object:
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(engine_rebuild, "rebuild", counted)


class _WaitLog:
    """The `warn_throttled` calls the wait makes (the throttle table is process-wide, so the log level proves nothing)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        real = engine.warn_throttled

        def spy(logger: logging.Logger, key: str, msg: str, *args: object, **kwargs: object) -> None:
            if "engine lock" in msg:
                self.calls.append((key, kwargs))
            real(logger, key, msg, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine, "warn_throttled", spy)


def test_a_start_that_waits_on_a_held_lock_starts_when_it_frees_and_finds_the_holders_rebuild_done(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    waits = _WaitLog(monkeypatch)
    holder = _OtherHolder(persona, on_ask={2: lambda: (make_live(persona, SWITCHED_BY_THE_HOLDER), holder.release())})
    ensure_engine_ready(persona, stop_requested=holder)
    assert holder.asks == 2  # two steps ran out, the third took the lock
    assert len(waits.calls) == 2 and all(kwargs == {"exc_info": False} for _, kwargs in waits.calls)
    assert rebuilt.calls == 0  # the re-check under the lock saw the holder's switch-over
    assert state_of(persona).switch_over_at == SWITCHED_BY_THE_HOLDER


def test_a_start_that_waits_on_a_holder_that_failed_runs_the_rebuild_itself(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    holder = _OtherHolder(persona, on_ask={1: lambda: holder.release()})  # the holder lets go and has written nothing
    ensure_engine_ready(persona, stop_requested=holder)
    assert rebuilt.calls == 1
    assert state_of(persona).switch_over_at == NOW


def test_a_stop_asked_for_during_the_wait_stops_start_up_with_a_named_reason_at_info_and_runs_nothing(
    persona: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    holder = _OtherHolder(persona, answers=[False, True])
    try:
        with caplog.at_level(logging.INFO, logger="brain.emotion.engine"), pytest.raises(EngineStartupError) as stopped:
            ensure_engine_ready(persona, stop_requested=holder)
    finally:
        holder.release()
    assert stopped.value.what == "the engine lock"
    assert "stopped while waiting for the rebuild" in str(stopped.value)
    assert holder.asks == 2  # no further step after the stop
    assert rebuilt.calls == 0
    assert not engine_files.state_path(persona).exists()
    engine_records = [r for r in caplog.records if r.name == "brain.emotion.engine"]
    assert not [r for r in engine_records if r.levelno >= logging.ERROR]  # a stop is not a failure: no error line, no traceback
    stop_lines = [r for r in engine_records if r.levelno == logging.INFO and "stopped while waiting" in r.getMessage()]
    assert len(stop_lines) == 1 and stop_lines[0].exc_info is None


def test_a_free_lock_is_taken_and_the_rebuild_runs_even_when_a_stop_is_already_asked_for(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rebuilt = _RebuildSpy(monkeypatch)
    asked: list[int] = []

    def stop_requested() -> bool:
        asked.append(1)
        return True

    ensure_engine_ready(persona, stop_requested=stop_requested)
    assert asked == []  # asked only after a step ran out
    assert rebuilt.calls == 1


def test_a_lock_timeout_raised_inside_the_rebuild_is_not_waited_for(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.utils.file_lock import LockTimeoutError

    calls: list[int] = []

    def rebuild_that_times_out(*_args: object, **_kwargs: object) -> None:
        calls.append(1)
        raise LockTimeoutError("a second lock inside the rebuild did not come")

    monkeypatch.setattr(engine_rebuild, "rebuild", rebuild_that_times_out)
    asked: list[int] = []
    with pytest.raises(EngineStartupError) as failed:
        # True, so that code which wrongly waited would stop at once (a False would make it loop for good)
        ensure_engine_ready(persona, stop_requested=lambda: asked.append(1) or True)
    assert failed.value.what == "the first rebuild"
    assert calls == [1] and asked == []  # one rebuild, never retried, the stop check never asked


BUSY_LINE = "The emotion engine is busy with another job. Try again when it finishes."


def test_a_command_line_caller_still_waits_one_step_and_refuses_and_passes_no_stop_check(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[dict[str, object]] = []
    real = engine.ensure_engine_ready

    def recording(persona_dir: Path, **kwargs: object) -> None:
        seen.append(kwargs)
        real(persona_dir, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(brain_cli.emotion_engine, "ensure_engine_ready", recording)
    rebuilt = _RebuildSpy(monkeypatch)
    with engine_files.state_lock(persona):
        refused = brain_cli._engine_ready_or_refuse(persona, False)
    assert refused == 2  # Hana's point 4 of #633: a lock that does not come is a refusal, not a crash
    assert capsys.readouterr().err == BUSY_LINE + "\n"
    assert seen == [{}]  # no stop_requested argument
    assert rebuilt.calls == 0


@pytest.mark.no_base_failure(reason="the positive control of the refusal tests: with the lock free the command goes on, as on the base code")
def test_a_command_goes_on_when_the_lock_comes(persona: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert brain_cli._engine_ready_or_refuse(persona, False) is None  # the same persona, nobody holds the lock: the first rebuild runs
    assert capsys.readouterr().err == ""
    assert engine_files.read_state(persona) is not None


def test_nell_chat_stops_on_a_refusal_and_goes_no_further(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(brain_cli, "get_persona_dir", lambda name: persona)
    monkeypatch.setattr(brain_cli, "_resolve_routing", lambda persona_dir, args: ("fake", "fake"))
    monkeypatch.setattr(brain_cli, "_start_floor_bootstrap_for_direct_chat", lambda persona_dir, *, blocking: None)
    reached: list[str] = []
    monkeypatch.setattr(brain_cli, "get_provider", lambda name: reached.append("provider") or pytest.fail("the provider was fetched"))
    args = SimpleNamespace(persona="p", message="hello", close=False, provider=None, model=None, dry_run=False)
    with engine_files.state_lock(persona):
        result = brain_cli._chat_direct_mode(args)
    assert result == 2
    assert capsys.readouterr().err == BUSY_LINE + "\n"
    assert reached == []


@pytest.mark.no_base_failure(reason="a retention pin: a start-up failure that is not a lock timeout raises out of the helper today and still does")
def test_another_start_up_failure_still_raises_out_of_the_helper(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise engine_rebuild.RebuildCheckError("boom")

    monkeypatch.setattr(engine_rebuild, "rebuild", failing)
    with pytest.raises(EngineStartupError) as failed:
        brain_cli._engine_ready_or_refuse(persona, False)
    assert failed.value.what == "the first rebuild"


@pytest.mark.no_base_failure(reason="a retention pin: a lock file that cannot be opened raises out of the helper today and still does")
def test_a_lock_file_that_cannot_be_opened_still_raises_out_of_the_helper(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unopenable(persona_dir: Path) -> None:
        raise PermissionError("no access")

    monkeypatch.setattr(engine_files, "state_lock", unopenable)
    with pytest.raises(EngineStartupError) as failed:
        brain_cli._engine_ready_or_refuse(persona, False)
    assert failed.value.what == "the engine lock"
    assert isinstance(failed.value.cause, PermissionError)
