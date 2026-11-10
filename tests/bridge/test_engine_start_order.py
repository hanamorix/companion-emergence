"""Engine 6, G2: where `ensure_engine_ready` runs at start-up, and what a dry run does.

The bridge lifespan calls it between the vocabulary load and the engine's start-up drain, and a failure stops start-up before the
supervisor thread is spawned. Six `nell` commands (chat in-process, heartbeat, dream, reflex, research, soul review) call it
through one helper right after their own vocabulary load; five of them take `--dry-run`, which writes nothing and so cannot do the
switch-over: on a persona that is not live (not switched, or an emotion state that cannot be read) it refuses with a plain
message and runs nothing. Personas here are tmp directories with synthetic rows; nothing touches a real persona.

Two shapes of test: with the shared recording stub (`emotion_engine_stub`) for ORDER ("ensure" between "vocabulary" and the first
real work), and with the real engine for what the first start leaves behind.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import cli as brain_cli
from brain.bridge import server
from brain.bridge import supervisor as supervisor_mod
from brain.bridge.server import build_app
from brain.cli import main
from brain.emotion import engine, engine_files
from brain.emotion.engine import EngineStartupError
from brain.paths import get_log_dir, get_persona_dir
from tests.unit.brain.emotion._engine_support import (
    LockSpy,
    local_day,
    local_noon,
    make_live,
    make_persona,
    row,
    ruin_the_state,
)

T35 = local_noon(local_day(35))

# (id, argv) of the six commands that call `ensure`; "--persona nell" is added. Each one's first real work after `ensure` is a
# call of `brain.cli.get_provider`, which the tests replace with a recorder that stops the command.
SIX_COMMANDS = [
    ("chat", ["chat", "--no-bridge", "hello"]),
    ("heartbeat", ["heartbeat"]),
    ("dream", ["dream"]),
    ("reflex", ["reflex"]),
    ("research", ["research"]),
    ("soul-review", ["soul", "review"]),
]
FIVE_DRY_RUN_COMMANDS = [(name, argv) for name, argv in SIX_COMMANDS if name != "chat"]


class _StopError(Exception):
    """Raised by the recorder that stands in for a command's first real work, so a test ends the command there."""


@pytest.fixture
def cli_persona(tmp_path: Path) -> Path:
    """The persona `nell` under the test's home (the bridge conftest points `KINDLED_HOME` at a tmp directory), with a database of one row."""
    persona = get_persona_dir("nell")
    make_persona(persona, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    (persona / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    (persona / "emotion_vocabulary.json").write_text('{"version": 1, "emotions": []}', encoding="utf-8")
    return persona


def _record_vocabulary_and_stop_at_the_first_work(monkeypatch: pytest.MonkeyPatch, order: list[str], at_work=None) -> None:  # noqa: ANN001
    """`order` gets "vocabulary" for the command's vocabulary load and "work" at its first real work, where the command stops."""
    from brain.emotion import persona_loader

    real_load = persona_loader.load_persona_vocabulary

    def load(*args: object, **kwargs: object):
        order.append("vocabulary")
        return real_load(*args, **kwargs)

    def first_work(*_args: object, **_kwargs: object):
        order.append("work")
        if at_work is not None:
            at_work()
        raise _StopError

    monkeypatch.setattr(brain_cli, "load_persona_vocabulary", load)  # the five handlers import the name
    monkeypatch.setattr(persona_loader, "load_persona_vocabulary", load)  # the chat handler imports it inside its body
    monkeypatch.setattr(brain_cli, "get_provider", first_work)


def _files_under(persona: Path) -> set[str]:
    """Every file and folder name under the persona directory, at any depth (what a refused dry run must leave as it found it)."""
    return {str(p.relative_to(persona)) for p in persona.rglob("*")}


def _record_ensure(stub, order: list[str]) -> None:  # noqa: ANN001
    real = stub.ensure_ready
    stub.ensure_ready = lambda: (order.append("ensure"), real())[1]


# ================================================================================================
# The commands
# ================================================================================================


@pytest.mark.parametrize(("name", "argv"), SIX_COMMANDS, ids=[name for name, _ in SIX_COMMANDS])
def test_each_command_calls_ensure_after_its_vocabulary_load_and_before_its_first_work(
    name: str, argv: list[str], cli_persona: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    _record_ensure(emotion_engine_stub, order)
    with pytest.raises(_StopError):
        main([*argv, "--persona", "nell"])
    assert order == ["vocabulary", "ensure", "work"]
    assert [call for call in emotion_engine_stub.recorded_calls if call[0] == "ensure_ready"] == [("ensure_ready",)]


@pytest.mark.parametrize(("name", "argv"), SIX_COMMANDS, ids=[name for name, _ in SIX_COMMANDS])
def test_a_first_start_through_each_command_leaves_the_switch_before_the_first_work_and_takes_the_lock_once(
    name: str, argv: list[str], cli_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    switched_at_work: list[bool] = []
    _record_vocabulary_and_stop_at_the_first_work(
        monkeypatch, order, at_work=lambda: switched_at_work.append(engine.engine_for(cli_persona).is_live())
    )
    spy = LockSpy(monkeypatch)
    with pytest.raises(_StopError):
        main([*argv, "--persona", "nell"])
    assert switched_at_work == [True]
    assert len([call for call in spy.within if call[0] == engine_files.state_path(cli_persona)]) == 1  # the engine lock, once


@pytest.mark.parametrize(("name", "argv"), FIVE_DRY_RUN_COMMANDS, ids=[name for name, _ in FIVE_DRY_RUN_COMMANDS])
def test_a_dry_run_refuses_with_a_plain_message_and_runs_nothing_while_the_persona_is_not_switched_and_goes_on_once_it_is(
    name: str, argv: list[str], cli_persona: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    _record_ensure(emotion_engine_stub, order)
    emotion_engine_stub.live = False
    files_before = _files_under(cli_persona)
    assert main([*argv, "--persona", "nell", "--dry-run"]) == 2
    assert order == ["vocabulary"]  # no switch-over, and none of the command's work (no reader)
    assert not (cli_persona / "state").exists()  # nothing of the engine was written
    assert _files_under(cli_persona) == files_before  # no store of the command's engine was opened or created, nothing written anywhere
    captured = capsys.readouterr()
    assert "not switched on yet" in captured.err and "without --dry-run" in captured.err and captured.out == ""
    assert "—" not in captured.err
    # the same command on a switched persona goes on as today, and still does not call ensure (a dry run never switches over)
    order.clear()
    emotion_engine_stub.live = True
    with pytest.raises(_StopError):
        main([*argv, "--persona", "nell", "--dry-run"])
    assert order == ["vocabulary", "work"]


@pytest.mark.parametrize(("name", "argv"), FIVE_DRY_RUN_COMMANDS, ids=[name for name, _ in FIVE_DRY_RUN_COMMANDS])
def test_a_dry_run_on_a_persona_whose_state_cannot_be_read_refuses_the_same_way(
    name: str, argv: list[str], cli_persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_live(cli_persona, T35)  # switched over, but no copy of the state can be read: cannot tell, so not switched
    ruin_the_state(cli_persona)
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    files_before = _files_under(cli_persona)
    assert main([*argv, "--persona", "nell", "--dry-run"]) == 2
    assert order == ["vocabulary"]
    assert _files_under(cli_persona) == files_before
    message = capsys.readouterr().err
    assert "cannot be read" in message and engine_files.state_path(cli_persona).name in message  # the cause is named, not "run it first"
    assert "without --dry-run" not in message


def test_a_failing_ensure_stops_the_command_and_the_crash_log_gets_the_error(
    cli_persona: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)

    def fail() -> None:
        raise EngineStartupError("the first rebuild", OSError("disk full"))

    emotion_engine_stub.ensure_ready = fail
    with pytest.raises(EngineStartupError):
        main(["heartbeat", "--persona", "nell"])
    assert order == ["vocabulary"]  # the command's work never started
    crash_log = (get_log_dir() / "cli-crash.log").read_text(encoding="utf-8")
    assert "EngineStartupError" in crash_log and "the first rebuild" in crash_log


# ================================================================================================
# The bridge lifespan
# ================================================================================================


def _supervisor_spawns(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the supervisor loop with a recorder that waits for the stop; its list says whether the thread was spawned."""
    ran: list[str] = []

    def fake_run_supervised(stop_event, **_kw):  # noqa: ANN001
        ran.append("supervisor")
        stop_event.wait()

    monkeypatch.setattr(supervisor_mod, "run_supervised", fake_run_supervised)
    return ran


def test_lifespan_calls_ensure_between_the_vocabulary_load_and_the_start_up_drain(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_load = server.ensure_persona_vocabulary_loaded
    monkeypatch.setattr(server, "ensure_persona_vocabulary_loaded", lambda *a, **k: (order.append("vocabulary"), real_load(*a, **k))[1])
    _record_ensure(emotion_engine_stub, order)
    real_note_active = emotion_engine_stub.note_active
    emotion_engine_stub.note_active = lambda now: (order.append("drain"), real_note_active(now))[1]
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert order == ["vocabulary", "ensure", "drain"]


def test_a_failing_ensure_stops_the_bridge_before_the_supervisor_is_spawned(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned = _supervisor_spawns(monkeypatch)

    def fail() -> None:
        raise EngineStartupError("the first rebuild", OSError("disk full"))

    emotion_engine_stub.ensure_ready = fail
    app = build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)
    with pytest.raises(EngineStartupError), TestClient(app):
        pass
    assert spawned == []
    assert app.state.bridge.supervisor_thread is None and app.state.bridge.migration_thread is None
    assert [call for call in emotion_engine_stub.recorded_calls if call[0] == "note_active"] == []  # the drain never ran either


def test_a_first_start_of_the_bridge_switches_the_persona_over_before_the_supervisor_is_spawned(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    switched_when_spawned: list[bool] = []

    def fake_run_supervised(stop_event, **_kw):  # noqa: ANN001
        switched_when_spawned.append(engine.engine_for(persona_dir).is_live())
        stop_event.wait()

    monkeypatch.setattr(supervisor_mod, "run_supervised", fake_run_supervised)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)):
        pass
    assert switched_when_spawned == [True]
    assert engine_files.read_state(persona_dir).state is not None


def test_a_bridge_whose_state_cannot_be_read_starts_and_skips_the_switch_over(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    make_live(persona_dir, T35 - timedelta(days=1))
    ruin_the_state(persona_dir)
    spawned = _supervisor_spawns(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="brain.emotion.engine"):
        with TestClient(build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)):
            assert engine.engine_for(persona_dir).is_live() is False
    assert spawned == ["supervisor"]  # an unreadable state is not a failure: the bridge comes up
    skip_lines = [r.getMessage() for r in caplog.records if "skips the switch-over" in r.getMessage()]
    assert skip_lines and "cannot be read" in skip_lines[0]
    # `ensure` rebuilt nothing and recorded no switch-over (no x10 either); the start-up drain that follows it starts the
    # unreadable state over from an empty one (accepted edge (a), Planning B25 SW2), so the next start runs the switch-over again
    after = engine_files.read_state(persona_dir).state
    assert after is not None and after.switch_over_at is None and after.emotions == {}
