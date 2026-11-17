"""Engine 6, G2 and S107 (HS7): where a command's rebuild step runs, what a dry run does, and what the bridge no longer does.

The six `nell` commands (chat in-process, heartbeat, dream, reflex, research, soul review) call `_engine_ready_or_refuse` right
after their own vocabulary load. A real run registers the command and runs the rebuild step (`daemon.run_rebuild_step`): the
rebuild is its own process (`nell supervisor rebuild`), no longer an in-process `ensure_engine_ready`. Five of them take
`--dry-run`, which writes nothing and so cannot do the switch-over: on a persona that is not live (not switched, or an emotion
state that cannot be read) it refuses with a plain message and runs nothing. The bridge lifespan no longer switches the persona
over: its starter runs the rebuild process first, and a bridge started around the supervisor starts on the saved state. Personas
here are tmp directories with synthetic rows; nothing touches a real persona.

Two shapes of test: with the shared recording stub (`emotion_engine_stub`) and recorders for the step, for ORDER ("register" and
"step" between "vocabulary" and the first real work), and with the real engine for what a start leaves behind.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain import cli as brain_cli
from brain.bridge import daemon, server
from brain.bridge import supervisor as supervisor_mod
from brain.bridge.server import build_app
from brain.cli import main
from brain.emotion import engine, engine_files
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

# (id, argv) of the six commands that run the step; "--persona nell" is added. Each one's first real work after the step is a
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


def _record_step(monkeypatch: pytest.MonkeyPatch, order: list[str], answer: int | None = None) -> None:
    """`order` gets "register" for the command's run record and "step" for its rebuild step, which answers `answer` (None: go on)."""
    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: order.append("register"))
    monkeypatch.setattr(daemon, "run_rebuild_step", lambda persona_dir: (order.append("step"), answer)[1])


# ================================================================================================
# The commands
# ================================================================================================


@pytest.mark.parametrize(("name", "argv"), SIX_COMMANDS, ids=[name for name, _ in SIX_COMMANDS])
def test_each_command_registers_and_runs_the_step_after_its_vocabulary_load_and_before_its_first_work(
    name: str, argv: list[str], cli_persona: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    _record_step(monkeypatch, order)
    with pytest.raises(_StopError):
        main([*argv, "--persona", "nell"])
    assert order == ["vocabulary", "register", "step", "work"]
    assert ("ensure_ready",) not in emotion_engine_stub.recorded_calls  # a command no longer rebuilds in-process


@pytest.mark.real_rebuild_step(reason="the real rebuild step reaches the recorder that stands in for the child")
@pytest.mark.parametrize(("name", "argv"), SIX_COMMANDS, ids=[name for name, _ in SIX_COMMANDS])
def test_a_first_start_through_each_command_runs_the_rebuild_child_once_before_the_first_work_and_takes_no_engine_lock_itself(
    name: str, argv: list[str], cli_persona: Path, monkeypatch: pytest.MonkeyPatch, rebuild_spawns
) -> None:
    order: list[str] = []
    spawned_at_work: list[int] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order, at_work=lambda: spawned_at_work.append(len(rebuild_spawns.calls)))
    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: None)  # no run record left in the tmp persona
    spy = LockSpy(monkeypatch)
    with pytest.raises(_StopError):
        main([*argv, "--persona", "nell"])
    assert spawned_at_work == [1]  # the child was spawned (and waited for) before the command's first work
    assert rebuild_spawns.calls[0][0][-4:] == ["supervisor", "rebuild", "--persona", "nell"]
    assert [call for call in spy.within if call[0] == engine_files.state_path(cli_persona)] == []  # the rebuild is the child's, not ours


@pytest.mark.parametrize("code", [1, 2])
@pytest.mark.parametrize(("name", "argv"), SIX_COMMANDS, ids=[name for name, _ in SIX_COMMANDS])
def test_each_command_returns_a_refused_steps_code_and_does_no_work(
    name: str, argv: list[str], code: int, cli_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    _record_step(monkeypatch, order, answer=code)
    assert main([*argv, "--persona", "nell"]) == code
    assert order == ["vocabulary", "register", "step"]  # the command's work never started


@pytest.mark.no_base_failure(reason="pins existing behaviour: nell chat --no-bridge starts no bridge, before and after the rebuild moved out of process")
@pytest.mark.real_rebuild_step(reason="the real rebuild step runs (against the recorder) so a bridge start by the step would show")
def test_nell_chat_no_bridge_starts_no_bridge(cli_persona: Path, monkeypatch: pytest.MonkeyPatch, rebuild_spawns) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: None)
    started: list[str] = []
    monkeypatch.setattr(daemon, "cmd_start", lambda *a, **k: started.append("cmd_start") or 0)
    monkeypatch.setattr(daemon, "spawn_detached", lambda *a, **k: started.append("spawn_detached") or 1)
    with pytest.raises(_StopError):
        main(["chat", "--no-bridge", "hello", "--persona", "nell"])
    assert order == ["vocabulary", "work"]
    assert started == []


@pytest.mark.parametrize(("name", "argv"), FIVE_DRY_RUN_COMMANDS, ids=[name for name, _ in FIVE_DRY_RUN_COMMANDS])
def test_a_dry_run_refuses_with_a_plain_message_and_runs_nothing_while_the_persona_is_not_switched_and_goes_on_once_it_is(
    name: str, argv: list[str], cli_persona: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)
    _record_step(monkeypatch, order)
    emotion_engine_stub.live = False
    files_before = _files_under(cli_persona)
    assert main([*argv, "--persona", "nell", "--dry-run"]) == 2
    assert order == ["vocabulary"]  # no switch-over, no step, and none of the command's work (no reader)
    assert not (cli_persona / "state").exists()  # nothing of the engine was written
    assert _files_under(cli_persona) == files_before  # no store of the command's engine was opened or created, nothing written anywhere
    captured = capsys.readouterr()
    assert "not switched on yet" in captured.err and "without --dry-run" in captured.err and captured.out == ""
    assert "—" not in captured.err
    # the same command on a switched persona goes on as today, and still runs no step (a dry run never switches over)
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


def test_a_step_that_raises_stops_the_command_and_the_crash_log_gets_the_error(
    cli_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    _record_vocabulary_and_stop_at_the_first_work(monkeypatch, order)

    def broken(persona_dir: Path) -> None:
        raise RuntimeError("the rebuild step broke")

    monkeypatch.setattr(engine_files, "register_command_run", lambda persona_dir: None)
    monkeypatch.setattr(daemon, "run_rebuild_step", broken)
    with pytest.raises(RuntimeError):
        main(["heartbeat", "--persona", "nell"])
    assert order == ["vocabulary"]  # the command's work never started
    crash_log = (get_log_dir() / "cli-crash.log").read_text(encoding="utf-8")
    assert "RuntimeError" in crash_log and "the rebuild step broke" in crash_log


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


def test_lifespan_runs_the_start_up_drain_after_the_vocabulary_load_and_no_switch_over(
    persona_dir: Path, emotion_engine_stub, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_load = server.ensure_persona_vocabulary_loaded
    monkeypatch.setattr(server, "ensure_persona_vocabulary_loaded", lambda *a, **k: (order.append("vocabulary"), real_load(*a, **k))[1])
    real_ensure = emotion_engine_stub.ensure_ready
    emotion_engine_stub.ensure_ready = lambda: (order.append("ensure"), real_ensure())[1]
    real_note_active = emotion_engine_stub.note_active
    emotion_engine_stub.note_active = lambda now: (order.append("drain"), real_note_active(now))[1]
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests")):
        pass
    assert order == ["vocabulary", "drain"]  # the bridge no longer calls ensure (S107: the starter's rebuild process does)


def test_a_bridge_on_a_persona_that_is_not_switched_starts_and_does_not_switch_it_over(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    live_when_spawned: list[bool] = []

    def fake_run_supervised(stop_event, **_kw):  # noqa: ANN001
        live_when_spawned.append(engine.engine_for(persona_dir).is_live())
        stop_event.wait()

    monkeypatch.setattr(supervisor_mod, "run_supervised", fake_run_supervised)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)):
        pass
    assert live_when_spawned == [False]
    assert engine_files.read_state(persona_dir).state is None  # nothing was rebuilt or written: the saved state is what it was


def test_a_bridge_whose_state_cannot_be_read_starts_and_leaves_it_unreadable(
    persona_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_persona(persona_dir, [row(T35, emotions={"joy": 6.0}, mid="seed")])
    make_live(persona_dir, T35 - timedelta(days=1))
    ruin_the_state(persona_dir)
    spawned = _supervisor_spawns(monkeypatch)
    with TestClient(build_app(persona_dir=persona_dir, client_origin="tests", background_threads=True)):
        assert engine.engine_for(persona_dir).is_live() is False
    assert spawned == ["supervisor"]
    # The repair is the rebuild process's (its first rebuild writes a readable state in place of the unreadable one: the real
    # child's ruined-state case in test_rebuild_real_child.py); the bridge itself neither reads nor rewrites it.
    assert engine_files.read_state(persona_dir).state is None
    assert engine_files.read_state(persona_dir).problem == "unreadable"
