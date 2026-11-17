"""Engine 6, S107 step (d), HS4 and HS6: stopping the bridge before the rebuild, and where the rebuild step runs.

HS4: `daemon._rebuild_needed(persona_dir, bridge_running)`, `daemon._stop_for_rebuild`, `cmd_stop(..., kill_pre_pr=False)` and the
`attach` flag of `nell chat`'s already-running start. HS6: `cmd_start` (rebuild under the start lock, bridge stopped first), `cmd_run`
(the service start) and `run_rebuild_step` (the commands). The rebuild step itself (`_rebuild_under_lock`) is tested in
test_rebuild_under_lock.py and test_rebuild_real_child.py; here it is a spy unless a test says otherwise. Synthetic personas only,
under the bridge conftest's tmp home.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from brain.bridge import daemon, kill, runner, shutdown_progress, start_backoff, state_file
from brain.emotion import engine, engine_rebuild
from brain.paths import get_persona_dir, runtime_log_path
from tests.conftest import RebuildPopenRecorder
from tests.unit.brain.emotion._engine_support import make_live

REPO = Path(__file__).resolve().parents[2]
BUSY = "A start or an update of your memories is under way. Try again when it has finished."
PREVIOUS_DID_NOT_STOP = "The previous version of the bridge did not stop."


def _did_not_stop(name: str) -> str:
    return (
        "The running bridge did not stop, so nothing was updated and no new bridge was started. "
        "A bridge from a previous version may still be stopping. "
        f'If it is stuck, end it with "nell supervisor stop --persona {name} --kill", then start again.'
    )


@pytest.fixture
def persona() -> Path:
    path = get_persona_dir("nell")
    path.mkdir(parents=True)
    return path


@pytest.fixture(autouse=True)
def _root_logger_is_left_as_found() -> Any:
    root = logging.getLogger()
    level = root.level
    yield
    root.setLevel(level)


def _lock_is_free(persona: Path) -> bool:
    fd = daemon.acquire_lock(persona)
    if fd is None:
        return False
    daemon.release_lock(persona, fd)
    return True


def _start_args(**extra: Any) -> SimpleNamespace:
    return SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="cli", **extra)


class _Calls:
    """The order of the calls a start makes, and what the lock looked like at each."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.lock_free_at: dict[str, bool] = {}


@pytest.fixture
def start_calls(persona: Path, monkeypatch: pytest.MonkeyPatch) -> _Calls:
    """A start with no service installed: the spawn and the readiness wait are stubs, `cmd_stop` is a stub that answers 0 and
    the rebuild step is a stub that answers None (go on). Each records its call and whether the start lock was free then."""
    calls = _Calls()

    def record(name: str, answer: Any) -> Any:
        def stub(*args: Any, **kwargs: Any) -> Any:
            calls.order.append(name)
            calls.lock_free_at[name] = _lock_is_free(persona)
            return answer

        return stub

    monkeypatch.setattr(daemon, "spawn_detached", record("spawn", 4242))
    monkeypatch.setattr(daemon, "_await_readiness", record("ready", 0))
    monkeypatch.setattr(daemon, "cmd_stop", record("stop", 0))
    monkeypatch.setattr(daemon, "_rebuild_under_lock", record("step", None))
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: False)
    monkeypatch.setattr(state_file, "is_running", lambda pd: False)
    return calls


def _write_running_record(persona: Path) -> None:
    state_file.write(
        persona,
        state_file.BridgeState(
            persona="nell", pid=4242, port=51999, started_at="t", stopped_at=None, shutdown_clean=False, client_origin="cli"
        ),
    )


# --- HS4: the need check ----------------------------------------------------------------------------------------------------


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_an_unswitched_persona_needs_a_rebuild_with_or_without_a_bridge(persona: Path) -> None:
    assert daemon._rebuild_needed(persona, False) is True
    assert daemon._rebuild_needed(persona, True) is True


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_a_switched_persona_needs_none(persona: Path) -> None:
    from datetime import UTC, datetime

    make_live(persona, datetime(2030, 1, 1, tzinfo=UTC))
    assert daemon._rebuild_needed(persona, False) is False
    assert daemon._rebuild_needed(persona, True) is False


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_a_roll_forward_need_is_read_only_when_no_bridge_runs(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """While a bridge runs, the roll-forward detector cannot tell the bridge's own writes from rows an older brain wrote, so a switched persona
    under a running bridge needs nothing; with no bridge the same detector answer is a need."""
    from datetime import UTC, datetime

    make_live(persona, datetime(2030, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(engine_rebuild, "rolled_forward", lambda *args, **kwargs: True)
    assert daemon._rebuild_needed(persona, False) is True
    assert daemon._rebuild_needed(persona, True) is False


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_a_check_that_raises_is_logged_to_the_runtime_log_and_the_start_goes_on(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(persona_dir: Path) -> bool:
        raise ValueError("a state that cannot be read")

    monkeypatch.setattr(engine, "start_rebuild_needed", boom)
    before = list(logging.getLogger().handlers)
    assert daemon._rebuild_needed(persona, False) is False
    assert list(logging.getLogger().handlers) == before
    text = runtime_log_path("nell").read_text(encoding="utf-8")
    assert "ValueError" in text and "a state that cannot be read" in text


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_a_raising_check_logs_once_when_the_persistent_runtime_log_is_attached(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(persona_dir: Path) -> bool:
        raise ValueError("a state that cannot be read")

    monkeypatch.setattr(engine, "start_rebuild_needed", boom)
    runner._setup_runtime_logging(persona)
    before = list(logging.getLogger().handlers)
    assert daemon._rebuild_needed(persona, False) is False
    assert list(logging.getLogger().handlers) == before  # the persistent handler stays, no transient one was added
    assert runtime_log_path("nell").read_text(encoding="utf-8").count("a state that cannot be read") == 1


def _break_runner_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """`from brain.bridge import runner` now raises ImportError (the module entry is None and the package attribute is gone)."""
    import brain.bridge as bridge_package

    monkeypatch.delattr(bridge_package, "runner", raising=False)
    monkeypatch.setitem(sys.modules, "brain.bridge.runner", None)


@pytest.mark.real_rebuild_step(reason="the need check under test")
def test_a_check_that_raises_still_answers_false_when_the_runtime_log_cannot_be_imported(
    persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The log line is best effort: an import failure of the bridge stack must not turn a need check that never stops a start into
    one that does."""

    def boom(persona_dir: Path) -> bool:
        raise ValueError("a state that cannot be read")

    monkeypatch.setattr(engine, "start_rebuild_needed", boom)
    _break_runner_import(monkeypatch)
    assert daemon._rebuild_needed(persona, False) is False


@pytest.mark.no_base_failure(reason="the lifted helper exists on base (HS5, 00f7ab7d); the need check's logging relies on it")
def test_runtime_log_attached_finds_the_persistent_handler_only(persona: Path) -> None:
    assert runner._runtime_log_attached(persona) is False
    transient = runner._attach_transient_runtime_log(persona)
    try:
        assert runner._runtime_log_attached(persona) is False  # the append-and-close handler is not it
    finally:
        runner._detach_runtime_log(transient)
    runner._setup_runtime_logging(persona)
    assert runner._runtime_log_attached(persona) is True


_IMPORT_SCRIPT = textwrap.dedent(
    """
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    from brain.bridge import daemon
    from brain.paths import get_persona_dir
    from tests.unit.brain.emotion._engine_support import make_live

    persona = get_persona_dir("nell")
    persona.mkdir(parents=True)
    make_live(persona, datetime(2030, 1, 1, tzinfo=UTC))
    for bridge_running in (False, True):
        assert daemon._rebuild_needed(persona, bridge_running) is False
    for name in ("brain.bridge.runner", "uvicorn", "fastapi"):
        assert name not in sys.modules, name + " was imported by the need check"
    print("ok")
    """
)


def test_a_need_check_that_finds_nothing_never_imports_the_bridge_stack(tmp_path: Path) -> None:
    """In a fresh process (an in-process check cannot work, and the autouse fixture does not reach it)."""
    import os

    env = dict(os.environ, KINDLED_HOME=str(tmp_path / "home"), PYTHONPATH=str(REPO))
    done = subprocess.run([sys.executable, "-c", _IMPORT_SCRIPT], env=env, capture_output=True, text=True, timeout=120, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# --- HS4: stopping the bridge -------------------------------------------------------------------------------------------------


def test_stop_for_rebuild_is_the_restarts_stop_with_no_kill_of_a_pre_pr_bridge(persona: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []

    def stop(args: Any, **kwargs: Any) -> int:
        seen.append((args, kwargs))
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", stop)
    args = _start_args()
    assert daemon._stop_for_rebuild(persona, args) == 0
    assert seen == [(args, {"reason": None, "kill_pre_pr": False})]


def test_a_stop_that_fails_prints_one_plain_line_and_returns_its_code(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(daemon, "cmd_stop", lambda args, **kwargs: 1)
    assert daemon._stop_for_rebuild(persona, _start_args()) == 1
    captured = capsys.readouterr()
    assert captured.err == _did_not_stop("nell") + "\n"
    assert captured.out == ""


def _legacy_bridge(persona: Path, monkeypatch: pytest.MonkeyPatch, *, request_fails: bool = True) -> list[Any]:
    """A proven pre-PR bridge record (no identity token) that stays ALIVE; returns the list `kill_bridge` is called with."""
    state_file.write(
        persona,
        state_file.BridgeState(
            persona="nell", pid=4242, port=1, started_at="t", stopped_at=None, shutdown_clean=False, client_origin="cli"
        ),
    )
    assert state_file.is_legacy(state_file.read(persona))
    monkeypatch.setattr(state_file, "bridge_liveness", lambda s, persona_dir: state_file.ALIVE)
    monkeypatch.setattr(daemon, "_PRE_PR_REQUEST_BUDGET_S", 0.0)

    def request(s: Any, **kwargs: Any) -> Any:
        if request_fails:
            raise OSError("down")
        return None

    monkeypatch.setattr(daemon, "_request_shutdown_via_http", request)
    killed: list[Any] = []

    def kill_bridge(persona_dir: Path, persona: Any = None, **kwargs: Any) -> kill.KillResult:
        killed.append(persona_dir)
        return kill.KillResult("killed", "killed", [], True)

    monkeypatch.setattr(kill, "kill_bridge", kill_bridge)
    return killed


def test_a_pre_pr_bridge_that_does_not_stop_is_not_killed_when_kill_pre_pr_is_false(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    killed = _legacy_bridge(persona, monkeypatch)
    args = SimpleNamespace(persona="nell", timeout=5.0, force=False, kill=False, skip_hard_exit_request=False)
    assert daemon.cmd_stop(args, reason=None, kill_pre_pr=False) == 1
    assert killed == []
    captured = capsys.readouterr()
    assert PREVIOUS_DID_NOT_STOP in captured.err
    assert "Stopped the previous version" not in captured.out


@pytest.mark.no_base_failure(reason="pins existing behaviour: stop and restart still kill a pre-PR bridge that does not stop")
def test_a_pre_pr_bridge_that_does_not_stop_is_still_killed_by_stop_and_restart(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    killed = _legacy_bridge(persona, monkeypatch)
    args = SimpleNamespace(persona="nell", timeout=5.0, force=False, kill=False, skip_hard_exit_request=False)
    assert daemon.cmd_stop(args) == 0
    assert killed == [persona]
    assert "Stopped the previous version" in capsys.readouterr().out


@pytest.mark.no_base_failure(reason="pins existing behaviour: restart's stop keeps the kill of a pre-PR bridge")
def test_restart_still_stops_with_the_kill_of_a_pre_pr_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def stop(args: Any, **kwargs: Any) -> int:
        seen.append(kwargs)
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", stop)
    monkeypatch.setattr(daemon, "cmd_start", lambda args, **kwargs: 0)
    assert daemon.cmd_restart(_start_args()) == 0
    assert seen == [{"reason": None}]  # no kill_pre_pr: the default (True) keeps the kill


def test_a_pre_pr_bridge_that_stops_in_time_is_reported_stopped_with_kill_pre_pr_false(
    persona: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    killed = _legacy_bridge(persona, monkeypatch)
    args = SimpleNamespace(persona="nell", timeout=5.0, force=False, kill=False, skip_hard_exit_request=False)
    # ALIVE at cmd_stop's own look, gone at the wait
    answers = iter([state_file.ALIVE, state_file.GONE, state_file.GONE])
    monkeypatch.setattr(state_file, "bridge_liveness", lambda s, persona_dir: next(answers, state_file.GONE))
    monkeypatch.setattr(daemon, "_PRE_PR_REQUEST_BUDGET_S", 5.0)
    assert daemon.cmd_stop(args, reason=None, kill_pre_pr=False) == 0
    assert killed == []
    assert "Stopped the previous version" in capsys.readouterr().out


# --- HS4: the attach flag ---------------------------------------------------------------------------------------------------


def _chat_start_args(monkeypatch: pytest.MonkeyPatch, *, running: bool) -> list[Any]:
    from brain import cli

    seen: list[Any] = []
    monkeypatch.setattr(state_file, "is_running", lambda pd: running)
    monkeypatch.setattr(daemon, "cmd_start", lambda args, out=None: seen.append(args) or 0)
    monkeypatch.setattr(cli, "_chat_via_bridge", lambda *a, **k: 0)
    cli._chat_handler(SimpleNamespace(persona="nell", no_bridge=False, bridge_only=False))
    return seen


def test_chats_already_running_start_carries_the_attach_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    (args,) = _chat_start_args(monkeypatch, running=True)
    assert getattr(args, "attach", False) is True


@pytest.mark.no_base_failure(reason="pins existing behaviour: the auto-spawn start is a plain start, not an attach")
def test_chats_auto_spawn_start_is_not_an_attach(monkeypatch: pytest.MonkeyPatch) -> None:
    (args,) = _chat_start_args(monkeypatch, running=False)
    assert getattr(args, "attach", False) is False


# --- HS6 (a): cmd_start -------------------------------------------------------------------------------------------------------


def test_a_start_that_needs_a_rebuild_runs_the_step_under_the_lock_then_starts_the_bridge(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    needed_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: needed_calls.append((pd, running)) or True)
    assert daemon.cmd_start(_start_args()) == 0
    assert start_calls.order == ["step", "spawn", "ready"]
    assert start_calls.lock_free_at["step"] is False  # held during the step
    assert start_calls.lock_free_at["spawn"] is True  # handed to the child runner, as before
    assert needed_calls == [(persona, False)]


@pytest.mark.no_base_failure(reason="pins existing behaviour: a start that needs no rebuild runs no step")
def test_a_start_that_needs_no_rebuild_runs_no_step(persona: Path, start_calls: _Calls) -> None:
    assert daemon.cmd_start(_start_args()) == 0
    assert start_calls.order == ["spawn", "ready"]


def test_a_failed_step_releases_the_lock_returns_its_code_and_starts_nothing(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", lambda pd: 1)
    assert daemon.cmd_start(_start_args()) == 1
    assert start_calls.order == []
    assert _lock_is_free(persona)


def test_a_step_interrupted_by_ctrl_c_still_releases_the_lock(persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(pd: Path) -> int | None:
        raise KeyboardInterrupt

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", interrupted)
    with pytest.raises(KeyboardInterrupt):
        daemon.cmd_start(_start_args())
    assert start_calls.order == []
    assert _lock_is_free(persona)


def test_a_second_start_during_the_step_gets_exit_2(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    second: list[int] = []

    def step(pd: Path) -> int | None:
        second.append(daemon.cmd_start(_start_args()))
        return None

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", step)
    assert daemon.cmd_start(_start_args()) == 0
    assert second == [2]
    assert "bridge already starting (lockfile held)" in capsys.readouterr().err


def test_a_failed_step_leaves_the_stop_intent_and_the_back_off_alone(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step runs before the start clears a kill's stop intent and starts the back-off over: a start that never got going keeps both."""
    shutdown_progress.write_record(persona, shutdown_progress.INTENT_FILENAME, {"pid": 1})
    resets: list[Path] = []
    monkeypatch.setattr(start_backoff, "reset", lambda pd: resets.append(pd))
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", lambda pd: 1)
    assert daemon.cmd_start(_start_args()) == 1
    assert (persona / shutdown_progress.INTENT_FILENAME).exists()
    assert resets == []


def test_a_running_bridge_with_a_rebuild_needed_is_stopped_first_then_the_step_runs_then_the_bridge_starts(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_running_record(persona)
    stops: list[tuple[Any, dict[str, Any]]] = []

    def stop(args: Any, **kwargs: Any) -> int:
        start_calls.order.append("stop")
        stops.append((args, kwargs))
        return 0

    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    monkeypatch.setattr(daemon, "cmd_stop", stop)
    needed_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: needed_calls.append((pd, running)) or True)
    args = _start_args()
    assert daemon.cmd_start(args) == 0
    assert start_calls.order == ["stop", "step", "spawn", "ready"]
    assert stops == [(args, {"reason": None, "kill_pre_pr": False})]
    assert needed_calls == [(persona, True)]


def test_a_stop_that_fails_starts_nothing_kills_nothing_and_takes_no_lock(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_running_record(persona)
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    monkeypatch.setattr(daemon, "cmd_stop", lambda args, **kwargs: 1)
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    killed: list[str] = []
    monkeypatch.setattr(kill, "kill_bridge", lambda *a, **k: killed.append("bridge"))
    monkeypatch.setattr(kill, "kill_command", lambda *a, **k: killed.append("command"))
    assert daemon.cmd_start(_start_args()) == 1
    assert start_calls.order == []  # no step, no spawn, no readiness wait
    assert killed == []
    assert _lock_is_free(persona)
    assert capsys.readouterr().err == _did_not_stop("nell") + "\n"


@pytest.mark.no_base_failure(reason="pins existing behaviour: a start over a running bridge with no rebuild needed refuses with exit 2")
def test_a_running_bridge_with_no_rebuild_needed_is_refused_as_today(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_running_record(persona)
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    out: dict[str, Any] = {}
    assert daemon.cmd_start(_start_args(), out=out) == 2
    assert start_calls.order == []
    assert "bridge already running on port 51999 (pid 4242)" in capsys.readouterr().err
    assert out["readiness"].pid == 4242


@pytest.mark.no_base_failure(reason="pins existing behaviour: an attach start over a running bridge is refused and stops nothing")
def test_an_attach_start_over_a_running_bridge_is_refused_before_any_need_check(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_running_record(persona)
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    checks: list[Any] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: checks.append(running) or True)
    out: dict[str, Any] = {}
    assert daemon.cmd_start(_start_args(attach=True), out=out) == 2
    assert checks == []
    assert start_calls.order == []  # no stop, no step, no spawn
    assert "bridge already running on port 51999 (pid 4242)" in capsys.readouterr().err
    assert out["readiness"].pid == 4242


def test_a_bridge_that_a_zero_stop_left_behind_is_refused_by_the_lock(
    persona: Path, start_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stop that answers 0 while a bridge is somehow still there (stubbed here): a live bridge holds the start lock, so this start gets 2
    and runs no step."""
    _write_running_record(persona)
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    holder = daemon.acquire_lock(persona)
    assert holder is not None
    try:
        assert daemon.cmd_start(_start_args()) == 2
    finally:
        daemon.release_lock(persona, holder)
    assert start_calls.order == ["stop"]  # the stub stop ran and answered 0; nothing after it
    assert "bridge already starting (lockfile held)" in capsys.readouterr().err


# --- HS6 (b): cmd_run ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def run_calls(persona: Path, monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()

    def record(name: str, answer: Any) -> Any:
        def stub(*args: Any, **kwargs: Any) -> Any:
            calls.order.append(name)
            calls.lock_free_at[name] = _lock_is_free(persona)
            return answer

        return stub

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", record("bridge", 0))
    monkeypatch.setattr(daemon, "_rebuild_under_lock", record("step", None))
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: False)
    monkeypatch.setattr(start_backoff, "wait_before_start", lambda pd, interrupted=None: True)
    monkeypatch.setattr(start_backoff, "note_start", record("note_start", None))
    return calls


def _run_args(origin: str) -> SimpleNamespace:
    return SimpleNamespace(persona="nell", idle_shutdown=0, client_origin=origin)


def test_a_service_run_that_needs_a_rebuild_runs_the_step_under_the_lock_before_the_bridge(
    persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    needed_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: needed_calls.append((pd, running)) or True)
    assert daemon.cmd_run(_run_args("task-scheduler")) == 0
    assert run_calls.order == ["step", "note_start", "bridge"]
    assert run_calls.lock_free_at["step"] is False
    assert needed_calls == [(persona, False)]
    assert _lock_is_free(persona)


def test_the_step_runs_before_the_systemd_unit_refresh(persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr("brain.service.systemd.refresh_installed_unit", lambda persona_name: run_calls.order.append("refresh"))
    assert daemon.cmd_run(_run_args("systemd")) == 0
    assert run_calls.order == ["step", "refresh", "note_start", "bridge"]


@pytest.mark.parametrize("origin", ["task-scheduler", "cli"])
def test_a_failed_run_step_returns_its_code_releases_the_lock_and_notes_a_start_only_when_guarded(
    persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", lambda pd: 1)
    assert daemon.cmd_run(_run_args(origin)) == 1
    assert ("note_start" in run_calls.order) is (origin == "task-scheduler")
    assert "bridge" not in run_calls.order
    assert _lock_is_free(persona)


def test_a_guarded_run_whose_step_raises_notes_a_start_and_the_exception_still_propagates(
    persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception (not a returned code) out of the step is a failed start for the back-off too: systemd restarts the run at
    RestartSec=2, and without the noted start its restart limit is the only brake."""

    def unforeseen(pd: Path) -> int | None:
        raise RuntimeError("an unforeseen bug in the step")

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", unforeseen)
    with pytest.raises(RuntimeError, match="an unforeseen bug in the step"):
        daemon.cmd_run(_run_args("task-scheduler"))
    assert run_calls.order == ["note_start"]
    assert _lock_is_free(persona)


@pytest.mark.no_base_failure(reason="pins that an app or CLI run (not guarded) notes no start on the exception path; base notes none either")
def test_an_unguarded_run_whose_step_raises_notes_no_start_and_the_exception_still_propagates(
    persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unforeseen(pd: Path) -> int | None:
        raise RuntimeError("an unforeseen bug in the step")

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", unforeseen)
    with pytest.raises(RuntimeError, match="an unforeseen bug in the step"):
        daemon.cmd_run(_run_args("cli"))
    assert run_calls.order == []
    assert _lock_is_free(persona)


@pytest.mark.no_base_failure(reason="pins the new path's guard; on base the exception path calls no note_start, so nothing can mask")
def test_a_note_start_that_fails_on_the_exception_path_does_not_hide_the_steps_exception(
    persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unforeseen(pd: Path) -> int | None:
        raise RuntimeError("an unforeseen bug in the step")

    def broken_note(pd: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", unforeseen)
    monkeypatch.setattr(start_backoff, "note_start", broken_note)
    with pytest.raises(RuntimeError, match="an unforeseen bug in the step"):
        daemon.cmd_run(_run_args("task-scheduler"))
    assert _lock_is_free(persona)


@pytest.mark.no_base_failure(reason="pins that the step sits after the guard checks; on base there is no step to place")
def test_a_run_the_guard_stops_runs_no_step(persona: Path, run_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    """The step comes after the guard checks: a service run that is not starting (a user's kill of this boot) rebuilds nothing."""
    from brain.bridge import respawn_guard

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(respawn_guard, "guard_fires", lambda pd: True)
    assert daemon.cmd_run(_run_args("task-scheduler")) == 0
    assert run_calls.order == []


@pytest.mark.no_base_failure(reason="pins existing behaviour: a service run that needs no rebuild runs no step")
def test_a_service_run_that_needs_no_rebuild_runs_no_step(persona: Path, run_calls: _Calls) -> None:
    assert daemon.cmd_run(_run_args("task-scheduler")) == 0
    assert run_calls.order == ["note_start", "bridge"]


# --- HS6 (c): the commands ----------------------------------------------------------------------------------------------------


@pytest.fixture
def step_calls(persona: Path, monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()

    def step(pd: Path) -> int | None:
        calls.order.append("step")
        calls.lock_free_at["step"] = _lock_is_free(persona)
        return None

    monkeypatch.setattr(daemon, "_rebuild_under_lock", step)
    monkeypatch.setattr(daemon, "spawn_detached", lambda *a, **k: calls.order.append("spawn") or 1)
    monkeypatch.setattr(state_file, "is_running", lambda pd: False)
    return calls


@pytest.mark.no_base_failure(reason="pins existing behaviour: a command does nothing under a running bridge")
def test_a_command_under_a_running_bridge_does_nothing(persona: Path, step_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_file, "is_running", lambda pd: True)
    checks: list[Any] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: checks.append(running) or True)
    assert daemon.run_rebuild_step(persona) is None
    assert step_calls.order == [] and checks == []


def test_a_command_on_a_live_persona_takes_no_lock_and_runs_no_step(
    persona: Path, step_calls: _Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: False)
    taken: list[Any] = []
    real_acquire = daemon.acquire_lock
    monkeypatch.setattr(daemon, "acquire_lock", lambda *a, **k: taken.append(a) or real_acquire(*a, **k))
    assert daemon.run_rebuild_step(persona) is None
    assert taken == [] and step_calls.order == []


@pytest.mark.parametrize("answer", [None, 1])
def test_a_command_that_needs_a_rebuild_runs_the_step_under_the_start_lock_and_releases_it(
    persona: Path, step_calls: _Calls, monkeypatch: pytest.MonkeyPatch, answer: int | None
) -> None:
    needed_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: needed_calls.append((pd, running)) or True)

    def step(pd: Path) -> int | None:
        step_calls.order.append("step")
        step_calls.lock_free_at["step"] = _lock_is_free(persona)
        return answer

    monkeypatch.setattr(daemon, "_rebuild_under_lock", step)
    assert daemon.run_rebuild_step(persona) == answer
    assert step_calls.order == ["step"]  # and it starts no bridge
    assert step_calls.lock_free_at["step"] is False
    assert needed_calls == [(persona, False)]
    assert _lock_is_free(persona)


def test_a_command_releases_the_lock_when_the_step_raises(persona: Path, step_calls: _Calls, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(pd: Path) -> int | None:
        raise KeyboardInterrupt

    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    monkeypatch.setattr(daemon, "_rebuild_under_lock", interrupted)
    with pytest.raises(KeyboardInterrupt):
        daemon.run_rebuild_step(persona)
    assert _lock_is_free(persona)


def test_a_command_is_refused_at_once_when_a_start_holds_the_lock(
    persona: Path, step_calls: _Calls, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(daemon, "_rebuild_needed", lambda pd, running: True)
    holder = daemon.acquire_lock(persona)
    assert holder is not None
    try:
        assert daemon.run_rebuild_step(persona) == 2
    finally:
        daemon.release_lock(persona, holder)
    assert step_calls.order == []
    captured = capsys.readouterr()
    assert captured.err == BUSY + "\n" and captured.out == ""
    assert _lock_is_free(persona)  # the holder's release left it free; the refused command took nothing


@pytest.mark.real_rebuild_step(reason="the whole step through the recorder")
def test_a_command_on_an_unswitched_persona_runs_the_rebuild_child_once_and_goes_on(
    persona: Path, rebuild_spawns: RebuildPopenRecorder
) -> None:
    before = list(logging.getLogger().handlers)
    assert daemon.run_rebuild_step(persona) is None
    assert len(rebuild_spawns.calls) == 1
    assert rebuild_spawns.calls[0][0][-6:] == ["-m", "brain.cli", "supervisor", "rebuild", "--persona", "nell"]
    assert list(logging.getLogger().handlers) == before
    assert _lock_is_free(persona)
