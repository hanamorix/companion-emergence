"""F-006 coverage sweep for `brain.bridge.runner`.

Covers the lifecycle surface that `test_runner_logging.py` leaves uncovered:
- `_allocate_port` success / retry / exhaustion
- `_write_clean_shutdown` no-op + idempotency + exception swallow
- `run_bridge_foreground` writes initial state, calls uvicorn, then writes
  clean shutdown in the finally block
- `main()` argparse → `run_bridge_foreground` dispatch

Notes on environmental isolation:

`run_bridge_foreground` registers an `atexit` handler and an OS-level
SIGTERM handler. We monkeypatch both `atexit.register` and `signal.signal`
to no-op so we don't pollute the pytest process's signal handlers or
leave a callable scheduled to run at interpreter exit.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from brain.bridge import runner, state_file

# ---------------------------------------------------------------------------
# _allocate_port
# ---------------------------------------------------------------------------


def test_allocate_port_succeeds_first_attempt() -> None:
    """Happy path: first bind succeeds, returns a usable ephemeral port."""
    port = runner._allocate_port()
    assert isinstance(port, int)
    # Ephemeral range — kernel chose freely from 1024+.
    assert 1024 <= port <= 65535


def test_allocate_port_retries_then_succeeds(monkeypatch) -> None:
    """First bind raises OSError; second attempt succeeds; backoff slept once.

    Asserts the contract: on transient failure, _allocate_port sleeps with
    the documented backoff schedule (10ms on attempt 0) and retries rather
    than propagating.
    """
    calls = {"socket": 0, "sleep": []}
    real_socket = socket.socket

    def flaky_socket(*args, **kwargs):
        calls["socket"] += 1
        if calls["socket"] == 1:
            # Return a sock whose bind() raises.
            mock_sock = MagicMock()
            mock_sock.bind.side_effect = OSError("address in use")
            return mock_sock
        return real_socket(*args, **kwargs)

    def fake_sleep(seconds: float) -> None:
        calls["sleep"].append(seconds)

    monkeypatch.setattr(runner.socket, "socket", flaky_socket)
    monkeypatch.setattr("time.sleep", fake_sleep)

    port = runner._allocate_port(max_attempts=3)
    assert isinstance(port, int)
    assert 1024 <= port <= 65535
    assert calls["socket"] == 2
    # Backoff schedule: (10 ** attempt) / 1000.0; attempt=0 → 0.001s.
    assert calls["sleep"] == [pytest.approx(0.001)]


def test_allocate_port_exhausts_attempts_raises(monkeypatch) -> None:
    """All attempts fail → RuntimeError with the last OSError attached."""

    def always_failing_socket(*args, **kwargs):
        m = MagicMock()
        m.bind.side_effect = OSError("address in use")
        return m

    sleeps: list[float] = []
    monkeypatch.setattr(runner.socket, "socket", always_failing_socket)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(RuntimeError, match="failed to bind a free port after 3 attempts"):
        runner._allocate_port(max_attempts=3)

    # Backoff schedule: (10 ** attempt) / 1000.0, slept only BETWEEN attempts (#481: nothing follows the last one, so it
    # no longer sleeps): attempts 0 and 1 sleep, attempt 2 does not.
    assert sleeps == [pytest.approx(0.001), pytest.approx(0.01)]


# ---------------------------------------------------------------------------
# _write_stopped (#456: replaces _write_clean_shutdown; written only for this bridge's own record)
# ---------------------------------------------------------------------------


def test_write_stopped_no_state_file_noop(tmp_path: Path) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    runner._write_stopped(persona_dir)
    assert not (persona_dir / state_file.STATE_FILENAME).exists()


def test_write_stopped_leaves_another_process_record_alone(tmp_path: Path) -> None:
    """A record that is not this process's (another bridge took over) is never written (main's ruling, C42 a)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    original = state_file.BridgeState(
        persona="persona", pid=os.getpid() + 1, port=1, started_at="2026-05-10T00:00:00+00:00", stopped_at=None,
        shutdown_clean=False, client_origin="tests", auth_token="t", pid_token="linux:x:1:1", phase="running",
    )
    state_file.write(persona_dir, original)
    before = (persona_dir / state_file.STATE_FILENAME).read_bytes()
    runner._write_stopped(persona_dir)
    assert (persona_dir / state_file.STATE_FILENAME).read_bytes() == before


def test_write_stopped_swallows_exceptions(tmp_path: Path, monkeypatch, caplog) -> None:
    """Exit-path code must never re-raise."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    def boom(*_a, **_kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runner.state_file, "update", boom)
    with caplog.at_level("WARNING", logger="brain.bridge.runner"):
        runner._write_stopped(persona_dir)  # must not raise
    assert any("stopped-mark write failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_bridge_foreground
# ---------------------------------------------------------------------------


def test_run_bridge_foreground_writes_initial_state_and_runs_uvicorn(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end contract for run_bridge_foreground:
    - initial BridgeState written with pid/port/persona before uvicorn starts
    - uvicorn.Server.run is called with the chosen port
    - finally block writes clean shutdown (shutdown_clean=True, pid/port cleared)
    - atexit/signal registration happens but is isolated from the test
      process (we monkeypatch both to no-op).
    """
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    captured: dict[str, object] = {"port_at_uvicorn": None, "state_at_uvicorn": None}

    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner, "build_app", lambda **_kw: object())

    # Isolate signal handlers so we don't touch the pytest process.
    monkeypatch.setattr(runner.signal, "signal", lambda _sig, _handler: None)
    from brain.bridge import shutdown_progress

    shutdown_progress.reset_for_tests()

    class FakeConfig:
        def __init__(self, app, *, host, port, log_level, timeout_graceful_shutdown):
            captured["port_at_uvicorn"] = port

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False

        def run(self):
            # Snapshot what's persisted at the moment the server would begin serving.
            st = state_file.read(persona_dir)
            captured["state_at_uvicorn"] = st

    monkeypatch.setattr(runner.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(runner.uvicorn, "Server", FakeServer)

    rc = runner.run_bridge_foreground(persona_dir, client_origin="tests")
    assert rc == 0

    # uvicorn was handed the allocated port.
    assert captured["port_at_uvicorn"] == 54321

    # Initial state was persisted BEFORE uvicorn started.
    initial = captured["state_at_uvicorn"]
    assert initial is not None
    assert initial.persona == "nell"
    assert initial.port == 54321
    assert initial.pid is not None  # current process pid
    assert initial.shutdown_clean is False
    assert initial.client_origin == "tests"
    assert initial.auth_token is not None and len(initial.auth_token) > 0

    # After the finally block ran, the state is marked clean.
    final = state_file.read(persona_dir)
    assert final is not None
    assert final.shutdown_clean is True
    assert final.pid is None
    assert final.port is None
    assert final.stopped_at is not None
    assert final.phase == "stopped"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_run_bridge_foreground_binds_shutdown_controller_to_uvicorn_server(
    tmp_path: Path,
    monkeypatch,
) -> None:
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner.signal, "signal", lambda *_a, **_kw: None)

    def fake_build_app(**kwargs):
        captured["shutdown_controller"] = kwargs["shutdown_controller"]
        return object()

    class FakeConfig:
        def __init__(self, app, *, host, port, log_level, timeout_graceful_shutdown):
            captured["config"] = (app, host, port, log_level, timeout_graceful_shutdown)

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            captured["server"] = self

        def run(self):
            controller = captured["shutdown_controller"]
            assert controller.request("test") is True
            assert self.should_exit is True

    monkeypatch.setattr(runner, "build_app", fake_build_app)
    monkeypatch.setattr(runner.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(runner.uvicorn, "Server", FakeServer)

    rc = runner.run_bridge_foreground(persona_dir, client_origin="tests")

    assert rc == 0
    assert captured["server"].should_exit is True
    assert captured["config"][4] == 30  # timeout_graceful_shutdown


@pytest.mark.parametrize("ending", ["return", "system_exit", "keyboard_interrupt"])
def test_after_run_decision_runs_however_server_run_ends(tmp_path: Path, monkeypatch, ending) -> None:
    """C42 (d): the after-run decision runs when server.run() returns, ends by SystemExit (re-raised SIGTERM) or by
    KeyboardInterrupt (re-raised SIGINT); exit status 0; nothing running -> "stopped"."""
    from brain.bridge import shutdown_progress

    shutdown_progress.reset_for_tests()
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner, "build_app", lambda **_kw: object())
    monkeypatch.setattr(runner.signal, "signal", lambda *_a, **_kw: None)

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False

        def run(self):
            if ending == "system_exit":
                raise SystemExit(0)
            if ending == "keyboard_interrupt":
                raise KeyboardInterrupt

    monkeypatch.setattr(runner.uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(runner.uvicorn, "Server", FakeServer)
    assert runner.run_bridge_foreground(persona_dir, client_origin="tests") == 0
    assert state_file.read(persona_dir).phase == "stopped"


@pytest.mark.parametrize("stop_begun", [True, False])
def test_after_run_with_work_still_running_takes_the_hard_exit_path(tmp_path: Path, monkeypatch, stop_begun) -> None:
    """Plan I4 runner: run() ended but a task still runs -> the hard-exit path at once, never "stopped". Review S7: a
    run that ended with no stop begun (no request, no signal) is recorded as a crash, not a deliberate hard exit, so
    the app's crash watchdog restarts it."""
    from brain.bridge import shutdown as shutdown_mod
    from brain.bridge import shutdown_progress

    reg = shutdown_progress.reset_for_tests()
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner, "build_app", lambda **_kw: object())
    monkeypatch.setattr(runner.signal, "signal", lambda *_a, **_kw: None)
    exits: list[str] = []

    class _Exited(BaseException):
        pass

    def fake_hard_exit(pd, reason="hard_exit"):
        exits.append(reason)
        raise _Exited

    monkeypatch.setattr(shutdown_mod, "hard_exit", fake_hard_exit)
    # The real hard exit never returns, so nothing stops the monitor here: keep its thread from starting, or it would
    # outlive this test and apply the noted signal (a process-wide stop) during later tests.
    monkeypatch.setattr(shutdown_mod.Monitor, "start", lambda self: None)

    class FakeServer:
        def __init__(self, config):
            self.should_exit = False

        def run(self):
            reg.register("stub task")
            if stop_begun:
                self.controller.note_signal()  # the SIGINT that uvicorn re-raises as KeyboardInterrupt
                raise KeyboardInterrupt
            raise RuntimeError("uvicorn's loop ended unexpectedly")

    monkeypatch.setattr(runner.uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(runner.uvicorn, "Server", FakeServer)
    with pytest.raises(_Exited):
        runner.run_bridge_foreground(persona_dir, client_origin="tests")
    assert exits == (["hard_exit"] if stop_begun else ["crash"])
    assert state_file.read(persona_dir).phase == "running"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_parses_args_and_dispatches(monkeypatch, tmp_path: Path) -> None:
    """main() parses CLI args and forwards them to run_bridge_foreground."""
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--persona-dir",
            str(persona_dir),
            "--client-origin",
            "tauri",
            "--idle-shutdown-seconds",
            "12.5",
        ],
    )

    received: dict = {}

    def fake_run_bridge_foreground(persona_dir, *, client_origin, idle_shutdown_seconds):
        received["persona_dir"] = persona_dir
        received["client_origin"] = client_origin
        received["idle_shutdown_seconds"] = idle_shutdown_seconds
        return 0

    monkeypatch.setattr(runner, "run_bridge_foreground", fake_run_bridge_foreground)

    rc = runner.main()
    assert rc == 0
    assert received["persona_dir"] == persona_dir
    # argparse Path conversion happened.
    assert isinstance(received["persona_dir"], Path)
    assert received["client_origin"] == "tauri"
    assert received["idle_shutdown_seconds"] == 12.5
