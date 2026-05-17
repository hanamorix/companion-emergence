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

    # Backoff schedule: (10 ** attempt) / 1000.0; attempts 0..2.
    assert sleeps == [pytest.approx(0.001), pytest.approx(0.01), pytest.approx(0.1)]


# ---------------------------------------------------------------------------
# _write_clean_shutdown
# ---------------------------------------------------------------------------


def test_write_clean_shutdown_no_state_file_noop(tmp_path: Path) -> None:
    """If bridge.json doesn't exist, _write_clean_shutdown returns without
    writing anything."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    # No state file present.
    runner._write_clean_shutdown(persona_dir)
    assert not (persona_dir / state_file.STATE_FILENAME).exists()


def test_write_clean_shutdown_already_clean_noop(tmp_path: Path) -> None:
    """If shutdown_clean is already True, the function returns without
    rewriting the file (stopped_at must NOT change)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    original = state_file.BridgeState(
        persona="persona",
        pid=None,
        port=None,
        started_at="2026-05-10T00:00:00+00:00",
        stopped_at="2026-05-10T00:00:01+00:00",
        shutdown_clean=True,
        client_origin="tests",
        auth_token="t",
    )
    state_file.write(persona_dir, original)

    runner._write_clean_shutdown(persona_dir)

    after = state_file.read(persona_dir)
    assert after is not None
    assert after.shutdown_clean is True
    assert after.stopped_at == "2026-05-10T00:00:01+00:00"  # unchanged


def test_write_clean_shutdown_swallows_exceptions(tmp_path: Path, monkeypatch, caplog) -> None:
    """If state_file.read raises, the helper logs a warning instead of
    propagating — exit-path code must never re-raise."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    def boom(*_a, **_kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runner.state_file, "read", boom)
    with caplog.at_level("WARNING", logger="brain.bridge.runner"):
        runner._write_clean_shutdown(persona_dir)  # must not raise

    assert any("clean-shutdown write failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_bridge_foreground
# ---------------------------------------------------------------------------


def test_run_bridge_foreground_writes_initial_state_and_runs_uvicorn(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end contract for run_bridge_foreground:
    - initial BridgeState written with pid/port/persona before uvicorn starts
    - uvicorn.run is called with the chosen port
    - finally block writes clean shutdown (shutdown_clean=True, pid/port cleared)
    - atexit/signal registration happens but is isolated from the test
      process (we monkeypatch both to no-op).
    """
    persona_dir = tmp_path / "personas" / "nell"
    persona_dir.mkdir(parents=True)

    captured = {"port_at_uvicorn": None, "state_at_uvicorn": None}

    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner, "build_app", lambda **_kw: object())

    # Isolate atexit + signal so we don't touch the pytest process.
    atexit_captured: list = []
    monkeypatch.setattr(
        runner.atexit,
        "register",
        lambda fn, *a, **kw: atexit_captured.append((fn, a, kw)),
    )
    monkeypatch.setattr(runner.signal, "signal", lambda _sig, _handler: None)

    def fake_uvicorn_run(app, *, host, port, log_level):
        # Snapshot what's persisted at the moment uvicorn would begin serving.
        captured["port_at_uvicorn"] = port
        st = state_file.read(persona_dir)
        captured["state_at_uvicorn"] = st
        return None

    monkeypatch.setattr(runner.uvicorn, "run", fake_uvicorn_run)

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

    # atexit registration was captured (proves the line ran) but cleaning
    # ourselves so the bound callable doesn't fire at interpreter exit.
    assert len(atexit_captured) == 1
    assert atexit_captured[0][0] is runner._write_clean_shutdown


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
