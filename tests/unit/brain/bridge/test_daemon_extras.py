"""Unit tests for cmd_restart and cmd_tail_log — the two new daemon handlers
added with the `nell supervisor` rename. The four existing handlers
(cmd_start/stop/status/tail) are covered by tests/bridge/test_lifecycle.py
at the integration layer; we don't duplicate that here."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from brain.bridge import daemon


def _args(persona: str, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(persona=persona, idle_shutdown=30, client_origin="cli", timeout=180.0)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


# ---------- cmd_restart ----------


def test_cmd_restart_calls_stop_then_start_when_running(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: stop returns 0, start returns 0, restart returns 0 with two-phase output."""
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 0

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 0
    assert calls == ["stop", "start"]
    out = capsys.readouterr().out
    assert "stopping bridge" in out
    assert "starting bridge" in out


def test_cmd_restart_bails_when_stop_timed_out_on_wedged_bridge(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop returns 1 (SIGTERM timeout / wedged bridge) — restart must NOT call start.

    cmd_stop returns 0 for clean-stop AND no-bridge-running, so the only way
    to see stop=1 is a wedge. Spawning a second bridge over a wedged first
    one is the exact footgun the supervisor exists to prevent.
    """
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 1

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 1
    assert calls == ["stop"]
    err = capsys.readouterr().err
    assert "restart aborted" in err
    assert "stop failed" in err


def test_cmd_restart_bails_when_stop_returns_unexpected_high_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop returns 2 (lock held) — restart must NOT call start; returns stop's code."""
    calls: list[str] = []

    def fake_stop(args):
        calls.append("stop")
        return 2

    def fake_start(args):
        calls.append("start")
        return 0

    monkeypatch.setattr(daemon, "cmd_stop", fake_stop)
    monkeypatch.setattr(daemon, "cmd_start", fake_start)

    rc = daemon.cmd_restart(_args("nell"))
    assert rc == 2
    assert calls == ["stop"]
    err = capsys.readouterr().err
    assert "restart aborted" in err
    assert "stop failed" in err


def test_cmd_restart_propagates_start_failure_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop succeeds but start returns 1 — restart returns 1, not coerced."""
    monkeypatch.setattr(daemon, "cmd_stop", lambda a: 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda a: 1)
    assert daemon.cmd_restart(_args("nell")) == 1


def test_cmd_restart_propagates_start_already_running_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop succeeds but start returns 2 (race: another process spawned) — restart returns 2."""
    monkeypatch.setattr(daemon, "cmd_stop", lambda a: 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda a: 2)
    assert daemon.cmd_restart(_args("nell")) == 2


# ---------- cmd_tail_log ----------


def _make_persona(tmp_path: Path, name: str = "nell") -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / name
    persona_dir.mkdir(parents=True)
    return persona_dir


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, persona: str = "nell") -> Path:
    """Wire NELLBRAIN_HOME so get_persona_dir AND get_log_dir resolve under tmp_path/home.

    get_log_dir() honors NELLBRAIN_HOME (returning <HOME>/logs) since the
    paths.py override fix; setting the env var is sufficient.
    """
    home = tmp_path / "home"
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _make_persona(tmp_path, persona)
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    return log_dir


def test_acquire_lock_does_not_unlink_new_lock_after_stale_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    lock_path = persona_dir / daemon.LOCKFILE
    lock_path.write_text("999999", encoding="utf-8")

    def fake_pid_is_alive(_pid: int) -> bool:
        lock_path.write_text("123456", encoding="utf-8")
        return False

    monkeypatch.setattr(daemon.state_file, "pid_is_alive", fake_pid_is_alive)

    assert daemon.acquire_lock(persona_dir) is None
    assert lock_path.read_text(encoding="utf-8") == "123456"


# ---------- cmd_run ----------


def test_cmd_run_calls_foreground_runner_without_detaching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreground service mode runs in-process; launchd must own this process."""
    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    calls: dict[str, object] = {}

    def fail_detach(*args, **kwargs):
        pytest.fail("cmd_run must not call spawn_detached")

    def fake_foreground(persona_dir_arg, *, client_origin, idle_shutdown_seconds):
        calls["persona_dir"] = persona_dir_arg
        calls["client_origin"] = client_origin
        calls["idle_shutdown_seconds"] = idle_shutdown_seconds
        return 0

    monkeypatch.setattr(daemon, "spawn_detached", fail_detach)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda persona_dir: None)
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", fake_foreground)

    rc = daemon.cmd_run(_args("nell", idle_shutdown=0, client_origin="launchd"))

    assert rc == 0
    assert calls == {
        "persona_dir": persona_dir,
        "client_origin": "launchd",
        "idle_shutdown_seconds": None,
    }
    assert not (persona_dir / daemon.LOCKFILE).exists()


def test_cmd_run_converts_idle_shutdown_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    calls: dict[str, object] = {}

    def fake_foreground(persona_dir_arg, *, client_origin, idle_shutdown_seconds):
        calls["idle_shutdown_seconds"] = idle_shutdown_seconds
        return 0

    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda persona_dir: None)
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", fake_foreground)

    rc = daemon.cmd_run(_args("nell", idle_shutdown=2.5, client_origin="cli"))

    assert rc == 0
    assert calls["idle_shutdown_seconds"] == 150.0


def test_cmd_run_refuses_when_bridge_already_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=os.getpid(),
            port=51234,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="launchd",
        ),
    )

    def fail_foreground(*args, **kwargs):
        pytest.fail("cmd_run must not start a second foreground runner")

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", fail_foreground)

    rc = daemon.cmd_run(_args("nell", idle_shutdown=0, client_origin="launchd"))

    assert rc == 2
    assert "bridge already running" in capsys.readouterr().err


def _tail_args(persona: str, **kw) -> argparse.Namespace:
    ns = argparse.Namespace(persona=persona, lines=50, follow=False)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_cmd_tail_log_prints_last_n_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(1, 101)) + "\n")

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=3))
    assert rc == 0
    out = capsys.readouterr().out
    assert "line 98" in out
    assert "line 99" in out
    assert "line 100" in out
    assert "line 97" not in out


def test_cmd_tail_log_default_50_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("\n".join(f"line {i}" for i in range(1, 101)) + "\n")

    rc = daemon.cmd_tail_log(_tail_args("nell"))  # lines=50 default
    assert rc == 0
    out = capsys.readouterr().out
    assert "line 51" in out
    assert "line 50" not in out


def test_cmd_tail_log_n_zero_prints_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("alpha\nbeta\n")

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=0))
    assert rc == 0
    out = capsys.readouterr().out
    assert out == ""


def test_cmd_tail_log_missing_file_returns_1_with_helpful_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_paths(monkeypatch, tmp_path)  # log dir exists; log file does not

    rc = daemon.cmd_tail_log(_tail_args("nell"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "bridge log not found" in err
    assert "supervisor ever started" in err


def test_cmd_tail_log_persona_not_found_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    rc = daemon.cmd_tail_log(_tail_args("ghost"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "persona directory not found" in err


def test_cmd_tail_log_follow_mode_emits_new_lines_then_exits_on_keyboard_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Follow mode polls the log; KeyboardInterrupt exits cleanly with code 0.

    Timing: the writer thread sleeps 350ms, appends two lines, sleeps
    another 350ms, then sets the stop_event. The daemon polls every
    200ms. Widened from 150ms on 2026-05-08 — the macOS CI runner
    started flaking under the bigger workspace, the writer's first
    sleep needs to be greater than the daemon's poll interval to avoid
    the daemon checking stop_event before the writer has flushed.
    """
    import threading
    import time

    log_dir = _patch_paths(monkeypatch, tmp_path)
    log_path = log_dir / "bridge-nell.log"
    log_path.write_text("seed\n")

    stop_event = threading.Event()
    monkeypatch.setattr(daemon, "_follow_should_stop", stop_event)

    # writer thread appends two lines after a short delay, then signals interrupt
    def writer():
        time.sleep(0.35)
        with log_path.open("a") as f:
            f.write("new1\n")
            f.write("new2\n")
            f.flush()
        time.sleep(0.35)
        stop_event.set()

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    rc = daemon.cmd_tail_log(_tail_args("nell", lines=1, follow=True))
    t.join(timeout=2.0)
    assert rc == 0
    out = capsys.readouterr().out
    assert "seed" in out
    assert "new1" in out
    assert "new2" in out


# ---------- cmd_tail (I-1 follow-up audit: subprotocol auth, no ?token=) ----------


def test_cmd_tail_uses_subprotocol_auth_not_url_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cmd_tail must authenticate via Sec-WebSocket-Protocol (subprotocols
    arg), NOT via ?token=<...> URL query string.

    Pre-fix (I-1 in 2026-05-05 follow-up audit): cmd_tail built
    'ws://...?token=<token>'. The server only reads the bearer token from
    Sec-WebSocket-Protocol so tail was silently broken under auth, AND
    the token leaked into URL space (process listing, proxy logs).
    """
    from contextlib import contextmanager

    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr(
        "brain.paths.get_persona_dir", lambda name: persona_dir
    )

    s = state_file.BridgeState(
        persona="nell", pid=12345, port=50000,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None, shutdown_clean=False, client_origin="cli",
        auth_token="secret-token-aaa",
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    captured: dict[str, object] = {}

    @contextmanager
    def fake_connect(url, *, subprotocols=None, **kw):
        captured["url"] = url
        captured["subprotocols"] = subprotocols
        # Raise KeyboardInterrupt immediately to bail out of the recv loop
        class _WS:
            def recv(self): raise KeyboardInterrupt()
        yield _WS()

    monkeypatch.setattr(
        "websockets.sync.client.connect", fake_connect
    )

    rc = daemon.cmd_tail(_args("nell"))
    assert rc == 0
    assert captured["url"] == "ws://127.0.0.1:50000/events"
    assert "?token=" not in captured["url"]
    assert "secret-token-aaa" not in captured["url"]
    # Auth via subprotocols ["bearer", "<token>"]
    assert captured["subprotocols"] == ["bearer", "secret-token-aaa"]


def test_cmd_tail_no_subprotocols_when_token_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If auth_token is None (auth-disabled config), pass subprotocols=None."""
    from contextlib import contextmanager

    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr(
        "brain.paths.get_persona_dir", lambda name: persona_dir
    )

    s = state_file.BridgeState(
        persona="nell", pid=12345, port=50000,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None, shutdown_clean=False, client_origin="cli",
        auth_token=None,
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    captured: dict[str, object] = {}

    @contextmanager
    def fake_connect(url, *, subprotocols=None, **kw):
        captured["subprotocols"] = subprotocols
        class _WS:
            def recv(self): raise KeyboardInterrupt()
        yield _WS()

    monkeypatch.setattr("websockets.sync.client.connect", fake_connect)
    rc = daemon.cmd_tail(_args("nell"))
    assert rc == 0
    assert captured["subprotocols"] is None


# ---- Bug B (audit-3): readiness handoff via cmd_start out= dict ----


def test_cmd_start_populates_readiness_when_already_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the bridge is already running, cmd_start returns 2 AND populates
    out['readiness'] with the live BridgeReadiness so callers can connect
    without re-reading state_file (which would race against state_file
    rewrites by the supervisor). Bug B from the 2026-05-05 audit-3."""
    from brain.bridge import daemon, state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr(
        "brain.paths.get_persona_dir", lambda name: persona_dir
    )

    s = state_file.BridgeState(
        persona="nell", pid=4321, port=51234,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None, shutdown_clean=False, client_origin="cli",
        auth_token="tok-aaa",
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    out: dict = {}
    rc = daemon.cmd_start(_args("nell"), out=out)
    assert rc == 2  # already-running
    assert "readiness" in out
    r = out["readiness"]
    assert r.pid == 4321
    assert r.port == 51234
    assert r.auth_token == "tok-aaa"


def test_cmd_start_readiness_is_optional_for_legacy_callers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cmd_start without out= behaves exactly as before (argparse handlers
    don't pass out=). The new readiness handoff is purely additive."""
    from brain.bridge import daemon, state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    monkeypatch.setattr(
        "brain.paths.get_persona_dir", lambda name: persona_dir
    )

    s = state_file.BridgeState(
        persona="nell", pid=4321, port=51234,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None, shutdown_clean=False, client_origin="cli",
        auth_token=None,
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    rc = daemon.cmd_start(_args("nell"))  # no out=
    assert rc == 2
