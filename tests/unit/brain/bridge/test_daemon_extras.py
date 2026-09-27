"""Unit tests for cmd_restart and cmd_tail_log — the two new daemon handlers
added with the `nell supervisor` rename. The four existing handlers
(cmd_start/stop/status/tail) are covered by tests/bridge/test_lifecycle.py
at the integration layer; we don't duplicate that here."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
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
    # S56: release_lock never unlinks — the lock file persists after release,
    # only the OS-level lock on it is dropped (a fresh acquire must succeed).
    assert (persona_dir / daemon.LOCKFILE).exists()
    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    daemon.release_lock(persona_dir, fd)


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
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)

    s = state_file.BridgeState(
        persona="nell",
        pid=12345,
        port=50000,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
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
            def recv(self):
                raise KeyboardInterrupt()

        yield _WS()

    monkeypatch.setattr("websockets.sync.client.connect", fake_connect)

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
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)

    s = state_file.BridgeState(
        persona="nell",
        pid=12345,
        port=50000,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
        auth_token=None,
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    captured: dict[str, object] = {}

    @contextmanager
    def fake_connect(url, *, subprotocols=None, **kw):
        captured["subprotocols"] = subprotocols

        class _WS:
            def recv(self):
                raise KeyboardInterrupt()

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
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)

    s = state_file.BridgeState(
        persona="nell",
        pid=4321,
        port=51234,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
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
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)

    s = state_file.BridgeState(
        persona="nell",
        pid=4321,
        port=51234,
        started_at="2026-05-05T00:00:00+00:00",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
        auth_token=None,
    )
    state_file.write(persona_dir, s)
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    rc = daemon.cmd_start(_args("nell"))  # no out=
    assert rc == 2


# ============================================================================
# F-005 sweep — daemon.py coverage 54% → ~95%
#
# 23-test sweep covering recovery triggers, lock acquire/release round-trips,
# spawn_detached wiring, cmd_start orphan-drain + readiness verify + child-kill,
# cmd_stop signal handling, cmd_status health-probe, cmd_run persona validation,
# cmd_tail not-running + KeyboardInterrupt, cmd_tail_log OSError path.
# ============================================================================


def _seed_persona_config(persona_dir: Path) -> None:
    """Write a minimal persona_config.json so PersonaConfig.load returns
    something non-default (helps recovery exercise its full path)."""
    from brain.persona_config import PersonaConfig

    PersonaConfig(
        provider="echo",
        searcher="fake",
        mcp_audit_log_level="info",
        user_name="Hana",
    ).save(persona_dir / "persona_config.json")


# ---------- run_recovery_if_needed (lines 52-78) ----------


def test_run_recovery_returns_none_when_state_file_missing(
    tmp_path: Path,
) -> None:
    """No bridge.json at all → recovery not needed → returns None."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    assert daemon.run_recovery_if_needed(persona_dir) is None


def test_run_recovery_returns_none_when_previous_shutdown_clean(
    tmp_path: Path,
) -> None:
    """Previous bridge shutdown_clean=True → recovery not needed → returns None."""
    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=99999,
            port=50000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at="2026-05-08T01:00:00+00:00",
            shutdown_clean=True,
            client_origin="cli",
        ),
    )
    assert daemon.run_recovery_if_needed(persona_dir) is None


def test_run_recovery_returns_none_when_dirty_but_pid_still_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dirty shutdown but the recorded pid is still alive → bridge actually
    running, recovery would clobber it. Must return None."""
    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=12345,
            port=50000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: True)
    assert daemon.run_recovery_if_needed(persona_dir) is None


def test_run_recovery_runs_drain_when_dirty_with_dead_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dirty shutdown + dead pid → recovery fires, drains stale sessions,
    returns the report count."""
    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _seed_persona_config(persona_dir)
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=99999,
            port=50000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: False)

    captured: dict[str, object] = {}

    def fake_snapshot_stale(
        persona_dir_arg,
        *,
        silence_minutes,
        store,
        hebbian,
        provider,
    ):
        captured["persona_dir"] = persona_dir_arg
        captured["silence_minutes"] = silence_minutes
        # Report list of 2 to assert returned count.
        return [object(), object()]

    monkeypatch.setattr("brain.bridge.daemon.snapshot_stale_sessions", fake_snapshot_stale)

    drained = daemon.run_recovery_if_needed(persona_dir)
    assert drained == 2
    assert captured["persona_dir"] == persona_dir
    assert captured["silence_minutes"] == 0


def test_run_recovery_fires_on_drain_errors_even_when_shutdown_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """drain_errors>0 arm: shutdown_clean=True but the last drain failed,
    so the buffer was retained — recovery must run anyway."""
    from brain.bridge import state_file

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _seed_persona_config(persona_dir)
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=99999,
            port=50000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at="2026-05-08T01:00:00+00:00",
            shutdown_clean=True,
            client_origin="cli",
            drain_errors=3,
        ),
    )
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(
        "brain.bridge.daemon.snapshot_stale_sessions",
        lambda *a, **kw: [],
    )
    drained = daemon.run_recovery_if_needed(persona_dir)
    assert drained == 0  # ran, but no orphan sessions to snapshot


# ---------- acquire_lock / release_lock (lines 81-114) ----------


def test_acquire_lock_succeeds_on_fresh_persona_dir(tmp_path: Path) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    assert (persona_dir / daemon.LOCKFILE).exists()
    daemon.release_lock(persona_dir, fd)
    # S56: release_lock NEVER unlinks — the file persists, only the OS lock
    # on it is dropped.
    assert (persona_dir / daemon.LOCKFILE).exists()


def test_acquire_lock_content_is_irrelevant_only_the_os_lock_matters(
    tmp_path: Path,
) -> None:
    """A pre-existing lockfile with arbitrary/garbage content (a leftover pid,
    non-integer garbage, or nothing at all) does not itself block a fresh
    acquire — S50 removed all pid/content-based staleness guessing. The ONLY
    thing that can refuse an acquire is another process actually holding the
    OS-level lock right now (covered by the contention tests below)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / daemon.LOCKFILE).write_text("not-a-pid", encoding="utf-8")

    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    # Content is truncated and replaced with our own pid on success.
    assert (persona_dir / daemon.LOCKFILE).read_bytes().count(str(os.getpid()).encode()) == 1
    daemon.release_lock(persona_dir, fd)


def test_acquire_lock_c18a_kill_then_restart_acquires_immediately_no_archive(
    tmp_path: Path,
) -> None:
    """C18(a): kill -9 the bridge (simulated: the OS drops the flock/msvcrt
    lock the instant the holding fd is closed, exactly what happens when a
    process is SIGKILLed — no cleanup code runs either way); the next start
    acquires the lock immediately (< 1s), with no 120s wait and no
    `.stale-*` archive file created (that whole mechanism, and the file it
    used to write, no longer exists)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    fd1 = daemon.acquire_lock(persona_dir)
    assert fd1 is not None
    # Simulate a SIGKILL: the process dies without running release_lock, but
    # the kernel still releases the OS-level lock on fd close. Closing the
    # bare fd (bypassing release_lock's unlock+close) reproduces exactly that.
    os.close(fd1)

    started = time.time()
    fd2 = daemon.acquire_lock(persona_dir)
    elapsed = time.time() - started

    assert fd2 is not None
    assert elapsed < 1.0
    assert list(persona_dir.glob("bridge.json.lock.stale-*")) == []
    daemon.release_lock(persona_dir, fd2)


def test_acquire_lock_c18b_reboot_with_pid_reuse_acquires_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C18(b): simulated reboot with pid reuse — the lock file's leftover
    content names the pid of a live, unrelated process (a reboot clears all
    OS locks but the lock FILE itself, being on disk, survives with its old
    content). No process holds the OS lock. A start must still acquire
    immediately: pid liveness is never consulted."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    # Leftover content names a pid that IS alive (this test process itself)
    # but is unrelated to any bridge — the reboot dropped the real lock.
    (persona_dir / daemon.LOCKFILE).write_text(str(os.getpid()), encoding="utf-8")
    # Even if something still consulted liveness, it would say "alive" here —
    # proving the acquire below does NOT consult it at all.
    monkeypatch.setattr(daemon.state_file, "pid_is_alive", lambda _pid: True)

    started = time.time()
    fd = daemon.acquire_lock(persona_dir)
    elapsed = time.time() - started

    assert fd is not None
    assert elapsed < 1.0
    daemon.release_lock(persona_dir, fd)


def test_acquire_lock_c18c_pid_readable_by_another_reader_while_held(
    tmp_path: Path,
) -> None:
    """C18(c): while a bridge holds the lock, a second process can still read
    the pid from the lock file. On POSIX the pid sits at offset 0 (flock is
    advisory; reads are never blocked). On Windows the locked byte is at
    offset 0 and the pid is written starting at offset 1 specifically so it
    stays readable — this assertion is offset-agnostic (it just looks for
    the pid's digits anywhere in the file) so it holds unmodified on a real
    Windows CI runner too."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    try:
        raw = (persona_dir / daemon.LOCKFILE).read_bytes()
        assert str(os.getpid()).encode() in raw
    finally:
        daemon.release_lock(persona_dir, fd)


def test_acquire_lock_c19_two_racing_acquires_exactly_one_wins(tmp_path: Path) -> None:
    """C19(b)-equivalent at the acquire_lock layer (the exclusive point
    runner.main relies on, S57): two threads racing acquire_lock on the same
    persona dir at once — exactly one gets a live fd, the other gets None.
    A real two-child-process race (tests/bridge/test_runner_mutual_exclusion.py)
    exercises the same guarantee through runner.main end to end; this proves
    the exclusion itself, at the primitive the plan names as the one
    exclusive point."""
    import threading

    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    barrier = threading.Barrier(2)
    results: list[int | None] = [None, None]

    def _race(i: int) -> None:
        barrier.wait()
        results[i] = daemon.acquire_lock(persona_dir)

    threads = [threading.Thread(target=_race, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    winners = [fd for fd in results if fd is not None]
    losers = [fd for fd in results if fd is None]
    assert len(winners) == 1
    assert len(losers) == 1
    daemon.release_lock(persona_dir, winners[0])


def test_release_lock_never_unlinks_and_inode_is_stable(tmp_path: Path) -> None:
    """C19(c): the lock file's inode is the same before and after acquire +
    release, and release never unlinks it — spied via os.unlink."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    lock_path = persona_dir / daemon.LOCKFILE
    inode_before = lock_path.stat().st_ino

    unlink_calls: list[Path] = []
    real_unlink = os.unlink

    def spy_unlink(path, *a, **kw):
        unlink_calls.append(Path(str(path)))
        return real_unlink(path, *a, **kw)

    import unittest.mock

    with unittest.mock.patch("os.unlink", side_effect=spy_unlink):
        daemon.release_lock(persona_dir, fd)

    assert unlink_calls == []
    assert lock_path.exists()
    assert lock_path.stat().st_ino == inode_before


def test_release_lock_is_idempotent_after_external_unlink(
    tmp_path: Path,
) -> None:
    """release_lock must not raise if the lockfile was already removed.

    POSIX permits unlinking a file while its fd is still open, but Windows
    raises WinError 32. Close the fd first so the test exercises the same
    release_lock idempotence contract without relying on POSIX-only semantics.
    """
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    fd = daemon.acquire_lock(persona_dir)
    assert fd is not None
    os.close(fd)
    (persona_dir / daemon.LOCKFILE).unlink()  # simulate external removal
    daemon.release_lock(persona_dir, fd)  # must not raise on closed fd + missing path


# ---------- spawn_detached (lines 117-148) ----------


def test_spawn_detached_invokes_popen_with_detach_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    log_path = tmp_path / "logs" / "bridge-nell.log"

    captured: dict[str, object] = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *, stdout, stderr, stdin, start_new_session):
        captured["cmd"] = cmd
        captured["start_new_session"] = start_new_session
        captured["stderr"] = stderr
        captured["stdin"] = stdin
        # stdout is a file handle — confirm it's not a TTY
        captured["stdout_is_file"] = hasattr(stdout, "write")
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)

    pid = daemon.spawn_detached(
        persona_dir,
        idle_shutdown_seconds=300.0,
        client_origin="cli",
        log_path=log_path,
    )
    assert pid == 4242
    assert captured["start_new_session"] is True
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["stdout_is_file"] is True
    cmd = captured["cmd"]
    assert sys.executable in cmd
    assert "-m" in cmd and "brain.bridge.runner" in cmd
    assert "--persona-dir" in cmd
    assert "--client-origin" in cmd and "cli" in cmd
    assert "--idle-shutdown-seconds" in cmd and "300.0" in cmd
    # Log directory was auto-created.
    assert log_path.parent.exists()


def test_spawn_detached_isolates_the_bridge_from_the_callers_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`-m` puts the cwd on sys.path, and the bridge inherits the caller's cwd: run
    `nell supervisor start` from a checkout and the runtime's python would import
    the checkout's brain/. -P, like the MCP server spawn in provider.py (#285)."""
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 1

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    daemon.spawn_detached(persona_dir, None, "cli", tmp_path / "bridge.log")
    assert captured["cmd"][:4] == [sys.executable, "-P", "-m", "brain.bridge.runner"]


def test_spawn_detached_omits_idle_arg_when_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    log_path = tmp_path / "logs" / "bridge-nell.log"
    captured: dict[str, object] = {}

    class FakeProc:
        pid = 5555

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    pid = daemon.spawn_detached(persona_dir, None, "launchd", log_path)
    assert pid == 5555
    assert "--idle-shutdown-seconds" not in captured["cmd"]


# ---------- cmd_start (lines 169-253) ----------


def test_cmd_start_returns_1_when_persona_dir_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    rc = daemon.cmd_start(_args("ghost"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "persona directory not found" in err


def test_cmd_start_returns_2_when_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """is_running=False but acquire_lock returns None — another starter is
    mid-flight. Must surface code 2, not spawn."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon.state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(daemon, "acquire_lock", lambda _p: None)

    def fail_spawn(*a, **kw):
        pytest.fail("cmd_start must not spawn when lock is held")

    monkeypatch.setattr(daemon, "spawn_detached", fail_spawn)
    rc = daemon.cmd_start(_args("nell"))
    assert rc == 2
    assert "lockfile held" in capsys.readouterr().err


def test_cmd_start_prints_drain_message_when_recovery_ran(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recovery returns 2 drained sessions → user-visible message + spawn proceeds."""
    from brain.bridge import state_file

    log_dir = _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda _p: 2)

    written_pid = 7777

    def fake_spawn(persona_dir_arg, idle, client_origin, log_path):
        # Write state_file as the runner would so verify loop sees a fresh state.
        state_file.write(
            persona_dir_arg,
            state_file.BridgeState(
                persona="nell",
                pid=written_pid,
                port=51001,
                started_at="2026-05-08T00:00:00+00:00",
                stopped_at=None,
                shutdown_clean=False,
                client_origin=client_origin,
                auth_token="tok",
            ),
        )
        return written_pid

    monkeypatch.setattr(daemon, "spawn_detached", fake_spawn)

    class FakeResp:
        status_code = 200

        def json(self):
            return {}

    monkeypatch.setattr("httpx.get", lambda *a, **kw: FakeResp())

    rc = daemon.cmd_start(_args("nell"))
    assert rc == 0
    captured = capsys.readouterr()
    assert "snapshotted 2 active sessions" in captured.out
    assert "bridge started on port 51001" in captured.out
    assert log_dir.exists()


def test_cmd_start_prints_no_drain_message_when_recovery_ran_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recovery ran but drained 0 sessions → "no orphan sessions to drain" message."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda _p: 0)

    def fake_spawn(persona_dir_arg, idle, client_origin, log_path):
        state_file.write(
            persona_dir_arg,
            state_file.BridgeState(
                persona="nell",
                pid=8888,
                port=51002,
                started_at="2026-05-08T00:00:00+00:00",
                stopped_at=None,
                shutdown_clean=False,
                client_origin=client_origin,
            ),
        )
        return 8888

    monkeypatch.setattr(daemon, "spawn_detached", fake_spawn)

    class FakeResp:
        status_code = 200

        def json(self):
            return {}

    monkeypatch.setattr("httpx.get", lambda *a, **kw: FakeResp())

    rc = daemon.cmd_start(_args("nell"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "no orphan sessions to drain" in out


def test_cmd_start_kills_orphan_child_when_health_never_responds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """H-D regression: bridge spawned, /health never came up — must SIGTERM
    the orphan child and return 1 with a "killed orphan child" message."""
    import httpx

    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(daemon, "run_recovery_if_needed", lambda _p: None)

    orphan_pid = 9090

    def fake_spawn(persona_dir_arg, idle, client_origin, log_path):
        # Write state_file but health never comes up.
        state_file.write(
            persona_dir_arg,
            state_file.BridgeState(
                persona="nell",
                pid=orphan_pid,
                port=51003,
                started_at="2026-05-08T00:00:00+00:00",
                stopped_at=None,
                shutdown_clean=False,
                client_origin=client_origin,
            ),
        )
        return orphan_pid

    monkeypatch.setattr(daemon, "spawn_detached", fake_spawn)

    # /health always raises ConnectError so we run the full deadline loop.
    def fake_get(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", fake_get)
    # Short-circuit the 50s readiness deadline (Bug 2b widened it from 5s).
    monkeypatch.setattr("brain.bridge.daemon.time.sleep", lambda _s: None)
    fake_now = iter([0.0, 1.0, 2.0, 51.0])
    monkeypatch.setattr("brain.bridge.daemon.time.time", lambda: next(fake_now))

    killed: dict[str, object] = {}

    def fake_kill(pid, sig):
        killed["pid"] = pid
        killed["sig"] = sig

    monkeypatch.setattr("brain.bridge.daemon.os.kill", fake_kill)

    rc = daemon.cmd_start(_args("nell"))
    assert rc == 1
    assert killed["pid"] == orphan_pid
    import signal as _signal

    assert killed["sig"] == _signal.SIGTERM
    err = capsys.readouterr().err
    assert "killed orphan child" in err
    assert "Inspect log at" in err


# ---------- cmd_run (lines 256-307) ----------


def test_cmd_run_persona_not_found_returns_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bonus test: cmd_run mirrors cmd_start's persona-not-found path."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))
    rc = daemon.cmd_run(_args("ghost", idle_shutdown=0, client_origin="launchd"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "persona directory not found" in err


def test_cmd_run_returns_2_when_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """is_running=False but acquire_lock=None — surface 2, do not call runner."""
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(daemon.state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(daemon, "acquire_lock", lambda _p: None)

    def fail_runner(*a, **kw):
        pytest.fail("cmd_run must not start runner when lock is held")

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", fail_runner)
    rc = daemon.cmd_run(_args("nell", idle_shutdown=0, client_origin="launchd"))
    assert rc == 2
    assert "lockfile held" in capsys.readouterr().err


# ---------- cmd_stop (lines 310-331) ----------


def test_cmd_stop_returns_0_when_no_state_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    rc = daemon.cmd_stop(_args("nell"))
    assert rc == 0
    assert "bridge not running" in capsys.readouterr().out


def test_cmd_stop_returns_0_when_pid_already_dead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """State file present but pid is dead → no-op clean exit."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=99999,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: False)
    rc = daemon.cmd_stop(_args("nell"))
    assert rc == 0
    assert "bridge not running" in capsys.readouterr().out


def _pin_daemon_posix(monkeypatch: pytest.MonkeyPatch, kill) -> None:
    """Swap the daemon module's ``os`` for a posix-shaped proxy exposing only
    what cmd_stop touches (``name`` + ``kill``).

    Patching the GLOBAL ``os.name`` breaks platform code in both directions —
    pathlib explodes on macOS when pinned to "nt" (see the note near the
    task-scheduler tests) and ``os.O_DIRECTORY`` doesn't exist on Windows when
    pinned to "posix" (state_file's fsync path; broke windows-latest CI). So
    the pin is scoped to ``brain.bridge.daemon`` alone."""
    import types

    monkeypatch.setattr(daemon, "os", types.SimpleNamespace(name="posix", kill=kill))


def test_cmd_stop_falls_back_to_sigterm_on_posix_when_http_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HTTP shutdown attempt fails (ConnectError), so cmd_stop falls back to
    POSIX SIGTERM. pid_is_alive returns True initially, False after SIGTERM.
    cmd_stop polls and exits 0. No real subprocess or network call made.

    The daemon's os is pinned posix-shaped (via _pin_daemon_posix) so this
    exercises the POSIX branch on every platform — on real Windows cmd_stop
    deliberately refuses the kill (see the windows-branch tests below)."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=33333,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )

    monkeypatch.setattr(
        "brain.bridge.daemon.httpx.post",
        lambda *_a, **_kw: (_ for _ in ()).throw(httpx.ConnectError("no route")),
    )

    alive_state = {"alive": True}
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: alive_state["alive"])

    killed: dict[str, object] = {}

    def fake_kill(pid, sig):
        killed["pid"] = pid
        killed["sig"] = sig
        # Simulate the child dying after SIGTERM.
        alive_state["alive"] = False

    _pin_daemon_posix(monkeypatch, fake_kill)
    # Replace sleep so test runs instantly.
    monkeypatch.setattr("brain.bridge.daemon.time.sleep", lambda _s: None)

    rc = daemon.cmd_stop(_args("nell", timeout=5.0))
    import signal as _signal

    assert killed["pid"] == 33333
    assert killed["sig"] == _signal.SIGTERM
    assert rc == 0
    assert "bridge stopped" in capsys.readouterr().out


def test_cmd_stop_returns_1_when_timeout_exceeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SIGTERM sent but pid never dies within timeout → return 1.

    Daemon os pinned posix-shaped — the SIGTERM fallback only exists there."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=44444,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )

    # Stays alive forever.
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: True)
    _pin_daemon_posix(monkeypatch, lambda pid, sig: None)
    monkeypatch.setattr("brain.bridge.daemon.time.sleep", lambda _s: None)
    # Force the deadline to be exceeded immediately.
    times = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr("brain.bridge.daemon.time.time", lambda: next(times))

    rc = daemon.cmd_stop(_args("nell", timeout=1.0))
    assert rc == 1
    assert "did not stop within" in capsys.readouterr().err


def test_cmd_stop_handles_processlookuperror_during_kill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the process dies between liveness probe and SIGTERM, os.kill raises
    ProcessLookupError — cmd_stop must catch and report no-op.

    Daemon os pinned posix-shaped — the SIGTERM fallback only exists there."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=55555,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: True)

    def fake_kill(pid, sig):
        raise ProcessLookupError()

    _pin_daemon_posix(monkeypatch, fake_kill)

    rc = daemon.cmd_stop(_args("nell"))
    assert rc == 0
    assert "bridge not running" in capsys.readouterr().out


# ---------- cmd_status (lines 334-361) ----------


def test_cmd_status_no_state_file_prints_not_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_paths(monkeypatch, tmp_path)
    rc = daemon.cmd_status(_args("nell"))
    assert rc == 0
    assert "not running (no state file)" in capsys.readouterr().out


def test_cmd_status_running_prints_health_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Live bridge: /health succeeds, status prints pid/port + health fields."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=11111,
            port=52000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
            auth_token="tok-status",
        ),
    )
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    captured: dict[str, object] = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "uptime_s": 42,
                "sessions_active": 3,
                "supervisor_thread": "alive",
                "pending_alarms": 1,
            }

    def fake_get(url, *, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResp()

    monkeypatch.setattr("httpx.get", fake_get)
    rc = daemon.cmd_status(_args("nell"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "pid=11111" in out
    assert "port=52000" in out
    assert "uptime_s: 42" in out
    assert "sessions_active: 3" in out
    assert "supervisor: alive" in out
    assert "pending_alarms: 1" in out
    assert captured["headers"] == {"Authorization": "Bearer tok-status"}


def test_cmd_status_returns_1_when_health_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """is_running=True but /health raises → return 1 + stderr."""
    import httpx

    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=11111,
            port=52000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)

    def fake_get(*a, **kw):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr("httpx.get", fake_get)
    rc = daemon.cmd_status(_args("nell"))
    assert rc == 1
    assert "/health unreachable" in capsys.readouterr().err


def test_cmd_status_reports_dirty_crash_when_recovery_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """State file present, not running, recovery_needed=True → dirty-crash message."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=66666,
            port=53000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(state_file, "recovery_needed", lambda _p: True)
    rc = daemon.cmd_status(_args("nell"))
    assert rc == 0
    assert "previous process crashed dirty" in capsys.readouterr().out


def test_cmd_status_reports_clean_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """State file present, not running, recovery_needed=False → clean stop message."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell",
            pid=77777,
            port=53000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at="2026-05-08T01:00:00+00:00",
            shutdown_clean=True,
            client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    monkeypatch.setattr(state_file, "recovery_needed", lambda _p: False)
    rc = daemon.cmd_status(_args("nell"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "stopped cleanly at 2026-05-08T01:00:00+00:00" in out


# ---------- cmd_tail (lines 364-390) ----------


def test_cmd_tail_returns_1_when_bridge_not_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No state file → cmd_tail prints "bridge not running" to stderr, returns 1."""
    _patch_paths(monkeypatch, tmp_path)
    rc = daemon.cmd_tail(_args("nell"))
    assert rc == 1
    assert "bridge not running" in capsys.readouterr().err


# ---------- Task 4: cmd_stop uses /supervisor/shutdown endpoint ----------


def test_cmd_stop_posts_shutdown_endpoint_instead_of_sigterm(
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
            pid=33333,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
            auth_token="secret",
        ),
    )

    alive = {"value": True}
    calls: list[tuple[str, dict]] = []

    def fake_post(url, *, headers, timeout):
        calls.append((url, {"headers": headers, "timeout": timeout}))
        alive["value"] = False

        class Response:
            status_code = 202

            def raise_for_status(self):
                return None

        return Response()

    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: alive["value"])
    monkeypatch.setattr("brain.bridge.daemon.httpx.post", fake_post)
    monkeypatch.setattr("brain.bridge.daemon.os.kill", lambda *_a: (_ for _ in ()).throw(AssertionError("must not signal")))
    monkeypatch.setattr("brain.bridge.daemon.time.sleep", lambda _s: None)

    rc = daemon.cmd_stop(_args("nell", timeout=5.0))

    assert rc == 0
    assert calls == [
        (
            "http://127.0.0.1:51000/supervisor/shutdown",
            {"headers": {"Authorization": "Bearer secret"}, "timeout": 3.0},
        )
    ]
    assert "bridge stopped" in capsys.readouterr().out


def test_cmd_stop_on_windows_does_not_fallback_to_sigterm_when_http_fails(
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
            pid=44444,
            port=51000,
            started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="task-scheduler",
            auth_token="secret",
        ),
    )

    # Patch get_persona_dir directly so pathlib doesn't try to construct a
    # WindowsPath on Python 3.13 macOS when os.name is patched to "nt".
    monkeypatch.setattr(daemon.os, "name", "nt")
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda _name: persona_dir)
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr("brain.bridge.daemon.httpx.post", lambda *_a, **_kw: (_ for _ in ()).throw(httpx.ConnectError("no route")))
    monkeypatch.setattr("brain.bridge.daemon.os.kill", lambda *_a: (_ for _ in ()).throw(AssertionError("windows stop must not TerminateProcess")))

    rc = daemon.cmd_stop(_args("nell", timeout=0.1))

    assert rc == 1
    assert "shutdown endpoint unreachable" in capsys.readouterr().err


def test_cmd_start_refusal_wording_is_byte_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """C28: both refusal strings are byte-identical to today's wording — the
    state-file pre-check ("bridge already running on port … (pid …)") and
    the lock branch ("bridge already starting (lockfile held)"). Neither
    changed shape when the lock mechanism underneath them did."""
    from brain.bridge import state_file

    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell", pid=555, port=51500, started_at="2026-05-08T00:00:00+00:00",
            stopped_at=None, shutdown_clean=False, client_origin="cli",
        ),
    )
    monkeypatch.setattr(state_file, "is_running", lambda _p: True)
    rc = daemon.cmd_start(_args("nell"))
    assert rc == 2
    assert "bridge already running on port 51500 (pid 555)" in capsys.readouterr().err

    # Second refusal path: is_running False but the OS lock is already held.
    monkeypatch.setattr(state_file, "is_running", lambda _p: False)
    held_fd = daemon.acquire_lock(persona_dir)
    assert held_fd is not None
    try:
        monkeypatch.setattr(daemon, "acquire_lock", lambda _p: None)
        rc = daemon.cmd_start(_args("nell"))
        assert rc == 2
        assert "bridge already starting (lockfile held)" in capsys.readouterr().err
    finally:
        daemon.release_lock(persona_dir, held_fd)


def test_cmd_stop_on_windows_with_force_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from brain.bridge import state_file
    _patch_paths(monkeypatch, tmp_path)
    persona_dir = tmp_path / "home" / "personas" / "nell"
    state_file.write(persona_dir, state_file.BridgeState(
        persona="nell", pid=44444, port=51000, started_at="2026-05-08T00:00:00+00:00",
        stopped_at=None, shutdown_clean=False, client_origin="task-scheduler", auth_token="secret"))
    # Patch get_persona_dir directly so pathlib doesn't try to construct a
    # WindowsPath on Python 3.13 macOS when os.name is patched to "nt".
    monkeypatch.setattr(daemon.os, "name", "nt")
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda _name: persona_dir)
    alive = {"v": True}
    monkeypatch.setattr(state_file, "pid_is_alive", lambda _pid: alive["v"])
    monkeypatch.setattr("brain.bridge.daemon.httpx.post", lambda *_a, **_kw: (_ for _ in ()).throw(httpx.ConnectError("no route")))
    killed = []
    def fake_kill(pid, sig):
        killed.append(pid)
        alive["v"] = False
    monkeypatch.setattr("brain.bridge.daemon.os.kill", fake_kill)
    monkeypatch.setattr("brain.bridge.daemon.time.sleep", lambda _s: None)
    rc = daemon.cmd_stop(_args("nell", timeout=5.0, force=True))
    assert rc == 0
    assert killed == [44444]
    assert "forcing Windows termination" in capsys.readouterr().err
