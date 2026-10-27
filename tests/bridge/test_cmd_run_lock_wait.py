"""`nell supervisor run` (the OS-service entry point) waits for the bridge lock instead of exiting 2 (C27, D4), and the
other start paths keep their exit codes (C28)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import process_identity as pi
from brain.bridge import daemon, state_file

REPO = Path(__file__).resolve().parents[2]  # not the cwd: pytest may run from anywhere

_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from brain.bridge.daemon import acquire_lock
    pd, ready, go = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    fd = acquire_lock(pd)
    assert fd is not None
    ready.write_text("held")
    deadline = time.time() + 60
    while not go.exists() and time.time() < deadline:
        time.sleep(0.02)
    """
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    persona_dir = tmp_path / "home" / "personas" / "nell"
    persona_dir.mkdir(parents=True)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: persona_dir)
    return persona_dir


def _holder(persona_dir: Path, tmp_path: Path):
    ready, go = tmp_path / "ready", tmp_path / "go"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, str(persona_dir), str(ready), str(go)], env=env)
    deadline = time.time() + 30
    while not ready.exists():
        assert time.time() < deadline and p.poll() is None, "lock holder never took the lock"
        time.sleep(0.02)
    return p, go


def _run_in_thread(monkeypatch, persona_dir, client_origin: str = "tests"):
    ran = threading.Event()
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", lambda pd, **kw: ran.set() or 0)
    result: dict = {}
    args = SimpleNamespace(persona="nell", idle_shutdown=0, client_origin=client_origin)
    t = threading.Thread(target=lambda: result.setdefault("rc", daemon.cmd_run(args)), daemon=True)
    t.start()
    return t, ran, result


@pytest.mark.parametrize("live_record", [True, False], ids=["live-bridge-record", "starter-no-record"])
def test_cmd_run_waits_for_the_lock_then_runs(home, tmp_path, monkeypatch, live_record):
    holder, go = _holder(home, tmp_path)
    try:
        if live_record:
            state_file.write(
                home,
                state_file.BridgeState(
                    persona="nell", pid=holder.pid, port=51234, started_at="t", stopped_at=None,
                    shutdown_clean=False, client_origin="cli", pid_token=pi.token_of(holder.pid), identity="token",
                ),
            )
        t, ran, result = _run_in_thread(monkeypatch, home)
        t.join(timeout=1.5)
        assert t.is_alive(), f"cmd_run returned {result.get('rc')} instead of waiting for the lock"
        assert not ran.is_set()
        go.write_text("go")
        t.join(timeout=30)
        assert not t.is_alive()
        assert result["rc"] == 0 and ran.is_set()
    finally:
        go.write_text("go")
        holder.wait(timeout=30)


def test_cmd_run_over_a_stale_reused_pid_record_runs_at_once(home, tmp_path, monkeypatch):
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state_file.write(
            home,
            state_file.BridgeState(
                persona="nell", pid=sleeper.pid, port=51234, started_at="t", stopped_at=None, shutdown_clean=False,
                client_origin="cli", pid_token=f"linux:other:{sleeper.pid}:1", identity="token",
            ),
        )
        t, ran, result = _run_in_thread(monkeypatch, home)
        t.join(timeout=30)
        assert result.get("rc") == 0 and ran.is_set()
    finally:
        sleeper.kill()
        sleeper.wait()


def _live_record(persona_dir: Path, holder) -> None:
    state_file.write(
        persona_dir,
        state_file.BridgeState(
            persona="nell", pid=holder.pid, port=51234, started_at="t", stopped_at=None, shutdown_clean=False,
            client_origin="cli", pid_token=pi.token_of(holder.pid), identity="token", phase="running",
        ),
    )


def _stopped_on_request(persona_dir: Path, reason: str) -> None:
    """What the bridge it waited for writes at the end of a requested stop (runner._write_stopped)."""

    def _mark(st: state_file.BridgeState) -> None:
        st.phase, st.stop_reason, st.shutdown_clean = "stopped", reason, True
        st.pid = st.port = st.pid_token = None

    cur = state_file.read(persona_dir)
    _mark(cur)
    state_file.write(persona_dir, cur)


def test_a_service_run_waiting_for_a_bridge_does_not_undo_its_plain_stop(home, tmp_path, monkeypatch):
    """Pre-merge review O-2: a service-origin `nell supervisor run` queued behind a running bridge must not start a new
    bridge once that bridge has ended on a plain `nell supervisor stop` (user_stop): it exits 0 (success to every
    service manager, so nothing restarts it) without running."""
    holder, go = _holder(home, tmp_path)
    try:
        _live_record(home, holder)
        t, ran, result = _run_in_thread(monkeypatch, home, client_origin="task-scheduler")
        t.join(timeout=1.5)
        assert t.is_alive(), f"cmd_run returned {result.get('rc')} instead of waiting for the lock"
        _stopped_on_request(home, "user_stop")
        go.write_text("go")
        t.join(timeout=30)
        assert not t.is_alive()
        assert result["rc"] == 0 and not ran.is_set()
    finally:
        go.write_text("go")
        holder.wait(timeout=30)


@pytest.mark.parametrize(
    "origin, ended",
    [
        ("task-scheduler", "crash"),
        ("task-scheduler", "died"),
        ("task-scheduler", "idle_timeout"),
        ("task-scheduler", "os_signal"),
        ("task-scheduler", "app_restart"),
        ("task-scheduler", "manual_restart"),
        ("task-scheduler", "hard_exit"),
        ("tests", "user_stop"),
    ],
    ids=["service-after-crash", "service-after-death", "service-after-idle-stop", "service-after-os-signal",
         "service-after-app-restart", "service-after-manual-restart", "service-after-hard-exit",
         "terminal-run-after-stop"],
)
def test_a_waiting_run_still_starts_after_a_crash_or_when_run_by_hand(home, tmp_path, monkeypatch, origin, ended):
    """The other side of O-2: a bridge that crashed (reason "crash") or died with its record still "running" is
    replaced by the waiting service run (#177). An idle stop is nobody's request to keep the service down (an
    always-on service bridge queued behind the app's bridge must start when that bridge idles out, fix-check
    must-fix 1), and neither is an OS signal, which cannot tell a user's kill from a logout or a manager's stop. A
    restart's stop (app_restart, manual_restart from `nell supervisor restart`) or a hard exit (which may end a
    restart) is followed by a start: if the queued run skipped, the restart's own start would find the lock held or
    the service job exiting, and the persona would be left with no bridge (maintainer's Q1). A run started by hand in
    a terminal is an explicit start and runs after the stop it waited for."""
    holder, go = _holder(home, tmp_path)
    try:
        _live_record(home, holder)
        t, ran, result = _run_in_thread(monkeypatch, home, client_origin=origin)
        t.join(timeout=1.5)
        assert t.is_alive()
        if ended == "crash":
            st = state_file.read(home)
            st.phase, st.stop_reason = "hard_exit", "crash"
            state_file.write(home, st)
        elif ended in ("user_stop", "idle_timeout", "os_signal", "app_restart", "manual_restart", "hard_exit"):
            _stopped_on_request(home, ended)
        go.write_text("go")
        t.join(timeout=30)
        assert not t.is_alive()
        assert result["rc"] == 0 and ran.is_set()
    finally:
        go.write_text("go")
        holder.wait(timeout=30)
