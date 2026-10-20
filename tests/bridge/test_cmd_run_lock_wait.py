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
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, str(persona_dir), str(ready), str(go)], env=env)
    deadline = time.time() + 30
    while not ready.exists():
        assert time.time() < deadline and p.poll() is None, "lock holder never took the lock"
        time.sleep(0.02)
    return p, go


def _run_in_thread(monkeypatch, persona_dir):
    ran = threading.Event()
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", lambda pd, **kw: ran.set() or 0)
    result: dict = {}
    args = SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="tests")
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
