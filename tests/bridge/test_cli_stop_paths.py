"""CLI stop paths around the watcher: escalation only after identity (C22), `--kill` (C35), pre-PR bridges (C43 b-d),
service uninstall and install with a running bridge (C54, C58 d); D10, D16, D20, D22.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from brain import process_identity as pi
from brain.bridge import daemon, kill, state_file, stop_watch

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX legs; Windows in CI")


@pytest.fixture
def pd(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    p = tmp_path / "home" / "personas" / "nell"
    (p / "active_conversations").mkdir(parents=True)
    return p


def _stop_args(**kw):
    base = {"persona": "nell", "timeout": 5.0, "force": False, "kill": False, "skip_hard_exit_request": False}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def sleeper():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p
    p.kill()
    p.wait()


def _write(pd: Path, **kw) -> None:
    base = {"persona": "nell", "pid": None, "port": 1, "started_at": "t", "stopped_at": None,
            "shutdown_clean": False, "client_origin": "cli"}
    base.update(kw)
    state_file.write(pd, state_file.BridgeState(**base))


# ---------------------------------------------------------------- C22: escalation only after identity


@posix
def test_failed_request_with_identity_holding_falls_back_to_sigterm(pd, sleeper, monkeypatch, capsys):
    _write(pd, pid=sleeper.pid, pid_token=pi.token_of(sleeper.pid), identity="token")
    monkeypatch.setattr(daemon, "_request_shutdown_via_http", lambda s, **k: (_ for _ in ()).throw(OSError("down")))
    monkeypatch.setattr(stop_watch, "_interactive", lambda: False)
    assert daemon.cmd_stop(_stop_args()) == 0
    sleeper.wait(10)
    assert sleeper.returncode == -signal.SIGTERM


def test_failed_request_with_unknown_identity_signals_nothing(pd, sleeper, monkeypatch, capsys):
    _write(pd, pid=sleeper.pid, pid_token="linux:x:1:1", identity="token")
    monkeypatch.setattr(pi, "token_of", lambda pid: pi.UNKNOWN)
    monkeypatch.setattr(daemon, "_request_shutdown_via_http", lambda s, **k: (_ for _ in ()).throw(OSError("down")))
    signals: list = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: signals.append(sig) or True)
    assert daemon.cmd_stop(_stop_args()) == 1
    assert signals == [] and sleeper.poll() is None


# ---------------------------------------------------------------- C35: --kill


def test_kill_on_a_stale_record_kills_nothing(pd, sleeper, capsys):
    _write(pd, pid=sleeper.pid, pid_token=f"linux:other:{sleeper.pid}:1", identity="token")
    assert daemon.cmd_stop(_stop_args(kill=True)) == 0
    assert "bridge not running (stale record)" in capsys.readouterr().out
    assert sleeper.poll() is None


def test_kill_on_a_running_bridge_uses_the_helper(pd, sleeper, monkeypatch):
    _write(pd, pid=sleeper.pid, pid_token=pi.token_of(sleeper.pid), identity="token")
    calls: list = []
    monkeypatch.setattr(kill, "kill_bridge", lambda persona_dir, persona=None, **kw: calls.append(kw)
                        or kill.KillResult("killed", "bridge killed"))
    assert daemon.cmd_stop(_stop_args(kill=True, skip_hard_exit_request=True)) == 0
    assert calls == [{"timeout": 5.0, "skip_hard_exit_request": True, "expect_pid": None, "expect_token": None}]


def test_kill_with_survivors_exits_nonzero(pd, sleeper, monkeypatch, capsys):
    # Review of #458 (finding 1): a kill that leaves processes running is not a success. Exit 0 here made the app's
    # force restart and `restart` start a new bridge beside the old tree.
    _write(pd, pid=sleeper.pid, pid_token=pi.token_of(sleeper.pid), identity="token")
    monkeypatch.setattr(kill, "kill_bridge", lambda persona_dir, persona=None, **kw: kill.KillResult(
        "killed", "processes still running after the kill: [4242]", [4242], True))
    assert daemon.cmd_stop(_stop_args(kill=True)) == 1
    assert "[4242]" in capsys.readouterr().out


# ---------------------------------------------------------------- C43: pre-PR bridges (D22)

_STUB_RUNNER = textwrap.dedent(
    """
    # A pre-PR bridge stand-in: writes an old-format bridge.json (no pid_token), answers the shutdown request with 202
    # and then ignores it, and has a `claude` child.
    import http.server, json, os, socket, subprocess, sys, threading, time
    from pathlib import Path
    args = sys.argv[1:]
    pd = Path(args[args.index("--persona-dir") + 1])
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "claude"])
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(202); self.end_headers(); self.wfile.write(b"{}")
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    (pd / "bridge.json").write_text(json.dumps({"persona": pd.name, "pid": os.getpid(), "port": port,
        "started_at": "t", "stopped_at": None, "shutdown_clean": False, "client_origin": "cli", "auth_token": None}))
    (pd / "stub-child").write_text(str(child.pid))
    srv.serve_forever()
    """
)


@posix
def test_a_proven_pre_pr_bridge_is_asked_then_killed_with_its_tree(pd, tmp_path, monkeypatch, capsys):
    """C43 (b)."""
    stub_root = tmp_path / "stubpkg" / "brain" / "bridge"
    stub_root.mkdir(parents=True)
    for d in (stub_root.parent, stub_root):
        (d / "__init__.py").write_text("")
    (stub_root / "runner.py").write_text(_STUB_RUNNER)
    proc = subprocess.Popen([sys.executable, "-m", "brain.bridge.runner", "--persona-dir", str(pd)],
                            env=dict(os.environ, PYTHONPATH=str(tmp_path / "stubpkg")), cwd=str(tmp_path))
    try:
        deadline = time.monotonic() + 30
        while not (pd / "stub-child").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        child = int((pd / "stub-child").read_text())
        assert state_file.is_legacy(state_file.read(pd))
        asked: list = []
        real_post = daemon._request_shutdown_via_http
        monkeypatch.setattr(daemon, "_request_shutdown_via_http", lambda s, **k: asked.append(1) or real_post(s, **k))
        prompts: list = []
        monkeypatch.setattr(stop_watch, "_prompt", lambda text: prompts.append(text) or "")
        t0 = time.monotonic()
        assert daemon.cmd_stop(_stop_args(timeout=10)) == 0
        assert asked == [1] and prompts == []
        assert time.monotonic() - t0 >= daemon._PRE_PR_REQUEST_BUDGET_S - 0.1
        out = capsys.readouterr().out
        assert "Stopped the previous version of the bridge." in out
        assert "Waiting for" not in out  # no task list
        proc.wait(10)
        for pid in (child,):
            try:
                assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                pass
        assert state_file.recovery_needed(pd)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_an_unproven_pre_pr_record_is_not_running_everywhere(pd, sleeper, monkeypatch, capsys):
    """C43 (c)."""
    _write(pd, pid=sleeper.pid)  # old format: no identity, pid of an unrelated live process
    assert state_file.is_running(pd) is False
    assert state_file.recovery_needed(pd) is True
    signals: list = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: signals.append(sig) or True)
    assert daemon.cmd_stop(_stop_args()) == 0
    assert "bridge not running (stale record)" in capsys.readouterr().out
    assert daemon.cmd_stop(_stop_args(kill=True)) == 0
    assert signals == [] and sleeper.poll() is None
    st = daemon.status_dict(pd)
    assert st["legacy"] is True and st["proven"] is False and st["alive"] == state_file.GONE


def test_an_unreadable_pre_pr_cmdline_is_unproven(pd, sleeper, monkeypatch):
    """C43 (d)."""
    _write(pd, pid=sleeper.pid)

    def denied(self):
        raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil.Process, "cmdline", denied)
    assert state_file.prove_bridge_cmdline(sleeper.pid, pd, "nell") is False
    assert daemon.status_dict(pd)["proven"] is False


# ---------------------------------------------------------------- C54 / C58 (d): service uninstall and install


class _Backend:
    def __init__(self):
        self.calls: list[str] = []

    def uninstall_service(self, *, persona, keep_plist=False):
        self.calls.append("uninstall")
        return Path("/x")

    def install_service(self, **kw):
        from brain.service import lifecycle

        lifecycle.before_install(kw["persona"])
        self.calls.append("install")
        return Path("/x")

    def resolve_nell_path(self, p):
        return Path("/bin/true")


@pytest.mark.parametrize("tty", [True, False])
def test_uninstall_stops_the_bridge_through_the_request_first(pd, monkeypatch, tty):
    """C54: the shutdown request and the watcher first; uninstall only once the bridge has stopped; without a TTY a
    bridge still stopping after --timeout leaves the service installed."""
    from brain import cli

    backend = _Backend()
    monkeypatch.setattr("brain.service.current_backend", lambda: backend)
    monkeypatch.setattr(cli, "_service_backend", lambda action: backend)
    order: list[str] = []
    monkeypatch.setattr(daemon, "_stop_before_service_change",
                        lambda persona, timeout: order.append(f"stop tty={stop_watch._interactive()}") or tty)
    monkeypatch.setattr(stop_watch, "_interactive", lambda: tty)
    monkeypatch.setattr(state_file, "liveness", lambda p: state_file.ALIVE)
    rc = cli.main(["service", "uninstall", "--persona", "nell"])
    assert order == [f"stop tty={tty}"]
    if tty:
        assert rc == 0 and backend.calls == ["uninstall"]
    else:
        assert rc == 1 and backend.calls == []


def test_install_with_a_tty_stops_the_running_bridge_first(pd, monkeypatch):
    """C58 (d) TTY half: the watcher's stop runs first, then the install."""
    from brain import cli
    from brain.service import lifecycle

    backend = _Backend()
    monkeypatch.setattr(cli, "_service_backend", lambda action: backend)
    order: list[str] = []
    running = {"v": True}
    monkeypatch.setattr(lifecycle, "bridge_running", lambda persona: running["v"])
    monkeypatch.setattr(lifecycle, "_interactive", lambda: True)

    def fake_stop(persona, timeout):
        order.append("stop")
        running["v"] = False
        return True

    monkeypatch.setattr(daemon, "_stop_before_service_change", fake_stop)
    rc = cli.main(["service", "install", "--persona", "nell", "--nell-path", "/bin/true"])
    assert rc == 0 and order == ["stop"] and backend.calls == ["install"]


def test_windows_uninstall_does_not_end_the_task_itself(monkeypatch, tmp_path):
    """C54: /End is only the kill path's hard-exit step; uninstall of a stopped bridge just deletes the task."""
    from brain.service import windows_service

    calls: list = []
    monkeypatch.setattr(windows_service, "run_schtasks", lambda args: calls.append(args) or subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(windows_service, "task_xml_dir", lambda: tmp_path)
    windows_service.uninstall_service(persona="nell")
    assert all(c[0] != "/End" for c in calls)



def test_stop_kill_passes_the_expected_bridge_to_the_helper(monkeypatch, tmp_path):
    """Review S8 pin: `nell supervisor stop --kill --expect-pid P --expect-token T` hands P and T to the kill helper,
    and a bridge_changed refusal exits 1 with its message."""
    from brain import cli
    from brain.bridge import daemon, kill

    seen: dict = {}

    def _fake_kill(persona_dir, persona=None, **kw):
        seen.update(kw)
        return kill.KillResult("bridge_changed", "bridge_changed: another bridge runs now; nothing killed")

    monkeypatch.setattr(kill, "kill_bridge", _fake_kill)
    monkeypatch.setattr(daemon.state_file, "read", lambda pd: daemon.state_file.BridgeState(
        persona="nell", pid=4242, port=1, started_at="t", stopped_at=None, shutdown_clean=False, client_origin="tauri",
        pid_token="linux:b:4242:1", identity="token", phase="running"))
    monkeypatch.setattr(daemon.state_file, "bridge_liveness", lambda st, pd: daemon.state_file.ALIVE)
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    rc = cli.main(["supervisor", "stop", "--persona", "nell", "--kill", "--skip-hard-exit-request",
                   "--expect-pid", "4242", "--expect-token", "linux:b:4242:0"])
    assert rc == 1
    assert seen["expect_pid"] == 4242 and seen["expect_token"] == "linux:b:4242:0"
    assert seen["skip_hard_exit_request"] is True


def test_restart_asks_for_a_restart_not_a_plain_stop(pd, monkeypatch):
    """Maintainer's Q1: `nell supervisor restart`'s stop is recorded as a restart (no reason -> "manual_restart"),
    not as a plain stop (user_stop), so a service run queued behind the bridge starts instead of skipping."""
    reasons: list = []
    monkeypatch.setattr(daemon, "cmd_stop", lambda args, *, reason="user_stop": reasons.append(reason) or 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda args: 0)
    assert daemon.cmd_restart(_stop_args()) == 0
    assert reasons == [None]
