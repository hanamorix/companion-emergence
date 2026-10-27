"""The respawn guard and the start routing (C65 a-g, C58 a-c; D20, D25).

After a user kill of a service-owned bridge, the service manager may start the bridge again on its own. The guard
(`nell supervisor run` with a service origin) honours a committed "user kill" intent of this boot whose bridge is gone:
it exits 0 without running, before the lock and again right after acquiring it, and never deletes the intent. Every
explicit start clears the intent and runs; a start for a persona whose service is installed goes through the manager.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import process_identity as pi
from brain.bridge import daemon, kill, respawn_guard, shutdown_progress, state_file


@pytest.fixture
def pd(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    p = tmp_path / "home" / "personas" / "nell"
    (p / "active_conversations").mkdir(parents=True)
    return p


@pytest.fixture(scope="session")
def gone_pid():
    """The pid of a process that has exited. Every test names it with a made-up token, which
    ``process_identity.matches`` reports as GONE whatever process later takes the number."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _intent(pd: Path, *, pid: int, token: str | None, state: str = "committed", boot: object = "current",
            at: datetime | None = None, reason: str = "user_kill") -> dict:
    it = {"pid": pid, "pid_token": token, "reason": reason, "state": state, "nonce": "n1",
          "at": (at or datetime.now(UTC)).isoformat(), "boot": pi.boot_id() if boot == "current" else boot,
          "legacy": False}
    kill._write_intent(pd, it)
    return it


def _run_args(origin: str):
    return SimpleNamespace(persona="nell", idle_shutdown=0, client_origin=origin)


@pytest.fixture
def ran(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", lambda pd, **kw: calls.append("ran") or 0)
    return calls


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux boot id leg")


@linux_only
@pytest.mark.parametrize("origin", ["systemd", "launchd", "task-scheduler"])
def test_a_committed_kill_of_this_boot_stops_a_service_respawn_before_the_lock(pd, gone_pid, ran, monkeypatch, origin):
    """C65 (a)."""
    it = _intent(pd, pid=gone_pid, token=f"linux:{pi.boot_id()}:{gone_pid}:1")
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    took: list[int] = []
    real = daemon.acquire_lock
    monkeypatch.setattr(daemon, "acquire_lock", lambda *a, **k: took.append(1) or real(*a, **k))
    assert daemon.cmd_run(_run_args(origin)) == 0
    assert ran == [] and took == []
    assert kill.read_intent(pd) == it  # never deleted by the guard


@linux_only
def test_a_respawn_already_waiting_for_the_lock_checks_again_after_it(pd, gone_pid, ran, monkeypatch):
    """C65 (b): the intent was written while this service process waited for the lock (D4)."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    holder_fd = daemon.acquire_lock(pd)  # the lock, held on a second descriptor of this process
    assert holder_fd is not None
    waiting = threading.Event()
    real_acquire = daemon.acquire_lock

    def _acquire(persona_dir, *, blocking=False):
        if blocking:
            waiting.set()  # cmd_run is about to block on the lock
        return real_acquire(persona_dir, blocking=blocking)

    monkeypatch.setattr(daemon, "acquire_lock", _acquire)
    t = None
    try:
        out: dict = {}
        t = threading.Thread(target=lambda: out.setdefault("rc", daemon.cmd_run(_run_args("systemd"))), daemon=True)
        t.start()
        deadline = time.monotonic() + 30
        while not waiting.wait(0.02):
            assert t.is_alive(), f"cmd_run returned {out.get('rc')} instead of waiting for the lock"
            assert time.monotonic() < deadline, "cmd_run never reached the lock wait"
        t.join(0.3)
        assert t.is_alive()  # waiting for the lock
        it = _intent(pd, pid=gone_pid, token=f"linux:{pi.boot_id()}:{gone_pid}:1")
        (pd / shutdown_progress.PROGRESS_FILENAME).write_text("{}")  # a watcher file the start would delete
        daemon.release_lock(pd, holder_fd)
        holder_fd = None
        t.join(30)
        assert out["rc"] == 0 and ran == []
        assert kill.read_intent(pd) == it
        assert (pd / shutdown_progress.PROGRESS_FILENAME).exists()  # checked BEFORE the watcher-file cleanup
        fd = real_acquire(pd)
        assert fd is not None  # released
        daemon.release_lock(pd, fd)
    finally:
        if holder_fd is not None:
            daemon.release_lock(pd, holder_fd)
        if t is not None:
            t.join(10)


@linux_only
def test_a_pending_intent_is_ignored(pd, gone_pid, ran, monkeypatch):
    """C65 (c), (e)."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    _intent(pd, pid=gone_pid, token=f"linux:{pi.boot_id()}:{gone_pid}:1", state="pending")
    assert daemon.cmd_run(_run_args("systemd")) == 0 and ran == ["ran"]


@linux_only
@pytest.mark.parametrize("boot", [None, "another-boot"])
def test_linux_an_intent_of_another_boot_or_no_boot_is_ignored(pd, gone_pid, ran, monkeypatch, boot):
    """C65 (d) Linux."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    _intent(pd, pid=gone_pid, token="linux:x:1:1", boot=boot)
    assert daemon.cmd_run(_run_args("systemd")) == 0 and ran == ["ran"]


@linux_only
def test_linux_an_unreadable_boot_id_runs(pd, gone_pid, ran, monkeypatch):
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    _intent(pd, pid=gone_pid, token="linux:x:1:1")
    monkeypatch.setattr(pi, "boot_id", lambda: None)
    assert daemon.cmd_run(_run_args("systemd")) == 0 and ran == ["ran"]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
@pytest.mark.parametrize("delta", [-0.6, 0.6])
def test_mac_windows_boot_time_jitter_still_fires(pd, gone_pid, ran, monkeypatch, platform, delta):
    """C65 (d1): the guard's boot_time() reading differs from the writer's by under a second either way: fires."""
    import psutil

    monkeypatch.setattr(respawn_guard, "_platform", lambda: platform)
    real_bt = psutil.boot_time()
    _intent(pd, pid=gone_pid, token=f"x:{gone_pid}:1", boot=None, at=datetime.now(UTC))
    monkeypatch.setattr(psutil, "boot_time", lambda: real_bt + delta)
    monkeypatch.setattr(pi, "matches", lambda pid, tok: pi.GONE)
    assert daemon.cmd_run(_run_args("launchd")) == 0 and ran == []


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_mac_windows_an_intent_of_an_earlier_boot_is_ignored(pd, gone_pid, ran, monkeypatch, platform):
    """C65 (d2)."""
    import psutil

    monkeypatch.setattr(respawn_guard, "_platform", lambda: platform)
    _intent(pd, pid=gone_pid, token="x", boot=None, at=datetime.now(UTC) - timedelta(days=2))
    monkeypatch.setattr(psutil, "boot_time", lambda: (datetime.now(UTC) - timedelta(hours=1)).timestamp())
    monkeypatch.setattr(pi, "matches", lambda pid, tok: pi.GONE)
    assert daemon.cmd_run(_run_args("launchd")) == 0 and ran == ["ran"]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_mac_windows_a_raising_boot_time_runs(pd, gone_pid, ran, monkeypatch, platform):
    """C65 (d3)."""
    import psutil

    monkeypatch.setattr(respawn_guard, "_platform", lambda: platform)
    _intent(pd, pid=gone_pid, token="x", boot=None)

    def boom():
        raise OSError("no boot time")

    monkeypatch.setattr(psutil, "boot_time", boom)
    monkeypatch.setattr(pi, "matches", lambda pid, tok: pi.GONE)
    assert daemon.cmd_run(_run_args("launchd")) == 0 and ran == ["ran"]


@linux_only
def test_a_live_bridge_named_by_the_intent_is_not_a_reason_to_stay_down(pd, ran, monkeypatch):
    """The guard fires only when the killed bridge is GONE."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _intent(pd, pid=p.pid, token=pi.token_of(p.pid))
        assert respawn_guard.guard_fires(pd) is False
    finally:
        p.kill()
        p.wait()


@linux_only
@pytest.mark.parametrize("origin", ["cli", "tauri", "tests"])
def test_app_or_cli_origin_never_applies_the_guard(pd, gone_pid, ran, monkeypatch, origin):
    """C65 (g)."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    _intent(pd, pid=gone_pid, token="linux:x:1:1")
    assert daemon.cmd_run(_run_args(origin)) == 0 and ran == ["ran"]


# ---------------------------------------------------------------- explicit starts (C65 f, C58 a-c)


def _start_args():
    return SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="cli")


def test_cmd_start_while_running_touches_neither_the_bridge_nor_the_intent(pd, monkeypatch):
    """C58 (b), C65 (f): a launch check during an in-flight kill does not delete the intent."""
    it = _intent(pd, pid=1, token="t", state="pending")
    monkeypatch.setattr(state_file, "is_running", lambda p: True)
    monkeypatch.setattr(state_file, "liveness", lambda p: state_file.ALIVE)
    state_file.write(pd, state_file.BridgeState(persona="nell", pid=1, port=5, started_at="t", stopped_at=None,
                                                shutdown_clean=False, client_origin="cli"))
    started: list = []
    monkeypatch.setattr(daemon, "spawn_detached", lambda *a, **k: started.append("spawn") or 1)
    monkeypatch.setattr(daemon, "_start_service", lambda persona: started.append("service"))
    assert daemon.cmd_start(_start_args()) == 2
    assert started == [] and kill.read_intent(pd) == it


@pytest.mark.parametrize("installed", [True, False])
def test_cmd_start_clears_the_intent_then_routes_by_service(pd, monkeypatch, installed):
    """C58 (a, c), C65 (f): an installed service starts through its manager (then the readiness wait); otherwise
    today's spawn. The intent is gone before either."""
    _intent(pd, pid=1, token="t")
    seen: list = []
    monkeypatch.setattr(daemon, "_service_installed", lambda persona: installed)
    monkeypatch.setattr(daemon, "_start_service",
                        lambda persona: seen.append(("service", kill.read_intent(pd))))
    monkeypatch.setattr(daemon, "_await_service_readiness", lambda pd_, budget, out: 0)
    monkeypatch.setattr(daemon, "spawn_detached", lambda *a, **k: seen.append(("spawn", kill.read_intent(pd))) or 4242)
    monkeypatch.setattr(daemon, "_await_readiness", lambda *a, **k: 0)
    assert daemon.cmd_start(_start_args()) == 0
    assert seen == [("service" if installed else "spawn", None)]


def test_cmd_start_reports_a_failed_service_start_at_once(pd, monkeypatch, capsys):
    """Review S12: the manager's own failure (kickstart / systemctl start / schtasks /Run non-zero) is reported at once
    with its message, not after the 50 s readiness wait."""
    import subprocess

    monkeypatch.setattr(daemon, "_service_installed", lambda persona: True)
    monkeypatch.setattr(daemon, "_start_service", lambda persona: subprocess.CompletedProcess(
        args=["systemctl"], returncode=1, stdout="", stderr="Job for the unit failed."))
    waited: list = []
    monkeypatch.setattr(daemon, "_await_service_readiness", lambda *a, **k: waited.append(1) or 0)
    assert daemon.cmd_start(_start_args()) == 1
    assert waited == []
    err = capsys.readouterr().err
    assert "service start failed" in err and "Job for the unit failed." in err


@pytest.mark.parametrize("origin", ["systemd", "launchd", "task-scheduler"])
def test_a_crashed_service_bridge_is_not_held_back_by_the_guard(pd, gone_pid, ran, monkeypatch, origin):
    """Review R2-S1: a crash writes no stop intent, so the service manager's restart (`supervisor run`) runs."""
    monkeypatch.setattr(respawn_guard, "_platform", lambda: "linux")
    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=gone_pid, port=5, started_at="t", stopped_at=None, shutdown_clean=False,
        client_origin=origin, pid_token=f"linux:{pi.boot_id()}:{gone_pid}:1", identity="token", phase="hard_exit",
        stop_reason="crash"))
    assert respawn_guard.guard_fires(pd) is False
    assert daemon.cmd_run(_run_args(origin)) == 0
    assert ran == ["ran"]
