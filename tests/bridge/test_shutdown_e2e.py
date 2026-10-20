"""Real-process stop tests: a real `python -m brain.bridge.runner` on a synthetic persona (#456: C10-C13, C15, C42,
C55).

The runner starts with KINDLED_TEST_HOOKS=1: its supervisor is a stub, and N counted stub tasks each run until the test
creates their release file, so "work still running during the stop" is deterministic. KINDLED_HOME is sandboxed under
the test's tmp dir; the persona's provider is the fake one, so nothing can reach a real model.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from brain.bridge import shutdown_progress, state_file

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and flock; Windows legs run in CI")

REPO = Path(__file__).resolve().parents[2]
START_TIMEOUT_S = 60.0


class Bridge:
    def __init__(self, tmp: Path, *, tasks: int = 0, hold_supervisor: bool = False, idle_s: float | None = None,
                 sighup_ignored: bool = False, calls: tuple[str, ...] = (), provider: str = "fake",
                 new_group: bool = False, prepare=None):
        self.home = tmp / "home"
        self.pd = self.home / "personas" / "nell"
        (self.pd / "active_conversations").mkdir(parents=True, exist_ok=True)
        cfg = self.pd / "persona_config.json"
        if not cfg.exists():
            cfg.write_text(json.dumps({"provider": provider, "searcher": "noop"}), encoding="utf-8")
        if prepare is not None:
            prepare(self.pd)
        self.fake_dir = tmp / "fake-claude"
        self.fake_dir.mkdir(exist_ok=True)
        self.fake_release = tmp / "fake-claude-release"
        self.stub_dir = tmp / "stub"
        self.stub_dir.mkdir(exist_ok=True)
        self.hold = tmp / "supervisor-release"
        env = dict(os.environ)
        env.update({
            "KINDLED_HOME": str(self.home),
            "PYTHONPATH": str(REPO),
            "KINDLED_TEST_HOOKS": "1",
            "KINDLED_TEST_STUB_TASKS": str(tasks),
            "KINDLED_TEST_STUB_DIR": str(self.stub_dir),
            "KINDLED_TEST_SUPERVISOR_HOLD": str(self.hold) if hold_supervisor else "",
            "KINDLED_TEST_CALLS": ",".join(calls),
            "PATH": str(REPO / "tests" / "bridge" / "fake_claude") + os.pathsep + os.environ.get("PATH", ""),
            "FAKE_CLAUDE_DIR": str(self.fake_dir),
            "FAKE_CLAUDE_RELEASE": str(self.fake_release),
        })
        argv = [sys.executable, "-m", "brain.bridge.runner", "--persona-dir", str(self.pd), "--client-origin", "tests"]
        if idle_s is not None:
            argv += ["--idle-shutdown-seconds", str(idle_s)]

        def _pre():
            if sighup_ignored:
                signal.signal(signal.SIGHUP, signal.SIG_IGN)

        self.proc = subprocess.Popen(argv, env=env, cwd=str(tmp), preexec_fn=_pre, start_new_session=new_group,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.state = self._wait_ready()

    def _wait_ready(self) -> state_file.BridgeState:
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            assert self.proc.poll() is None, f"runner exited early: {self.proc.returncode}"
            st = state_file.read(self.pd)
            if st is not None and st.pid == self.proc.pid and st.port:
                try:
                    r = httpx.get(f"http://127.0.0.1:{st.port}/health",
                                  headers={"Authorization": f"Bearer {st.auth_token}"}, timeout=2)
                    if r.status_code == 200:
                        return st
                except httpx.HTTPError:
                    pass
            time.sleep(0.1)
        raise AssertionError("bridge did not become ready")

    def post(self, path: str, body: dict | None = None) -> httpx.Response:
        return httpx.post(f"http://127.0.0.1:{self.state.port}{path}", json=body,
                          headers={"Authorization": f"Bearer {self.state.auth_token}"}, timeout=10)

    def release(self, i: int) -> None:
        (self.stub_dir / f"release-{i}").write_text("go")

    def release_all(self, n: int) -> None:
        for i in range(n):
            self.release(i)
        self.hold.write_text("go")
        self.fake_release.write_text("go")

    def record(self) -> state_file.BridgeState | None:
        return state_file.read(self.pd)

    def progress(self) -> dict | None:
        return shutdown_progress.read_record(self.pd, shutdown_progress.PROGRESS_FILENAME)

    def lock_free(self) -> bool:
        from brain.bridge.daemon import acquire_lock, release_lock

        fd = acquire_lock(self.pd)
        if fd is None:
            return False
        release_lock(self.pd, fd)
        return True

    def wait_exit(self, timeout: float = 60.0) -> int:
        return self.proc.wait(timeout=timeout)

    def fake_pids(self) -> list[int]:
        pids: list[int] = []
        for f in self.fake_dir.glob("pids-*"):
            pids += [int(x) for x in f.read_text().split()]
        return pids

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(10)
        import psutil

        for pid in self.fake_pids():
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass


@pytest.fixture
def bridges():
    started: list[Bridge] = []
    yield started
    for b in started:
        b.release_all(8)
        b.kill()


def _until(pred, timeout: float = 30.0, step: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("condition not met in time")


def test_http_stop_waits_for_work_reports_progress_and_ends_honestly(tmp_path, bridges):
    """C10, C11, C12, C13 (user stop), C15 (HTTP): no internal time limit; "stopping" in the first read after the
    202, never "stopped" (nor the lock free) while a task runs; the progress record counts the tasks down with
    names, pid + token, request time and reason; exit status 0."""
    b = Bridge(tmp_path, tasks=3)
    bridges.append(b)
    token = b.state.pid_token
    assert b.post("/supervisor/shutdown", {"reason": "user_stop"}).status_code == 202
    first = b.record()
    assert first.phase == "stopping" and first.stop_reason == "user_stop" and first.stop_requested_at
    rec = _until(lambda: (p := b.progress()) and p.get("count") == 3 and p)
    assert rec["tasks"] == ["stub task 0", "stub task 1", "stub task 2"]
    assert rec["pid"] == b.proc.pid and rec["pid_token"] == token and rec["reason"] == "user_stop"
    assert rec["requested_at"] == first.stop_requested_at
    time.sleep(2.0)  # well past any internal limit an earlier bridge had (none must apply now)
    assert b.proc.poll() is None and b.record().phase == "stopping" and not b.lock_free()
    for i, left in ((0, 2), (1, 1), (2, 0)):
        b.release(i)
        rec = _until(lambda left=left: (p := b.progress()) and p.get("count") == left and p)
        assert len(rec["tasks"]) == left
        if left:
            assert b.record().phase == "stopping" and not b.lock_free()
    assert b.wait_exit() == 0
    end = b.record()
    assert end.phase == "stopped" and end.shutdown_clean and end.pid is None and end.stop_reason == "user_stop"
    assert b.lock_free()


@pytest.mark.parametrize("reason", ["app_restart", None])
def test_http_stop_reasons(tmp_path, bridges, reason):
    """C13: app restart, and no body = "manual_restart"."""
    b = Bridge(tmp_path)
    bridges.append(b)
    assert b.post("/supervisor/shutdown", {"reason": reason} if reason else None).status_code == 202
    assert b.wait_exit() == 0
    assert b.record().stop_reason == (reason or "manual_restart")


def test_sigterm_stop_marks_stopping_within_one_poll_and_exits_0(tmp_path, bridges):
    """C11 (SIGTERM leg), C13 (OS signal), C15 (SIGTERM)."""
    b = Bridge(tmp_path, tasks=1)
    bridges.append(b)
    b.proc.send_signal(signal.SIGTERM)
    time.sleep(0.3)  # one monitor poll is 0.1 s
    st = b.record()
    assert st.phase == "stopping" and st.stop_reason == "os_signal"
    assert b.proc.poll() is None
    b.release(0)
    assert b.wait_exit() == 0
    assert b.record().phase == "stopped" and b.record().shutdown_clean


def test_idle_stop_records_its_reason(tmp_path, bridges):
    b = Bridge(tmp_path, idle_s=1.0)
    bridges.append(b)
    assert b.wait_exit(90) == 0
    assert b.record().stop_reason == "idle_timeout" and b.record().phase == "stopped"


@pytest.mark.parametrize("gap", ["during-wait", "immediate"])
def test_second_sigint_is_a_hard_exit(tmp_path, bridges, gap):
    """C42 (c) [P2]: a second SIGINT during the lifespan wait, or within 0.1 s of the first, takes the hard-exit
    path: not clean, reason hard exit, exit 0; "stopped" is never written while the task runs."""
    b = Bridge(tmp_path, tasks=1)
    bridges.append(b)
    b.proc.send_signal(signal.SIGINT)
    if gap == "during-wait":
        _until(lambda: (st := b.record()) and st.phase == "stopping")
        time.sleep(0.3)
    else:
        # Within the first 0.1 s, but not while the first is still pending (the kernel merges two pending SIGINTs
        # into one delivery).
        time.sleep(0.03)
    b.proc.send_signal(signal.SIGINT)
    assert b.wait_exit() == 0
    st = b.record()
    assert st.phase == "hard_exit" and st.shutdown_clean is False and st.stop_reason == "hard_exit"


def test_hard_exit_request_naming_the_bridge_ends_it(tmp_path, bridges):
    """C13 (hard exit) and the request channel (D6 b): a request for this pid + token ends the bridge at once, exit 0,
    not clean; a request naming another token is ignored."""
    b = Bridge(tmp_path, tasks=1)
    bridges.append(b)
    assert b.post("/supervisor/shutdown", {"reason": "user_stop"}).status_code == 202
    shutdown_progress.write_record(b.pd, shutdown_progress.REQUEST_FILENAME,
                                   {"pid": b.proc.pid, "pid_token": "linux:other:1:1"})
    time.sleep(0.5)
    assert b.proc.poll() is None
    shutdown_progress.write_record(b.pd, shutdown_progress.REQUEST_FILENAME,
                                   {"pid": b.proc.pid, "pid_token": b.state.pid_token})
    assert b.wait_exit(30) == 0
    st = b.record()
    assert st.phase == "hard_exit" and not st.shutdown_clean and st.stop_reason == "hard_exit"


def test_watcher_files_are_deleted_only_by_the_lock_holder(tmp_path, bridges):
    """C55, C43 (a): files left by an earlier process are deleted by the runner that takes the lock; a runner that
    loses the lock (exit 2) deletes nothing."""
    b = Bridge(tmp_path)
    bridges.append(b)
    for name in shutdown_progress.WATCHER_FILENAMES:
        shutdown_progress.write_record(b.pd, name, {"pid": 1, "pid_token": "old"})
    env = dict(os.environ, KINDLED_HOME=str(b.home), PYTHONPATH=str(REPO))
    rc = subprocess.run([sys.executable, "-m", "brain.bridge.runner", "--persona-dir", str(b.pd)], env=env,
                        timeout=60, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    assert rc == 2
    assert all((b.pd / n).exists() for n in shutdown_progress.WATCHER_FILENAMES)
    assert b.post("/supervisor/shutdown", {"reason": "user_stop"}).status_code == 202
    assert b.wait_exit() == 0
    for name in shutdown_progress.WATCHER_FILENAMES:
        if name != shutdown_progress.PROGRESS_FILENAME:
            shutdown_progress.write_record(b.pd, name, {"pid": 1, "pid_token": "old"})
    b2 = Bridge(tmp_path)
    bridges.append(b2)
    assert not (b2.pd / shutdown_progress.REQUEST_FILENAME).exists()
    assert not (b2.pd / shutdown_progress.INTENT_FILENAME).exists()


def test_a_waiting_cmd_run_keeps_its_fresh_record(tmp_path, bridges):
    """C42 (a, b): the old bridge writes nothing after it releases the lock, so a `nell supervisor run` waiting for the
    lock keeps the record it writes when it takes over."""
    b = Bridge(tmp_path, tasks=1)
    bridges.append(b)
    env = dict(os.environ, KINDLED_HOME=str(b.home), PYTHONPATH=str(REPO), KINDLED_TEST_HOOKS="1",
               KINDLED_TEST_STUB_TASKS="0")
    waiter = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from types import SimpleNamespace; from brain.bridge import daemon; "
         "sys.exit(daemon.cmd_run(SimpleNamespace(persona='nell', idle_shutdown=0, client_origin='tests')))"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3.0)
        assert waiter.poll() is None
        assert b.post("/supervisor/shutdown", {"reason": "app_restart"}).status_code == 202
        b.release(0)
        assert b.wait_exit() == 0
        st = _until(lambda: (s := b.record()) and s.pid == waiter.pid and s.phase == "running" and s, timeout=60)
        time.sleep(1.0)
        assert b.record().pid == waiter.pid and b.record().phase == "running"
        assert st.pid_token
    finally:
        waiter.send_signal(signal.SIGTERM)
        try:
            waiter.wait(60)
        except subprocess.TimeoutExpired:
            waiter.kill()
            waiter.wait(10)


def test_sighup_starts_a_graceful_stop(tmp_path, bridges):
    """C64 (bridge half; the fake `claude` child half is in the I5 tests): SIGHUP to a terminal-run runner starts a
    normal stop: the stopping mark, waits for the task, exit 0."""
    b = Bridge(tmp_path, tasks=1)
    bridges.append(b)
    b.proc.send_signal(signal.SIGHUP)
    _until(lambda: (st := b.record()) and st.phase == "stopping")
    time.sleep(0.5)
    assert b.proc.poll() is None
    b.release(0)
    assert b.wait_exit() == 0
    assert b.record().phase == "stopped" and b.record().stop_reason == "os_signal"


def test_a_runner_started_with_sighup_ignored_keeps_ignoring_it(tmp_path, bridges):
    """C64 (S5-h): nohup-style start (SIGHUP ignored) -> SIGHUP does not stop the bridge."""
    b = Bridge(tmp_path, sighup_ignored=True)
    bridges.append(b)
    b.proc.send_signal(signal.SIGHUP)
    time.sleep(1.0)
    assert b.proc.poll() is None and b.record().phase == "running"
    assert b.post("/supervisor/shutdown", {"reason": "user_stop"}).status_code == 202
    assert b.wait_exit() == 0
