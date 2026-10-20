"""launchd service kill end to end, with a fake launchd modelled on the real one (C62; D17, D20, D25).

The fake launchd:
- ``bootout`` sends SIGTERM to the job and returns when the job exits, or (for our call) at the cap;
- SIGKILLs the job's process group only at ``ExitTimeOut`` (300 s, on an injected clock);
- respawns a loaded job that exits with any status other than 0 after a short throttle (``KeepAlive {Crashed,
  SuccessfulExit: false}``); a job that exits 0 is not respawned; a respawn runs the real ``nell supervisor run
  --client-origin launchd`` (with the respawn guard);
- ``print`` shows ``pid = N`` for a loaded job and fails for an unloaded one; ``bootstrap`` loads (and, RunAtLoad,
  starts) the job; plain ``kickstart`` starts a loaded job that is not running.
Modes: (m1) unloaded when ``bootout`` starts; (m2) still listed (and respawn-eligible) until the job exits; (m3)
``bootout`` returns error 5 without unloading; (m4) ``bootout`` succeeds within the cap (the job honours SIGTERM; its
children, in their own sessions, stay for the helper).

The killed job is a stub bridge that ignores the hard-exit request, with a fake ``claude`` child (own session) and an
MCP-named grandchild. The kill must leave no bridge and no descendant, and stay so across the respawn window and past
``ExitTimeOut``. Then every start path clears the stop intent, runs ``print`` -> ``bootstrap`` (when unloaded) ->
plain ``kickstart``, and reaches a healthy bridge.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from brain import process_identity as pi
from brain.bridge import daemon, kill, shutdown_progress, state_file

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the fake launchd runs on Linux CI/dev")

REPO = Path(__file__).resolve().parents[2]
EXIT_TIMEOUT_S = 300.0
THROTTLE_S = 0.5

_STUB_BRIDGE = textwrap.dedent(
    """
    # A launchd-owned stub bridge: records itself (new format, with its identity token), starts a fake `claude` in its
    # own session with an MCP-named grandchild, ignores the hard-exit request; honours SIGTERM only when told to.
    import os, signal, subprocess, sys, time
    from pathlib import Path
    from brain import process_identity as pi
    from brain.bridge import state_file
    pd, honour = Path(sys.argv[1]), sys.argv[2] == "1"
    # honour: SIGTERM ends the job with status 0 (so launchd does not respawn it); otherwise SIGTERM is ignored.
    signal.signal(signal.SIGTERM, (lambda *a: os._exit(0)) if honour else signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, "-c",
        "import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', "
        "'brain.mcp_server']); open(sys.argv[1], 'w').write(str(g.pid)); time.sleep(600)",
        str(pd / "stub-grand"), "claude"], start_new_session=True)
    (pd / "stub-child").write_text(str(child.pid))
    state_file.write(pd, state_file.BridgeState(persona="nell", pid=os.getpid(), port=1, started_at="t",
        stopped_at=None, shutdown_clean=False, client_origin="launchd", pid_token=pi.own_token(), identity="token",
        phase="running"))
    while True:
        time.sleep(0.1)
    """
)


class FakeLaunchd:
    def __init__(self, pd: Path, env: dict, mode: str):
        self.pd, self.env, self.mode = pd, env, mode
        self.loaded = True
        self.job: subprocess.Popen | None = None
        self.respawns: list[int] = []
        self.calls: list[list[str]] = []
        self.clock = 0.0
        self.bootout_at: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watch = threading.Thread(target=self._keepalive, daemon=True)
        self._watch.start()

    # -- the job
    def start_stub(self, *, honour_sigterm: bool) -> None:
        self.job = subprocess.Popen([sys.executable, "-c", _STUB_BRIDGE, str(self.pd), "1" if honour_sigterm else "0"],
                                    env=self.env, start_new_session=True)

    def _start_real(self) -> None:
        code = ("import sys; from types import SimpleNamespace; from brain.bridge import daemon; "
                "sys.exit(daemon.cmd_run(SimpleNamespace(persona='nell', idle_shutdown=0, client_origin='launchd')))")
        self.job = subprocess.Popen([sys.executable, "-c", code], env=self.env, start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _keepalive(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job = self.job
            if job is not None and job.poll() is not None:
                rc = job.returncode
                with self._lock:
                    self.job = None
                    respawn = self.loaded and rc != 0
                if respawn:
                    time.sleep(THROTTLE_S)
                    with self._lock:
                        if self.loaded:
                            self._start_real()
                            self.respawns.append(self.job.pid)
            time.sleep(0.05)

    def close(self) -> None:
        self._stop.set()
        if self.job is not None and self.job.poll() is None:
            self.job.kill()
            self.job.wait(10)

    # -- launchctl
    def launchctl(self, args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        verb = args[0]
        if verb == "print":
            with self._lock:
                if not self.loaded:
                    return subprocess.CompletedProcess(args, 113, "", "Could not find service")
                pid = self.job.pid if self.job is not None and self.job.poll() is None else None
            return subprocess.CompletedProcess(args, 0, f"state = running\n\tpid = {pid}\n" if pid else "state = waiting\n", "")
        if verb == "bootout":
            return self._bootout(timeout or 15.0)
        if verb == "bootstrap":
            with self._lock:
                self.loaded = True
                if self.job is None:
                    self._start_real()  # RunAtLoad
            return subprocess.CompletedProcess(args, 0, "", "")
        if verb == "kickstart":
            assert "-k" not in args
            with self._lock:
                if self.loaded and self.job is None:
                    self._start_real()
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 1, "", "unknown verb")

    def _bootout(self, cap: float) -> subprocess.CompletedProcess:
        self.bootout_at = self.clock
        if self.mode == "m3":
            return subprocess.CompletedProcess(["bootout"], 5, "", "Boot-out failed: 5: Input/output error")
        with self._lock:
            if self.mode == "m1":
                self.loaded = False
            job = self.job
        if job is not None and job.poll() is None:
            job.send_signal(signal.SIGTERM)
            try:
                job.wait(cap)
            except subprocess.TimeoutExpired:
                return subprocess.CompletedProcess(["bootout"], 124, "", "timed out")
        with self._lock:
            self.loaded = False
        return subprocess.CompletedProcess(["bootout"], 0, "", "")

    def advance_past_exit_timeout(self) -> None:
        """ExitTimeOut: SIGKILL the job's process group (only), if bootout had started and the job still runs."""
        self.clock += EXIT_TIMEOUT_S + 1
        if self.bootout_at is not None and self.job is not None and self.job.poll() is None:
            try:
                os.killpg(self.job.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def env_pd(tmp_path, monkeypatch):
    home = tmp_path / "home"
    pd = home / "personas" / "nell"
    (pd / "active_conversations").mkdir(parents=True)
    (pd / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}', encoding="utf-8")
    monkeypatch.setenv("KINDLED_HOME", str(home))
    env = dict(os.environ, KINDLED_HOME=str(home), PYTHONPATH=str(REPO), KINDLED_TEST_HOOKS="1",
               KINDLED_TEST_STUB_TASKS="0")
    return env, pd


@pytest.fixture
def launchd_seams(monkeypatch, tmp_path):
    from brain.service import launchd

    holder: dict = {}
    monkeypatch.setattr(launchd, "run_launchctl", lambda args, timeout=None: holder["fake"].launchctl(args, timeout))
    plist = tmp_path / "agent.plist"
    plist.write_text("<plist/>")
    monkeypatch.setattr(launchd, "paths_for_persona", lambda persona: SimpleNamespace(plist_path=plist, label="x"))
    real_owned, real_stop = kill.owned_by_service, kill.manager_stop
    monkeypatch.setattr(kill, "service_installed", lambda persona, platform=None: True)
    monkeypatch.setattr(kill, "owned_by_service", lambda p, pid, origin: real_owned(p, pid, origin, platform="darwin"))
    monkeypatch.setattr(kill, "manager_stop", lambda p: real_stop(p, platform="darwin"))
    monkeypatch.setattr(kill, "MANAGER_CAP_S", 2.0)
    monkeypatch.setattr(daemon, "_service_installed", lambda persona: True)
    monkeypatch.setattr(daemon, "_start_service", lambda persona: launchd.start_service(persona=persona))
    return holder


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_file(path: Path, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text().strip():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.05)
    return int(path.read_text())


def _no_bridge(pd: Path) -> bool:
    st = state_file.read(pd)
    return st is None or state_file.bridge_liveness(st, pd) != state_file.ALIVE


@pytest.mark.parametrize("mode", ["m1", "m2", "m3", "m4"])
def test_a_launchd_owned_bridge_is_killed_with_its_tree_and_stays_down(env_pd, launchd_seams, mode):
    env, pd = env_pd
    fake = FakeLaunchd(pd, env, mode)
    launchd_seams["fake"] = fake
    try:
        fake.start_stub(honour_sigterm=(mode == "m4"))
        child = _wait_file(pd / "stub-child")
        grand = _wait_file(pd / "stub-grand")
        stub = fake.job.pid
        _wait_until = time.monotonic() + 30
        while state_file.read(pd) is None or state_file.read(pd).pid != stub:
            assert time.monotonic() < _wait_until
            time.sleep(0.05)
        t0 = time.monotonic()
        result = kill.kill_bridge(pd, timeout=1.0)
        assert time.monotonic() - t0 <= kill.kill_bound_s(1.0) + 5.0
        assert result.outcome == "killed", result.message
        for pid in (stub, child, grand):
            assert not _alive(pid), (mode, pid)
        assert kill.read_intent(pd)["state"] == "committed"
        time.sleep(THROTTLE_S + 8.0)  # the respawn window: a respawn runs and the guard ends it
        fake.advance_past_exit_timeout()
        time.sleep(1.0)
        for pid in (stub, child, grand):
            assert not _alive(pid)
        assert _no_bridge(pd)
        expected_respawns = 1 if mode in ("m2", "m3") else 0
        assert len(fake.respawns) == expected_respawns, (mode, fake.respawns)
        for pid in fake.respawns:
            assert not _alive(pid)
        assert kill.read_intent(pd) is not None  # the guard never deletes it
    finally:
        fake.close()


def _healthy(pd: Path, timeout: float = 60.0) -> state_file.BridgeState:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = state_file.read(pd)
        if st is not None and st.port and state_file.bridge_liveness(st, pd) == state_file.ALIVE:
            try:
                if httpx.get(f"http://127.0.0.1:{st.port}/health",
                             headers={"Authorization": f"Bearer {st.auth_token}"}, timeout=2).status_code == 200:
                    return st
            except httpx.HTTPError:
                pass
        time.sleep(0.1)
    raise AssertionError("no healthy bridge")


def _stop(pd: Path, fake: FakeLaunchd) -> None:
    rc = daemon.cmd_stop(SimpleNamespace(persona="nell", timeout=30.0, force=False, kill=False,
                                         skip_hard_exit_request=False))
    assert rc == 0
    deadline = time.monotonic() + 30
    while fake.job is not None and time.monotonic() < deadline:
        time.sleep(0.05)


def test_every_start_path_after_a_kill_reaches_a_healthy_bridge(env_pd, launchd_seams, monkeypatch):
    """C62 (start half), C58 (c): after the kill (m2), each start path clears the stop intent and runs print ->
    bootstrap (when unloaded) -> plain kickstart, and the bridge is healthy (not stopped by the guard)."""
    from brain import cli

    env, pd = env_pd
    fake = FakeLaunchd(pd, env, "m1")  # m1 leaves the job unloaded: the first start must bootstrap
    launchd_seams["fake"] = fake
    monkeypatch.setattr(cli, "_chat_via_bridge", lambda args, persona_dir, readiness=None: 0)
    try:
        fake.start_stub(honour_sigterm=False)
        _wait_file(pd / "stub-child")
        _wait_file(pd / "stub-grand")
        assert kill.kill_bridge(pd, timeout=1.0).outcome == "killed"
        assert not fake.loaded
        starts = {
            "supervisor start": lambda: daemon.cmd_start(SimpleNamespace(persona="nell", idle_shutdown=0,
                                                                         client_origin="tauri")),
            "restart": lambda: daemon.cmd_restart(SimpleNamespace(persona="nell", idle_shutdown=0,
                                                                  client_origin="cli", timeout=30.0, force=False,
                                                                  kill=False, skip_hard_exit_request=False)),
            "chat auto-start": lambda: cli._chat_handler(SimpleNamespace(persona="nell", no_bridge=False,
                                                                         bridge_only=False)),
            "app (ensure_bridge_running argv)": lambda: cli.main(["supervisor", "start", "--persona", "nell",
                                                                  "--client-origin", "tauri"]),
        }
        for name, start in starts.items():
            fake.calls.clear()
            unloaded = not fake.loaded
            if name != "supervisor start":
                kill._write_intent(pd, shutdown_progress.stop_intent(
                    pid=999_999, pid_token="linux:x:999999:1", state="committed", nonce="n", at="2026-10-19T00:00:00",
                    boot=pi.boot_id(), legacy=False, helper_pid=999_998, helper_token="linux:x:999998:1"))
            assert start() == 0, name
            assert kill.read_intent(pd) is None, name
            verbs = [c[0] for c in fake.calls if c[0] in ("print", "bootstrap", "kickstart")]
            assert verbs[:3] == (["print", "bootstrap", "kickstart"] if unloaded else ["print", "kickstart"])[:3], (
                name, verbs)
            _healthy(pd)
            _stop(pd, fake)
    finally:
        fake.close()
