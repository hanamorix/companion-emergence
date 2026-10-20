"""The kill helper (`brain/bridge/kill.py`): C35 (Python), C45 a, b, c, d, g, h, i; D17, D22, D25.

A fake bridge is a real process tree: a sleeper "bridge" with a "claude" child that has an MCP-named grandchild. Its
bridge.json carries its real identity token. Service ownership and the manager call are spied (the Linux / macOS /
Windows manager legs of C45 e and C62 have their own tests).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

from brain import process_identity as pi
from brain.bridge import kill, shutdown_progress, state_file

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and suspend; Windows leg in CI")

_FAKE_BRIDGE = textwrap.dedent(
    """
    import os, subprocess, sys, time
    from pathlib import Path
    pids, request, honour = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
    child = subprocess.Popen([sys.executable, "-c",
        "import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', "
        "'brain.mcp_server']); print(g.pid, flush=True); time.sleep(600)", "claude"], stdout=subprocess.PIPE, text=True)
    grand = int(child.stdout.readline())
    pids.write_text(f"{os.getpid()} {child.pid} {grand}")
    while True:
        if honour and request.exists():
            os._exit(0)  # an accepted hard exit that leaves its child and grandchild behind
        time.sleep(0.05)
    """
)


class FakeBridge:
    def __init__(self, tmp: Path, *, honour_request: bool = False):
        self.pd = tmp / "nell"
        self.pd.mkdir(exist_ok=True)
        pids = tmp / f"pids-{time.monotonic_ns()}"
        request = shutdown_progress.path(self.pd, shutdown_progress.REQUEST_FILENAME)
        self.proc = subprocess.Popen([sys.executable, "-c", _FAKE_BRIDGE, str(pids), str(request),
                                      "1" if honour_request else "0"])
        deadline = time.monotonic() + 30
        while not pids.exists() or len(pids.read_text().split()) < 3:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        self.pid, self.child, self.grand = (int(x) for x in pids.read_text().split())
        state_file.write(self.pd, state_file.BridgeState(
            persona="nell", pid=self.pid, port=1, started_at="t", stopped_at=None, shutdown_clean=False,
            client_origin="cli", pid_token=pi.token_of(self.pid), identity="token", phase="running"))

    @property
    def pids(self) -> list[int]:
        return [self.pid, self.child, self.grand]

    def statuses(self) -> dict[int, str]:
        out = {}
        for pid in self.pids:
            try:
                out[pid] = psutil.Process(pid).status()
            except psutil.NoSuchProcess:
                out[pid] = "gone"
        return out

    def all_gone(self) -> bool:
        return all(s in ("gone", psutil.STATUS_ZOMBIE) for s in self.statuses().values())

    def cleanup(self) -> None:
        for pid in self.pids:
            try:
                p = psutil.Process(pid)
                p.resume()
                p.kill()
            except psutil.Error:
                pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            pass


@pytest.fixture
def fake(tmp_path):
    made: list[FakeBridge] = []

    def _make(**kw) -> FakeBridge:
        b = FakeBridge(tmp_path, **kw)
        made.append(b)
        return b

    yield _make
    for b in made:
        b.cleanup()


@pytest.fixture(autouse=True)
def _not_a_service(monkeypatch):
    monkeypatch.setattr(kill, "service_installed", lambda persona: False)


def _intent(pd: Path) -> dict | None:
    return kill.read_intent(pd)


def test_a_stale_or_unproven_record_is_not_running_and_nothing_is_written(tmp_path, fake):
    """C35 (Python): --kill kills only a running bridge whose token matches."""
    b = fake()
    st = state_file.read(b.pd)
    st.pid_token = f"linux:other:{b.pid}:1"
    state_file.write(b.pd, st)
    r = kill.kill_bridge(b.pd, timeout=2)
    assert r.outcome == "not_running"
    assert _intent(b.pd) is None
    assert all(s != "gone" for s in b.statuses().values())


def test_unknown_identity_refuses(tmp_path, fake, monkeypatch):
    b = fake()
    monkeypatch.setattr(pi, "token_of", lambda pid: pi.UNKNOWN)
    r = kill.kill_bridge(b.pd, timeout=2)
    assert r.outcome == "unknown_identity" and _intent(b.pd) is None


def test_own_kill_freezes_kills_the_bridge_first_then_the_tree_and_commits(tmp_path, fake, monkeypatch):
    """C45 (a, b, c own kill, h): pending -> committed before the first suspend; the bridge is suspended first, killed
    before its descendants; everything is gone; the committed intent stays; one psutil.Process for the bridge."""
    b = fake()
    order: list[str] = []
    objects: set[int] = set()
    real_suspend, real_kill = psutil.Process.suspend, psutil.Process.kill

    def _suspend(self):
        if self.pid == b.pid:
            objects.add(id(self))
        order.append(f"suspend {self.pid}")
        return real_suspend(self)

    def _kill(self):
        if self.pid == b.pid:
            objects.add(id(self))
        order.append(f"kill {self.pid}")
        return real_kill(self)

    monkeypatch.setattr(psutil.Process, "suspend", _suspend)
    monkeypatch.setattr(psutil.Process, "kill", _kill)
    states: list[str] = []
    r = kill.kill_bridge(b.pd, timeout=10, _hooks={
        "after_enumeration": lambda: states.append(_intent(b.pd)["state"]),
        "committed": lambda: states.append(_intent(b.pd)["state"]),
    })
    assert r.outcome == "killed" and not r.survivors
    assert states == ["pending", "committed"]
    assert order[0] == f"suspend {b.pid}"
    kills = [o for o in order if o.startswith("kill")]
    assert kills[0] == f"kill {b.pid}"
    assert b.all_gone()
    it = _intent(b.pd)
    assert it["state"] == "committed" and it["reason"] == "user_kill" and it["pid"] == b.pid
    assert it["pid_token"] == state_file.read(b.pd).pid_token and it["nonce"]
    assert len(objects) == 1  # (h) one psutil.Process object for the bridge, from the check to the kill


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGHUP", "SIGINT"])
@pytest.mark.parametrize("when", ["before_bridge_kill", "after_bridge_suspend"])
def test_a_signal_after_the_freeze_before_the_bridge_kill_resumes_everything(tmp_path, fake, sig, when):
    """C45 (a iii, g): the intent is deleted and every suspended process runs again."""
    b = fake()

    def _signal():
        os.kill(os.getpid(), getattr(signal, sig))
        time.sleep(0.2)

    with pytest.raises((kill._KillInterrupted, KeyboardInterrupt)):
        kill.kill_bridge(b.pd, timeout=5, _hooks={when: _signal})
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values()), b.statuses()


def test_a_second_signal_while_resuming_is_ignored(tmp_path, fake):
    """C45 (g): signals are masked while the finally resumes."""
    b = fake()
    calls = {"n": 0}

    def _first():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)

    def _second():
        calls["n"] += 1
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)

    with pytest.raises(kill._KillInterrupted):
        kill.kill_bridge(b.pd, timeout=5, _hooks={"before_bridge_kill": _first, "finally_start": _second})
    assert calls["n"] == 1
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())


def test_a_signal_after_the_bridge_kill_returned_kills_the_tree_and_keeps_the_intent(tmp_path, fake):
    """C45 (g): after the bridge's kill call returned, the finally kills the frozen descendants (no resume)."""
    b = fake()

    def _signal():
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.2)

    with pytest.raises(kill._KillInterrupted):
        kill.kill_bridge(b.pd, timeout=5, _hooks={"after_bridge_kill": _signal})
    deadline = time.monotonic() + 10
    while not b.all_gone() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert b.all_gone()
    assert _intent(b.pd)["state"] == "committed"


def test_a_raising_suspend_resumes_and_deletes_the_intent(tmp_path, fake, monkeypatch):
    """C45 (a): an error before the bridge kill (a raising suspend) cleans up and is reported to the caller."""
    b = fake()
    real_suspend = psutil.Process.suspend

    def _suspend(self):
        if self.pid == b.child:
            raise RuntimeError("suspend failed")
        return real_suspend(self)

    monkeypatch.setattr(psutil.Process, "suspend", _suspend)
    with pytest.raises(RuntimeError):
        kill.kill_bridge(b.pd, timeout=5)
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())


def test_a_committed_intent_of_another_helper_is_never_downgraded(tmp_path, fake):
    """C45 (i)."""
    b = fake()
    st = state_file.read(b.pd)
    theirs = {"pid": b.pid, "pid_token": st.pid_token, "reason": "user_kill", "state": "committed", "nonce": "aaaa",
              "at": "t", "boot": None, "legacy": False}
    kill._write_intent(b.pd, theirs)
    r = kill.kill_bridge(b.pd, timeout=2)
    assert r.outcome == "in_progress"
    assert _intent(b.pd) == theirs
    assert all(s != "gone" for s in b.statuses().values())


def test_no_commit_when_another_bridge_started_meanwhile(tmp_path, fake):
    """C45 (i): another live bridge for the persona appeared during the helper's run: no commit, the pending intent is
    deleted, nothing killed."""
    b = fake()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        def _new_bridge():
            state_file.write(b.pd, state_file.BridgeState(
                persona="nell", pid=other.pid, port=2, started_at="t2", stopped_at=None, shutdown_clean=False,
                client_origin="cli", pid_token=pi.token_of(other.pid), identity="token", phase="running"))

        r = kill.kill_bridge(b.pd, timeout=2, _hooks={"after_enumeration": _new_bridge})
        assert r.outcome == "other_bridge"
        assert _intent(b.pd) is None
        assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())
    finally:
        other.kill()
        other.wait()


def test_an_unreadable_boot_id_stores_null_and_the_kill_proceeds(tmp_path, fake, monkeypatch):
    """C45 (a)."""
    b = fake()
    monkeypatch.setattr(pi, "boot_id", lambda: None)
    seen: list = []
    r = kill.kill_bridge(b.pd, timeout=10, _hooks={"after_enumeration": lambda: seen.append(_intent(b.pd))})
    assert seen[0]["boot"] is None and r.outcome == "killed" and b.all_gone()


def test_a_call_off_the_main_thread_installs_no_handlers(tmp_path, fake, monkeypatch):
    """C45 (a): handlers only from the main thread."""
    import threading

    b = fake()
    installed: list = []
    real_signal = signal.signal
    monkeypatch.setattr(signal, "signal", lambda s, h: installed.append(s) or real_signal(s, h))
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("r", kill.kill_bridge(b.pd, timeout=10)))
    t.start()
    t.join(30)
    assert out["r"].outcome == "killed" and installed == []


# ---------------------------------------------------------------- service-owned (spied manager)


@pytest.fixture
def owned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(kill, "service_installed", lambda persona: True)
    monkeypatch.setattr(kill, "owned_by_service", lambda persona, pid, origin: True)
    return calls


def test_owned_bridge_that_accepts_the_hard_exit_leaves_no_descendant(tmp_path, fake, owned, monkeypatch):
    """C45 (c, accepted hard exit): the bridge exits 0 on the request, leaving its child and grandchild; the helper
    kills the remembered tree; no manager call was needed."""
    b = fake(honour_request=True)
    monkeypatch.setattr(kill, "manager_stop", lambda persona: owned.append("manager"))
    r = kill.kill_bridge(b.pd, timeout=10)
    assert r.outcome == "killed" and b.all_gone()
    assert owned == []
    assert _intent(b.pd)["state"] == "committed"


def test_owned_bridge_ignoring_the_request_gets_the_manager_then_the_helper_kills_the_rest(
    tmp_path, fake, owned, monkeypatch
):
    """C45 (c manager-ended, e systemd/launchd rc 0 ending ONLY the bridge, d bound)."""
    b = fake()

    def _manager_ends_only_the_bridge(persona):
        owned.append("manager")
        os.kill(b.pid, signal.SIGKILL)
        b.proc.wait(5)

    monkeypatch.setattr(kill, "manager_stop", _manager_ends_only_the_bridge)
    t0 = time.monotonic()
    r = kill.kill_bridge(b.pd, timeout=1.0)
    assert time.monotonic() - t0 <= kill.kill_bound_s(1.0) + 1.0
    assert owned == ["manager"] and r.outcome == "killed"
    assert b.all_gone()


@pytest.mark.parametrize("sig", ["SIGTERM", "SIGINT"])
@pytest.mark.parametrize("when", ["hard_exit_wait", "manager"])
def test_a_signal_during_the_wait_or_the_manager_call_freezes_nothing_and_deletes_the_intent(
    tmp_path, fake, owned, monkeypatch, sig, when
):
    """C45 (a i, ii)."""
    b = fake()

    def _signal(*_a):
        os.kill(os.getpid(), getattr(signal, sig))
        time.sleep(0.2)

    if when == "manager":
        monkeypatch.setattr(kill, "manager_stop", _signal)
        hooks = {}
    else:
        hooks = {"hard_exit_wait": _signal}
    with pytest.raises((kill._KillInterrupted, KeyboardInterrupt)):
        kill.kill_bridge(b.pd, timeout=0.5 if when == "manager" else 5, _hooks=hooks)
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())


def test_skip_hard_exit_request_sends_none_and_waits_not(tmp_path, fake, owned, monkeypatch):
    """C45 (d): with --skip-hard-exit-request there is no request and no first wait."""
    b = fake()
    monkeypatch.setattr(kill, "manager_stop", lambda persona: owned.append("manager"))
    t0 = time.monotonic()
    r = kill.kill_bridge(b.pd, timeout=30, skip_hard_exit_request=True)
    assert time.monotonic() - t0 < 10
    assert not shutdown_progress.path(b.pd, shutdown_progress.REQUEST_FILENAME).exists()
    assert r.outcome == "killed" and b.all_gone()


def test_kill_bound_matches_the_stated_formula():
    assert kill.kill_bound_s(180) == 180 + 2 * 15 + 180
    assert kill.kill_bound_s(180, skip_hard_exit_request=True) == 2 * 15 + 180



def test_an_unsuspendable_descendant_does_not_make_the_freeze_loop_spin(tmp_path, fake, monkeypatch):
    """C45 (d), review M3: a descendant whose suspend() raises stays listed; the freeze still ends (every pid tried is
    remembered), the bridge is killed first and the descendant is killed with the tree."""
    import threading

    b = fake()
    real_suspend = psutil.Process.suspend

    def _suspend(self):
        if self.pid == b.child:
            raise psutil.AccessDenied(self.pid)
        return real_suspend(self)

    monkeypatch.setattr(psutil.Process, "suspend", _suspend)
    out: dict = {}
    t = threading.Thread(target=lambda: out.setdefault("r", kill.kill_bridge(b.pd, timeout=5)), daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "the freeze loop never ended"
    assert out["r"].outcome == "killed"
    deadline = time.monotonic() + 10
    while not b.all_gone() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert b.all_gone()


def test_a_bridge_that_exits_just_before_its_suspend_still_has_its_tree_killed(tmp_path, fake, monkeypatch):
    """Review S4: the bridge ends between the liveness check and suspend(); the helper treats it as gone, kills the
    remembered descendants and keeps its committed intent (no traceback)."""
    b = fake()
    real_suspend = psutil.Process.suspend

    def _suspend(self):
        if self.pid == b.pid:
            os.kill(b.pid, signal.SIGKILL)
            b.proc.wait(5)
            raise psutil.NoSuchProcess(self.pid)
        return real_suspend(self)

    monkeypatch.setattr(psutil.Process, "suspend", _suspend)
    r = kill.kill_bridge(b.pd, timeout=5)
    assert r.outcome == "killed"
    deadline = time.monotonic() + 10
    while not b.all_gone() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert b.all_gone()
    assert _intent(b.pd)["state"] == "committed"
