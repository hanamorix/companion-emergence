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


def _signal_self(sig) -> None:
    """Send ``sig`` to this process, but only when a handler is installed for it.

    A signal whose disposition is the default (SIG_DFL) would terminate this process, which under
    pytest-xdist is the worker running a whole share of the suite. SIG_IGN is fine (``kill_bridge``
    ignores the stop signals while it resumes, and one test sends a second signal from there)."""
    handler = signal.getsignal(sig)
    assert handler is not signal.SIG_DFL, f"{sig!r} has no handler: sending it would kill this process"
    os.kill(os.getpid(), sig)


def test_signal_self_refuses_a_signal_that_would_kill_the_process():
    sig = signal.SIGUSR1
    seen = []
    previous = signal.signal(sig, signal.SIG_DFL)
    try:
        with pytest.raises(AssertionError, match="no handler"):
            _signal_self(sig)
        signal.signal(sig, signal.SIG_IGN)
        _signal_self(sig)  # ignored: allowed, and nothing happens
        signal.signal(sig, lambda s, f: seen.append(s))
        _signal_self(sig)
        deadline = time.monotonic() + 5
        while not seen and time.monotonic() < deadline:  # the handler runs in this (main) thread soon after the kill
            time.sleep(0.01)
        assert seen == [sig]
    finally:
        signal.signal(sig, previous)


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
        _signal_self(getattr(signal, sig))
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
        _signal_self(signal.SIGTERM)
        time.sleep(0.2)

    def _second():
        calls["n"] += 1
        _signal_self(signal.SIGTERM)
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
        _signal_self(signal.SIGTERM)
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
        _signal_self(getattr(signal, sig))
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


def test_a_signal_during_the_hard_exit_wait_withdraws_the_request(tmp_path, fake, owned):
    """Review of #458 (finding 2): a kill called off during the wait leaves no hard-exit request behind; the bridge,
    still running, would otherwise act on it later (an unrecorded hard exit after the user cancelled)."""
    b = fake()

    def _signal(*_a):
        _signal_self(signal.SIGINT)
        time.sleep(0.2)

    with pytest.raises((kill._KillInterrupted, KeyboardInterrupt)):
        kill.kill_bridge(b.pd, timeout=5, _hooks={"hard_exit_wait": _signal})
    assert not shutdown_progress.path(b.pd, shutdown_progress.REQUEST_FILENAME).exists()


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


def test_a_signal_while_the_pending_intent_is_written_deletes_it_and_restores_the_handlers(tmp_path, fake, monkeypatch):
    """C45 (a), review S2: a signal in the window between the handlers and the try leaves no intent behind and the
    previous handlers back in place."""
    b = fake()
    before = signal.getsignal(signal.SIGTERM)
    real_write = kill._write_intent

    def _write_then_signal(pd, intent):
        real_write(pd, intent)
        _signal_self(signal.SIGTERM)
        time.sleep(0.2)

    monkeypatch.setattr(kill, "_write_intent", _write_then_signal)
    with pytest.raises(kill._KillInterrupted):
        kill.kill_bridge(b.pd, timeout=5)
    assert _intent(b.pd) is None
    assert signal.getsignal(signal.SIGTERM) is before
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())


def test_only_the_first_stop_signal_interrupts_the_helper(tmp_path, fake):
    """C45 (g), review S2: a second signal arriving before the finally masks signals cannot skip the resume."""
    b = fake()
    seen: dict = {}

    def _two_signals():
        handler = signal.getsignal(signal.SIGTERM)
        try:
            handler(signal.SIGTERM, None)
        except kill._KillInterrupted as first:
            try:
                handler(signal.SIGTERM, None)
                seen["second"] = "ignored"
            except kill._KillInterrupted:
                seen["second"] = "raised"
            raise first

    with pytest.raises(kill._KillInterrupted):
        kill.kill_bridge(b.pd, timeout=5, _hooks={"after_bridge_suspend": _two_signals})
    assert seen["second"] == "ignored"
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())


def test_a_committed_intent_whose_helper_is_gone_does_not_block_a_new_kill(tmp_path, fake):
    """Review S3: a helper killed after its commit leaves a committed intent naming a live bridge; once that helper is
    gone the intent is stale and a later kill proceeds (and finishes a frozen bridge)."""
    b = fake()
    st = state_file.read(b.pd)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = {"pid": b.pid, "pid_token": st.pid_token, "reason": "user_kill", "state": "committed", "nonce": "bbbb",
             "at": "2026-10-19T21:04:00.000000+00:00", "boot": None, "legacy": False,
             "helper_pid": dead.pid, "helper_token": f"linux:gone:{dead.pid}:1"}
    kill._write_intent(b.pd, stale)
    psutil.Process(b.pid).suspend()  # the dead helper had frozen it
    r = kill.kill_bridge(b.pd, timeout=5)
    assert r.outcome == "killed"
    assert b.all_gone()


def test_the_intent_names_the_helper(tmp_path, fake):
    """Review S3: the intent carries the helper's own pid + identity token."""
    b = fake()
    seen: list = []
    kill.kill_bridge(b.pd, timeout=10, _hooks={"after_enumeration": lambda: seen.append(_intent(b.pd))})
    assert seen[0]["helper_pid"] == os.getpid()
    assert seen[0]["helper_token"] == pi.own_token()


def test_no_commit_when_another_process_holds_the_bridge_lock(tmp_path, fake):
    """C45 (i) lock clause, review S10: the record still names the target, but another process took the persona's
    bridge lock meanwhile: no commit, the pending intent is deleted, nothing killed."""
    b = fake()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from pathlib import Path; from brain.bridge.daemon import acquire_lock; "
         "fd = acquire_lock(Path(sys.argv[1])); print('held' if fd is not None else 'free', flush=True); "
         "time.sleep(600)", str(b.pd)],
        stdout=subprocess.PIPE, text=True)
    try:
        holder_ready: dict = {}

        def _taken():
            holder_ready["line"] = holder.stdout.readline().strip()

        r = kill.kill_bridge(b.pd, timeout=2, _hooks={"after_enumeration": _taken})
        assert holder_ready["line"] == "held"
        assert r.outcome == "other_bridge"
        assert _intent(b.pd) is None
        assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())
    finally:
        holder.kill()
        holder.wait()


def test_the_target_holding_its_own_lock_is_not_another_bridge(tmp_path, fake):
    """C45 (i): the lock held by the target's own pid does not stop the kill."""
    b = fake()
    (b.pd / "bridge.json.lock").write_text(str(b.pid))
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX); "
         "print('held', flush=True); time.sleep(600)", str(b.pd / "bridge.json.lock")],
        stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        r = kill.kill_bridge(b.pd, timeout=10)
        assert r.outcome == "killed" and b.all_gone()
    finally:
        holder.kill()
        holder.wait()


_MONITOR_BRIDGE = textwrap.dedent(
    """
    import os, subprocess, sys, time
    from pathlib import Path
    from brain import process_identity as pi
    from brain.bridge import shutdown, state_file
    pd = Path(sys.argv[1])
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", "claude"])
    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=os.getpid(), port=1, started_at="t", stopped_at=None, shutdown_clean=False,
        client_origin="launchd", pid_token=pi.own_token(), identity="token", phase="running"))
    c = shutdown.BridgeShutdownController()
    c.configure(pd)
    m = shutdown.Monitor(c, pd, pid=os.getpid(), pid_token=pi.own_token(),
                         hard_exit=lambda reason: shutdown.hard_exit(pd, reason))
    m.start()
    print(child.pid, flush=True)
    while True:
        time.sleep(0.1)
    """
)


def test_a_running_service_bridge_acts_on_the_helpers_request_at_once(tmp_path, owned, monkeypatch):
    """Review M2: the bridge side is the REAL stop monitor of a bridge that was never asked to stop (a running service
    bridge): it honours the helper's hard-exit request at its next poll, so `--kill` does not wait out `--timeout`, and
    no manager call is needed."""
    pd = tmp_path / "nell"
    pd.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", _MONITOR_BRIDGE, str(pd)], stdout=subprocess.PIPE, text=True)
    child = int(proc.stdout.readline())
    try:
        monkeypatch.setattr(kill, "manager_stop", lambda persona: owned.append("manager"))
        t0 = time.monotonic()
        r = kill.kill_bridge(pd, timeout=30)
        assert time.monotonic() - t0 < 10
        assert r.outcome == "killed" and owned == []
        assert proc.wait(5) == 0
        assert state_file.read(pd).phase == "hard_exit"
        deadline = time.monotonic() + 5
        while psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        for pid in (proc.pid, child):
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        proc.wait(5)


def test_a_second_ctrl_c_cannot_skip_the_resume_either(tmp_path, fake):
    """Review R2-S2 (C45 g): Ctrl-C has the same one-shot handler as SIGTERM / SIGHUP, so a second Ctrl-C landing
    before the finally masks signals cannot skip the resume."""
    b = fake()
    before = signal.getsignal(signal.SIGINT)
    seen: dict = {}

    def _two_ctrl_c():
        handler = signal.getsignal(signal.SIGINT)
        try:
            handler(signal.SIGINT, None)
        except BaseException as first:  # noqa: BLE001 - whatever the first Ctrl-C raises
            try:
                handler(signal.SIGINT, None)
                seen["second"] = "ignored"
            except BaseException:  # noqa: BLE001
                seen["second"] = "raised"
            raise first

    with pytest.raises((kill._KillInterrupted, KeyboardInterrupt)):
        kill.kill_bridge(b.pd, timeout=5, _hooks={"after_bridge_suspend": _two_ctrl_c})
    assert seen["second"] == "ignored"
    assert _intent(b.pd) is None
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())
    assert signal.getsignal(signal.SIGINT) is before


def test_an_expected_bridge_that_is_no_longer_named_is_not_killed(tmp_path, fake):
    """Review S8 pin (main ruling, D16 amended): with --expect-pid / --expect-token the helper refuses with
    bridge_changed, before writing anything, when the record names another bridge; it kills only the expected one."""
    b = fake()
    st = state_file.read(b.pd)
    r = kill.kill_bridge(b.pd, timeout=5, expect_pid=b.pid, expect_token="linux:watched-earlier:1:1")
    assert r.outcome == "bridge_changed" and r.message.startswith("bridge_changed")
    assert _intent(b.pd) is None
    assert not shutdown_progress.path(b.pd, shutdown_progress.REQUEST_FILENAME).exists()
    assert all(s not in ("gone", psutil.STATUS_STOPPED) for s in b.statuses().values())
    r = kill.kill_bridge(b.pd, timeout=5, expect_pid=b.pid + 1, expect_token=st.pid_token)
    assert r.outcome == "bridge_changed"
    r = kill.kill_bridge(b.pd, timeout=10, expect_pid=b.pid, expect_token=st.pid_token)
    assert r.outcome == "killed" and b.all_gone()


class _FakeKernel32:
    def __init__(self):
        self.terminated: list = []

    def OpenProcess(self, access, inherit, pid):  # noqa: N802 - the Win32 name
        return 7

    def TerminateProcess(self, handle, code):  # noqa: N802
        self.terminated.append((handle, code))

    def CloseHandle(self, handle):  # noqa: N802
        pass


class _ReusedPid:
    """A remembered process that has exited; its pid now belongs to another process."""

    pid = 4242

    def is_running(self):
        return False


def test_windows_kill_never_ends_a_process_that_reused_the_pid(monkeypatch):
    """Review of #458 (finding 3): the Windows exit-0 terminate opens the pid, then confirms it is still the remembered
    process (psutil's create-time check) before ending it."""
    k32 = _FakeKernel32()
    monkeypatch.setattr(kill.sys, "platform", "win32")
    monkeypatch.setattr(kill, "_kernel32", lambda: k32)
    kill._kill_proc(_ReusedPid())
    assert k32.terminated == []


def test_windows_kill_ends_the_remembered_process_with_exit_code_0(monkeypatch):
    k32 = _FakeKernel32()
    monkeypatch.setattr(kill.sys, "platform", "win32")
    monkeypatch.setattr(kill, "_kernel32", lambda: k32)

    class _Same(_ReusedPid):
        def is_running(self):
            return True

    kill._kill_proc(_Same())
    assert k32.terminated == [(7, 0)]
