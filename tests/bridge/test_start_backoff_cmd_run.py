"""`nell supervisor run` waits out the start back-off for service origins only (C67, C68 b-c, C69; D26).

The clock and the sleep are injected (`start_backoff._now` / `_sleep`): no test here waits in real time, except the one
real-subprocess SIGTERM case, which ends the sleeping process at once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import daemon, respawn_guard, shutdown, start_backoff, state_file
from brain.bridge import runner as _runner
from brain.bridge.shutdown import POLL_S

SERVICE_ORIGINS = ("systemd", "launchd", "task-scheduler")


real_setup_runtime_logging = _runner._setup_runtime_logging  # captured before any fixture patches it


class Clock:
    def __init__(self) -> None:
        self.t = 2_000_000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(start_backoff, "_now", c.now)
    monkeypatch.setattr(start_backoff, "_monotonic", c.now)
    monkeypatch.setattr(start_backoff, "_sleep", c.sleep)
    return c


@pytest.fixture
def pd(tmp_path, monkeypatch) -> Path:
    p = tmp_path / "home" / "personas" / "nell"
    p.mkdir(parents=True)
    monkeypatch.setattr("brain.paths.get_persona_dir", lambda name: p)
    # The service-origin extras of cmd_run (the unit refresh, the launchd log truncation) are not under test here.
    monkeypatch.setattr("brain.service.systemd.refresh_installed_unit", lambda persona: False)
    monkeypatch.setattr("brain.service.launchd.truncate_launchd_logs_if_large", lambda persona: None)
    monkeypatch.setattr("brain.bridge.runner._setup_runtime_logging", lambda p: None)  # the real one: logging tests
    return p


def _args(origin: str):
    return SimpleNamespace(persona="nell", idle_shutdown=0, client_origin=origin)


def _record(pd: Path, starts: int, clock: Clock) -> None:
    for _ in range(starts):
        start_backoff.note_start(pd)


def _fake_run(monkeypatch, *, rc: int = 0, on_run=None) -> list[float]:
    ran: list[float] = []

    def run(persona_dir, **kw):
        ran.append(start_backoff._now())
        if on_run is not None:
            on_run(persona_dir)
        return rc

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", run)
    return ran


# ---- C67 (a): the wait, per origin and per step --------------------------------------------------------------------


@pytest.mark.parametrize("origin", SERVICE_ORIGINS)
@pytest.mark.parametrize(("starts", "step"), [(1, 30), (2, 60), (3, 300), (4, 600), (5, 600)])
def test_a_service_run_waits_out_the_step_in_poll_slices_then_runs(pd, clock, monkeypatch, origin, starts, step):
    _record(pd, starts, clock)
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args(origin)) == 0
    assert ran, "the bridge never ran"
    assert clock.slept and all(0 < s <= POLL_S + 1e-9 for s in clock.slept)
    assert sum(clock.slept) == pytest.approx(step, abs=POLL_S)


@pytest.mark.parametrize("origin", SERVICE_ORIGINS)
def test_a_service_run_with_no_failures_runs_at_once(pd, clock, monkeypatch, origin):
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args(origin)) == 0
    assert ran and clock.slept == []


# ---- C67 (b): no lock, no watcher file, bridge.json untouched during the wait ---------------------------------------


def test_the_wait_holds_no_lock_and_writes_no_bridge_files(pd, clock, monkeypatch):
    _record(pd, 2, clock)
    seen: list[str] = []
    real = clock.sleep

    def sleep(s):
        fd = daemon.acquire_lock(pd)
        seen.append("free" if fd is not None else "HELD")
        if fd is not None:
            daemon.release_lock(pd, fd)
        assert state_file.read(pd) is None
        real(s)

    monkeypatch.setattr(start_backoff, "_sleep", sleep)
    _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    assert seen and set(seen) == {"free"}
    # Only the record and the lock file (the probe's own) were ever there: no watcher file, no bridge.json.
    assert {p.name for p in pd.iterdir()} <= {start_backoff.FILENAME, daemon.LOCKFILE}


# ---- C67 (c): app / cli origins neither wait nor note -----------------------------------------------------------------


@pytest.mark.parametrize("origin", ["cli", "app", "tests"])
def test_a_non_service_run_never_waits_never_notes_and_leaves_the_record(pd, clock, monkeypatch, origin):
    _record(pd, 3, clock)
    snapshot = start_backoff.read(pd)
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args(origin)) == 0
    assert ran and clock.slept == []
    assert start_backoff.read(pd) == snapshot


# ---- C67 (d): the start is noted after both guard checks, under the lock ------------------------------------------------


@pytest.mark.parametrize("origin", SERVICE_ORIGINS)
def test_the_start_is_noted_under_the_lock_right_before_the_run(pd, clock, monkeypatch, origin):
    notes: list[str] = []
    real_note = start_backoff.note_start

    def note(persona_dir):
        fd = daemon.acquire_lock(persona_dir)
        notes.append("lock-not-held" if fd is not None else "lock-held")
        if fd is not None:
            daemon.release_lock(persona_dir, fd)
        real_note(persona_dir)

    monkeypatch.setattr(start_backoff, "note_start", note)
    _fake_run(monkeypatch, on_run=lambda p: notes.append(f"run:{start_backoff.read(p).unproven_starts}"))
    assert daemon.cmd_run(_args(origin)) == 0
    assert notes == ["lock-held", "run:1"]


def test_a_guard_that_fires_only_after_the_lock_leaves_the_count_alone(pd, clock, monkeypatch):
    _record(pd, 1, clock)
    state = {"locked": False}
    real_acquire = daemon.acquire_lock

    def acquire(*a, **k):
        fd = real_acquire(*a, **k)
        state["locked"] = fd is not None
        return fd

    monkeypatch.setattr(daemon, "acquire_lock", acquire)
    monkeypatch.setattr(respawn_guard, "guard_fires", lambda p: state["locked"])
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    assert not ran
    assert start_backoff.read(pd).unproven_starts == 1


# ---- C67 (e): an explicit start (record deleted) ends the wait ---------------------------------------------------------


def test_the_record_deleted_during_the_wait_starts_the_run_at_once(pd, clock, monkeypatch):
    _record(pd, 4, clock)  # a 600 s wait
    real = clock.sleep
    n = {"calls": 0}

    def sleep(s):
        n["calls"] += 1
        real(s)
        if n["calls"] == 3:
            start_backoff.reset(pd)  # `nell supervisor start`

    monkeypatch.setattr(start_backoff, "_sleep", sleep)
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    assert ran and n["calls"] == 3


# ---- C67 (f): the respawn guard firing during the wait ends it -----------------------------------------------------------


def test_the_guard_firing_during_the_wait_ends_it_without_taking_the_lock(pd, clock, monkeypatch, capsys):
    _record(pd, 3, clock)
    polls = {"n": 0}

    def fires(p):
        polls["n"] += 1
        return polls["n"] >= 3  # not before the wait, not at the first poll, then yes

    monkeypatch.setattr(respawn_guard, "guard_fires", fires)
    monkeypatch.setattr(daemon, "acquire_lock", lambda *a, **k: pytest.fail("the lock was taken"))
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    assert not ran
    assert "not starting: stopped on request" in capsys.readouterr().err
    assert start_backoff.read(pd).unproven_starts == 3


def test_a_keyboard_interrupt_in_the_wait_exits_zero_and_keeps_the_record(pd, clock, monkeypatch):
    _record(pd, 2, clock)

    def sleep(s):
        raise KeyboardInterrupt

    monkeypatch.setattr(start_backoff, "_sleep", sleep)
    ran = _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    assert not ran
    assert start_backoff.read(pd).unproven_starts == 2


# ---- C67 (h): the D4 lock wait follows, unchanged ----------------------------------------------------------------------


def test_after_the_wait_a_held_lock_is_still_waited_for(pd, clock, monkeypatch):
    _record(pd, 1, clock)
    fd = daemon.acquire_lock(pd)
    assert fd is not None
    ran = _fake_run(monkeypatch)
    result: dict = {}
    # The holder is this process's own fd: acquire_lock(blocking=True) on a second fd blocks until it is released.
    t = threading.Thread(target=lambda: result.setdefault("rc", daemon.cmd_run(_args("systemd"))), daemon=True)
    t.start()
    t.join(timeout=2.0)
    assert t.is_alive() and not ran, "the run did not wait for the lock"
    daemon.release_lock(pd, fd)
    t.join(timeout=30)
    assert not t.is_alive() and ran and result["rc"] == 0


# ---- C68 (b): what resets and what does not -----------------------------------------------------------------------------


@pytest.fixture
def run_env(pd, monkeypatch, tmp_path):
    """run_bridge_foreground with the server faked (the pattern of test_runner_extras)."""
    from brain.bridge import runner, shutdown_progress

    shutdown_progress.reset_for_tests()
    monkeypatch.setattr(runner, "_allocate_port", lambda: 54321)
    monkeypatch.setattr(runner, "_setup_runtime_logging", lambda _p: None)
    monkeypatch.setattr(runner, "build_app", lambda **_kw: object())
    monkeypatch.setattr(runner.signal, "signal", lambda *_a, **_kw: None)
    monkeypatch.setattr(shutdown.Monitor, "start", lambda self: None)
    monkeypatch.setattr(runner.uvicorn, "Config", lambda *a, **k: None)
    return runner


def test_a_clean_stop_resets_the_record(pd, clock, run_env, monkeypatch):
    _record(pd, 3, clock)

    class Server:
        def __init__(self, config):
            self.should_exit = False
            self.started = True

        def run(self):
            self.controller.begin_stop("user_stop")

    monkeypatch.setattr(run_env.uvicorn, "Server", Server)
    assert run_env.run_bridge_foreground(pd, client_origin="systemd") == 0
    assert not (pd / start_backoff.FILENAME).exists()


def test_a_crash_leaves_the_record(pd, clock, run_env, monkeypatch):
    _record(pd, 3, clock)

    class _Exited(BaseException):
        pass

    def fake_hard_exit(persona_dir, reason="hard_exit"):
        raise _Exited

    monkeypatch.setattr(shutdown, "hard_exit", fake_hard_exit)

    class Server:
        def __init__(self, config):
            self.should_exit = False
            self.started = False

        def run(self):
            return  # a lifespan startup that raised: uvicorn's run() returns, nothing asked for a stop

    monkeypatch.setattr(run_env.uvicorn, "Server", Server)
    with pytest.raises(_Exited):
        run_env.run_bridge_foreground(pd, client_origin="systemd")
    assert start_backoff.read(pd).unproven_starts == 3


@pytest.mark.parametrize(("reason", "resets"), [("hard_exit", True), ("user_stop", True), ("crash", False)])
def test_hard_exit_resets_the_record_unless_it_is_a_crash(pd, clock, monkeypatch, reason, resets):
    _record(pd, 2, clock)

    class _Exited(BaseException):
        pass

    def fake_exit(code):
        raise _Exited

    monkeypatch.setattr(shutdown.os, "_exit", fake_exit)
    monkeypatch.setattr(shutdown, "_kill_descendants", lambda: None)
    with pytest.raises(_Exited):
        shutdown.hard_exit(pd, reason)
    assert (not (pd / start_backoff.FILENAME).exists()) is resets


def test_the_success_watch_runs_for_service_origins_only_and_ends_with_the_run(pd, clock, run_env, monkeypatch):
    started: list[str] = []
    stopped: list[str] = []

    class Watch:
        def __init__(self, persona_dir, server, **kw):
            pass

        def start(self):
            started.append("x")

        def stop(self):
            stopped.append("x")

    monkeypatch.setattr(start_backoff, "SuccessWatch", Watch)

    class Server:
        def __init__(self, config):
            self.should_exit = False
            self.started = True

        def run(self):
            self.controller.begin_stop("user_stop")

    monkeypatch.setattr(run_env.uvicorn, "Server", Server)
    run_env.run_bridge_foreground(pd, client_origin="cli")
    assert started == [] and stopped == []
    run_env.run_bridge_foreground(pd, client_origin="systemd")
    assert started == ["x"] and stopped == ["x"]


# ---- C68 (c): explicit starts reset ----------------------------------------------------------------------------------


def test_cmd_start_resets_before_routing_to_the_service_and_not_when_refused(pd, clock, monkeypatch):
    _record(pd, 3, clock)
    seen: list[int] = []
    monkeypatch.setattr(daemon, "_service_installed", lambda persona: True)
    monkeypatch.setattr(
        daemon,
        "_start_service",
        lambda persona: seen.append(start_backoff.read(pd).unproven_starts) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(daemon, "_await_service_readiness", lambda *a, **k: 0)
    args = SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="cli")
    monkeypatch.setattr("brain.paths.get_log_dir", lambda: pd.parent)
    assert daemon.cmd_start(args) == 0
    assert seen == [0]
    assert not (pd / start_backoff.FILENAME).exists()


def test_a_refused_cmd_start_leaves_the_record(pd, clock, monkeypatch):
    _record(pd, 3, clock)
    monkeypatch.setattr(state_file, "is_running", lambda p: True)
    monkeypatch.setattr(daemon, "_refuse_already_running", lambda p, out: 2)
    args = SimpleNamespace(persona="nell", idle_shutdown=0, client_origin="cli")
    assert daemon.cmd_start(args) == 2
    assert start_backoff.read(pd).unproven_starts == 3


# ---- C69: the headline case, a fake service manager over a bridge that never starts ----------------------------------------


class _Server:
    def __init__(self, started: bool) -> None:
        self.started = started


def _manager_loop(pd: Path, clock: Clock, monkeypatch, runs: list[dict], restart_sec: float = 2.0) -> list[float]:
    """`Restart=on-failure` modelled: cmd_run again `restart_sec` after every non-zero return. Each `runs` item says
    how long that run lived and whether it was healthy for 600 s (then the success watch resets the record)."""
    starts: list[float] = []
    plan = iter(runs)

    def run(persona_dir, **kw):
        spec = next(plan)
        starts.append(clock.t)
        if spec.get("healthy"):
            mono = SimpleNamespace(t=0.0)
            watch = start_backoff.SuccessWatch(persona_dir, _Server(True), clock=lambda: mono.t)
            watch.poll_once()
            mono.t += start_backoff.HEALTHY_RESET_S
            watch.poll_once()
        clock.t += spec["lived"]
        return 1

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", run)
    for _ in runs:
        daemon.cmd_run(_args("systemd"))
        clock.t += restart_sec
    return starts


def test_a_bridge_that_never_starts_is_spaced_on_the_apps_ladder_forever(pd, clock, monkeypatch):
    starts = _manager_loop(pd, clock, monkeypatch, [{"lived": 4}] * 10)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    floor = [30, 60, 300, 600, 600, 600, 600, 600, 600]
    assert all(g >= f - 1e-6 for g, f in zip(gaps, floor, strict=True)), gaps
    # No 10 s window holds more than 2 starts: systemd's default start limit (5 in 10 s) cannot trip.
    for i, s in enumerate(starts):
        assert sum(1 for x in starts[i:] if x - s < 10) <= 2


def test_a_crash_after_a_healthy_run_waits_30_s_and_the_ladder_goes_on(pd, clock, monkeypatch):
    """Main's ruling after the D26 review: after a healthy run the app still waits 30 s after a crash, and a quick
    second failure is followed by its 60 s."""
    runs = [{"lived": 4}, {"lived": 4}, {"lived": 4}, {"lived": 700, "healthy": True}, {"lived": 4}, {"lived": 4}]
    starts = _manager_loop(pd, clock, monkeypatch, runs)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    tol = POLL_S + 1e-6
    assert gaps[0] == pytest.approx(30, abs=tol) and gaps[1] == pytest.approx(60, abs=tol)
    assert gaps[2] == pytest.approx(300, abs=tol)
    assert gaps[3] == pytest.approx(700 + 2 + 30, abs=tol)  # healthy 700 s, crash, 2 s restart, then the first step
    assert gaps[4] == pytest.approx(60, abs=tol)  # the quick failure of that start: the second step, not 600


def test_a_clean_stop_after_a_healthy_run_means_no_wait_at_the_next_start(pd, clock, monkeypatch):
    starts: list[float] = []

    def run(persona_dir, **kw):
        starts.append(clock.t)
        mono = SimpleNamespace(t=0.0)
        watch = start_backoff.SuccessWatch(persona_dir, _Server(True), clock=lambda: mono.t)
        watch.poll_once()
        mono.t += start_backoff.HEALTHY_RESET_S
        watch.poll_once()
        clock.t += 700
        start_backoff.reset(persona_dir)  # the clean stop's _write_stopped
        return 0

    monkeypatch.setattr("brain.bridge.runner.run_bridge_foreground", run)
    for _ in range(2):
        daemon.cmd_run(_args("systemd"))
    assert clock.slept == [] and len(starts) == 2


def test_a_run_that_crashes_after_20_s_is_spaced_from_its_start(pd, clock, monkeypatch):
    starts = _manager_loop(pd, clock, monkeypatch, [{"lived": 20}] * 4)
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert gaps[0] == pytest.approx(30, abs=POLL_S + 1e-6)  # not 20 + 30


def test_an_unwritable_record_directory_keeps_the_old_behaviour_and_logs(pd, clock, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(start_backoff, "write_json", boom)
    with caplog.at_level("WARNING"):
        starts = _manager_loop(pd, clock, monkeypatch, [{"lived": 4}] * 3)
    assert clock.slept == [] and len(starts) == 3
    assert any("start back-off" in r.message for r in caplog.records)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; the Windows leg is CI")
def test_sigterm_during_the_wait_ends_the_process_at_once_with_nothing_held(tmp_path):
    home = tmp_path / "home"
    persona = home / "personas" / "nell"
    persona.mkdir(parents=True)
    (persona / start_backoff.FILENAME).write_text(
        f'{{"unproven_starts": 4, "at": {time.time()}}}', encoding="utf-8"
    )  # a 600 s wait
    err = tmp_path / "err.txt"
    env = dict(os.environ, PYTHONPATH=os.getcwd(), KINDLED_HOME=str(home))
    with err.open("w") as ef:
        p = subprocess.Popen(
            [sys.executable, "-m", "brain.cli", "supervisor", "run", "--persona", "nell", "--client-origin", "systemd",
             "--idle-shutdown", "0"],
            env=env, stdout=subprocess.DEVNULL, stderr=ef,
        )
    try:
        deadline = time.time() + 60
        while time.time() < deadline and p.poll() is None and "before starting" not in err.read_text():
            time.sleep(0.2)
        assert p.poll() is None, f"the run ended on its own: {err.read_text()}"
        assert "before starting" in err.read_text(), "the run never reached its wait"
        t0 = time.time()
        p.terminate()
        p.wait(timeout=10)
        assert time.time() - t0 < 10
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)
    assert not (persona / "bridge.json").exists()
    fd = daemon.acquire_lock(persona)
    assert fd is not None
    daemon.release_lock(persona, fd)
    assert start_backoff.read(persona).unproven_starts == 4


# ---- the back-off lines reach the bridge log (review of D26, SF6) ------------------------------------------------------


@pytest.fixture
def root_handlers():
    import logging

    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield root
    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


def test_the_wait_line_reaches_the_runtime_log(pd, clock, monkeypatch, tmp_path, root_handlers):
    """The runtime log is set up before the wait, so its INFO line (and the unit refresh's) is not lost."""
    log_dir = tmp_path / "logs"
    monkeypatch.setattr("brain.paths.get_log_dir", lambda: log_dir)
    monkeypatch.setattr("brain.bridge.runner._setup_runtime_logging", real_setup_runtime_logging)  # the real one
    _record(pd, 1, clock)
    _fake_run(monkeypatch)
    assert daemon.cmd_run(_args("systemd")) == 0
    for h in root_handlers.handlers:
        h.flush()
    text = (log_dir / "runtime-nell.log").read_text(encoding="utf-8")
    assert "start back-off: waiting" in text


def test_setting_up_the_runtime_log_twice_adds_one_handler(pd, monkeypatch, tmp_path, root_handlers):
    from logging.handlers import RotatingFileHandler

    monkeypatch.setattr("brain.paths.get_log_dir", lambda: tmp_path / "logs")
    n0 = sum(isinstance(h, RotatingFileHandler) for h in root_handlers.handlers)
    real_setup_runtime_logging(pd)
    real_setup_runtime_logging(pd)
    assert sum(isinstance(h, RotatingFileHandler) for h in root_handlers.handlers) == n0 + 1
