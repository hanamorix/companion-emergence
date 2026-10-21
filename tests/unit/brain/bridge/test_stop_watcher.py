"""The terminal stop watcher (C16 a-j; Q-A, D18) and the CLI stop paths around it (C22, C35).

The watcher reads only files (the progress record, bridge.json) and the bridge's identity: the bridge's HTTP listener
is closed once a stop begins. The clock, stdin and the bridge's liveness are injected.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import daemon, shutdown_progress, state_file, stop_watch

PID, TOKEN = 4242, "linux:b:4242:7"
T0 = 1_000_000.0


class Clock:
    def __init__(self, start: float = T0):
        self.t = start
        self.events: list[tuple[float, object]] = []

    def now(self) -> float:
        return self.t

    def at(self, when: float, fn) -> None:
        self.events.append((when, fn))

    def sleep(self, dt: float) -> None:
        self.t += dt
        due = [e for e in self.events if e[0] <= self.t]
        self.events = [e for e in self.events if e[0] > self.t]
        for _, fn in due:
            fn()
        time.sleep(0.001)  # let the prompt thread run


class Bridge:
    def __init__(self, pd: Path):
        self.pd = pd
        self.alive = True

    def record(self, tasks: list[str], *, requested_at: float = T0, pid: int = PID, token: str = TOKEN,
               reason: str = "user_stop") -> None:
        from datetime import UTC, datetime

        shutdown_progress.write_record(self.pd, shutdown_progress.PROGRESS_FILENAME, shutdown_progress.progress_record(
            pid=pid, pid_token=token, persona="nell", reason=reason, tasks=tasks, updated_at="u",
            requested_at=datetime.fromtimestamp(requested_at, UTC).isoformat()))


@pytest.fixture
def pd(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    p = tmp_path / "home" / "personas" / "nell"
    p.mkdir(parents=True)
    return p


@pytest.fixture
def bridge(pd, monkeypatch):
    b = Bridge(pd)
    monkeypatch.setattr(stop_watch, "_liveness", lambda pid, token: "alive" if b.alive else "gone")
    return b


def _watch(pd, clock, *, interactive=True, answer=None, timeout=180.0, kill=None):
    out: list[str] = []
    answered = threading.Event()

    def prompt(text: str) -> str:
        out.append(text)
        if answer is None:
            answered.wait()  # never answered in this test
            return ""
        return answer

    rc = stop_watch.watch_stop(
        pd, pid=PID, token=TOKEN, timeout=timeout, interactive=interactive, now=clock.now, sleep=clock.sleep,
        prompt=prompt, out=out.append, kill=kill or (lambda **k: SimpleNamespace(outcome="killed", survivors=[])),
    )
    answered.set()
    return rc, out


def test_prints_the_tasks_as_they_change_and_ends_when_gone(pd, bridge):
    """C16 (a), (d)."""
    clock = Clock()
    bridge.record(["chat reply", "background job: compaction"])
    clock.at(T0 + 5, lambda: bridge.record(["background job: compaction"]))
    clock.at(T0 + 10, lambda: bridge.record([]))
    clock.at(T0 + 10.4, lambda: setattr(bridge, "alive", False))
    rc, out = _watch(pd, clock)
    assert rc == 0
    assert out[0] == "Stopping. Waiting for 2 tasks to finish: chat reply, background job: compaction."
    assert out[1] == "Stopping. Waiting for 1 task to finish: background job: compaction."
    assert out[-1] == "Everything has stopped."
    assert not any("Hard exit now?" in line for line in out)  # (b) no offer before 180 s


def test_the_offer_comes_at_180s_from_the_recorded_request_and_y_hard_exits(pd, bridge):
    """C16 (c), (e)."""
    clock = Clock()
    bridge.record(["background job: compaction"], requested_at=T0 - 30)  # the stop began 30 s before we watch

    def on_request():
        req = shutdown_progress.read_record(pd, shutdown_progress.REQUEST_FILENAME)
        if req and req["pid"] == PID and req["pid_token"] == TOKEN:
            bridge.alive = False

    seen: dict = {}

    def check_offer_time():
        seen["no_offer_at_149"] = True

    clock.at(T0 + 149, check_offer_time)
    clock.at(T0 + 151, on_request)
    rc, out = _watch(pd, clock, answer="y")
    offers = [line for line in out if "Hard exit now?" in line]
    assert offers == [
        "Still waiting for 1 task after 3 minutes: background job: compaction. Hard exit now? "
        "The next start will pick up anything left unfinished. [y/N] "
    ]
    assert rc == 0 and out[-1] == "Hard exit done. The next start will pick up anything left unfinished."


def test_n_keeps_waiting_and_the_offer_is_shown_once(pd, bridge):
    """C16 (f)."""
    clock = Clock()
    bridge.record(["soul review"])
    clock.at(T0 + 400, lambda: setattr(bridge, "alive", False))
    rc, out = _watch(pd, clock, answer="n")
    assert rc == 0
    assert sum("Hard exit now?" in line for line in out) == 1
    assert not shutdown_progress.path(pd, shutdown_progress.REQUEST_FILENAME).exists()


def test_without_a_tty_no_prompt_no_hard_exit_and_1_after_the_timeout(pd, bridge):
    """C16 (g)."""
    clock = Clock()
    bridge.record(["soul review"])
    kills: list = []
    rc, out = _watch(pd, clock, interactive=False, timeout=60, kill=lambda **k: kills.append(k))
    assert rc == 1
    assert not any("Hard exit now?" in line for line in out)
    assert kills == [] and not shutdown_progress.path(pd, shutdown_progress.REQUEST_FILENAME).exists()


def test_an_ignored_hard_exit_escalates_to_the_kill_helper_after_the_timeout(pd, bridge):
    """C16 (e) escalation, C44 (b): the request was accepted but the bridge never acted on it."""
    clock = Clock()
    bridge.record(["stuck"])
    kills: list = []

    def fake_kill(**kw):
        kills.append(kw)
        bridge.alive = False
        return SimpleNamespace(outcome="killed", survivors=[])

    rc, out = _watch(pd, clock, answer="y", timeout=30, kill=fake_kill)
    assert rc == 0 and len(kills) == 1 and kills[0]["skip_hard_exit_request"] is True


def test_a_record_of_another_bridge_is_ignored(pd, bridge):
    """C16 (i)."""
    clock = Clock()
    bridge.record(["old task"], pid=PID, token="linux:other:4242:1")
    clock.at(T0 + 2, lambda: setattr(bridge, "alive", False))
    rc, out = _watch(pd, clock)
    assert rc == 0 and not any("old task" in line for line in out)


def test_attach_to_a_stop_already_in_progress_offers_at_the_recorded_time(pd, bridge, monkeypatch):
    """C16 (j): an idle stop began 100 s ago; the POST fails (listener closed); the CLI attaches: task list, the
    offer 80 s later, no signal."""
    clock = Clock()
    state_file.write(pd, state_file.BridgeState(
        persona="nell", pid=PID, port=1, started_at="t", stopped_at=None, shutdown_clean=False, client_origin="cli",
        pid_token=TOKEN, identity="token", phase="stopping", stop_reason="idle_timeout"))
    bridge.record(["background job: maintenance"], requested_at=T0 - 100, reason="idle_timeout")
    signals: list = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: signals.append(sig) or True)
    monkeypatch.setattr(daemon, "_request_shutdown_via_http", lambda s, **k: (_ for _ in ()).throw(OSError("closed")))
    monkeypatch.setattr(state_file, "bridge_liveness", lambda st, p: state_file.ALIVE)
    offered_at: list[float] = []

    def prompt(text):
        offered_at.append(clock.now())
        bridge.alive = False
        return "n"

    monkeypatch.setattr(stop_watch, "_clock", clock)
    monkeypatch.setattr(stop_watch, "_interactive", lambda: True)
    monkeypatch.setattr(stop_watch, "_prompt", prompt)
    rc = daemon.cmd_stop(SimpleNamespace(persona="nell", timeout=180.0, force=False, kill=False,
                                         skip_hard_exit_request=False))
    assert rc == 0 and signals == []
    assert offered_at and 79.5 <= offered_at[0] - T0 <= 80.5


def test_restart_starts_only_after_the_old_bridge_is_gone(pd, bridge, monkeypatch):
    """C16 (h)."""
    order: list[str] = []
    monkeypatch.setattr(daemon, "cmd_stop", lambda args, **_kw: order.append("stopped") or 0)
    monkeypatch.setattr(daemon, "cmd_start", lambda args: order.append("started") or 0)
    assert daemon.cmd_restart(SimpleNamespace(persona="nell")) == 0
    assert order == ["stopped", "started"]
