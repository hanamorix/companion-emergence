from __future__ import annotations

from brain.bridge.shutdown import BridgeShutdownController


class FakeServer:
    should_exit = False


def test_request_marks_bound_uvicorn_server_for_exit() -> None:
    server = FakeServer()
    controller = BridgeShutdownController()
    controller.bind_server(server)

    accepted = controller.request("manual_restart")

    assert accepted is True
    assert server.should_exit is True
    assert controller.requested is True
    assert controller.reason == "manual_restart"


def test_request_is_idempotent_and_keeps_first_reason() -> None:
    server = FakeServer()
    controller = BridgeShutdownController()
    controller.bind_server(server)

    assert controller.request("idle_timeout") is True
    assert controller.request("manual_restart") is True

    assert server.should_exit is True
    assert controller.reason == "idle_timeout"


def test_request_without_server_returns_false() -> None:
    controller = BridgeShutdownController()

    assert controller.request("manual_restart") is False
    assert controller.requested is False
    assert controller.reason is None


# ---------------------------------------------------------------- #456: begin_stop, the signal path, the monitor

import os  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from brain import process_identity as _pi  # noqa: E402
from brain.bridge import cli_throttle, shutdown_progress, state_file  # noqa: E402
from brain.bridge import shutdown as shutdown_mod  # noqa: E402


@pytest.fixture
def own_record(tmp_path: Path):
    cli_throttle.reset()
    shutdown_progress.reset_for_tests()
    pd = tmp_path / "nell"
    pd.mkdir()
    state_file.write(
        pd,
        state_file.BridgeState(
            persona="nell", pid=os.getpid(), port=5000, started_at="t", stopped_at=None, shutdown_clean=False,
            client_origin="tests", pid_token=_pi.own_token(), identity="token", phase="running",
        ),
    )
    yield pd
    cli_throttle.reset()
    shutdown_progress.reset_for_tests()


def test_begin_stop_sets_flag_events_and_mark_before_it_returns(own_record):
    """C6 (HTTP / idle half): everything is in place when begin_stop returns (before the 202)."""
    c = BridgeShutdownController()
    c.configure(own_record)
    server = FakeServer()
    c.bind_server(server)
    ev = threading.Event()
    c.add_stop_event(ev)
    assert c.begin_stop("user_stop") is True
    assert cli_throttle.shutdown_requested() and ev.is_set() and server.should_exit
    st = state_file.read(own_record)
    assert st.phase == "stopping" and st.stop_reason == "user_stop" and st.stop_requested_at


def test_a_later_stop_never_overwrites_the_first_reason(own_record):
    """C13 (lifespan belt)."""
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    c.begin_stop("idle_timeout")
    assert c.begin_stop("unknown") is False
    assert c.reason == "idle_timeout"
    assert state_file.read(own_record).stop_reason == "idle_timeout"


def _server_class():
    import uvicorn

    return shutdown_mod.make_server_class(uvicorn.Server)


def test_signal_handler_takes_no_lock_sets_no_event_and_does_no_io(monkeypatch):
    """C6 (signal half): the handler only writes attributes (fail-demo: a handler calling Event.set())."""
    import builtins

    import uvicorn

    c = BridgeShutdownController()
    server_cls = _server_class()
    server = server_cls(uvicorn.Config(app=lambda *a: None))
    server.controller = c
    seen: list[str] = []

    class _Spy:
        def __init__(self, name, real):
            self._name, self._real = name, real

        def __call__(self, *a, **k):
            obj = self._real(*a, **k)
            name = self._name

            class _Wrapped:
                def acquire(self, *a, **k):
                    seen.append(f"{name}.acquire")
                    return obj.acquire(*a, **k)

                def __enter__(self):
                    seen.append(f"{name}.enter")
                    return obj.__enter__()

                def __exit__(self, *e):
                    return obj.__exit__(*e)

                def set(self):
                    seen.append(f"{name}.set")
                    return obj.set()

                def __getattr__(self, attr):
                    return getattr(obj, attr)

            return _Wrapped()

    # Any lock or event the handler might reach goes through these spies; and any file opened.
    for name in ("Lock", "RLock", "Event"):
        monkeypatch.setattr(threading, name, _Spy(name, getattr(threading, name)))
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda *a, **k: seen.append("open") or real_open(*a, **k))
    monkeypatch.setattr(os, "replace", lambda *a, **k: seen.append("replace"))
    # The controller's own lock/events were created before the patch; route them through spies too.
    c._lock = threading.Lock()
    c._stop_events = [threading.Event()]
    seen.clear()
    import signal as _signal

    server.handle_exit(_signal.SIGTERM, None)
    server.handle_exit(_signal.SIGTERM, None)
    assert seen == []
    assert c.signal_count == 2 and c.signal_reason == "os_signal" and server.should_exit
    assert server._captured_signals == [_signal.SIGTERM]  # once per signal number
    assert server.force_exit is False


def _monitor(pd, c, exits):
    return shutdown_mod.Monitor(
        c, pd, pid=os.getpid(), pid_token=_pi.own_token(), hard_exit=lambda reason: exits.append(reason)
    )


def test_monitor_applies_a_signal_within_one_poll(own_record):
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = _monitor(own_record, c, exits)
    c.note_signal()
    m.poll_once()
    assert c.stopping and c.reason == "os_signal" and cli_throttle.shutdown_requested()
    assert state_file.read(own_record).phase == "stopping"
    assert exits == []


def test_a_second_signal_is_a_hard_exit(own_record):
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = _monitor(own_record, c, exits)
    c.note_signal()
    m.poll_once()
    c.note_signal()
    m.poll_once()
    assert exits == ["hard_exit"]


def test_progress_record_follows_the_tasks_and_names_this_bridge(own_record):
    """C12 (unit half): N tasks ending one by one -> N, N-1, ..., 0, with names, pid + token, time and reason."""
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = _monitor(own_record, c, exits)
    reg = shutdown_progress.registry
    handles = [reg.register(f"task {i}") for i in range(3)]
    c.begin_stop("user_stop")
    counts = []
    for h in [None, *handles]:
        if h is not None:
            reg.unregister(h)
        m.poll_once()
        rec = shutdown_progress.read_record(own_record, shutdown_progress.PROGRESS_FILENAME)
        counts.append((rec["count"], rec["tasks"]))
        assert rec["pid"] == os.getpid() and rec["pid_token"] == _pi.own_token()
        assert rec["reason"] == "user_stop" and rec["requested_at"] == c.requested_at
    assert [n for n, _ in counts] == [3, 2, 1, 0]
    assert counts[0][1] == ["task 0", "task 1", "task 2"]


def test_hard_exit_request_is_honoured_only_when_it_names_this_bridge(own_record):
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = _monitor(own_record, c, exits)
    c.begin_stop("user_stop")
    shutdown_progress.write_record(
        own_record, shutdown_progress.REQUEST_FILENAME, {"pid": os.getpid(), "pid_token": "linux:other:1:1"}
    )
    m.poll_once()
    assert exits == []
    shutdown_progress.write_record(
        own_record, shutdown_progress.REQUEST_FILENAME, {"pid": os.getpid(), "pid_token": _pi.own_token()}
    )
    m.poll_once()
    assert exits == ["hard_exit"]


def test_update_never_loses_a_competing_write(own_record, monkeypatch):
    """C18 (a): a competing in-process read-modify-write inside the window keeps both (fail-demo: unserialized)."""
    other = threading.Thread(target=lambda: state_file.update(own_record, lambda st: setattr(st, "stop_reason", "x")))

    def _between():
        monkeypatch.setattr(state_file, "_update_between", None)
        other.start()
        other.join(0.2)

    monkeypatch.setattr(state_file, "_update_between", _between)
    assert state_file.update(own_record, lambda st: setattr(st, "phase", "stopping"))
    other.join(5)
    st = state_file.read(own_record)
    assert st.phase == "stopping" and st.stop_reason == "x"


def test_update_skips_a_record_that_is_not_this_bridge(own_record):
    """C18 (b)."""
    st = state_file.read(own_record)
    st.pid_token = "linux:other:1:1"
    state_file.write(own_record, st)
    before = (own_record / "bridge.json").read_bytes()
    assert state_file.update(own_record, lambda s: setattr(s, "phase", "stopping")) is False
    assert (own_record / "bridge.json").read_bytes() == before


def test_the_monitor_survives_write_errors_and_still_honours_a_later_request(own_record, monkeypatch):
    """C44 (a): an injected OSError on the progress-record replace does not stop the monitor, and a later valid request
    is still honoured (fail-demo: a loop without a per-iteration catch)."""
    import time as _time

    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = shutdown_mod.Monitor(c, own_record, pid=os.getpid(), pid_token=_pi.own_token(),
                             hard_exit=lambda reason: exits.append(reason), poll_s=0.02)
    real_write = shutdown_progress.write_record
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError("disk full")
        return real_write(*a, **k)

    monkeypatch.setattr(shutdown_progress, "write_record", flaky)
    real_read = shutdown_progress.read_record
    reads = {"n": 0}

    def flaky_read(*a, **k):
        reads["n"] += 1
        if reads["n"] == 1:
            raise OSError("io error")  # an unexpected error escaping poll_once itself
        return real_read(*a, **k)

    monkeypatch.setattr(shutdown_progress, "read_record", flaky_read)
    c.begin_stop("user_stop")
    m.start()
    try:
        deadline = _time.monotonic() + 5
        while calls["n"] < 3 and _time.monotonic() < deadline:
            _time.sleep(0.02)
        real_write(own_record, shutdown_progress.REQUEST_FILENAME, {"pid": os.getpid(), "pid_token": _pi.own_token()})
        while not exits and _time.monotonic() < deadline:
            _time.sleep(0.02)
    finally:
        m.stop()
    assert exits == ["hard_exit"]
    assert calls["n"] >= 3


def test_a_hard_exit_request_is_honoured_by_a_bridge_that_is_not_stopping(own_record):
    """D6 (b), review M2: the request (naming this pid + token) is the only condition; a running bridge acts on it
    at the next poll (the kill helper writes it for a service bridge that was never asked to stop)."""
    c = BridgeShutdownController()
    c.configure(own_record)
    c.bind_server(FakeServer())
    exits: list[str] = []
    m = _monitor(own_record, c, exits)
    shutdown_progress.write_record(
        own_record, shutdown_progress.REQUEST_FILENAME, {"pid": os.getpid(), "pid_token": "linux:other:1:1"}
    )
    m.poll_once()
    assert exits == [] and not c.stopping
    shutdown_progress.write_record(
        own_record, shutdown_progress.REQUEST_FILENAME, {"pid": os.getpid(), "pid_token": _pi.own_token()}
    )
    m.poll_once()
    assert exits == ["hard_exit"]
