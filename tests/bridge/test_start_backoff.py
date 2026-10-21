"""The start back-off record and the success watch (C66, C68 a, C71; D26).

A bridge run by an OS service that keeps failing to start is spaced 30 s, 60 s, 300 s, then 600 s apart (the app
watchdog's ladder); the spacing resets after the same success (D7). These tests use an injected clock only: nothing
here sleeps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from brain.bridge import daemon, shutdown_progress, start_backoff

REPO = Path(__file__).resolve().parents[2]
WATCHDOG_TS = REPO / "app" / "src" / "hooks" / "useBridgeWatchdog.ts"


@pytest.fixture
def pd(tmp_path) -> Path:
    p = tmp_path / "personas" / "nell"
    p.mkdir(parents=True)
    return p


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(start_backoff, "_now", c.now)
    monkeypatch.setattr(start_backoff, "_sleep", c.sleep)
    return c


# ---- C66: the ladder, the record ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("starts", "step"),
    [(0, 0), (1, 30), (2, 60), (3, 300), (4, 600), (5, 600), (6, 600), (50, 600)],
)
def test_the_step_before_a_start_follows_the_apps_ladder(starts, step):
    assert start_backoff.step_s(starts) == step


def test_first_start_is_noted_as_one_and_each_later_start_adds_one(pd, clock):
    assert start_backoff.read(pd) == start_backoff.Record(0, 0.0)
    start_backoff.note_start(pd)
    assert start_backoff.read(pd) == start_backoff.Record(1, clock.t)
    clock.t += 31
    start_backoff.note_start(pd)
    assert start_backoff.read(pd) == start_backoff.Record(2, clock.t)


def test_remaining_wait_is_the_step_minus_the_time_since_the_last_start(pd, clock):
    start_backoff.note_start(pd)  # k = 1: the next start waits 30 s from now
    rec = start_backoff.read(pd)
    assert start_backoff.remaining_wait_s(rec, clock.t) == 30
    assert start_backoff.remaining_wait_s(rec, clock.t + 12) == 18
    assert start_backoff.remaining_wait_s(rec, clock.t + 30) == 0
    assert start_backoff.remaining_wait_s(rec, clock.t + 10_000) == 0


def test_a_clock_set_backwards_waits_the_step_never_more(pd, clock):
    for _ in range(4):
        start_backoff.note_start(pd)  # k = 4: 600 s
    rec = start_backoff.read(pd)
    assert start_backoff.remaining_wait_s(rec, clock.t - 3600) == 600


def test_no_record_means_no_wait(pd):
    assert start_backoff.remaining_wait_s(start_backoff.read(pd), 5.0) == 0


def test_the_record_lives_beside_bridge_json_through_json_io(pd, clock):
    start_backoff.note_start(pd)
    path = pd / start_backoff.FILENAME
    assert path.name == "bridge_start_backoff.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"unproven_starts": 1, "at": clock.t}


def test_the_record_is_not_a_watcher_file_and_survives_the_start_cleanup(pd, clock):
    assert start_backoff.FILENAME not in shutdown_progress.WATCHER_FILENAMES
    start_backoff.note_start(pd)
    shutdown_progress.delete_watcher_files(pd)
    assert start_backoff.read(pd).unproven_starts == 1


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"[1, 2]", b'"text"', b'{"unproven_starts": "two", "at": 1}', b'{"unproven_starts": -3, "at": 1}',
     b'{"unproven_starts": 2}', b'{"unproven_starts": true, "at": 1}', b"\xff\xfe\x00bad"],
)
def test_unreadable_or_invalid_content_reads_as_zero_failures(pd, content):
    (pd / start_backoff.FILENAME).write_bytes(content)
    assert start_backoff.read(pd).unproven_starts == 0


def test_a_failed_write_is_a_warning_and_never_raises(pd, clock, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(start_backoff, "write_json", boom)
    with caplog.at_level("WARNING"):
        start_backoff.note_start(pd)  # must not raise
    assert any("start back-off" in r.message for r in caplog.records)


def test_reset_deletes_the_record_and_is_safe_when_there_is_none(pd, clock):
    start_backoff.reset(pd)
    start_backoff.note_start(pd)
    start_backoff.reset(pd)
    assert not (pd / start_backoff.FILENAME).exists()


# ---- wait_before_start: the loop, with the injected clock ---------------------------------------------------------


def test_the_wait_sleeps_in_poll_slices_for_the_remaining_step_and_then_goes_on(pd, clock):
    from brain.bridge.shutdown import POLL_S

    start_backoff.note_start(pd)
    slices: list[float] = []
    real = clock.sleep
    clock_sleep = lambda s: (slices.append(s), real(s))  # noqa: E731
    import brain.bridge.start_backoff as sb

    sb._sleep = clock_sleep
    try:
        assert start_backoff.wait_before_start(pd, interrupted=lambda: False) is True
    finally:
        sb._sleep = clock.sleep
    assert slices, "no wait happened"
    assert all(0 < s <= POLL_S + 1e-9 for s in slices)
    assert sum(slices) == pytest.approx(30, abs=POLL_S)


def test_the_wait_ends_when_the_record_is_deleted(pd, clock, monkeypatch):
    start_backoff.note_start(pd)
    calls = {"n": 0}
    real = clock.sleep

    def sleep(s):
        calls["n"] += 1
        real(s)
        if calls["n"] == 3:
            start_backoff.reset(pd)  # an explicit start

    monkeypatch.setattr(start_backoff, "_sleep", sleep)
    assert start_backoff.wait_before_start(pd, interrupted=lambda: False) is True
    assert calls["n"] == 3


def test_the_wait_ends_when_interrupted_and_says_not_to_start(pd, clock, monkeypatch):
    start_backoff.note_start(pd)
    polls = iter([False, False, True])
    assert start_backoff.wait_before_start(pd, interrupted=lambda: next(polls)) is False
    assert start_backoff.read(pd).unproven_starts == 1  # untouched


def test_no_record_means_the_wait_does_not_sleep(pd, clock, monkeypatch):
    monkeypatch.setattr(start_backoff, "_sleep", lambda s: pytest.fail("slept"))
    assert start_backoff.wait_before_start(pd, interrupted=lambda: False) is True


# ---- C68 (a): the success watch ------------------------------------------------------------------------------------


class FakeServer:
    def __init__(self, started: bool = False) -> None:
        self.started = started


def test_the_watch_resets_only_600_s_after_the_server_started(pd, clock):
    start_backoff.note_start(pd)
    srv = FakeServer(started=False)
    mono = Clock(50.0)
    w = start_backoff.SuccessWatch(pd, srv, clock=mono.now)
    mono.t += 5000
    w.poll_once()  # the server never started: no clock runs
    assert start_backoff.read(pd).unproven_starts == 1
    srv.started = True
    w.poll_once()  # started now
    mono.t += start_backoff.HEALTHY_RESET_S - 1
    w.poll_once()
    assert start_backoff.read(pd).unproven_starts == 1
    mono.t += 1
    w.poll_once()
    assert not (pd / start_backoff.FILENAME).exists()


def test_the_watch_thread_ends_with_the_run(pd, clock):
    srv = FakeServer(started=False)
    w = start_backoff.SuccessWatch(pd, srv)
    w.start()
    w.stop()
    assert w._thread is not None and not w._thread.is_alive()  # noqa: SLF001


# ---- C71: the constants equal the app's -----------------------------------------------------------------------------


def _ts_const(name: str) -> str:
    m = re.search(rf"export const {name}\s*=\s*([^;]+);", WATCHDOG_TS.read_text(encoding="utf-8"))
    assert m, f"{name} not found in {WATCHDOG_TS}"
    return m.group(1)


def test_the_python_constants_equal_the_app_watchdogs():
    delays = [int(x.replace("_", "")) for x in re.findall(r"[\d_]+", _ts_const("CRASH_RESTART_DELAYS_MS").split("as")[0])]
    assert tuple(d // 1000 for d in delays) == start_backoff.DELAYS_S
    healthy = int(_ts_const("HEALTHY_RESET_MS").replace("_", ""))
    assert healthy // 1000 == start_backoff.HEALTHY_RESET_S


def test_daemon_module_still_imports():
    assert daemon.cmd_run
