"""The start commands' 50 s readiness budget pauses while an engine progress record names a live writer (Engine 6, HR5 part 4).

``daemon._await_readiness`` (a direct spawn) and ``daemon._await_service_readiness`` (a service-started bridge) share one
budget. A start that waits on another process's emotion rebuild must not be given up on at 50 s while that rebuild's record
is present (owner S101); with no record the budget is the plain 50 s as before. Time before the record appears counts.

Every test runs on a fake clock (``daemon.time`` is replaced, so no real wait) and the fake ``sleep`` raises after a fixed
number of polls: a loop that never ends fails the test instead of hanging it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import daemon, state_file

_MAX_POLLS = 20_000  # 2000 s of fake time at the 0.1 s poll; the longest scenario below needs about 1300 s


class _FakeTime:
    """Stands in for the ``time`` module inside ``brain.bridge.daemon``: sleep advances the clock and counts polls."""

    def __init__(self) -> None:
        self.now = 10_000.0
        self.start = self.now
        self.polls = 0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.polls += 1
        if self.polls > _MAX_POLLS:
            raise AssertionError("the wait never ended (bounded fallback)")
        self.now += seconds

    @property
    def elapsed(self) -> float:
        return self.now - self.start


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeTime:
    fake = _FakeTime()
    monkeypatch.setattr(daemon, "time", fake)
    monkeypatch.setattr(state_file, "read", lambda persona_dir: None)
    return fake


def _record_between(monkeypatch: pytest.MonkeyPatch, clock: _FakeTime, begin_s: float, end_s: float) -> None:
    """The engine progress record shows from ``begin_s`` to ``end_s`` after the start (the reader the daemon calls)."""
    from brain.emotion import engine_files

    def read(persona_dir: Path) -> dict | None:
        return {"phase": "rebuilding"} if begin_s <= clock.elapsed < end_s else None

    monkeypatch.setattr(engine_files, "read_engine_progress", read)


@pytest.fixture
def killed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    monkeypatch.setattr(daemon, "_kill_if_alive", lambda pid, sig: seen.append(pid))
    return seen


def _direct(tmp_path: Path) -> int:
    return daemon._await_readiness(tmp_path, 4242, tmp_path / "bridge.log", None)


def _service(tmp_path: Path, budget_s: float = 50.0) -> int:
    return daemon._await_service_readiness(tmp_path, budget_s, None)


@pytest.mark.no_base_failure(reason="pins the unchanged no-record behaviour (a plain 50 s), which the old code already had")
def test_direct_spawn_with_no_record_gives_up_at_fifty_seconds(tmp_path, clock, monkeypatch, killed):
    _record_between(monkeypatch, clock, 1e9, 1e9)  # never
    assert _direct(tmp_path) == 1
    assert 50.0 <= clock.elapsed < 50.5
    assert killed == [4242]


def test_direct_spawn_waits_while_a_record_shows_and_gives_up_after_it_is_gone(tmp_path, clock, monkeypatch, killed):
    # 30 s before the record counts (20 s left), then 170 s paused, then the 20 s that were left.
    _record_between(monkeypatch, clock, 30.0, 200.0)
    assert _direct(tmp_path) == 1
    assert 219.5 <= clock.elapsed < 221.0
    assert killed == [4242]


def test_direct_spawn_that_is_up_after_a_long_pause_succeeds(tmp_path, clock, monkeypatch, killed):
    _record_between(monkeypatch, clock, 0.0, 900.0)
    monkeypatch.setattr(
        daemon.state_file,
        "read",
        lambda persona_dir: SimpleNamespace(pid=4242, port=9, auth_token=None) if clock.elapsed >= 800.0 else None,
    )
    monkeypatch.setattr(daemon.httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    assert _direct(tmp_path) == 0
    assert killed == []


def test_direct_spawn_whose_child_exits_during_the_pause_stops_at_once(tmp_path, clock, monkeypatch, killed):
    _record_between(monkeypatch, clock, 0.0, 900.0)
    child = SimpleNamespace(poll=lambda: 1 if clock.elapsed >= 120.0 else None)
    monkeypatch.setitem(daemon._spawned_children, 4242, child)
    try:
        assert _direct(tmp_path) == 1
    finally:
        daemon._spawned_children.pop(4242, None)
    assert 120.0 <= clock.elapsed < 121.0
    assert killed == []


@pytest.mark.no_base_failure(reason="pins the unchanged no-record behaviour (the plain budget), which the old code already had")
def test_service_start_with_no_record_gives_up_at_its_budget(tmp_path, clock, monkeypatch):
    _record_between(monkeypatch, clock, 1e9, 1e9)
    assert _service(tmp_path, 35.0) == 1
    assert 35.0 <= clock.elapsed < 35.5


def test_service_start_waits_while_a_record_shows_and_gives_up_after_it_is_gone(tmp_path, clock, monkeypatch):
    _record_between(monkeypatch, clock, 30.0, 1000.0)
    assert _service(tmp_path) == 1
    assert 1019.5 <= clock.elapsed < 1021.0


def test_service_start_that_is_up_after_a_long_pause_succeeds(tmp_path, clock, monkeypatch):
    _record_between(monkeypatch, clock, 0.0, 900.0)
    monkeypatch.setattr(
        daemon.state_file,
        "read",
        lambda persona_dir: SimpleNamespace(pid=7, port=9, auth_token=None) if clock.elapsed >= 800.0 else None,
    )
    monkeypatch.setattr(daemon.state_file, "bridge_liveness", lambda s, persona_dir: state_file.ALIVE)
    monkeypatch.setattr(daemon.httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    assert _service(tmp_path) == 0
