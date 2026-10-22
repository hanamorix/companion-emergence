"""The task registry a stop waits on (C9, C19, C40 worker half, C41; D1, D21).

Every piece of work a stop must wait for is a counted task for its whole run; nothing new registers once the stop has
begun (background spawners) or once the registry is sealed (everything); the seal checks "empty" and seals in one step.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from brain.bridge import cli_throttle, shutdown_progress


@pytest.fixture(autouse=True)
def _fresh():
    cli_throttle.reset()
    reg = shutdown_progress.reset_for_tests()
    yield reg
    cli_throttle.reset()
    shutdown_progress.reset_for_tests()


def test_concurrent_registrations_are_all_counted(_fresh):
    """C19: a registration injected between another's read and write is not lost (fail-demo: unguarded)."""
    reg = _fresh
    other = threading.Thread(target=lambda: reg.register("other"))
    calls = {"n": 0}

    def _between():
        if calls["n"] == 0:
            calls["n"] += 1
            other.start()
            time.sleep(0.05)  # the other registration runs now, if nothing stops it

    reg._between = _between
    reg.register("first")
    reg._between = None
    other.join(5)
    assert reg.count() == 2
    assert sorted(reg.snapshot()) == ["first", "other"]


def test_seal_refuses_while_work_runs_and_nothing_registers_after_it(_fresh):
    """C41: seal is check-empty-and-mark in one step; a registration after it is refused (fail-demo: unsealed)."""
    reg = _fresh
    h = reg.register("chat reply")
    assert reg.seal() is False and not reg.sealed
    reg.unregister(h)
    assert reg.seal() is True and reg.sealed
    with pytest.raises(cli_throttle.ShutdownDeferred):
        reg.register("late close")
    assert reg.count() == 0


def test_background_spawners_are_refused_after_the_flag_request_work_is_not(_fresh):
    reg = _fresh
    cli_throttle.request_shutdown()
    with pytest.raises(cli_throttle.ShutdownDeferred):
        reg.register("floor bootstrap", spawner=True)
    h = reg.register("chat reply")  # a turn admitted before the stop still runs (D21)
    assert reg.snapshot() == ["chat reply"]
    reg.unregister(h)


def test_start_tracked_thread_counts_the_thread_and_starts_nothing_after_the_flag(_fresh):
    """C9."""
    reg = _fresh
    release = threading.Event()
    t = shutdown_progress.start_tracked_thread("backlog migration", lambda: release.wait(10), name="mig")
    assert t is not None
    assert reg.snapshot() == ["backlog migration"]
    cli_throttle.request_shutdown()
    assert shutdown_progress.start_tracked_thread("backlog migration", lambda: None, name="mig2") is None
    release.set()
    t.join(5)
    assert reg.count() == 0


def test_a_worker_stays_counted_after_its_request_task_is_cancelled(_fresh):
    """C40: uvicorn cancels request tasks during a stop; the worker thread keeps running and stays counted until it
    ends, so the stop waits for it."""
    reg = _fresh
    started, release = threading.Event(), threading.Event()
    done: list[int] = []

    def work():
        started.set()
        release.wait(10)
        done.append(1)

    async def main():
        task = asyncio.create_task(shutdown_progress.tracked_to_thread("session close", work))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert reg.snapshot() == ["session close"]
        assert reg.seal() is False
        release.set()
        for _ in range(500):
            if reg.count() == 0:
                break
            await asyncio.sleep(0.01)
        assert reg.count() == 0 and done == [1]
        assert reg.seal() is True

    asyncio.run(main())


def test_tracked_to_thread_returns_the_result_and_raises_the_error(_fresh):
    async def main():
        assert await shutdown_progress.tracked_to_thread("x", lambda a, b=0: a + b, 2, b=3) == 5
        with pytest.raises(ValueError):
            await shutdown_progress.tracked_to_thread("y", lambda: (_ for _ in ()).throw(ValueError("boom")))

    asyncio.run(main())
    assert _fresh.count() == 0


def test_tracked_to_thread_refuses_after_the_seal_without_running(_fresh):
    ran: list[int] = []
    assert _fresh.seal()

    async def main():
        with pytest.raises(cli_throttle.ShutdownDeferred):
            await shutdown_progress.tracked_to_thread("late", lambda: ran.append(1))

    asyncio.run(main())
    assert ran == []
