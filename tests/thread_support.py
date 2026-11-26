"""Waiting for the background threads a test starts, however slow the machine is (#540).

``Thread.join(timeout=...)`` returns silently when the time is up, and so does ``Event.wait(n)`` when its return value is
dropped. On a loaded CI runner a thread that needs longer than the bound makes the asserts after it run on half-written
state, and the still-running thread leaks into the next test on the worker, where it can call that test's monkeypatched
functions (CI run 40639503385: a floor-bootstrap retry thread that needed more than 10 s). The tests that wait for a thread
to FINISH use these helpers instead: no bound. A thread that really never ends is a deadlock, which pytest-timeout
(``timeout`` in pyproject.toml) ends, on Windows by killing the xdist worker.

A deliberate probe (``join(0.2)`` to show a thread is still blocked) and a bound that is itself the verdict (``join(5)``
followed by ``assert not t.is_alive(), "... blocked"``) are not completion waits and keep their time bound.
"""

from __future__ import annotations

import threading

import pytest


def finish_threads(*threads: threading.Thread) -> None:
    """Wait for every thread to finish, with no time bound (a thread that is not started raises, as ``join`` does)."""
    for thread in threads:
        thread.join()


def wait_at_gate(gate: threading.Event, owner: threading.Thread) -> None:
    """A background body parks on `gate` until the test opens it. Never on the test's own thread (`owner`): a regression
    that ran the body inline there would otherwise wait for a gate only that same thread can open, and hang instead of
    failing the test's asserts."""
    if threading.current_thread() is not owner:
        gate.wait()


def release_at_teardown(request: pytest.FixtureRequest, gate: threading.Event, threads: list) -> None:
    """Whatever assert ends the test early, open `gate` and wait for every thread in `threads` (None entries are skipped)
    before the test's patches are undone (finalizers run last-in first), so a parked body can never wake up inside the
    next test. `threads` is read when the finalizer runs, so the test can append to it after registering."""

    def _release() -> None:
        gate.set()
        finish_threads(*(t for t in threads if t is not None))

    request.addfinalizer(_release)
