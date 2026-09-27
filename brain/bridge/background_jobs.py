"""Registry of currently-running background work (ram-spike-fix INC-11,
spec §6, S15/S23/S38).

A thread-safe REFCOUNTED registry, published to the NellFace app over
``/persona/state`` (``brain.bridge.persona_state.build_persona_state``) so
the app can show a line naming whatever is running right now, and clear it
when nothing is. Membership is "actually executing", not "scheduled" or
"due": the central cadence function (``central_cadence.py``) marks a gated
job a member only for the duration of its ``run()`` call, and the
supervisor's heartbeat hook marks ``"heartbeat"`` a member only for the
duration of the tick it actually performs. A job that pauses (INC-10)
leaves the set the moment it returns ``PAUSED`` — a paused job is not
running (spec §6).

Refcounted, not a plain set (stage-6 red-team MINOR, fixed): today's three
call sites can each only have one live `running(name)` per name at a time
(central_cadence.py runs jobs sequentially on one thread; heartbeat.py's
wraps are both behind a cross-process file lock), so a plain set has never
been observed to misbehave — but this module's own docstring already
claims to not assume single-threaded/single-caller use (see C31(c)), and a
plain set breaks that claim the moment two holders ever mark the SAME name
running at once: whichever holder exits first unconditionally discards the
name, wiping out the still-live second holder's membership. A count avoids
that — the name is only removed once every concurrent holder for it has
exited. Same idiom this codebase already uses for the identical hazard:
``brain.bridge.cli_throttle``'s ``_inflight_replies``/``_inflight_background``
counters ("counter, not boolean, so two concurrent turns don't clear each
other").

Accessors: the central cadence function and the heartbeat hook (writers,
both on the single supervisor thread in production, but the lock below
does not assume that — see C31(c)); ``/persona/state`` (reader, a request
thread). ``snapshot()`` returns a copy so a reader never observes the
registry mutate mid-iteration.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_running: Counter[str] = Counter()


@contextmanager
def running(name: str) -> Iterator[None]:
    """Mark ``name`` as running for the duration of the with-block.

    Increments ``name``'s count on entry, decrements on exit (including
    when the wrapped code raises, so a job that errors out never leaves a
    stale entry) — ``name`` is a member of ``snapshot()`` iff its count is
    above zero, so two overlapping ``running(name)`` calls for the SAME
    name both have to exit before the name clears.
    """
    with _lock:
        _running[name] += 1
    try:
        yield
    finally:
        with _lock:
            _running[name] -= 1
            if _running[name] <= 0:
                del _running[name]


def snapshot() -> list[str]:
    """Sorted list of currently-running job/heartbeat names.

    Sorted (not insertion- or completion-order) so two snapshots taken
    with the same membership always compare equal, and so the app's poll
    doesn't reorder the line on every tick for no reason.
    """
    with _lock:
        return sorted(_running)


def _reset_for_tests() -> None:
    """Test-only: clear the registry between tests that don't otherwise
    tear down cleanly (mirrors the ``_reset_for_tests`` convention used
    elsewhere in this codebase, e.g. ``tunables``)."""
    with _lock:
        _running.clear()
