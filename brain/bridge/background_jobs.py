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
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

_lock = threading.Lock()
_running: Counter[str] = Counter()
# INC-11 follow-up (owner-set layout, longest-running-first ordering): the
# monotonic time.monotonic() at which each name's count went 0->1. A NESTED
# `running(name)` call (the count going 2, 3, ...) does NOT reset this — the
# elapsed time reported is for the name's current unbroken running streak,
# not any one holder's own duration.
_started_at: dict[str, float] = {}


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
        if _running[name] == 0:
            _started_at[name] = time.monotonic()
        _running[name] += 1
    try:
        yield
    finally:
        with _lock:
            _running[name] -= 1
            if _running[name] <= 0:
                del _running[name]
                _started_at.pop(name, None)


def snapshot() -> list[str]:
    """Sorted (alphabetical) list of currently-running job/heartbeat names.

    Sorted (not insertion- or completion-order) so two snapshots taken
    with the same membership always compare equal. Callers that need
    longest-running-first ordering (the NellFace line) want
    ``snapshot_with_elapsed`` instead — this one is kept for callers that
    only need membership.
    """
    with _lock:
        return sorted(_running)


def snapshot_with_elapsed(*, now: float | None = None) -> list[tuple[str, float]]:
    """``[(name, elapsed_seconds), ...]`` for every currently-running name,
    ordered LONGEST-RUNNING FIRST (ties broken alphabetically, for a
    deterministic order when two names started in the same clock tick).

    ``elapsed_seconds`` is measured from the ``time.monotonic()`` recorded
    when the name's refcount went 0->1 — a nested/duplicate ``running(name)``
    call does not reset it, so the reported duration is always "how long has
    ANYONE been marking this name running, unbroken" (see the module-level
    ``_started_at`` note), which is the quantity that's actually meaningful
    to show a human ("this has been going for N seconds"), not any single
    holder's own slice of it.
    """
    if now is None:
        now = time.monotonic()
    with _lock:
        items = [(name, now - _started_at[name]) for name in _running]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return items


def _reset_for_tests() -> None:
    """Test-only: clear the registry between tests that don't otherwise
    tear down cleanly (mirrors the ``_reset_for_tests`` convention used
    elsewhere in this codebase, e.g. ``tunables``)."""
    with _lock:
        _running.clear()
        _started_at.clear()
