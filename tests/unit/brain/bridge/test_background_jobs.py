"""ram-spike-fix INC-11 — the background-jobs registry (spec §6, S15/S23/S38).

Criteria covered here: C11(a) (publish/clear), C31(c) (registry is a new
shared-state accessor — no-lost-update under a concurrent interleaving,
ST1.5e).
"""

from __future__ import annotations

import threading
import time
from collections import Counter

import pytest

from brain.bridge import background_jobs


def setup_function() -> None:
    background_jobs._reset_for_tests()


def teardown_function() -> None:
    background_jobs._reset_for_tests()


# ---------------------------------------------------------------------------
# C11(a) — publish while running, clear after
# ---------------------------------------------------------------------------


def test_snapshot_is_empty_with_nothing_running() -> None:
    assert background_jobs.snapshot() == []


def test_name_is_present_only_for_the_duration_of_the_with_block() -> None:
    assert background_jobs.snapshot() == []
    with background_jobs.running("compaction"):
        assert background_jobs.snapshot() == ["compaction"]
    assert background_jobs.snapshot() == []


def test_multiple_concurrent_names_all_appear_sorted() -> None:
    with background_jobs.running("weekly_selftune"), background_jobs.running("heartbeat"):
        assert background_jobs.snapshot() == ["heartbeat", "weekly_selftune"]


def test_a_raising_job_still_clears_its_name() -> None:
    """A job that raises must not leave a stale entry — the central cadence
    function's own S44 rule (raise → still resolves) means `run()` can raise
    inside the `with` block; the registry must not depend on a clean return."""
    assert background_jobs.snapshot() == []
    try:
        with background_jobs.running("maintenance"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert background_jobs.snapshot() == []


def test_paused_job_is_not_a_member_once_run_returns() -> None:
    """S38: a paused job is not running. `run()` returning PAUSED still exits
    the `with` block normally (a return, not an exception), which already
    removes the name — this documents that the registry itself has no
    separate "paused" state, matching central_cadence.py's placement of the
    `with` around exactly the `job.run()` call."""
    with background_jobs.running("pass2"):
        pass  # simulates run() returning JobOutcome.PAUSED and the `with` exiting
    assert background_jobs.snapshot() == []


def test_two_overlapping_holders_of_the_same_name_both_have_to_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-6 red-team MINOR, fixed: with a plain `set`, two concurrent
    `running(name)` holders for the SAME name would have the first holder's
    exit wipe the name out from under the still-live second holder. The
    refcounted registry must survive this: the name stays present until
    EVERY overlapping holder has exited, deterministically forced here (no
    live race needed) by holding holder A open, entering and fully exiting
    holder B while A is still open, then exiting A."""
    with background_jobs.running("heartbeat"):  # holder A opens
        assert background_jobs.snapshot() == ["heartbeat"]
        with background_jobs.running("heartbeat"):  # holder B opens (same name)
            assert background_jobs.snapshot() == ["heartbeat"]
        # holder B has fully exited; A is still open — must still be a member.
        assert background_jobs.snapshot() == ["heartbeat"], (
            "the still-live holder A's membership must survive holder B's exit"
        )
    # both holders have now exited.
    assert background_jobs.snapshot() == []


# ---------------------------------------------------------------------------
# C31(c) — new shared-state accessor: no-lost-update under a hammer test.
# Deterministic-enough per ST1.5e's "stated number of runs with a pass-rate
# floor" allowance for a live-race check: a fixed duration, tight-loop hammer
# with no sleeps reliably interleaves via GIL switches. Must fail against the
# unguarded version (below) — the H6/H9 fail-first oracle self-test.
# ---------------------------------------------------------------------------

_NAMES = tuple(f"job{i}" for i in range(6))


class _NoOpLock:
    """Stand-in for `background_jobs._lock` that performs no exclusion — the
    mutant that proves the real lock is load-bearing, not decorative."""

    def __enter__(self):  # noqa: ANN204
        return None

    def __exit__(self, *_exc):  # noqa: ANN002, ANN204
        return False


def _hammer(*, duration_s: float) -> list[BaseException]:
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer(name: str) -> None:
        try:
            while not stop.is_set():
                with background_jobs.running(name):
                    pass
        except BaseException as exc:  # noqa: BLE001 — captured for the main thread
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                snap = background_jobs.snapshot()
                assert isinstance(snap, list)
                assert snap == sorted(snap), f"unsorted snapshot: {snap}"
                assert len(snap) == len(set(snap)), f"duplicate entries: {snap}"
                assert all(n in _NAMES for n in snap), f"unknown entry: {snap}"
        except BaseException as exc:  # noqa: BLE001 — captured for the main thread
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in _NAMES]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()
    return errors


def test_c31c_registry_survives_concurrent_running_and_snapshot() -> None:
    """6 writer threads (each entering/exiting `running()` for its own name in
    a tight loop) plus 4 reader threads (`snapshot()` in a tight loop) for
    1s: no exception, every snapshot well-formed (sorted, no dupes, only
    known names) throughout."""
    errors = _hammer(duration_s=1.0)
    assert errors == [], f"unexpected error(s) under concurrent access: {errors}"


class _InjectableIterCounter(Counter):
    """A `Counter` (matches `background_jobs._running`'s real type) whose
    iterator calls `on_first_yield` once, right after the underlying
    dict-iterator is live and mid-scan — the deterministic injection point
    ST1.5e asks for (inject the competing mutation into the guarded window,
    don't hope a live race lands there)."""

    def __init__(self, *args, on_first_yield=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._on_first_yield = on_first_yield

    def __iter__(self):  # noqa: D105
        it = super().__iter__()
        fired = False
        for item in it:
            if not fired and self._on_first_yield is not None:
                fired = True
                self._on_first_yield()
            yield item


def test_c31c_hammer_fails_without_the_lock(monkeypatch) -> None:
    """H6 self-test (ST1.5f/ST1.5e), deterministic injection: while
    `snapshot()`'s `sorted(_running)` is mid-iteration (the injection point
    below), a second thread runs the SAME `running()` context manager a real
    job uses, and is held with its entry (the `add`) OUTSTANDING — not yet
    undone by its matching `discard` — at the exact moment the reader
    resumes iterating. With the lock removed (the mutant), that leaves the
    registry's membership changed mid-scan and CPython raises `RuntimeError:
    dictionary changed size during iteration` (`_running` is a `Counter`, a
    dict subclass) — proving the lock is load-bearing (a same-thread
    add+discard round trip before resumption does NOT reproduce this —
    verified separately — so the hold is what makes the injection land, not
    incidental timing)."""
    added = threading.Event()
    may_exit = threading.Event()
    writer_thread: list[threading.Thread] = []

    def _mutate_registry() -> None:
        with background_jobs.running("interloper"):
            added.set()
            may_exit.wait(timeout=2.0)
        # `running()`'s discard fires here, after `may_exit` is set below.

    def _inject() -> None:
        t = threading.Thread(target=_mutate_registry)
        writer_thread.append(t)
        t.start()
        assert added.wait(timeout=2.0), "writer never reached the outstanding add"

    monkeypatch.setattr(
        background_jobs,
        "_running",
        _InjectableIterCounter({"a": 1, "b": 1, "c": 1}, on_first_yield=_inject),
    )
    monkeypatch.setattr(background_jobs, "_lock", _NoOpLock())

    try:
        with pytest.raises(RuntimeError, match="dictionary changed size during iteration"):
            background_jobs.snapshot()
    finally:
        may_exit.set()
        writer_thread[0].join(timeout=2.0)
        assert not writer_thread[0].is_alive()


def test_c31c_the_same_injection_is_safe_with_the_real_lock(monkeypatch) -> None:
    """Control for the test above: same injection, real `threading.Lock`.
    `snapshot()` holds the lock for the whole `sorted()` call, so the
    injected `running()` call on the other thread blocks trying to acquire
    it until `snapshot()` returns — no interleaving, no exception. This is
    what shows the lock (not just "a lock exists somewhere") is what closes
    the gap the mutant above opens."""

    def _mutate_registry() -> None:
        with background_jobs.running("interloper"):
            pass

    def _inject() -> None:
        t = threading.Thread(target=_mutate_registry)
        t.start()
        # The writer is blocked on the real lock; give it a moment to prove
        # it's actually blocked (not merely not-yet-scheduled), then let the
        # reader finish and release the lock so it can proceed.
        t.join(timeout=0.2)
        assert t.is_alive(), "writer should be blocked on the lock, not racing ahead"
        # (joined again, unconditionally, after the reader's `with` block below)

    monkeypatch.setattr(
        background_jobs,
        "_running",
        _InjectableIterCounter({"a": 1, "b": 1, "c": 1}, on_first_yield=_inject),
    )

    result = background_jobs.snapshot()
    assert result == ["a", "b", "c"]
    # Let the (now-unblocked) writer finish and confirm it did its work.
    for _ in range(50):
        if "interloper" not in background_jobs._running:  # noqa: SLF001
            break
        time.sleep(0.02)
    assert "interloper" not in background_jobs._running  # noqa: SLF001
