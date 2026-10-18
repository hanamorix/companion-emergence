"""#440 - ST1.5e: the per-model rejection records have several writers (provider threads, log_usage,
set_persona_model) and one reader (/persona/state). Every access goes through one lock."""

from __future__ import annotations

import sys
import threading

import pytest

from brain.bridge import model_error

REJ = "There's an issue with the selected model (claude-sonet-5). It may not exist or you may not have access to it."


@pytest.fixture(autouse=True)
def _reset():
    model_error.reset()
    yield
    model_error.reset()


class _SpyLock:
    """Counts acquisitions and records whether it is held, so a test can assert both."""

    def __init__(self) -> None:
        self._real = threading.Lock()
        self.acquired = 0
        self.held = False

    def __enter__(self):
        self._real.acquire()
        self.acquired += 1
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        self._real.release()
        return False


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --- ORACLE: structural ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: model_error.note_failure(REJ),
        lambda: model_error.note_success("claude-sonet-5"),
        lambda: model_error.switch("claude-sonet-5", "sonnet", keep=frozenset()),
        lambda: model_error.state(),
        lambda: model_error.state(only={"x"}),
        lambda: model_error.note_unconfirmed("claude-sonet-5"),
        lambda: model_error.unconfirmed(),
    ],
    ids=["note_failure", "note_success", "switch", "state", "state_only", "note_unconfirmed", "unconfirmed"],
)
def test_each_public_function_takes_the_lock_exactly_once(monkeypatch, call):
    spy = _SpyLock()
    monkeypatch.setattr(model_error, "_lock", spy)
    call()
    assert spy.acquired == 1


def test_warn_interval_is_read_before_the_lock_is_taken(monkeypatch):
    spy = _SpyLock()
    monkeypatch.setattr(model_error, "_lock", spy)
    held_during_read = []
    real = model_error.tunables.get_tunable

    def _read(*a, **k):
        held_during_read.append(spy.held)
        return real(*a, **k)

    monkeypatch.setattr(model_error.tunables, "get_tunable", _read)
    model_error.note_failure(REJ)
    assert held_during_read and not any(held_during_read)


def test_logging_happens_outside_the_lock(monkeypatch):
    spy = _SpyLock()
    monkeypatch.setattr(model_error, "_lock", spy)
    held = []
    monkeypatch.setattr(model_error.log, "warning", lambda *a, **k: held.append(spy.held))
    model_error.note_failure(REJ)
    assert held == [False]


# The oracle's ability to fail is shown at stage 8 by mutating model_error.py (removing a `with _lock`)
# and recording that the exactly-once test goes red; see 8-harness.md.


# --- SMOKE: threaded (may not fail on a lockless variant under the GIL; not evidence of atomicity) -


def test_threaded_counts_are_exact():
    threads, per = 8, 2000
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        ts = [threading.Thread(target=lambda: [model_error.note_failure(REJ) for _ in range(per)]) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
    (rec,) = model_error.state()
    assert rec["count"] == threads * per


def test_interleaved_writers_and_readers_stay_consistent():
    stop = threading.Event()
    errors: list[BaseException] = []
    last_count = [0]

    def writer():
        while not stop.is_set():
            model_error.note_failure(REJ)

    def clearer():
        while not stop.is_set():
            model_error.note_success("claude-sonet-5")
            model_error.switch("claude-sonet-5", "claude-sonet-5", keep=frozenset())

    def reader():
        try:
            while not stop.is_set():
                for rec in model_error.state():
                    assert set(rec) == {"model", "since", "last_at", "count"}
                    assert rec["count"] >= 1
                    last_count[0] = rec["count"]
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=f) for f in (writer, writer, clearer, reader)]
    for t in ts:
        t.start()
    threading.Event().wait(0.3)
    stop.set()
    for t in ts:
        t.join()
    assert not errors
