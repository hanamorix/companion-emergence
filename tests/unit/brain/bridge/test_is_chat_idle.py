"""ram-spike-fix INC-6: cli_throttle.is_chat_idle() — C24 (the semantics)
and C35 (AC23, fail-closed + rate-limited logging)."""
from __future__ import annotations

import logging
import threading
import time

import pytest

from brain.bridge import cli_throttle


@pytest.fixture(autouse=True)
def _reset():
    cli_throttle.reset()
    yield
    cli_throttle.reset()


# ---------------------------------------------------------------------------
# C24: is_chat_idle semantics
# ---------------------------------------------------------------------------


def test_fresh_process_is_idle_before_any_message():
    """S35: a fresh process (never seen a message) is idle from its very
    first instant — the anchor is -inf, so any finite `now` clears the lull."""
    assert cli_throttle.is_chat_idle(now=0.0) is True
    assert cli_throttle.is_chat_idle(now=1_000_000.0) is True


def test_not_idle_while_reply_in_flight_even_past_the_lull():
    """S35/S42: false while a reply is in flight, even when the last message
    is older than the lull (a turn longer than the lull itself)."""
    cli_throttle.note_user_message(at=0.0)
    # Far past the lull, but the reply hasn't ended yet (no note_reply_end).
    assert cli_throttle.is_chat_idle(now=cli_throttle._lull_seconds() + 100.0) is False


def test_idle_only_once_lull_elapses_after_reply_end():
    cli_throttle.note_user_message(at=0.0)
    cli_throttle.note_reply_end(at=1.0)
    lull = cli_throttle._lull_seconds()
    assert cli_throttle.is_chat_idle(now=1.0 + lull - 1.0) is False
    assert cli_throttle.is_chat_idle(now=1.0 + lull) is True


def test_multiple_overlapping_inflight_replies_tracked_by_count():
    """The in-flight state is a counter, not a boolean (S5x): two concurrent
    turns must both complete before is_chat_idle can go True."""
    cli_throttle.note_user_message(at=0.0)
    cli_throttle.note_user_message(at=0.0)
    cli_throttle.note_reply_end(at=0.0)
    far = cli_throttle._lull_seconds() + 100.0
    assert cli_throttle.is_chat_idle(now=far) is False, "one reply still in flight"
    cli_throttle.note_reply_end(at=0.0)
    assert cli_throttle.is_chat_idle(now=far) is True


def test_concurrent_overlapping_turns_from_real_threads_never_corrupt_the_counter():
    """Stage-6 code red-team MINOR (coverage gap): the in-flight counter's
    thread-safety must be exercised with REAL concurrent threads, not a
    sequential simulation — two overlapping "turns" started/ended by real
    threads racing on note_user_message/note_reply_end must never leave
    _inflight_replies negative or stuck positive, and is_chat_idle must
    reflect the true state at every observed point."""
    n_threads = 8
    start_barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def _one_turn():
        try:
            start_barrier.wait(timeout=5.0)
            cli_throttle.note_user_message()
            time.sleep(0.01)
            cli_throttle.note_reply_end()
        except BaseException as exc:  # noqa: BLE001 — surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=_one_turn) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"worker thread(s) raised: {errors}"
    assert cli_throttle._inflight_replies == 0, (  # noqa: SLF001 — the exact invariant under test
        f"counter must settle at 0 once every thread's turn ended, got "
        f"{cli_throttle._inflight_replies}"
    )
    far = cli_throttle._lull_seconds() + 100.0
    assert cli_throttle.is_chat_idle(now=time.monotonic() + far) is True


def test_clock_is_real_monotonic_by_default():
    """C24: 'clock is time.monotonic' — is_chat_idle with no now= argument
    uses the real monotonic clock (not wall clock)."""
    import time as _time

    cli_throttle.note_user_message()
    cli_throttle.note_reply_end()
    assert cli_throttle.is_chat_idle() is False  # just happened, real clock
    # Fast-forward using the real clock's own reference frame.
    assert cli_throttle.is_chat_idle(now=_time.monotonic() + cli_throttle._lull_seconds() + 1) is True


# ---------------------------------------------------------------------------
# C35 / AC23: fail-closed + rate-limited logging
# ---------------------------------------------------------------------------


def test_is_chat_idle_fails_closed_on_internal_error(monkeypatch):
    import brain.bridge.cli_throttle as mod

    def _boom():
        raise RuntimeError("clock broken")

    monkeypatch.setattr(mod.time, "monotonic", _boom)
    assert cli_throttle.is_chat_idle() is False


def test_every_gated_caller_defers_when_is_chat_idle_raises(monkeypatch):
    """Every caller that gates on is_chat_idle (should_yield/slot_available/
    acquire_background/background_slot) must report "not idle"/deny when the
    check itself is broken — not silently fail open just because THEY wrap
    their own try/except for the unrelated slot-cap half."""
    import brain.bridge.cli_throttle as mod

    monkeypatch.setattr(mod, "is_chat_idle", lambda **_: False)

    assert cli_throttle.should_yield() is True
    assert cli_throttle.slot_available() is False
    assert cli_throttle.acquire_background() is False
    with cli_throttle.background_slot() as ok:
        assert ok is False


class _FakeClock:
    """A controllable fake monotonic clock — used so the failure injection
    (below) breaks the IDLE COMPUTATION, not the rate-limiter's own timing
    bookkeeping (which also calls time.monotonic() internally); breaking both
    at once would make the rate-limiter's own elapsed-time math meaningless."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


def test_error_logging_is_rate_limited_to_at_most_four_lines_over_1000_failures(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
):
    """1,000 identical failures over a fake 30-min clock -> <= 4 ERROR lines
    (first + one per completed 10-min window), later ones carrying the
    suppressed count. Fault injected via _lull_seconds (a plausible real
    fault — a broken tunables lookup) so the fake clock below still drives
    the rate-limiter's own elapsed-time bookkeeping."""
    import brain.bridge.cli_throttle as mod

    clock = _FakeClock(0.0)
    monkeypatch.setattr(mod.time, "monotonic", clock)
    monkeypatch.setattr(mod, "_lull_seconds", lambda: (_ for _ in ()).throw(RuntimeError("persistent fault")))

    with caplog.at_level(logging.ERROR, logger="brain.bridge.cli_throttle"):
        for i in range(1000):
            clock.t = i * 1.8  # 1000 * 1.8s ≈ 1800s = 30 minutes
            assert cli_throttle.is_chat_idle() is False

    error_lines = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_lines) <= 4, f"expected <= 4 ERROR lines, got {len(error_lines)}"
    assert len(error_lines) >= 1
    if len(error_lines) > 1:
        assert "suppressed" in error_lines[-1].getMessage()


def test_error_type_change_logs_immediately_not_suppressed(monkeypatch: pytest.MonkeyPatch):
    """A CHANGED error (different type/message) logs immediately, not waiting
    for the rate-limit window — distinct fault, not a repeat."""
    import brain.bridge.cli_throttle as mod

    clock = _FakeClock(0.0)
    monkeypatch.setattr(mod.time, "monotonic", clock)

    calls = {"n": 0}

    def _flip():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("fault A")
        raise ValueError("fault B")

    monkeypatch.setattr(mod, "_lull_seconds", _flip)

    logger = logging.getLogger("brain.bridge.cli_throttle")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _Capture()
    logger.addHandler(h)
    try:
        assert cli_throttle.is_chat_idle() is False
        assert cli_throttle.is_chat_idle() is False
    finally:
        logger.removeHandler(h)
    errors = [r for r in records if r.levelno == logging.ERROR]
    assert len(errors) == 2, "a changed error type must log immediately, not be suppressed"


def test_recovery_logs_one_info_line_after_a_failure(monkeypatch: pytest.MonkeyPatch):
    import brain.bridge.cli_throttle as mod

    clock = _FakeClock(0.0)
    monkeypatch.setattr(mod.time, "monotonic", clock)
    monkeypatch.setattr(mod, "_lull_seconds", lambda: (_ for _ in ()).throw(RuntimeError("fault")))

    assert cli_throttle.is_chat_idle() is False

    monkeypatch.undo()  # restore real time.monotonic AND real _lull_seconds

    logger = logging.getLogger("brain.bridge.cli_throttle")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _Capture()
    logger.addHandler(h)
    prior_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        assert cli_throttle.is_chat_idle() is True  # real clock, fresh process = idle
    finally:
        logger.removeHandler(h)
        logger.setLevel(prior_level)

    info_lines = [r for r in records if r.levelno == logging.INFO]
    assert len(info_lines) == 1, f"expected exactly one recovery INFO line, got {len(info_lines)}"
