import time

from brain.bridge import cli_throttle


def test_background_defers_while_interactive_active():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    assert cli_throttle.acquire_background(now=time.monotonic()) is False


def test_background_runs_when_idle_long_enough():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.acquire_background(now=cli_throttle._lull_seconds() + 1.0) is True


def test_concurrency_semaphore_blocks_second_background():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    h1 = cli_throttle.acquire_background(now=far)
    h2 = cli_throttle.acquire_background(now=far)  # second blocked by N=1 cap
    try:
        assert h1 is True and h2 is False
    finally:
        cli_throttle.release_background()


def test_release_allows_next_background():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()
    assert cli_throttle.acquire_background(now=far) is True



def test_background_slot_acquires_and_releases():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    with cli_throttle.background_slot(now=far) as ok:
        assert ok is True
    # released on exit → next acquire succeeds
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()


def test_background_slot_yields_false_when_deferred():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    with cli_throttle.background_slot() as ok:
        assert ok is False


def test_acquire_fails_open_on_internal_error(monkeypatch):
    cli_throttle.reset()
    import brain.bridge.cli_throttle as mod
    monkeypatch.setattr(mod.time, "monotonic", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    # now=None path uses time.monotonic() inside is_chat_idle → is_chat_idle
    # itself fails CLOSED (not idle) → acquire_background's own slot-cap half
    # never even runs, so this exercises is_chat_idle's fail-closed posture,
    # NOT acquire_background's fail-open slot-cap wrapper (see
    # test_is_chat_idle_fails_closed_on_internal_error below for that split).
    assert cli_throttle.acquire_background() is False


def test_should_yield_true_when_chat_recent():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.should_yield(now=1.0) is True


def test_should_yield_false_when_idle():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.should_yield(now=cli_throttle._lull_seconds() + 1.0) is False


def test_should_yield_does_not_consume_a_slot():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    cli_throttle.should_yield(now=far)  # peeking must not take the slot
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()


def test_slot_available_true_when_idle_and_cap_free():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    assert cli_throttle.slot_available(now=far) is True


def test_slot_available_false_when_interactive_recent():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.slot_available(now=1.0) is False


def test_slot_available_false_when_cap_full():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    assert cli_throttle.acquire_background(now=far) is True  # take the only slot
    try:
        assert cli_throttle.slot_available(now=far) is False
    finally:
        cli_throttle.release_background()


def test_slot_available_does_not_consume_a_slot():
    cli_throttle.reset()
    far = cli_throttle._lull_seconds() + 100.0
    cli_throttle.slot_available(now=far)  # peeking must not take the slot
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()


# ---------------------------------------------------------------------------
# ram-spike-fix INC-6: is_chat_idle / the single lull / fail-closed (C4, C24, C35)
# ---------------------------------------------------------------------------
#
# NOTE: test_acquire_background_min_idle_override (the old per-caller
# min_idle override) is REMOVED here, not merely renamed — INC-6 retires
# min_idle entirely (C4): every caller now shares the one registered
# chat.idle_lull_seconds lull via is_chat_idle(), with no per-caller
# shorter/longer window. See test_cli_throttle_idle.py for the dedicated
# is_chat_idle/C24/C35 coverage.
