import time

from brain.bridge import cli_throttle


def test_background_defers_while_interactive_active():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    assert cli_throttle.acquire_background(now=time.monotonic()) is False


def test_background_runs_when_idle_long_enough():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.acquire_background(now=cli_throttle._IDLE_SECONDS + 1.0) is True


def test_concurrency_semaphore_blocks_second_background():
    cli_throttle.reset()
    far = cli_throttle._IDLE_SECONDS + 100.0
    h1 = cli_throttle.acquire_background(now=far)
    h2 = cli_throttle.acquire_background(now=far)  # second blocked by N=1 cap
    try:
        assert h1 is True and h2 is False
    finally:
        cli_throttle.release_background()


def test_release_allows_next_background():
    cli_throttle.reset()
    far = cli_throttle._IDLE_SECONDS + 100.0
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()
    assert cli_throttle.acquire_background(now=far) is True



def test_background_slot_acquires_and_releases():
    cli_throttle.reset()
    far = cli_throttle._IDLE_SECONDS + 100.0
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
    # now=None path uses time.monotonic() → raises → fail open
    assert cli_throttle.acquire_background() is True


def test_should_yield_true_when_chat_recent():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.should_yield(now=1.0) is True


def test_should_yield_false_when_idle():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active(at=0.0)
    assert cli_throttle.should_yield(now=cli_throttle._IDLE_SECONDS + 1.0) is False


def test_should_yield_does_not_consume_a_slot():
    cli_throttle.reset()
    far = cli_throttle._IDLE_SECONDS + 100.0
    cli_throttle.should_yield(now=far)  # peeking must not take the slot
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()
