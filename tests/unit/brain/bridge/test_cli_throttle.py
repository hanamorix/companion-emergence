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
    assert h1 is True and h2 is False
    cli_throttle.release_background()


def test_release_allows_next_background():
    cli_throttle.reset()
    far = cli_throttle._IDLE_SECONDS + 100.0
    assert cli_throttle.acquire_background(now=far) is True
    cli_throttle.release_background()
    assert cli_throttle.acquire_background(now=far) is True


def test_interactive_never_throttled():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    assert cli_throttle.interactive_allowed() is True
