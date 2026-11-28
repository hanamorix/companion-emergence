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
# #817 (spec 9, A26): the reply-only gate pass 2 uses instead of the quiet-chat lull
# ---------------------------------------------------------------------------


def test_reply_gate_is_open_inside_the_lull_and_closed_while_a_reply_is_in_flight():
    cli_throttle.reset()
    cli_throttle.note_user_message()
    assert cli_throttle.reply_gate_open() is False  # a reply is in flight
    cli_throttle.note_reply_end()
    assert cli_throttle.is_chat_idle() is False  # the lull has not passed
    assert cli_throttle.reply_gate_open() is True  # but no reply is in flight


def test_reply_gate_is_closed_while_the_bridge_is_stopping():
    cli_throttle.reset()
    cli_throttle.request_shutdown()
    try:
        assert cli_throttle.reply_gate_open() is False
        assert cli_throttle.acquire_background(lull_required=False) is False
    finally:
        cli_throttle.clear_shutdown()


def test_reply_gate_fails_closed_on_an_internal_error(monkeypatch):
    cli_throttle.reset()

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_throttle, "reply_in_flight", _boom)
    assert cli_throttle.reply_gate_open() is False


def test_slot_without_lull_is_granted_inside_the_lull_when_no_reply_is_in_flight():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()  # a message just now
    assert cli_throttle.slot_available() is False
    assert cli_throttle.acquire_background() is False
    assert cli_throttle.slot_available(lull_required=False) is True
    with cli_throttle.background_slot(lull_required=False) as ok:
        assert ok is True


def test_slot_without_lull_is_denied_while_a_reply_is_in_flight():
    cli_throttle.reset()
    cli_throttle.note_user_message()
    try:
        assert cli_throttle.slot_available(lull_required=False) is False
        assert cli_throttle.acquire_background(lull_required=False) is False
        with cli_throttle.background_slot(lull_required=False) as ok:
            assert ok is False
    finally:
        cli_throttle.note_reply_end()


def test_slot_without_lull_still_counts_against_the_one_background_slot():
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    assert cli_throttle.acquire_background(lull_required=False) is True
    try:
        assert cli_throttle.slot_available(lull_required=False) is False
        assert cli_throttle.acquire_background(lull_required=False) is False
        far = cli_throttle._lull_seconds() + 1e9
        assert cli_throttle.acquire_background(now=far) is False  # a lull-gated job shares that same slot
    finally:
        cli_throttle.release_background()


# ---------------------------------------------------------------------------
# #650 finding 5: waiting for the one background slot on a release notification (pass 2's own thread)
# ---------------------------------------------------------------------------


def test_wait_for_background_slot_returns_false_at_once_when_a_slot_is_free():
    cli_throttle.reset()
    assert cli_throttle.wait_for_background_slot() is False


def test_wait_for_background_slot_blocks_until_the_release_and_returns_true():
    import threading

    cli_throttle.reset()
    assert cli_throttle.acquire_background(lull_required=False) is True
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(cli_throttle.wait_for_background_slot()), daemon=True)
    t.start()
    t.join(0.2)  # a deliberate probe: still blocked
    assert t.is_alive() and result == []
    cli_throttle.release_background()
    t.join()
    assert result == [True]
    cli_throttle.reset()


def test_wait_for_background_slot_returns_when_a_stop_is_requested():
    import threading

    cli_throttle.reset()
    assert cli_throttle.acquire_background(lull_required=False) is True
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(cli_throttle.wait_for_background_slot()), daemon=True)
    t.start()
    t.join(0.2)  # a deliberate probe: still blocked
    assert t.is_alive()
    cli_throttle.request_shutdown()
    t.join()
    assert result == [True]
    cli_throttle.reset()


def test_wait_for_background_slot_with_two_slots_waits_only_when_both_are_held(monkeypatch):
    import threading

    cli_throttle.reset()
    monkeypatch.setattr(cli_throttle, "_max_concurrent_background", lambda: 2)
    assert cli_throttle.acquire_background(lull_required=False) is True
    assert cli_throttle.wait_for_background_slot() is False  # one of two is free
    assert cli_throttle.acquire_background(lull_required=False) is True
    result: list[bool] = []
    t = threading.Thread(target=lambda: result.append(cli_throttle.wait_for_background_slot()), daemon=True)
    t.start()
    t.join(0.2)  # a deliberate probe: both held, still blocked
    assert t.is_alive()
    cli_throttle.release_background()
    t.join()
    assert result == [True]
    cli_throttle.reset()


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
