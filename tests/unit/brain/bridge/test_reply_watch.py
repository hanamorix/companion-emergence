"""S218: the reply-activity signal the direct chat's RAM ladder watches.

`cli_throttle.reply_watch()` snapshots reply activity and returns `still_clear()`;
`wait_for_no_reply_in_flight()` blocks a background thread until no reply is in flight.
Real functions, real threads; synthetic only.
"""

from __future__ import annotations

import threading

import pytest

from brain.bridge import cli_throttle
from tests.thread_support import finish_threads


@pytest.fixture(autouse=True)
def _reset():
    cli_throttle.reset()
    yield
    # A test that failed mid-way must not leave a reply in flight for the next one.
    cli_throttle.reset()


# ---------------------------------------------------------------------------
# G1: reply_watch
# ---------------------------------------------------------------------------


def test_the_watch_is_clear_while_no_reply_starts():
    still_clear = cli_throttle.reply_watch()
    assert still_clear() is True
    assert still_clear() is True


def test_the_watch_turns_false_when_a_reply_starts_after_the_snapshot_and_stays_false():
    still_clear = cli_throttle.reply_watch()
    cli_throttle.note_user_message()
    assert still_clear() is False, "a reply in flight"
    cli_throttle.note_reply_end()
    assert still_clear() is False, "it stays false once that reply has ended"


def test_a_reply_that_started_and_ended_between_two_looks_is_still_seen():
    """The in-flight counter alone would miss this one (0 before, 0 after)."""
    still_clear = cli_throttle.reply_watch()
    cli_throttle.note_user_message()
    cli_throttle.note_reply_end()
    assert cli_throttle.reply_in_flight() is False
    assert still_clear() is False


def test_a_reply_already_in_flight_at_the_snapshot_makes_the_watch_false_even_after_it_ends():
    cli_throttle.note_user_message()
    still_clear = cli_throttle.reply_watch()
    assert still_clear() is False
    cli_throttle.note_reply_end()
    assert still_clear() is False


def test_a_snapshot_taken_after_the_reply_ended_is_clear_again():
    cli_throttle.note_user_message()
    cli_throttle.note_reply_end()
    assert cli_throttle.reply_watch()() is True


def test_a_watch_is_independent_of_the_other_watches():
    first = cli_throttle.reply_watch()
    cli_throttle.note_user_message()
    cli_throttle.note_reply_end()
    second = cli_throttle.reply_watch()
    assert (first(), second()) == (False, True)


def test_reset_leaves_the_started_count_alone_and_does_not_clear_a_watch():
    """A test helper's `reset()` between a snapshot and the check must not make an
    overlapped measurement look clear (the counter is compared by equality)."""
    before = cli_throttle._replies_started  # noqa: SLF001
    still_clear = cli_throttle.reply_watch()
    cli_throttle.note_user_message()
    cli_throttle.reset()
    assert cli_throttle._replies_started == before + 1  # noqa: SLF001
    assert still_clear() is False


def test_each_user_message_counts_one_started_reply():
    before = cli_throttle._replies_started  # noqa: SLF001
    cli_throttle.note_user_message()
    cli_throttle.note_user_message()
    cli_throttle.note_reply_end()
    cli_throttle.note_reply_end()
    assert cli_throttle._replies_started == before + 2  # noqa: SLF001


# ---------------------------------------------------------------------------
# G2: wait_for_no_reply_in_flight
# ---------------------------------------------------------------------------


def _waiter() -> tuple[threading.Thread, threading.Event]:
    returned = threading.Event()

    def _run() -> None:
        cli_throttle.wait_for_no_reply_in_flight()
        returned.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, returned


def test_the_wait_returns_at_once_when_no_reply_is_in_flight():
    thread, returned = _waiter()
    assert returned.wait(5.0)
    finish_threads(thread)


def test_the_wait_blocks_until_the_reply_ends():
    cli_throttle.note_user_message()
    thread, returned = _waiter()
    try:
        assert not returned.wait(0.3), "a reply is in flight: the wait must not return"
        cli_throttle.note_reply_end()
        assert returned.wait(5.0), "the end of the reply wakes the waiter"
    finally:
        cli_throttle.reset()  # releases a stuck waiter if an assertion failed
        finish_threads(thread)


def test_the_wait_ends_only_when_every_overlapping_reply_has_ended():
    cli_throttle.note_user_message()
    cli_throttle.note_user_message()
    thread, returned = _waiter()
    try:
        cli_throttle.note_reply_end()
        assert not returned.wait(0.3), "one reply is still in flight"
        cli_throttle.note_reply_end()
        assert returned.wait(5.0)
    finally:
        cli_throttle.reset()
        finish_threads(thread)


# ---------------------------------------------------------------------------
# S223 (T1): on_reply_start, the hook that stops a running measurement call
# ---------------------------------------------------------------------------


def test_a_callback_runs_once_on_the_replying_thread_after_the_counters_changed_and_outside_the_lock():
    watch = cli_throttle.reply_watch()
    seen: list[tuple[int, bool, bool, bool]] = []

    def _callback() -> None:
        # called back into the module: would deadlock if the module lock were still held
        seen.append(
            (threading.get_ident(), watch(), cli_throttle.reply_in_flight(), cli_throttle.reply_watch()())
        )

    watch.on_reply_start(_callback)
    assert seen == [], "nothing runs while no reply starts"
    cli_throttle.note_user_message()
    assert seen == [(threading.get_ident(), False, True, False)], "a fresh watch taken during a reply is not clear"
    cli_throttle.note_user_message()
    assert len(seen) == 1, "once per registration, even if more replies start"


def test_the_callback_runs_on_the_thread_that_starts_the_reply_not_the_registering_one():
    watch = cli_throttle.reply_watch()
    where: list[int] = []
    watch.on_reply_start(lambda: where.append(threading.get_ident()))
    replier = threading.Thread(target=cli_throttle.note_user_message)
    replier.start()
    finish_threads(replier)
    assert where == [replier.ident]


def test_registering_on_a_watch_that_is_not_clear_runs_the_callback_at_once_and_once():
    watch = cli_throttle.reply_watch()
    cli_throttle.note_user_message()  # a reply started after the snapshot
    calls: list[int] = []
    watch.on_reply_start(lambda: calls.append(threading.get_ident()))
    assert calls == [threading.get_ident()]
    cli_throttle.note_user_message()
    assert len(calls) == 1


def test_registering_while_a_reply_was_in_flight_at_the_snapshot_runs_at_once():
    cli_throttle.note_user_message()
    watch = cli_throttle.reply_watch()
    calls: list[int] = []
    watch.on_reply_start(lambda: calls.append(1))
    assert calls == [1]


def test_an_unregistered_callback_never_runs_and_unregister_is_idempotent():
    watch = cli_throttle.reply_watch()
    calls: list[int] = []
    unregister = watch.on_reply_start(lambda: calls.append(1))
    unregister()
    unregister()
    cli_throttle.note_user_message()
    assert calls == []


def test_a_raising_callback_does_not_break_the_reply_start_or_the_other_callbacks():
    watch = cli_throttle.reply_watch()
    calls: list[str] = []

    def _bad() -> None:
        raise RuntimeError("boom")

    watch.on_reply_start(_bad)
    watch.on_reply_start(lambda: calls.append("second"))
    cli_throttle.note_user_message()  # must not raise
    assert calls == ["second"]
    assert cli_throttle.reply_in_flight() is True, "the counters were updated before the callbacks"


def test_two_watches_register_independently():
    first, second = cli_throttle.reply_watch(), cli_throttle.reply_watch()
    calls: list[str] = []
    unregister_first = first.on_reply_start(lambda: calls.append("first"))
    second.on_reply_start(lambda: calls.append("second"))
    unregister_first()
    cli_throttle.note_user_message()
    assert calls == ["second"]


def test_reset_clears_registrations_and_a_late_unregister_does_not_raise():
    watch = cli_throttle.reply_watch()
    calls: list[int] = []
    watch.on_reply_start(lambda: calls.append(1))
    cli_throttle.reset()
    cli_throttle.note_user_message()
    assert calls == [], "reset cleared the registration (a stale one would have fired now)"


def test_a_late_unregister_after_reset_does_not_raise():
    watch = cli_throttle.reply_watch()
    unregister = watch.on_reply_start(lambda: None)
    cli_throttle.reset()
    unregister()
    unregister()
