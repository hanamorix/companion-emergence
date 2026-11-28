"""Pass 2 on its own short-lived thread (#650 finding 5, L2 item 32): `pass2_queue.drain_in_own_thread`.

Real `cli_throttle` state (replies in flight, the background slot, the stop flag), synthetic tmp persona dirs, `test_probe`
queue records (no model call). A deliberate probe (`join(0.2)` to show the thread is still waiting) keeps its time bound; every
wait for the thread to FINISH is unbounded (tests/thread_support.py).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from brain.bridge import background_jobs, cli_throttle
from brain.chat import pass2_queue
from brain.utils.file_lock import file_lock
from tests.thread_support import finish_threads


@pytest.fixture(autouse=True)
def _fresh_throttle():
    cli_throttle.reset()
    pass2_queue.reset()
    yield
    cli_throttle.reset()
    pass2_queue.reset()


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "persona"
    p.mkdir()
    return p


def _enqueue(persona_dir: Path, log: list[str], label: str, side=None) -> None:
    rid = pass2_queue.new_record_id()

    def _fx() -> None:
        log.append(label)
        if side is not None:
            side()

    pass2_queue.register_test_side_effect(rid, _fx)
    pass2_queue.enqueue({"id": rid, "kind": "test_probe"}, persona_dir=persona_dir)


def _start(persona_dir: Path) -> threading.Thread:
    t = threading.Thread(target=pass2_queue.drain_in_own_thread, args=(persona_dir,), daemon=True)
    t.start()
    return t


def _still_waiting(t: threading.Thread) -> bool:
    t.join(0.2)  # a deliberate probe, not a completion wait
    return t.is_alive()


def test_it_drains_the_whole_queue_and_ends_and_is_listed_as_running_while_it_works(persona_dir: Path) -> None:
    log: list[str] = []
    seen: list[list[str]] = []
    _enqueue(persona_dir, log, "a", side=lambda: seen.append(background_jobs.snapshot()))
    _enqueue(persona_dir, log, "b")
    _enqueue(persona_dir, log, "c")
    finish_threads(_start(persona_dir))
    assert log == ["a", "b", "c"]
    assert pass2_queue.queue_length(persona_dir) == 0
    assert seen == [["pass2"]], "the app's running-now line names pass 2 while the thread drains"
    assert background_jobs.snapshot() == []
    assert cli_throttle.slot_available(lull_required=False), "the slot is let go when it ends"


def test_it_never_runs_the_heartbeat_between_items_only_the_reply_gate(persona_dir: Path, monkeypatch) -> None:
    seen: dict = {}

    def fake_drain(_pd, *, should_pause=None, **_k):
        seen["pause"] = should_pause
        return 0

    monkeypatch.setattr(pass2_queue, "drain_all_locked", fake_drain)
    finish_threads(_start(persona_dir))
    pause = seen["pause"]
    assert pause() is False
    cli_throttle.note_user_message()
    assert pause() is True, "a reply in flight stops it between items"
    cli_throttle.note_reply_end()
    cli_throttle.request_shutdown()
    assert pause() is True, "a stop stops it between items"
    # The quiet-chat lull is not asked: a reply that just ended (inside the lull) does not hold the gate.
    cli_throttle.reset()
    cli_throttle.mark_interactive_active()
    assert pause() is False


def test_it_waits_for_the_reply_in_flight_to_end(persona_dir: Path) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    cli_throttle.note_user_message()
    t = _start(persona_dir)
    assert _still_waiting(t) and log == []
    cli_throttle.note_reply_end()
    finish_threads(t)
    assert log == ["a"]


def test_a_reply_starting_mid_drain_stops_it_between_items_and_it_resumes_after_that_reply(persona_dir: Path) -> None:
    log: list[str] = []
    started = threading.Event()

    def _reply_starts() -> None:
        cli_throttle.note_user_message()
        started.set()

    _enqueue(persona_dir, log, "a", side=_reply_starts)
    _enqueue(persona_dir, log, "b")
    t = _start(persona_dir)
    started.wait()
    assert _still_waiting(t), "the thread waits for that reply, it does not end with work left"
    assert log == ["a"] and pass2_queue.queue_length(persona_dir) == 1
    assert cli_throttle.slot_available(lull_required=False) is False  # a reply is in flight: gate shut
    cli_throttle.note_reply_end()
    finish_threads(t)
    assert log == ["a", "b"] and pass2_queue.queue_length(persona_dir) == 0


def test_it_waits_for_the_slot_to_be_released_without_polling(persona_dir: Path) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    assert cli_throttle.acquire_background(lull_required=False)  # another job (a dream, reflex, research call) holds it
    t = _start(persona_dir)
    assert _still_waiting(t) and log == []
    cli_throttle.release_background()
    finish_threads(t)
    assert log == ["a"]


def test_with_two_slots_allowed_it_takes_the_free_one_at_once(persona_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_throttle, "_max_concurrent_background", lambda: 2)
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    assert cli_throttle.acquire_background(lull_required=False)
    finish_threads(_start(persona_dir))
    assert log == ["a"]
    cli_throttle.release_background()


def test_a_stop_ends_a_thread_waiting_for_the_slot_with_the_queue_untouched(persona_dir: Path) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    assert cli_throttle.acquire_background(lull_required=False)
    t = _start(persona_dir)
    assert _still_waiting(t)
    cli_throttle.request_shutdown()
    finish_threads(t)
    assert log == [] and pass2_queue.queue_length(persona_dir) == 1
    cli_throttle.release_background()


def test_a_stop_while_a_reply_is_in_flight_ends_the_thread_when_that_reply_ends(persona_dir: Path) -> None:
    """Replies admitted before a stop run to completion; the thread parked on that reply then sees the stop and ends."""
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    cli_throttle.note_user_message()
    t = _start(persona_dir)
    assert _still_waiting(t)
    cli_throttle.request_shutdown()
    assert _still_waiting(t), "still waiting for the admitted reply"
    cli_throttle.note_reply_end()
    finish_threads(t)
    assert log == [] and pass2_queue.queue_length(persona_dir) == 1


def test_a_refusal_from_an_item_ends_the_thread_and_the_item_stays_queued(persona_dir: Path, monkeypatch) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    _enqueue(persona_dir, log, "b")

    def refuse(_record, *, persona_dir):  # noqa: ARG001
        raise cli_throttle.ThrottleDeferred("login expired")

    monkeypatch.setattr(pass2_queue, "_dispatch", refuse)
    finish_threads(_start(persona_dir))
    assert pass2_queue.queue_length(persona_dir) == 2, "nothing was popped"
    assert cli_throttle.slot_available(lull_required=False), "the slot was let go"


def test_a_drain_lock_held_by_another_process_ends_it_without_spinning(persona_dir: Path) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    with file_lock(persona_dir / "pass2_drain"):
        finish_threads(_start(persona_dir))  # returns: drained 0 with work queued ends the thread
    assert log == [] and pass2_queue.queue_length(persona_dir) == 1


def test_an_item_queued_before_the_drain_lock_was_let_go_is_taken_by_the_same_thread(persona_dir: Path, monkeypatch) -> None:
    """The edge of L2 item 32: the drain returned having run something while the queue is not empty (an item queued just
    before the lock was let go), and no reply is in flight: the thread looks again instead of ending."""
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    real = pass2_queue.drain_all_locked
    calls: list[int] = []

    def first_call_leaves_work(pd, **kw):
        calls.append(1)
        if len(calls) == 1:
            _enqueue(pd, log, "late")  # queued after this drain's last look at the queue
            return 1
        return real(pd, **kw)

    monkeypatch.setattr(pass2_queue, "drain_all_locked", first_call_leaves_work)
    finish_threads(_start(persona_dir))
    assert len(calls) == 2 and log == ["a", "late"] and pass2_queue.queue_length(persona_dir) == 0


def test_an_unexpected_error_ends_the_thread_logged_with_the_queue_saved(persona_dir: Path, monkeypatch, caplog) -> None:
    log: list[str] = []
    _enqueue(persona_dir, log, "a")

    def boom(*_a, **_k):
        raise RuntimeError("the drain broke")

    monkeypatch.setattr(pass2_queue, "drain_all_locked", boom)
    finish_threads(_start(persona_dir))
    assert pass2_queue.queue_length(persona_dir) == 1
    assert "pass-2 thread failed" in caplog.text
    assert cli_throttle.slot_available(lull_required=False)


def test_a_failing_gate_does_not_spin(persona_dir: Path, monkeypatch) -> None:
    """acquire refused with no reply, no stop and a free slot (the gate fails closed): it ends rather than loop."""
    log: list[str] = []
    _enqueue(persona_dir, log, "a")
    monkeypatch.setattr(cli_throttle, "reply_gate_open", lambda: False)
    finish_threads(_start(persona_dir))
    assert log == [] and pass2_queue.queue_length(persona_dir) == 1
