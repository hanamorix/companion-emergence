"""Tests for brain.chat.pass2_queue — persisted, throttled pass-2 work queue.

ram-spike-fix INC-8: the queue is now backed by `<persona_dir>/pass2_queue.json`
(S64) instead of a pure in-memory deque. These tests drive the synchronous
`drain_pending()`/`drain_all_locked()` entry-points against a `tmp_path`
persona_dir; the daemon worker thread is never spawned here (conftest
inhibits it). Mechanism-level tests (FIFO order, overflow, throttle-yield,
error isolation) use the test-only `kind="test_probe"` record
(`pass2_queue.register_test_side_effect`) instead of a real "monologue"/
"attunement" record — those two kinds are covered by
test_tool_loop_pass2_spawn.py / test_attunement_pass2_spawn.py and the
INC-8-specific persistence/lock/at-least-once tests below.
"""
from __future__ import annotations

import logging
import time

from brain.bridge import cli_throttle
from brain.chat import pass2_queue


def _make_recorder() -> tuple[list[str], callable]:
    """Return (record_list, factory) where factory(label) -> a test_probe
    record dict whose registered side effect appends label to record_list
    when dispatched."""
    records: list[str] = []

    def factory(label: str) -> dict:
        record_id = pass2_queue.new_record_id()

        def fn() -> None:
            records.append(label)

        pass2_queue.register_test_side_effect(record_id, fn)
        return {"id": record_id, "kind": "test_probe", "label": label}

    return records, factory


class TestFIFO:
    def test_items_run_in_enqueue_order(self, tmp_path):
        records, make = _make_recorder()
        for i in range(5):
            pass2_queue.enqueue(make(str(i)), persona_dir=tmp_path)

        pass2_queue.drain_pending(tmp_path)

        assert records == ["0", "1", "2", "3", "4"]

    def test_drain_pending_max_items(self, tmp_path):
        records, make = _make_recorder()
        for i in range(10):
            pass2_queue.enqueue(make(str(i)), persona_dir=tmp_path)

        # drain only first 3
        pass2_queue.drain_pending(tmp_path, max_items=3)
        assert records == ["0", "1", "2"]

        # drain the rest
        pass2_queue.drain_pending(tmp_path)
        assert records == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# ---------------------------------------------------------------------------
# 2. Overflow: drop oldest + WARNING logged
# ---------------------------------------------------------------------------


class TestOverflow:
    def test_overflow_drops_oldest_and_warns(self, tmp_path, caplog):
        records, make = _make_recorder()
        cap = pass2_queue._MAX_QUEUE

        # Fill to cap
        for i in range(cap):
            pass2_queue.enqueue(make(str(i)), persona_dir=tmp_path)

        # Enqueue one more — should drop item-0 (oldest)
        with caplog.at_level(logging.WARNING, logger="brain.chat.pass2_queue"):
            pass2_queue.enqueue(make("extra"), persona_dir=tmp_path)

        # Queue should still be exactly cap items
        assert pass2_queue._queue_size(tmp_path) == cap

        # WARNING was logged
        overflow_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "overflow" in r.message.lower()
        ]
        assert len(overflow_warns) >= 1

        # Drain and confirm item-0 is absent, item-extra is present
        pass2_queue.drain_pending(tmp_path)
        assert "0" not in records
        assert "extra" in records

    def test_overflow_preserves_newest(self, tmp_path, caplog):
        """After N overflows the queue should hold the N most-recent items."""
        records, make = _make_recorder()
        cap = pass2_queue._MAX_QUEUE

        with caplog.at_level(logging.WARNING, logger="brain.chat.pass2_queue"):
            for i in range(cap + 5):
                pass2_queue.enqueue(make(str(i)), persona_dir=tmp_path)

        pass2_queue.drain_pending(tmp_path)
        # The first 5 items (0-4) should have been dropped; 5..cap+4 should remain
        expected_first = str(5)
        assert records[0] == expected_first
        assert len(records) == cap


# ---------------------------------------------------------------------------
# 3. Yields to active chat (throttle slot denied → items stay queued)
# ---------------------------------------------------------------------------

class TestThrottleYield:
    def test_worker_style_pre_flight_gate_does_not_drain_while_chat_active(self, tmp_path):
        """The OUTER per-call throttle gate (2-plan.md §3.5a) is the
        CALLER's job, not drain_all_locked's own. This test calls the same
        `cli_throttle.acquire_background()` / `drain_all_locked` primitives
        `pass2_queue._worker_loop` composes, directly, so the assertions
        below are legible on their own; it does NOT start the real worker
        thread — see `TestWorkerLoop` below for an end-to-end test that
        does, starting the actual daemon thread via `_ensure_worker`."""
        records, make = _make_recorder()
        pass2_queue.enqueue(make("x"), persona_dir=tmp_path)

        # chat just happened → the pre-flight gate denies; drain_all_locked
        # is never even called.
        cli_throttle.mark_interactive_active()
        assert cli_throttle.acquire_background() is False
        assert records == []
        assert pass2_queue._queue_size(tmp_path) == 1  # item preserved, not dropped

        # once the throttle is idle again, the pre-flight gate grants a slot
        # and the drain runs (should_pause then governs mid-drain yielding,
        # tested separately below).
        cli_throttle.reset()
        assert cli_throttle.acquire_background() is True
        try:
            assert pass2_queue.drain_all_locked(
                tmp_path, should_pause=lambda: not cli_throttle.is_chat_idle()
            ) == 1
        finally:
            cli_throttle.release_background()
        assert records == ["x"]

    def test_should_pause_stops_mid_drain_once_true_after_an_item(self, tmp_path):
        """The INNER, per-item pause check (S14): once draining has
        started, it stops at the next item boundary — never before or
        during the item it's already running."""
        records, make = _make_recorder()
        pass2_queue.enqueue(make("a"), persona_dir=tmp_path)
        pass2_queue.enqueue(make("b"), persona_dir=tmp_path)

        # Checked AFTER every item, including the first — so returning True
        # unconditionally stops the loop right after item "a" runs, never
        # before it and never mid-item.
        done = pass2_queue.drain_all_locked(tmp_path, should_pause=lambda: True)
        assert done == 1  # stopped after item "a"; "b" stays queued
        assert records == ["a"]
        assert pass2_queue._queue_size(tmp_path) == 1


class TestWorkerLoop:
    """End-to-end: the REAL daemon worker thread (`_ensure_worker`/
    `_worker_loop`), started for real, composing `cli_throttle.
    acquire_background()` + `drain_all_locked` + `release_background()` —
    not a hand-simulated stand-in. The autouse `_reset_pass2_queue` fixture
    (tests/conftest.py) sets `_worker_inhibited = True` for every other
    test in the suite specifically so enqueue() never starts this thread;
    this test opts back in deliberately and restores the inhibition after."""

    def _poll_until(self, predicate, *, timeout: float = 3.0, interval: float = 0.02) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return predicate()

    def test_worker_thread_drains_once_idle_but_not_while_chat_active(self, tmp_path):
        records, make = _make_recorder()
        pass2_queue._worker_inhibited = False
        try:
            cli_throttle.mark_interactive_active()  # not idle
            pass2_queue.enqueue(make("x"), persona_dir=tmp_path)  # starts the real worker thread

            # Give the real worker thread several poll cycles; it must NOT
            # drain while cli_throttle denies the background slot.
            # Several multiples of the worker's own _POLL_SECONDS (0.5s), so
            # this actually observes more than one denied poll cycle rather
            # than just the worker's very first attempt (re-review nitpick).
            drained_too_early = self._poll_until(
                lambda: pass2_queue._queue_size(tmp_path) == 0,
                timeout=pass2_queue._POLL_SECONDS * 2.5,
            )
            assert not drained_too_early, "worker thread drained while chat was active"
            assert records == []

            # Once idle, the same running thread must drain it without any
            # further enqueue.
            cli_throttle.reset()
            drained = self._poll_until(lambda: pass2_queue._queue_size(tmp_path) == 0, timeout=3.0)
            assert drained, "worker thread never drained the item once idle"
            assert records == ["x"]
        finally:
            pass2_queue._worker_inhibited = True
            pass2_queue.reset()


# ---------------------------------------------------------------------------
# 4. A raising item must not kill the drain (error isolation)
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_raising_item_does_not_stop_the_drain(self, tmp_path, caplog):
        records, make = _make_recorder()

        boom_id = pass2_queue.new_record_id()

        def boom():
            raise RuntimeError("pass-2 extraction blew up")

        pass2_queue.register_test_side_effect(boom_id, boom)
        pass2_queue.enqueue({"id": boom_id, "kind": "test_probe", "label": "boom"}, persona_dir=tmp_path)
        pass2_queue.enqueue(make("ok"), persona_dir=tmp_path)
        cli_throttle.reset()

        with caplog.at_level(logging.ERROR, logger="brain.chat.pass2_queue"):
            ran = pass2_queue.drain_pending(tmp_path)

        assert ran == 2  # both attempted (boom ran + was caught)
        assert records == ["ok"]  # the survivor ran
        assert any("failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 5. Persistence (S64): the queue survives across "restarts" (fresh reads)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_enqueued_record_round_trips_through_the_file(self, tmp_path):
        record = {"id": pass2_queue.new_record_id(), "kind": "monologue",
                  "monologue_text": "m", "visible_reply": "r", "recent_user_msgs": ["u"]}
        pass2_queue.enqueue(record, persona_dir=tmp_path)

        path = tmp_path / "pass2_queue.json"
        assert path.exists()
        import json

        on_disk = json.loads(path.read_text())
        assert on_disk == [record]

    def test_item_removed_only_after_processing(self, tmp_path):
        """S64: peek shows the item present until AFTER it's dispatched."""
        seen_present_during_run = {}
        record_id = pass2_queue.new_record_id()

        def fn():
            import json

            on_disk = json.loads((tmp_path / "pass2_queue.json").read_text())
            seen_present_during_run["present"] = any(it["id"] == record_id for it in on_disk)

        pass2_queue.register_test_side_effect(record_id, fn)
        pass2_queue.enqueue({"id": record_id, "kind": "test_probe"}, persona_dir=tmp_path)

        pass2_queue.drain_pending(tmp_path)

        assert seen_present_during_run["present"] is True  # still queued mid-run
        assert pass2_queue._queue_size(tmp_path) == 0  # removed only after it ran

    def test_a_fresh_read_of_the_file_shows_the_full_backlog(self, tmp_path):
        """Proves persistence directly against the file (not this process's
        transient `_test_side_effects` registry, which — like any in-memory
        state — genuinely would NOT survive a real restart): enqueue two
        items, then read the queue back via a fresh, independent parse of
        `pass2_queue.json`, exactly as a restarted process would. Full
        crash/restart/resume behaviour (kill -9, mid-item repeat) is
        covered end-to-end via real subprocesses in
        test_pass2_queue_cross_process.py."""
        record_a = {"id": pass2_queue.new_record_id(), "kind": "monologue",
                    "monologue_text": "a", "visible_reply": "ra", "recent_user_msgs": []}
        record_b = {"id": pass2_queue.new_record_id(), "kind": "attunement",
                    "turn_id": "t", "user_message": "u", "reply_text": "r", "buffer_slice": []}
        pass2_queue.enqueue(record_a, persona_dir=tmp_path)
        pass2_queue.enqueue(record_b, persona_dir=tmp_path)

        import json

        on_disk = json.loads((tmp_path / "pass2_queue.json").read_text())
        assert on_disk == [record_a, record_b]
        assert pass2_queue._queue_size(tmp_path) == 2


# ---------------------------------------------------------------------------
# 6. Queue-file lock (C31b): a concurrent enqueue survives an in-progress
#    removal-by-id, because the removal re-reads the file fresh.
# ---------------------------------------------------------------------------


class TestQueueFileLockFreshRead:
    def test_enqueue_during_drain_of_a_different_item_is_preserved(self, tmp_path, monkeypatch):
        """Item A's side effect enqueues item B (from "another thread", here
        simulated in-line since the mechanism under test is the fresh-read
        pop, not thread scheduling) WHILE item A is being processed — B must
        survive A's pop-by-id."""
        a_id = pass2_queue.new_record_id()
        b_id = pass2_queue.new_record_id()
        ran: list[str] = []

        def fn_a():
            ran.append("a")
            # Enqueue B mid-run, before A's own pop-by-id happens.
            pass2_queue.register_test_side_effect(b_id, lambda: ran.append("b"))
            pass2_queue.enqueue({"id": b_id, "kind": "test_probe"}, persona_dir=tmp_path)

        pass2_queue.register_test_side_effect(a_id, fn_a)
        pass2_queue.enqueue({"id": a_id, "kind": "test_probe"}, persona_dir=tmp_path)

        pass2_queue.drain_pending(tmp_path)  # drains A (which enqueues B), then B
        assert ran == ["a", "b"]
        assert pass2_queue._queue_size(tmp_path) == 0

    def test_a_real_second_thread_enqueues_mid_drain_and_survives(self, tmp_path):
        """C31(b), genuinely concurrent this time (a real `threading.Thread`,
        not an in-line call from within item A's own dispatch): item A's
        side effect blocks until a second, real OS thread has enqueued item
        B and confirmed the file write completed — then A's own pop-by-id
        runs. B must survive, proving the removal really does re-read the
        file fresh rather than working off a snapshot taken before B's
        thread wrote it."""
        import threading

        a_id = pass2_queue.new_record_id()
        b_id = pass2_queue.new_record_id()
        b_enqueued = threading.Event()
        ran: list[str] = []

        def enqueue_b_from_another_thread():
            pass2_queue.register_test_side_effect(b_id, lambda: ran.append("b"))
            pass2_queue.enqueue({"id": b_id, "kind": "test_probe"}, persona_dir=tmp_path)
            b_enqueued.set()

        def fn_a():
            ran.append("a")
            t = threading.Thread(target=enqueue_b_from_another_thread)
            t.start()
            assert b_enqueued.wait(timeout=5.0), "second thread never finished enqueuing B"
            t.join(timeout=5.0)

        pass2_queue.register_test_side_effect(a_id, fn_a)
        pass2_queue.enqueue({"id": a_id, "kind": "test_probe"}, persona_dir=tmp_path)

        pass2_queue.drain_pending(tmp_path)  # drains A (spawns the real thread for B), then B
        assert ran == ["a", "b"]
        assert pass2_queue._queue_size(tmp_path) == 0
