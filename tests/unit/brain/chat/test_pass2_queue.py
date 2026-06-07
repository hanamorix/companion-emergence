"""Tests for brain.chat.pass2_queue — FIFO throttled pass-2 work queue.

All tests drive the synchronous `drain_pending()` entry-point; the daemon
worker thread is never spawned here.  cli_throttle state is set via the
public helpers (mark_interactive_active / reset) to control slot availability
deterministically.
"""
from __future__ import annotations

import logging

from brain.bridge import cli_throttle
from brain.chat import pass2_queue


def _make_recorder() -> tuple[list[str], callable]:
    """Return (record_list, factory) where factory(label) -> callable appends
    label to record_list when called."""
    records: list[str] = []

    def factory(label: str):
        def fn():
            records.append(label)
        return fn

    return records, factory


class TestFIFO:
    def test_items_run_in_enqueue_order(self):
        records, make = _make_recorder()
        for i in range(5):
            pass2_queue.enqueue(make(str(i)), label=f"item-{i}")

        pass2_queue.drain_pending()

        assert records == ["0", "1", "2", "3", "4"]

    def test_drain_pending_max_items(self):
        records, make = _make_recorder()
        for i in range(10):
            pass2_queue.enqueue(make(str(i)), label=f"item-{i}")

        # drain only first 3
        pass2_queue.drain_pending(max_items=3)
        assert records == ["0", "1", "2"]

        # drain the rest
        pass2_queue.drain_pending()
        assert records == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]



# Overflow: drop oldest + WARNING logged


class TestOverflow:
    def test_overflow_drops_oldest_and_warns(self, caplog):
        records, make = _make_recorder()
        cap = pass2_queue._MAX_QUEUE

        # Fill to cap
        for i in range(cap):
            pass2_queue.enqueue(make(str(i)), label=f"item-{i}")

        # Enqueue one more — should drop item-0 (oldest)
        with caplog.at_level(logging.WARNING, logger="brain.chat.pass2_queue"):
            pass2_queue.enqueue(make("extra"), label="item-extra")

        # Queue should still be exactly cap items
        assert pass2_queue._queue_size() == cap

        # WARNING was logged
        overflow_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "overflow" in r.message.lower()
        ]
        assert len(overflow_warns) >= 1

        # Drain and confirm item-0 is absent, item-extra is present
        pass2_queue.drain_pending()
        assert "0" not in records
        assert "extra" in records

    def test_overflow_preserves_newest(self, caplog):
        """After N overflows the queue should hold the N most-recent items."""
        records, make = _make_recorder()
        cap = pass2_queue._MAX_QUEUE

        with caplog.at_level(logging.WARNING, logger="brain.chat.pass2_queue"):
            for i in range(cap + 5):
                pass2_queue.enqueue(make(str(i)), label=f"item-{i}")

        pass2_queue.drain_pending()
        # The first 5 items (0-4) should have been dropped; 5..cap+4 should remain
        expected_first = str(5)
        assert records[0] == expected_first
        assert len(records) == cap


# ---------------------------------------------------------------------------
# 3. Yields to active chat (throttle slot denied → items stay queued)
# ---------------------------------------------------------------------------

class TestThrottleYield:
    def test_does_not_drain_while_chat_active(self):
        records, make = _make_recorder()
        pass2_queue.enqueue(make("x"), label="x")

        # chat just happened → the throttle denies the background slot
        cli_throttle.mark_interactive_active()
        assert pass2_queue.drain_pending() == 0
        assert records == []
        assert pass2_queue._queue_size() == 1  # item preserved, not dropped

        # once the throttle is idle again, it drains
        cli_throttle.reset()
        assert pass2_queue.drain_pending() == 1
        assert records == ["x"]

    def test_drain_releases_slot_each_item(self):
        records, make = _make_recorder()
        pass2_queue.enqueue(make("a"), label="a")
        cli_throttle.reset()
        pass2_queue.drain_pending()
        # the slot was released after the item → a fresh background acquire works
        assert cli_throttle.acquire_background() is True
        cli_throttle.release_background()


# ---------------------------------------------------------------------------
# 4. A raising item must not kill the drain (error isolation)
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    def test_raising_item_does_not_stop_the_drain(self, caplog):
        records, make = _make_recorder()

        def boom():
            raise RuntimeError("pass-2 extraction blew up")

        pass2_queue.enqueue(boom, label="boom")
        pass2_queue.enqueue(make("ok"), label="ok")
        cli_throttle.reset()

        with caplog.at_level(logging.ERROR, logger="brain.chat.pass2_queue"):
            ran = pass2_queue.drain_pending()

        assert ran == 2  # both attempted (boom ran + was caught)
        assert records == ["ok"]  # the survivor ran
        assert any("failed" in r.message for r in caplog.records)
