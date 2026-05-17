"""Bridge event bus — module-level publisher + EventBus class."""

from __future__ import annotations

import asyncio

from brain.bridge import events


def test_publish_is_noop_when_no_publisher_registered():
    """When set_publisher hasn't been called, publish() returns silently."""
    events.set_publisher(None)
    # Must not raise.
    events.publish("anything", foo="bar")


def test_publish_calls_registered_publisher_with_envelope():
    captured: list[dict] = []
    events.set_publisher(captured.append)
    try:
        events.publish("dream_complete", dream_id="d1", duration_ms=42)
    finally:
        events.set_publisher(None)
    assert len(captured) == 1
    e = captured[0]
    assert e["type"] == "dream_complete"
    assert e["dream_id"] == "d1"
    assert e["duration_ms"] == 42
    assert "at" in e and e["at"].endswith("Z")  # iso UTC


def test_publish_swallows_publisher_exception(caplog):
    import logging

    caplog.set_level(logging.WARNING)

    def boom(_event):
        raise RuntimeError("publisher crashed")

    events.set_publisher(boom)
    try:
        events.publish("anything")  # must not raise
    finally:
        events.set_publisher(None)
    assert "event publish failed" in caplog.text


def test_event_bus_publish_is_noop_when_loop_unbound():
    """EventBus.publish before bind_loop should drop silently."""
    bus = events.EventBus()
    q = bus.subscribe()
    bus.publish({"type": "x", "at": "..."})  # loop not bound
    # No event should reach the queue.
    assert q.empty()


def test_event_bus_dispatches_to_subscribers():
    asyncio.run(_dispatch_body())


async def _dispatch_body():
    bus = events.EventBus()
    bus.bind_loop(asyncio.get_running_loop())
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.publish({"type": "ping", "at": "now"})
    await asyncio.sleep(0)  # let call_soon_threadsafe run
    e1 = await asyncio.wait_for(q1.get(), timeout=1)
    e2 = await asyncio.wait_for(q2.get(), timeout=1)
    assert e1 == {"type": "ping", "at": "now"}
    assert e2 == {"type": "ping", "at": "now"}


def test_event_bus_drops_oldest_on_overflow(caplog):
    asyncio.run(_overflow_body(caplog))


async def _overflow_body(caplog):
    bus = events.EventBus()
    bus.QUEUE_MAX = 2  # shrink for the test
    bus.bind_loop(asyncio.get_running_loop())
    q = bus.subscribe()
    # Don't drain — fill past capacity.
    for i in range(5):
        bus.publish({"type": "n", "i": i, "at": "."})
    await asyncio.sleep(0.01)  # let call_soon_threadsafe land
    received = []
    while not q.empty():
        received.append(q.get_nowait())
    # Oldest dropped: we keep the last 2.
    assert [r["i"] for r in received] == [3, 4]
    assert bus._dropped_total >= 3
