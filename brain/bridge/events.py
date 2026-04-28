"""Bridge event bus.

Two layers:

  1. Module-level `set_publisher(fn)` / `publish(type, **payload)` — engines
     call publish; it's a free no-op when no publisher is registered (CLI mode).
  2. EventBus class — bridge instantiates one at startup, calls bind_loop in
     lifespan, then sets it as the module-level publisher. Thread-safe; drops
     oldest event on per-subscriber queue overflow.

OG reference: NellBrain/nell_bridge.py:317-369 (EventBroadcaster).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level publisher contract
# ---------------------------------------------------------------------------

_publisher: Callable[[dict[str, Any]], None] | None = None


def set_publisher(fn: Callable[[dict[str, Any]], None] | None) -> None:
    """Bridge calls this on lifespan startup; sets None on teardown."""
    global _publisher
    _publisher = fn


def publish(event_type: str, **payload: Any) -> None:
    """Engines call this. Free no-op when no publisher is registered.

    Drop semantics: if the publisher raises, the exception is caught and
    logged at WARN. Engines never fail because a publish failed.
    """
    if _publisher is None:
        return
    event = {"type": event_type, "at": _now_iso(), **payload}
    try:
        _publisher(event)
    except Exception:
        logger.warning("event publish failed for type=%s", event_type, exc_info=True)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# EventBus — bridge-side fan-out with thread-safety + drop-on-overflow
# ---------------------------------------------------------------------------


class EventBus:
    """In-process pub/sub for the bridge daemon.

    Subscribers are asyncio.Queues. Publishers may run in any thread (the
    supervisor runs in a non-daemon thread); publish() uses
    call_soon_threadsafe to dispatch onto the bridge's main event loop.

    Per-subscriber queue is bounded at QUEUE_MAX; overflow drops the OLDEST
    event so live clients keep receiving fresh data instead of stale.
    """

    QUEUE_MAX = 64

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dropped_total = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def publish(self, event: dict) -> None:
        """Dispatch event to every subscriber. Thread-safe.

        Drops silently if loop is not yet bound (shouldn't happen in normal
        bridge lifecycle but is a guard for engine publishes during very
        early startup or very late shutdown).
        """
        if self._loop is None:
            return
        for q in list(self._subscribers):
            self._loop.call_soon_threadsafe(self._enqueue, q, event)

    def _enqueue(self, q: asyncio.Queue, event: dict) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(event)
                self._dropped_total += 1
                if self._dropped_total % 10 == 1:
                    logger.warning(
                        "event queue overflow, dropped=%d", self._dropped_total
                    )
            except asyncio.QueueEmpty:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.QUEUE_MAX)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def subscriber_count(self) -> int:
        return len(self._subscribers)
