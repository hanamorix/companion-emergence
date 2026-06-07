"""Process-global FIFO queue + single daemon worker for pass-2 extraction work.

Per-turn pass-2 extractions (monologue, attunement) are enqueued here instead of
spawning a thread per turn. A single worker drains them serially, acquiring the
``cli_throttle`` slot before each item — so interactive chat always has priority
and the worker serializes with the cadence engines — and **never drops** work
(except an overflow backstop). During active chat the worker waits and the queue
accumulates; when chat goes quiet it drains the backlog (consolidate during rest).

In-memory by design: un-drained items are lost on restart. Accepted — the raw
monologue is persisted synchronously to ``monologue_digest.jsonl`` at record time;
only the async EXTRACTION of not-yet-drained items is lost, a rare low-impact edge.

Public surface
--------------
enqueue(fn, *, label)
    Append a zero-arg callable. Overflow: at ``_MAX_QUEUE`` items, drop the OLDEST
    and log a WARNING (memory-safety backstop; debounce limits normal inflow).
    Lazily starts the single daemon worker, unless inhibited (tests).
drain_pending(max_items=None) -> int
    Synchronous drain (test helper): run items via the same ``_drain_one`` the
    worker uses, until empty / max reached / the throttle denies a slot. Returns
    the count run.
reset() -> None
    Test helper: stop the worker, empty the queue, clear state.
_queue_size() -> int
    Test helper: current queue length.

The drain logic lives in ONE place (``_drain_one``); both ``drain_pending`` and
the worker loop call it, so the per-item behaviour (throttle, pop, run, error
isolation, release) is identical and fully covered by the synchronous tests.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

from brain.bridge import cli_throttle

log = logging.getLogger(__name__)

_MAX_QUEUE: int = 200
_POLL_SECONDS: float = 0.5

_lock = threading.Lock()
_queue: deque[tuple[Callable[[], None], str]] = deque()  # (fn, label)
_worker_thread: threading.Thread | None = None
_shutdown = threading.Event()
# Set True by the test conftest so enqueue() doesn't spawn the worker thread —
# tests drive drain_pending() synchronously. Production leaves it False.
_worker_inhibited: bool = False


def enqueue(fn: Callable[[], None], *, label: str) -> None:
    """Append a work item; drop the oldest + WARN on overflow; start the worker."""
    with _lock:
        if len(_queue) >= _MAX_QUEUE:
            _, dropped_label = _queue.popleft()  # drop oldest
            log.warning(
                "pass2_queue overflow (cap=%d); dropped oldest (label=%s)",
                _MAX_QUEUE,
                dropped_label,
            )
        _queue.append((fn, label))
    if not _worker_inhibited:
        _ensure_worker()


def _drain_one() -> bool:
    """Run a single queued item IF the throttle grants a slot.

    Returns True if an item ran, False if the queue is empty OR the slot was
    denied (chat recently active / concurrency cap full — items stay queued).
    The single shared drain primitive: the worker loop and drain_pending both
    call this, so per-item behaviour is identical and test-covered.
    """
    with _lock:
        if not _queue:
            return False
    if not cli_throttle.acquire_background():
        return False  # yield to chat / cap — leave the item queued for later
    try:
        with _lock:
            if not _queue:  # reset() raced between the check and the acquire
                return False
            fn, label = _queue.popleft()
        try:
            fn()
        except Exception:  # noqa: BLE001 — an item failure must not kill the worker
            log.error("pass2_queue item failed (label=%s); continuing", label, exc_info=True)
        return True
    finally:
        cli_throttle.release_background()


def drain_pending(max_items: int | None = None) -> int:
    """Synchronous drain (test helper / on-demand): returns the count run."""
    executed = 0
    while max_items is None or executed < max_items:
        if not _drain_one():
            break
        executed += 1
    return executed


def reset() -> None:
    """Test helper: empty the queue and stop / clear the worker thread."""
    global _worker_thread
    _shutdown.set()
    with _lock:
        _queue.clear()
    thread = _worker_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    with _lock:
        _worker_thread = None
    _shutdown.clear()


def _queue_size() -> int:
    """Test helper: current number of items in the queue."""
    with _lock:
        return len(_queue)


def _ensure_worker() -> None:
    """Lazily start the single daemon worker thread if not already running."""
    global _worker_thread
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_thread = threading.Thread(
            target=_worker_loop, name="pass2-queue-worker", daemon=True
        )
        _worker_thread.start()


def _worker_loop() -> None:
    """Drain forever via _drain_one; sleep when there's nothing to do / no slot."""
    while not _shutdown.is_set():
        if not _drain_one():
            time.sleep(_POLL_SECONDS)
