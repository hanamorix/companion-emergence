"""Persisted FIFO queue + single daemon worker for pass-2 extraction work.

ram-spike-fix INC-8 (S64, S76-S80): the queue used to be purely in-memory
(un-drained items were lost on restart). It is now persisted to
``<persona_dir>/pass2_queue.json`` as a list of JSON-serializable records
(``{"id", "kind", ...}``); an item is removed from that file only AFTER it
has been processed (S64 — "removed only after it is processed"). This makes
delivery **at-least-once, not exactly-once** (S76, owner-accepted): if the
process dies in the split second between an item's effects running and its
removal being persisted, that item runs again on restart. See the
``PASS2-AT-LEAST-ONCE`` comments at the pop-after-process step below and at
the three repeat-sensitive call sites (extractor.py, attunement/budget.py,
attunement/store.py) — and GitHub #240 (emotion-system untangle) for the
durable fix. No per-item idempotency mechanism is built (S76 dissolves the
round-3/round-4 red-team findings against the prior per-store-marker design
by not building it, per the owner's direct ruling).

Two DIFFERENT locks, both the existing ``brain.utils.file_lock.file_lock``
primitive applied to different sidecar paths:

* ``pass2_queue.json.lock`` (blocking) — guards every read-modify-write of
  the queue FILE itself: each ``enqueue()`` append and each drained item's
  pop-after-process. Held briefly per operation, never across a dispatched
  item's own run (which can take up to ~137s, 2-plan.md §4.1) — that would
  block a concurrent enqueue for the whole item. A drainer always re-reads
  the file fresh under this lock before removing an item BY ID, so an
  enqueue landing mid-drain from another thread or process is never lost
  (C31b).
* ``pass2_drain.lock`` (non-blocking, S77) — guards which PROCESS may be
  actively draining at all: ``drain_all_locked()`` takes it for the whole
  drain; a process that can't acquire it returns 0 immediately and leaves
  the queue completely untouched (no peek, no read) rather than racing a
  concurrent drainer.

Public surface
--------------
new_record_id() -> str
    A fresh record id (uuid4 hex) for callers building a queue record.
enqueue(record, *, persona_dir)
    Append a serializable record; persist; drop-oldest + WARN at
    ``_MAX_QUEUE``; lazily starts the single daemon worker for this
    persona (unless inhibited — tests).
drain_all_locked(persona_dir, *, should_pause=None, on_progress=None,
                 time_budget_s=None) -> int
    Take ``pass2_drain.lock`` (non-blocking); drain queued items serially,
    peek -> dispatch -> pop-and-persist, until the queue is empty or
    ``should_pause``/``time_budget_s`` says stop (checked AFTER every item,
    including the first — never before/during one). Returns the count run.
    Two callers in this codebase: the bridge's worker thread (below, INC-8;
    INC-9 makes this the central-cadence pass-2 job instead) and
    ``nell chat --no-bridge``'s exit drain (cli.py, S78/S80 — no
    ``should_pause`` at all, a bounded ``time_budget_s`` instead).
drain_pending(persona_dir, max_items=None) -> int
    Synchronous drain (test helper / on-demand): thin wrapper that calls
    ``drain_all_locked`` with a ``should_pause`` that stops after
    ``max_items`` (if given).
reset(persona_dir=None) -> None
    Test helper: stop the worker, and if ``persona_dir`` is given, empty its
    persisted queue.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from brain.bridge import cli_throttle
from brain.utils.file_lock import file_lock

log = logging.getLogger(__name__)

_MAX_QUEUE: int = 200
_POLL_SECONDS: float = 0.5

_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_worker_persona_dir: Path | None = None
_shutdown = threading.Event()
# Set True by the test conftest so enqueue() doesn't spawn the worker thread —
# tests drive drain_all_locked()/drain_pending() synchronously. Production
# leaves it False.
_worker_inhibited: bool = False

# Test-only: an in-memory side-effect registry keyed by record id, NOT
# persisted and NOT used by production code. Low-level tests that only care
# about the queue MECHANISM (FIFO order, overflow, throttle-yield, error
# isolation) can enqueue a lightweight `kind="test_probe"` record and
# register a plain callable for it here instead of building a full
# "monologue"/"attunement" record — production records are always
# JSON-serializable data, never callables, since the queue must survive a
# process restart (S64); this registry exists so unit tests can still drive
# the mechanism directly without also exercising real extraction machinery.
_test_side_effects: dict[str, Callable[[], None]] = {}


def register_test_side_effect(record_id: str, fn: Callable[[], None]) -> None:
    """Test-only: associate ``fn`` with ``record_id`` for a ``"test_probe"``
    kind record (see ``_test_side_effects`` above)."""
    _test_side_effects[record_id] = fn


def new_record_id() -> str:
    """A fresh record id for a caller building a queue record."""
    return uuid.uuid4().hex


def _queue_path(persona_dir: Path) -> Path:
    return Path(persona_dir) / "pass2_queue.json"


def _load_queue_unlocked(persona_dir: Path) -> list[dict[str, Any]]:
    """Read the queue file. Missing/corrupt -> empty list (fail toward no
    backlog rather than wedging a drainer on a bad file)."""
    try:
        data = json.loads(_queue_path(persona_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _save_queue_unlocked(persona_dir: Path, items: list[dict[str, Any]]) -> None:
    """Atomic temp+replace write (never a torn read of a partial file)."""
    path = _queue_path(persona_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(items), encoding="utf-8")
    tmp.replace(path)


def enqueue(record: dict[str, Any], *, persona_dir: Path) -> None:
    """Append a serializable pass-2 work record; persist; start the worker.

    ``record`` must be JSON-serializable and carry ``"id"`` (see
    ``new_record_id()``) and ``"kind"`` (``"monologue"`` or ``"attunement"``
    today — see ``_dispatch``). Overflow: at ``_MAX_QUEUE`` items, drop the
    OLDEST and log a WARNING (memory-safety backstop, unchanged from before
    this revision — applied to the persisted list now instead of an
    in-memory deque).
    """
    persona_dir = Path(persona_dir)
    with _lock, file_lock(_queue_path(persona_dir)):
        items = _load_queue_unlocked(persona_dir)
        if len(items) >= _MAX_QUEUE:
            dropped = items.pop(0)
            log.warning(
                "pass2_queue overflow (cap=%d); dropped oldest (id=%s, kind=%s)",
                _MAX_QUEUE,
                dropped.get("id"),
                dropped.get("kind"),
            )
        items.append(record)
        _save_queue_unlocked(persona_dir, items)
    if not _worker_inhibited:
        _ensure_worker(persona_dir)


def _peek_head_locked(persona_dir: Path) -> dict[str, Any] | None:
    """Fresh read of the queue file; return the head item or None if empty."""
    with _lock, file_lock(_queue_path(persona_dir)):
        items = _load_queue_unlocked(persona_dir)
    return items[0] if items else None


def _pop_by_id_locked(persona_dir: Path, item_id: str) -> None:
    """Remove the item with this id from a FRESH read of the queue file,
    then persist — never from a stale snapshot (C31b): a concurrent enqueue
    landing between the peek and this pop is preserved because this removal
    keys on id, not position, against whatever the file holds right now."""
    with _lock, file_lock(_queue_path(persona_dir)):
        items = _load_queue_unlocked(persona_dir)
        items = [it for it in items if it.get("id") != item_id]
        _save_queue_unlocked(persona_dir, items)


def _queue_len_locked(persona_dir: Path) -> int:
    with _lock, file_lock(_queue_path(persona_dir)):
        return len(_load_queue_unlocked(persona_dir))


def _dispatch(record: dict[str, Any], *, persona_dir: Path) -> None:
    """Map a record's "kind" to today's two run bodies — UNCHANGED bodies
    and signatures (S76): no ``item_id`` kwarg added anywhere.

    Deferred import: ``tool_loop`` imports this module at top level (it
    calls ``enqueue``), so importing ``tool_loop`` back at THIS module's
    top level would be circular. Importing here, inside the function, is
    safe because by the time any record is dispatched ``tool_loop`` has
    already finished loading.
    """
    from brain.chat import tool_loop

    kind = record.get("kind")
    if kind == "monologue":
        try:
            tool_loop.run_pass2_monologue(
                monologue_text=record["monologue_text"],
                visible_reply=record["visible_reply"],
                recent_user_msgs=tuple(record.get("recent_user_msgs", ())),
                persona_dir=persona_dir,
            )
        except Exception:  # noqa: BLE001 — an item failure must not kill the drain
            log.error(
                "pass2_queue item failed (id=%s, kind=monologue); continuing",
                record.get("id"),
                exc_info=True,
            )
    elif kind == "attunement":
        from brain.attunement.store import BufferTurn

        try:
            buffer_slice = [
                BufferTurn(id=bt["id"], content=bt["content"])
                for bt in record.get("buffer_slice", ())
            ]
            tool_loop._run_attunement_pass2(
                persona_dir,
                record["turn_id"],
                record["user_message"],
                record["reply_text"],
                buffer_slice,
            )
        except Exception:  # noqa: BLE001 — an item failure must not kill the drain
            log.error(
                "pass2_queue item failed (id=%s, kind=attunement); continuing",
                record.get("id"),
                exc_info=True,
            )
    elif kind == "test_probe":
        fn = _test_side_effects.pop(record.get("id"), None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001 — an item failure must not kill the drain
                log.error(
                    "pass2_queue item failed (id=%s, kind=test_probe); continuing",
                    record.get("id"),
                    exc_info=True,
                )
    else:
        log.error(
            "pass2_queue: unknown record kind %r (id=%s); dropping",
            kind,
            record.get("id"),
        )


def drain_all_locked(
    persona_dir: Path,
    *,
    should_pause: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    time_budget_s: float | None = None,
) -> int:
    """Drain the persisted queue for ``persona_dir``; return the count run.

    Cross-process drain exclusivity (S77): takes ``pass2_drain.lock``
    (non-blocking). If another process already holds it, returns 0
    immediately — the queue is left completely untouched, not merely
    unmodified (no peek happens at all).

    Loop shape: for each queued item, dispatch it, THEN pop-and-persist its
    removal (S64), THEN check ``on_progress``/``should_pause``/
    ``time_budget_s`` — all checked AFTER every item including the first,
    never before or during one (round-6 wording fix). A single item whose
    own duration exceeds the entire ``time_budget_s`` still runs to
    completion before the budget is ever seen as exceeded — the real
    worst-case bound is ``time_budget_s`` + one item's duration, not
    ``time_budget_s`` alone (round-7 minor).
    """
    persona_dir = Path(persona_dir)
    with file_lock(persona_dir / "pass2_drain", blocking=False) as acquired:
        if not acquired:
            return 0  # another process is already draining (S77) — leave the queue alone
        start = time.monotonic()
        done = 0
        total = _queue_len_locked(persona_dir)
        while True:
            item = _peek_head_locked(persona_dir)
            if item is None:
                break
            _dispatch(item, persona_dir=persona_dir)
            # PASS2-AT-LEAST-ONCE: if the process dies between _dispatch()
            # returning (the item's effects already applied) and this
            # removal being persisted, this item re-runs on restart (one
            # extra emotion nudge / budget count / pattern-evidence
            # increment, at most). Owner-accepted 2026-09-26: "the odds of
            # it happening are tiny, the consequence if it happens are
            # small." See #240 (emotion-system untangle) for the durable
            # fix.
            _pop_by_id_locked(persona_dir, item["id"])
            done += 1
            if on_progress is not None:
                on_progress(done, total)
            if should_pause is not None and should_pause():
                break
            if time_budget_s is not None and (time.monotonic() - start) > time_budget_s:
                break
        return done


def drain_pending(persona_dir: Path, max_items: int | None = None) -> int:
    """Synchronous drain (test helper / on-demand): drains up to
    ``max_items`` (or the whole queue) via ``drain_all_locked``."""
    if max_items is None:
        return drain_all_locked(persona_dir)
    remaining = max_items

    def _stop_after_max() -> bool:
        nonlocal remaining
        remaining -= 1
        return remaining <= 0

    return drain_all_locked(persona_dir, should_pause=_stop_after_max)


def reset(persona_dir: Path | None = None) -> None:
    """Test helper: stop the worker; if ``persona_dir`` is given, empty its
    persisted queue file too."""
    global _worker_thread, _worker_persona_dir
    _shutdown.set()
    thread = _worker_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=2.0)
    with _lock:
        _worker_thread = None
        _worker_persona_dir = None
    _shutdown.clear()
    _test_side_effects.clear()
    if persona_dir is not None:
        persona_dir = Path(persona_dir)
        with _lock, file_lock(_queue_path(persona_dir)):
            _save_queue_unlocked(persona_dir, [])


def _queue_size(persona_dir: Path) -> int:
    """Test helper: current number of items in the persisted queue."""
    return _queue_len_locked(Path(persona_dir))


def _ensure_worker(persona_dir: Path) -> None:
    """Lazily start the single daemon worker thread for this persona if not
    already running."""
    global _worker_thread, _worker_persona_dir
    with _lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return
        _worker_persona_dir = Path(persona_dir)
        _worker_thread = threading.Thread(
            target=_worker_loop, args=(_worker_persona_dir,), name="pass2-queue-worker", daemon=True
        )
        _worker_thread.start()


def _worker_loop(persona_dir: Path) -> None:
    """Drain forever via drain_all_locked; sleep when there's nothing to do,
    the throttle denies a slot, or another process/thread already holds
    ``pass2_drain.lock``.

    The pre-flight ``cli_throttle.acquire_background()`` gate below is the
    "outer per-call throttle gate" of 2-plan.md §3.5a — it decides WHETHER
    to start draining at all (today's existing S32-table behaviour: pass 2
    yields to interactive chat and respects the concurrency cap). The
    ``should_pause`` passed into ``drain_all_locked`` is the SEPARATE,
    per-item pause check (S14): once draining has started, a returning user
    stops it at the next item boundary rather than only at the next
    worker-loop tick.
    """
    while not _shutdown.is_set():
        if not cli_throttle.acquire_background():
            time.sleep(_POLL_SECONDS)
            continue
        try:
            drained = drain_all_locked(
                persona_dir,
                should_pause=lambda: not cli_throttle.is_chat_idle(),
            )
        finally:
            cli_throttle.release_background()
        if drained == 0:
            time.sleep(_POLL_SECONDS)
