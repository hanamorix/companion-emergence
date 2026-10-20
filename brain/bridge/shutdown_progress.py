"""What is still running while the bridge stops, and the files a watcher reads (#456, D1, D6).

``TaskRegistry`` counts every piece of work that must finish before the bridge may say "stopped": the supervisor
thread, every request worker thread, the backlog migration, floor computations and probe threads. A stop waits until
the registry is empty and then seals it in one step, so nothing can register after the last check (C41).

The watcher files live next to ``bridge.json`` and are written and read only through ``json_io`` (atomic replace,
owner-only permissions):

- the progress record: written by the bridge's monitor while it stops (pid + token, request time, reason, tasks);
- the hard-exit request: written by the CLI or the app, honoured by the bridge only when its pid + token name it;
- the stop intent: written only by the kill helper (``brain/bridge/kill.py``).

Every reader checks the record's pid + token against the bridge it is asking about; a file left by an earlier process
is ignored, and the lock holder deletes all three at bridge start.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import Any

from brain.bridge import cli_throttle
from brain.health.json_io import Recovery, read_json, write_json

logger = logging.getLogger(__name__)

PROGRESS_FILENAME = "bridge_stop_progress.json"
REQUEST_FILENAME = "bridge_hard_exit_request.json"
INTENT_FILENAME = "bridge_stop_intent.json"
WATCHER_FILENAMES = (PROGRESS_FILENAME, REQUEST_FILENAME, INTENT_FILENAME)


class TaskRegistry:
    """Thread-safe count of running work, with a one-way seal.

    ``register`` refuses with ``ShutdownDeferred`` once sealed, and, for a background spawner (``spawner=True``), once
    the shutdown flag is set: a stopping bridge starts no new background work. Request workers register with
    ``spawner=False``: a turn admitted before the stop still runs (D21), but nothing registers after the seal.
    ``seal()`` checks "empty" and marks sealed in ONE critical section.

    ``_between`` is a test hook called inside the critical section between reading the next id and writing the entry
    (C19: an injected interleaving must not lose a registration).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[int, str] = {}
        self._next = 0
        self._sealed = False
        self._between: Callable[[], None] | None = None

    def register(self, label: str, *, spawner: bool = False) -> int:
        with self._lock:
            if self._sealed:
                raise cli_throttle.ShutdownDeferred(f"bridge stopped; {label} not started")
            if spawner and cli_throttle.shutdown_requested():
                raise cli_throttle.ShutdownDeferred(f"bridge is stopping; {label} not started")
            handle = self._next
            if self._between is not None:
                self._between()
            self._next = handle + 1
            self._tasks[handle] = label
            return handle

    def unregister(self, handle: int) -> None:
        with self._lock:
            self._tasks.pop(handle, None)

    @contextmanager
    def track(self, label: str, *, spawner: bool = False) -> Iterator[int]:
        handle = self.register(label, spawner=spawner)
        try:
            yield handle
        finally:
            self.unregister(handle)

    def set_activity(self, handle: int, label: str) -> None:
        with self._lock:
            if handle in self._tasks:
                self._tasks[handle] = label

    def snapshot(self) -> list[str]:
        with self._lock:
            return [self._tasks[h] for h in sorted(self._tasks)]

    def count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def seal(self) -> bool:
        """True (and sealed) when nothing is running; False (unchanged) otherwise."""
        with self._lock:
            if self._tasks:
                return False
            self._sealed = True
            return True

    @property
    def sealed(self) -> bool:
        with self._lock:
            return self._sealed


# The process-wide registry (one bridge per process).
registry = TaskRegistry()

# #456 (D17): set ONLY by the hard-exit path. Every `claude` spawn site checks it before spawning and after the
# subprocess returns, and then blocks until the process exits: no new `claude` starts and no caller sees a killed
# call as an answer (C14 f, C52).
abandoning = threading.Event()


def abandon_gate() -> None:
    """Block forever once the hard exit has begun (the process is about to end with ``os._exit``)."""
    if abandoning.is_set():
        while True:
            time.sleep(1)


def reset_for_tests() -> TaskRegistry:
    """A new bridge app in this process (tests): a fresh registry, not abandoning."""
    global registry
    registry = TaskRegistry()
    abandoning.clear()
    return registry


async def tracked_to_thread[T](
    label: str, fn: Callable[..., T], /, *args: Any, spawner: bool = False, **kwargs: Any
) -> T:
    """``asyncio.to_thread`` whose work is a counted task for its WHOLE run (C40).

    The task is registered on the event loop before the thread starts and unregistered by the thread itself when the
    work ends, so a request task cancelled by uvicorn during a stop does not uncount work that is still running, and a
    worker that never started cannot hold the stop forever. Registration refuses after the seal (and for a background
    spawner after the flag), raising ``ShutdownDeferred`` before any work starts.
    """
    reg = registry
    handle = reg.register(label, spawner=spawner)
    loop = asyncio.get_running_loop()
    fut: asyncio.Future[T] = loop.create_future()
    ctx = contextvars.copy_context()

    def _deliver(result: Any, exc: BaseException | None) -> None:
        if fut.done():
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    def _run() -> None:
        result: Any = None
        error: BaseException | None = None
        try:
            result = ctx.run(fn, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - delivered to the awaiting request
            error = exc
        finally:
            reg.unregister(handle)
        try:
            loop.call_soon_threadsafe(_deliver, result, error)
        except RuntimeError:  # the loop is closed: nobody is waiting any more
            pass

    try:
        threading.Thread(target=_run, name=f"bridge-task:{label}", daemon=False).start()
    except BaseException:
        reg.unregister(handle)
        raise
    return await fut


def start_tracked_thread(
    label: str, target: Callable[[], None], *, name: str, daemon: bool = True
) -> threading.Thread | None:
    """Start a background thread that is a counted task while it runs (backlog migration, floor computations).

    Returns None, starting nothing, when the stop has begun (C9: nothing new after the flag).
    """
    reg = registry
    try:
        handle = reg.register(label, spawner=True)
    except cli_throttle.ShutdownDeferred:
        logger.info("%s not started: the bridge is stopping", label)
        return None

    def _run() -> None:
        try:
            target()
        finally:
            reg.unregister(handle)

    t = threading.Thread(target=_run, name=name, daemon=daemon)
    try:
        t.start()
    except BaseException:
        reg.unregister(handle)
        raise
    return t


# ---------------------------------------------------------------- watcher files


def path(persona_dir: Path, filename: str) -> Path:
    base = persona_dir if isinstance(persona_dir, PurePath) else Path(persona_dir)
    return base / filename


def write_record(persona_dir: Path, filename: str, record: dict[str, Any]) -> None:
    write_json(path(persona_dir, filename), record, indent=2, sort_keys=True, fsync=False, mode=0o600)


def read_record(persona_dir: Path, filename: str) -> dict[str, Any] | None:
    """The record, or None when missing or unreadable as a JSON object (a reader never trusts a broken file)."""
    try:
        return read_json(path(persona_dir, filename), default=lambda: None, recover=Recovery.CONTENT, expect=dict)
    except Exception:  # noqa: BLE001 - an unreadable watcher file is "no record"; the writer retries next poll
        return None


def delete_record(persona_dir: Path, filename: str) -> bool:
    try:
        path(persona_dir, filename).unlink(missing_ok=True)
        return True
    except OSError:
        logger.warning("could not delete %s", filename, exc_info=True)
        return False


def delete_watcher_files(persona_dir: Path) -> None:
    """At bridge start, by the lock holder only (C55): files left by an earlier process are not this bridge's."""
    for name in WATCHER_FILENAMES:
        delete_record(persona_dir, name)


# ---- record shapes (C48: one golden fixture set, tests/fixtures/bridge_watcher_files/; Rust / TS readers parse it) ----

FORMAT_VERSION = 1


def progress_record(
    *, pid: int, pid_token: str | None, persona: str, requested_at: str | None, reason: str | None,
    tasks: list[str], updated_at: str, phase: str = "stopping",
) -> dict[str, Any]:
    """The progress record the monitor keeps while the bridge stops (and the hard exit marks)."""
    return {
        "version": FORMAT_VERSION,
        "pid": pid,
        "pid_token": pid_token,
        "persona": persona,
        "phase": phase,
        "requested_at": requested_at,
        "reason": reason,
        "tasks": list(tasks),
        "count": len(tasks),
        "updated_at": updated_at,
    }


def hard_exit_request(*, pid: int, pid_token: str | None, requested_at: str) -> dict[str, Any]:
    """The hard-exit request (CLI / app -> bridge); honoured only when pid + token are the bridge's own."""
    return {"version": FORMAT_VERSION, "pid": pid, "pid_token": pid_token, "requested_at": requested_at}


def stop_intent(
    *, pid: int, pid_token: str | None, state: str, nonce: str, at: str, boot: str | None, legacy: bool,
) -> dict[str, Any]:
    """The stop intent (kill helper only): ``state`` "pending" | "committed"; ``boot`` the Linux boot id or None."""
    return {
        "version": FORMAT_VERSION,
        "pid": pid,
        "pid_token": pid_token,
        "reason": "user_kill",
        "state": state,
        "nonce": nonce,
        "at": at,
        "boot": boot,
        "legacy": legacy,
    }


def names_this_bridge(record: dict[str, Any] | None, pid: int | None, token: str | None) -> bool:
    """True when ``record`` names exactly the bridge ``pid`` + ``token`` (C43 a: anything else is ignored)."""
    if not isinstance(record, dict) or pid is None or not token:
        return False
    return record.get("pid") == pid and record.get("pid_token") == token
