"""Cross-platform bridge shutdown controller.

The controller avoids using OS signals as the normal control plane. The runner binds the live uvicorn.Server
instance, and API/idle/CLI paths request shutdown by setting ``server.should_exit``.

#456 (D1, D6, D19): ``begin_stop`` is THE start of every stop. It records why (never overwritten once set), sets the
bridge-wide shutdown flag and every registered stop event, marks bridge.json "stopping", and asks uvicorn to exit.
HTTP and idle stops call it directly. An OS signal only sets plain attributes in the signal handler (no lock, Event or
file I/O there: C6); the monitor thread applies them within one poll. While stopping, the monitor keeps the progress
record current and honours a hard-exit request that names this bridge; a second OS signal is a hard exit (P2).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
from collections.abc import Callable, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# Monitor poll interval: the same 0.1 s the lifespan's own waits use (server.py `_wait_for_in_flight_drain`).
POLL_S = 0.1

# Stop reasons recorded in bridge.json and the progress record (C13).
REASON_USER_STOP = "user_stop"
REASON_APP_RESTART = "app_restart"
REASON_MANUAL_RESTART = "manual_restart"
REASON_IDLE = "idle_timeout"
REASON_OS_SIGNAL = "os_signal"
REASON_HARD_EXIT = "hard_exit"
REASON_UNKNOWN = "unknown"


class _UvicornServerLike(Protocol):
    should_exit: bool


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class BridgeShutdownController:
    """Thread-safe one-shot shutdown handle (the bridge's stop, from any path)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: _UvicornServerLike | None = None
        self._requested = False
        self._reason: str | None = None
        self.requested_at: str | None = None
        self.persona_dir: Path | None = None
        self._stop_events: list[threading.Event] = []
        # Written ONLY by the signal handler (main thread), read by the monitor: plain attributes, single writer.
        self.signal_count = 0
        self.signal_reason: str | None = None
        # Set by the runner's server once uvicorn's serve() has ended (its re-raised signal is not a new signal).
        self.serve_finished = False

    @property
    def requested(self) -> bool:
        return self._requested

    @property
    def stopping(self) -> bool:
        return self._requested

    @property
    def reason(self) -> str | None:
        return self._reason

    def configure(self, persona_dir: Path) -> None:
        self.persona_dir = Path(persona_dir)

    def add_stop_event(self, event: threading.Event) -> None:
        with self._lock:
            self._stop_events.append(event)
            if self._requested:
                event.set()

    def bind_server(self, server: _UvicornServerLike) -> None:
        with self._lock:
            self._server = server
            if self._requested:
                server.should_exit = True

    def begin_stop(self, reason: str) -> bool:
        """Start the stop (idempotent). Returns True when this call started it.

        Order (D1): reason, flag, stop events, then the bridge.json "stopping" mark, then uvicorn. A later call never
        overwrites the first reason (the lifespan belt passes "unknown").
        """
        from brain.bridge import cli_throttle

        with self._lock:
            if self._requested:
                if self._server is not None:
                    self._server.should_exit = True
                return False
            self._requested = True
            self._reason = reason
            self.requested_at = _now_iso()
            cli_throttle.request_shutdown()
            for ev in self._stop_events:
                ev.set()
            logger.info("bridge shutdown requested reason=%s", reason)
            self._mark_stopping()
            if self._server is not None:
                self._server.should_exit = True
            return True

    def request(self, reason: str) -> bool:
        """The pre-#456 entry point (idle watcher, /supervisor/shutdown): ``begin_stop``. False only when no server
        is bound yet AND no persona is configured (nothing could act on the request)."""
        with self._lock:
            unbound = self._server is None and self.persona_dir is None
        if unbound:
            logger.error("shutdown requested before uvicorn server was bound reason=%s", reason)
            return False
        self.begin_stop(reason)
        return True

    def _mark_stopping(self) -> None:
        if self.persona_dir is None:
            return
        from brain.bridge import state_file

        reason, at = self._reason, self.requested_at

        def _mark(st: state_file.BridgeState) -> None:
            st.phase = "stopping"
            if not st.stop_reason:
                st.stop_reason = reason
            if not st.stop_requested_at:
                st.stop_requested_at = at

        try:
            state_file.update(self.persona_dir, _mark)
        except Exception:  # noqa: BLE001 - the monitor re-applies the mark next poll
            logger.warning("could not mark bridge.json stopping; the monitor retries", exc_info=True)
            self._mark_pending = True
        else:
            self._mark_pending = False

    _mark_pending = False

    # ---------------------------------------------------------------- signal path (C6: attribute writes only)

    def note_signal(self) -> None:
        """Called from a signal handler: plain attribute writes, nothing else."""
        self.signal_count += 1
        if self.signal_reason is None:
            self.signal_reason = REASON_OS_SIGNAL


def make_server_class(base: type) -> type:
    """A uvicorn.Server subclass whose signal handling never forces uvicorn's quick exit (D19).

    ``handle_exit`` only writes attributes (the controller's signal count and reason, ``should_exit``) and keeps
    uvicorn's re-raise bookkeeping once per signal number, so a SIGTERM still ends the process with status 0 through
    the runner. It never calls the base ``handle_exit``, so ``force_exit`` is never set and the lifespan always runs.
    SIGHUP (POSIX) starts a normal stop too, unless the process was started with SIGHUP ignored (``nohup``), which is
    kept (S5-h). The controller learns when serve() has ended, so the re-raised signal is not counted again.
    """

    class _Server(base):  # type: ignore[misc, valid-type]
        controller: BridgeShutdownController

        def handle_exit(self, sig: int, frame: object) -> None:
            if sig != getattr(signal, "SIGHUP", None) and sig not in self._captured_signals:
                self._captured_signals.append(sig)
            self.controller.note_signal()
            self.should_exit = True

        @contextlib.contextmanager
        def capture_signals(self) -> Generator[None, None, None]:
            hup = getattr(signal, "SIGHUP", None)
            hup_prev = None
            on_main = threading.current_thread() is threading.main_thread()
            if hup is not None and on_main:
                prev = signal.getsignal(hup)
                if prev is not signal.SIG_IGN:
                    hup_prev = signal.signal(hup, self.handle_exit)
            try:
                with super().capture_signals():
                    try:
                        yield
                    finally:
                        self.controller.serve_finished = True
            finally:
                if hup_prev is not None:
                    signal.signal(hup, hup_prev)

    return _Server


def early_signal_handler(controller: BridgeShutdownController, server_holder: dict) -> Callable[[int, object], None]:
    """The runner's handler for SIGTERM/SIGINT before uvicorn installs its own, and again after uvicorn restores it.

    Before serve(): plain attribute writes (the stop starts as soon as uvicorn runs, which sees ``should_exit``).
    After serve() ended (uvicorn re-raises the captured signal into this restored handler): end with status 0 through
    the runner's after-run decision.
    """

    def _handler(signum: int, frame: object) -> None:
        if controller.serve_finished:
            raise SystemExit(0)
        controller.note_signal()
        server = server_holder.get("server")
        if server is not None:
            server.should_exit = True

    return _handler


class Monitor:
    """The bridge's watcher thread (D6): applies OS signals, writes the progress record, honours a hard-exit request.

    Polls every ``POLL_S``; each poll is fault-isolated; never dies while the bridge runs. ``hard_exit`` is called on a
    second OS signal (P2) or a hard-exit request naming this bridge.
    """

    def __init__(
        self,
        controller: BridgeShutdownController,
        persona_dir: Path,
        *,
        pid: int,
        pid_token: str | None,
        hard_exit: Callable[[str], None],
        poll_s: float = POLL_S,
    ) -> None:
        self.controller = controller
        self.persona_dir = Path(persona_dir)
        self.pid = pid
        self.pid_token = pid_token
        self.hard_exit = hard_exit
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_written: list[str] | None = None
        self._applied_signals = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="bridge-stop-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5 * self.poll_s + 1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - the monitor never dies; the next poll retries
                logger.warning("bridge stop monitor poll failed", exc_info=True)
            self._stop.wait(self.poll_s)

    def poll_once(self) -> None:
        from brain.bridge import shutdown_progress

        c = self.controller
        count = c.signal_count
        if count >= 1 and not c.stopping:
            c.begin_stop(c.signal_reason or REASON_OS_SIGNAL)
        if count >= 2:
            logger.warning("second stop signal: hard exit")
            self.hard_exit(REASON_HARD_EXIT)
            return
        # D6 (b): a request naming this pid + token is the only condition, whether or not a stop has begun (review
        # M2: the kill helper writes one for a running service bridge that was never asked to stop).
        request = shutdown_progress.read_record(self.persona_dir, shutdown_progress.REQUEST_FILENAME)
        if shutdown_progress.names_this_bridge(request, self.pid, self.pid_token):
            logger.warning("hard-exit request received")
            self.hard_exit(REASON_HARD_EXIT)
            return
        if not c.stopping:
            return
        if c._mark_pending:  # noqa: SLF001 - retry a failed "stopping" mark
            c._mark_stopping()  # noqa: SLF001
        tasks = shutdown_progress.registry.snapshot()
        if tasks != self._last_written:
            record = shutdown_progress.progress_record(
                pid=self.pid, pid_token=self.pid_token, persona=self.persona_dir.name,
                requested_at=c.requested_at, reason=c.reason, tasks=tasks, updated_at=_now_iso(),
            )
            try:
                shutdown_progress.write_record(self.persona_dir, shutdown_progress.PROGRESS_FILENAME, record)
                self._last_written = tasks
            except Exception:  # noqa: BLE001 - retried next poll
                logger.warning("could not write the stop progress record; retrying", exc_info=True)


# Test hook (C14 i): called between the hard exit's first and second enumeration of descendants.
_between_enumerations: Callable[[], None] | None = None


def _kill_descendants() -> None:
    """Kill every descendant of this process (no wait: C14 e)."""
    import psutil

    try:
        children = psutil.Process().children(recursive=True)
    except Exception:  # noqa: BLE001 - the exit must happen whatever the enumeration does
        return
    for child in children:
        try:
            child.kill()
        except Exception:  # noqa: BLE001 - gone already, or not ours to kill any more
            pass


def hard_exit(persona_dir: Path | None, reason: str = REASON_HARD_EXIT) -> None:
    """End this bridge at once: exit status 0, bridge.json left not clean with reason "hard_exit" (C14). Never returns.

    Order (plan I5): the abandoning flag first (no `claude` starts after it and no caller acts on a killed call), the
    record (best-effort: a failure there still exits, C14 g), the progress record, then every descendant killed without
    waiting (C14 e), logs flushed, a second enumeration for anything that started meanwhile (S5-d), ``os._exit(0)``.
    """
    try:
        from brain.bridge import shutdown_progress

        shutdown_progress.abandoning.set()
        if persona_dir is not None:
            try:
                from brain.bridge import state_file

                def _mark(st: state_file.BridgeState) -> None:
                    st.phase = "hard_exit"
                    st.shutdown_clean = False
                    st.stop_reason = reason

                state_file.update(persona_dir, _mark)
            except BaseException:  # noqa: BLE001
                logger.warning("hard exit: bridge.json not updated", exc_info=True)
            try:
                rec = shutdown_progress.read_record(persona_dir, shutdown_progress.PROGRESS_FILENAME) or {}
                from brain import process_identity

                rec = shutdown_progress.progress_record(
                    pid=os.getpid(), pid_token=process_identity.own_token(), persona=Path(persona_dir).name,
                    requested_at=rec.get("requested_at"), reason=reason, tasks=list(rec.get("tasks") or []),
                    updated_at=_now_iso(), phase="hard_exit",
                )
                shutdown_progress.write_record(persona_dir, shutdown_progress.PROGRESS_FILENAME, rec)
            except BaseException:  # noqa: BLE001
                pass
        _kill_descendants()
        try:
            for handler in logging.getLogger().handlers:
                handler.flush()
        except BaseException:  # noqa: BLE001
            pass
        if _between_enumerations is not None:
            try:
                _between_enumerations()
            except BaseException:  # noqa: BLE001
                pass
        _kill_descendants()
    finally:
        os._exit(0)
