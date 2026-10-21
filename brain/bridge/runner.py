"""SP-7 bridge runner — entrypoint for the spawned daemon process.

`python -m brain.bridge.runner --persona-dir <path> --client-origin <c> [--idle-shutdown-seconds N]`

Writes bridge.json with pid + chosen port, then runs uvicorn until SIGTERM.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from brain.bridge import respawn_guard, start_backoff, state_file
from brain.bridge.server import build_app
from brain.bridge.shutdown import BridgeShutdownController

logger = logging.getLogger(__name__)


def _allocate_port(max_attempts: int = 3) -> int:
    """Bind ephemeral, read assigned port, close. Retry with backoff if
    the *initial* bind itself fails.

    NOTE: this only retries the port-discovery bind here. The actual
    server bind happens later inside ``uvicorn.run(..., port=port)`` and
    is NOT wrapped in this retry loop — the bind→close→rebind race
    between our discovery and uvicorn's bind isn't actively defended,
    only narrowed (sub-millisecond window). The supervisor's startup
    handshake (``cmd_start`` waits for /health 200 and kills orphan
    children on timeout) is what closes the loop if the race does fire.

    Backoff schedule on failed discovery: 1ms, then 10ms between attempts
    (nothing sleeps after the last one, #481), so a failure surfaces at once.
    """
    import time

    last_exc: OSError | None = None
    for attempt in range(max_attempts):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:  # no sleep after the last attempt: nothing follows it (#481)
                time.sleep((10**attempt) / 1000.0)  # 0.001, 0.01, 0.1, ...
    raise RuntimeError(
        f"_allocate_port: failed to bind a free port after {max_attempts} attempts; "
        f"last error: {last_exc}",
    )


def _write_stopped(persona_dir: Path) -> None:
    """Record the honest end of a normal stop (#456 C11): "stopped", clean, pid/port cleared. Only this bridge's own
    record is written (``state_file.update``); a failure is logged, never raised."""

    def _mark(st: state_file.BridgeState) -> None:
        st.phase = "stopped"
        st.shutdown_clean = True
        st.stopped_at = datetime.now(UTC).isoformat()
        st.pid = None
        st.port = None
        st.pid_token = None

    try:
        if not state_file.update(persona_dir, _mark):
            logger.warning("bridge.json is not this bridge's record; the stopped mark was not written")
    except Exception:
        logger.warning("stopped-mark write failed", exc_info=True)
    start_backoff.reset(persona_dir)  # D26 / D7: a recorded stop is a success: the next start does not wait


def _after_run(persona_dir: Path, app: object, monitor: object) -> None:
    """The after-run decision (#456, plan I4): runs whatever way ``server.run()`` ended (normal return, SystemExit
    from the re-raised SIGTERM, KeyboardInterrupt from the re-raised SIGINT).

    A stop was asked for when a signal was noted or a stop path recorded its reason; the lifespan's own belt reason
    ("unknown") is not a request: it only means uvicorn ended for its own reasons. A run that ended with no stop asked
    for (a lifespan startup that raised, after which uvicorn's run() returns normally; a bind failure; an internal
    error) is a crash: the hard-exit path with reason "crash" (not clean, non-zero exit), so the app's crash watchdog
    and the service managers restart it (#177 point 1); a service run that keeps failing to start is spaced by the
    start back-off (D26), not restarted at once.

    Otherwise "stopped" only when nothing is running: the task registry is sealed (or, when the lifespan never ran,
    empty and sealed now) and the supervisor thread has ended. Work still running is the hard-exit path at once.
    """
    from brain.bridge import shutdown as shutdown_mod
    from brain.bridge import shutdown_progress

    c = getattr(monitor, "controller", None)
    deliberate = c is not None and (
        c.signal_count > 0 or (c.requested and c.reason not in (None, shutdown_mod.REASON_UNKNOWN))
    )
    if not deliberate:
        logger.error("bridge run ended with no stop requested (startup failure or server error); recording a crash")
        shutdown_mod.hard_exit(persona_dir, shutdown_mod.REASON_CRASH)
        return
    bridge_state = getattr(getattr(app, "state", None), "bridge", None)
    sup = getattr(bridge_state, "supervisor_thread", None)
    reg = shutdown_progress.registry
    sealed = reg.sealed or reg.seal()
    if sealed and (sup is None or not sup.is_alive()):
        monitor.stop()  # type: ignore[attr-defined]  # its final progress record names this bridge, then "stopped"
        _write_stopped(persona_dir)
        return
    logger.error("bridge run ended with work still running; hard exit")
    shutdown_mod.hard_exit(persona_dir, shutdown_mod.REASON_HARD_EXIT)


_RUNTIME_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_RUNTIME_LOG_BACKUPS = 3


def _setup_runtime_logging(persona_dir: Path) -> None:
    """Attach a size-rotating handler to the runner's root logger.

    launchd's ``StandardOutPath`` / ``StandardErrorPath`` (and the legacy
    ``spawn_detached`` log file) capture everything Python writes to
    stdout/stderr but **don't rotate** — a long-lived supervisor would
    grow those files unbounded. Adding a Python-side ``RotatingFileHandler``
    on the root logger gives us a primary log we control: 5MB cap with
    3 backups (~15MB ceiling per persona). The launchd-captured stdout/
    stderr files become a secondary catch-all for unhandled exceptions
    and uvicorn boot noise; those stay tiny in steady state.

    Idempotent: a second call for the same log file adds nothing. Path: ``<log_dir>/runtime-<persona>.log``. Best-effort — failures
    here must NOT prevent bridge startup, so any exception is swallowed
    after a stderr breadcrumb.
    """
    try:
        from logging.handlers import RotatingFileHandler

        from brain.paths import get_log_dir

        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"runtime-{persona_dir.name}.log"
        root = logging.getLogger()
        if any(isinstance(h, RotatingFileHandler) and h.baseFilename == str(path.resolve()) for h in root.handlers):
            return  # already set up (``supervisor run`` sets it up before the start back-off wait, D26)
        handler = RotatingFileHandler(
            path,
            maxBytes=_RUNTIME_LOG_MAX_BYTES,
            backupCount=_RUNTIME_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        handler.setLevel(logging.INFO)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
    except Exception as exc:  # noqa: BLE001
        print(f"runtime logging setup failed: {exc}", file=sys.stderr)


def run_bridge_foreground(
    persona_dir: Path,
    *,
    client_origin: str = "cli",
    idle_shutdown_seconds: float | None = None,
) -> int:
    """Run one bridge process in the foreground until shutdown.

    This is the long-lived process body used by both ``python -m
    brain.bridge.runner`` and the launchd-friendly ``nell supervisor run``
    command. It does not detach or fork: the caller owns this process and can
    supervise it directly.
    """
    import secrets

    _setup_runtime_logging(persona_dir)

    # #456 (C43 a, C55): watcher files (progress record, hard-exit request, stop intent) left by an earlier process
    # are not this bridge's. Both callers (runner.main, cmd_run) hold the bridge lock here; a runner that lost it has
    # already exited.
    from brain.bridge import shutdown_progress

    shutdown_progress.delete_watcher_files(persona_dir)

    port = _allocate_port()
    # H-C: ephemeral bearer token persisted in bridge.json. 32 bytes from
    # secrets.token_urlsafe — cryptographically random and subprotocol-safe.
    # bridge.json/state backups are chmod-hardened by state_file.write().
    auth_token = secrets.token_urlsafe(32)

    from brain import process_identity

    own_token = process_identity.own_token()
    if own_token is None:
        logger.error(
            "bridge identity unavailable on this platform; using the command-line check for liveness and kills"
        )
    initial_state = state_file.BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=port,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin=client_origin,
        auth_token=auth_token,
        pid_token=own_token,
        identity="token" if own_token else "unavailable",
        phase="running",
    )
    state_file.write(persona_dir, initial_state)

    from brain.bridge import shutdown as shutdown_mod

    shutdown_controller = BridgeShutdownController()
    shutdown_controller.configure(persona_dir)
    server_holder: dict = {}
    # Before uvicorn installs its own handlers (and after it restores this one): a signal only writes attributes;
    # uvicorn sees ``should_exit`` at start. After serve() ended, the re-raised SIGTERM ends with status 0 (C15).
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, shutdown_mod.early_signal_handler(shutdown_controller, server_holder))

    app = build_app(
        persona_dir=persona_dir,
        client_origin=client_origin,
        idle_shutdown_seconds=idle_shutdown_seconds,
        auth_token=auth_token,
        shutdown_controller=shutdown_controller,
    )
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=30,
    )
    server = shutdown_mod.make_server_class(uvicorn.Server)(config)
    server.controller = shutdown_controller
    server_holder["server"] = server
    shutdown_controller.bind_server(server)
    if shutdown_controller.signal_count:
        server.should_exit = True
    monitor = shutdown_mod.Monitor(
        shutdown_controller,
        persona_dir,
        pid=os.getpid(),
        pid_token=own_token,
        hard_exit=lambda reason: shutdown_mod.hard_exit(persona_dir, reason),
    )
    monitor.start()
    # D26 / D7: a service-run start counts as failed until the server has stayed up 600 s (or a stop is recorded).
    watch = None
    if client_origin in respawn_guard.SERVICE_ORIGINS:
        watch = start_backoff.SuccessWatch(persona_dir, server)
        watch.start()
    rc = 0
    try:
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # the re-raised SIGINT after a graceful stop: exit 0 (C42 d)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 0
    finally:
        if watch is not None:
            watch.stop()
        _after_run(persona_dir, app, monitor)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona-dir", required=True, type=Path)
    p.add_argument("--client-origin", default="cli")
    p.add_argument("--idle-shutdown-seconds", type=float, default=None)
    try:
        args = p.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error, but 2 is this entrypoint's S57
        # refusal code, which cmd_start reports as "already running/starting".
        # Keep 2 meaning only that; a bad argv is an ordinary failure.
        return 1 if exc.code == 2 else exc.code

    # Mutual-exclusion (Bug 2a, v0.0.36): the detached runner — spawned by
    # cmd_start on app open — must refuse to bind a SECOND bridge for a persona
    # that already has one. cmd_run (the task path) holds this same is_running +
    # lockfile guard; the runner lacked it, so two concurrent `supervisor start`
    # spawns each created a live bridge (two supervisors, parallel soul review).
    from brain.bridge.daemon import acquire_lock, release_lock

    if state_file.is_running(args.persona_dir):
        return 2
    fd = acquire_lock(args.persona_dir)
    if fd is None:
        return 2  # another starter holds the lock — lose cleanly, no second bridge
    try:
        return run_bridge_foreground(
            args.persona_dir,
            client_origin=args.client_origin,
            idle_shutdown_seconds=args.idle_shutdown_seconds,
        )
    finally:
        release_lock(args.persona_dir, fd)


if __name__ == "__main__":
    sys.exit(main())
