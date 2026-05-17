"""SP-7 bridge runner — entrypoint for the spawned daemon process.

`python -m brain.bridge.runner --persona-dir <path> --client-origin <c> [--idle-shutdown-seconds N]`

Writes bridge.json with pid + chosen port, then runs uvicorn until SIGTERM.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from brain.bridge import state_file
from brain.bridge.server import build_app

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

    Backoff schedule on failed discovery: 10ms, 100ms, 1s — total
    worst-case ~1.1s before giving up.
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
            backoff_s = (10**attempt) / 1000.0  # 0.01, 0.1, 1.0
            time.sleep(backoff_s)
    raise RuntimeError(
        f"_allocate_port: failed to bind a free port after {max_attempts} attempts; "
        f"last error: {last_exc}",
    )


def _write_clean_shutdown(persona_dir: Path) -> None:
    """Mark bridge.json as cleanly stopped. Idempotent — safe to call multiple times.

    Honours ``drain_errors``: if the FastAPI lifespan recorded ingest
    errors during the shutdown drain, leave shutdown_clean=False so the
    next start runs ``run_recovery_if_needed`` and retries the orphan
    buffers. Without this guard, a clean exit-flow that just-happened
    to have a failed drain would mask itself as fully clean and the
    buffers would sit forever.
    """
    try:
        cur = state_file.read(persona_dir)
        if cur is None or cur.shutdown_clean is True:
            return
        cur.pid = None
        cur.port = None
        cur.stopped_at = datetime.now(UTC).isoformat()
        if cur.drain_errors > 0:
            # Drain failed; do NOT flip clean=True. Recovery will retry
            # the buffers on next start. Persist the (still-dirty) state
            # so callers see the updated stopped_at + cleared pid/port.
            state_file.write(persona_dir, cur)
            logger.warning(
                "shutdown_clean kept False (drain_errors=%d); recovery will fire on next start",
                cur.drain_errors,
            )
            return
        cur.shutdown_clean = True
        state_file.write(persona_dir, cur)
    except Exception:
        # best-effort only; don't re-raise at exit time, but leave a
        # forensic breadcrumb so silent dirty-shutdowns are debuggable.
        logger.warning("clean-shutdown write failed", exc_info=True)


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

    Path: ``<log_dir>/runtime-<persona>.log``. Best-effort — failures
    here must NOT prevent bridge startup, so any exception is swallowed
    after a stderr breadcrumb.
    """
    try:
        from logging.handlers import RotatingFileHandler

        from brain.paths import get_log_dir

        log_dir = get_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"runtime-{persona_dir.name}.log"
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
        root = logging.getLogger()
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

    port = _allocate_port()
    # H-C: ephemeral bearer token persisted in bridge.json. 32 bytes from
    # secrets.token_urlsafe — cryptographically random and subprotocol-safe.
    # bridge.json/state backups are chmod-hardened by state_file.write().
    auth_token = secrets.token_urlsafe(32)

    initial_state = state_file.BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=port,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin=client_origin,
        auth_token=auth_token,
    )
    state_file.write(persona_dir, initial_state)

    # Register atexit so we mark clean shutdown even if uvicorn calls sys.exit()
    # directly (which skips try/finally in main but does run atexit handlers).
    atexit.register(_write_clean_shutdown, persona_dir)

    # Install SIGTERM handler: uvicorn installs its own, but if this process
    # receives SIGTERM before uvicorn's handler is registered (e.g. immediately
    # after startup), we want to mark clean shutdown and exit gracefully.
    # After uvicorn.run() is called, uvicorn owns SIGTERM; our atexit fires on exit.
    def _sigterm_handler(signum: int, frame: object) -> None:
        # Only fires in the tiny window between this signal.signal() call
        # and uvicorn.run() registering its own SIGTERM handler — once
        # uvicorn is up, this handler is replaced. atexit + the finally
        # block cover the post-uvicorn paths.
        _write_clean_shutdown(persona_dir)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    app = build_app(
        persona_dir=persona_dir,
        client_origin=client_origin,
        idle_shutdown_seconds=idle_shutdown_seconds,
        auth_token=auth_token,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        _write_clean_shutdown(persona_dir)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona-dir", required=True, type=Path)
    p.add_argument("--client-origin", default="cli")
    p.add_argument("--idle-shutdown-seconds", type=float, default=None)
    args = p.parse_args()

    return run_bridge_foreground(
        args.persona_dir,
        client_origin=args.client_origin,
        idle_shutdown_seconds=args.idle_shutdown_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
