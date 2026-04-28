"""SP-7 bridge runner — entrypoint for the spawned daemon process.

`python -m brain.bridge.runner --persona-dir <path> --client-origin <c> [--idle-shutdown-seconds N]`

Writes bridge.json with pid + chosen port, then runs uvicorn until SIGTERM.
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from brain.bridge import state_file
from brain.bridge.server import build_app


def _allocate_port() -> int:
    """Bind ephemeral, read assigned port, close. Race window is tiny."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _write_clean_shutdown(persona_dir: Path) -> None:
    """Mark bridge.json as cleanly stopped. Idempotent — safe to call multiple times."""
    try:
        cur = state_file.read(persona_dir)
        if cur is not None and cur.shutdown_clean is False:
            cur.pid = None
            cur.port = None
            cur.stopped_at = datetime.now(UTC).isoformat()
            cur.shutdown_clean = True
            state_file.write(persona_dir, cur)
    except Exception:
        pass  # best-effort only; don't re-raise at exit time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--persona-dir", required=True, type=Path)
    p.add_argument("--client-origin", default="cli")
    p.add_argument("--idle-shutdown-seconds", type=float, default=None)
    args = p.parse_args()

    persona_dir = args.persona_dir
    port = _allocate_port()

    initial_state = state_file.BridgeState(
        persona=persona_dir.name,
        pid=os.getpid(),
        port=port,
        started_at=datetime.now(UTC).isoformat(),
        stopped_at=None,
        shutdown_clean=False,
        client_origin=args.client_origin,
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
        client_origin=args.client_origin,
        idle_shutdown_seconds=args.idle_shutdown_seconds,
    )
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        _write_clean_shutdown(persona_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
