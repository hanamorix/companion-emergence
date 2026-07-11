"""Engine adapter — stand up the REAL bridge + drive it over the WS ``/stream/{sid}`` protocol.

Generalizes the hunt harness ``live_server.py`` + ``t*_driver.py`` drive core. Two layers:

- ``BridgeServer`` — starts the REAL app (``brain.bridge.server.build_app``) on a uvicorn thread
  inside the sandbox env. Only the token-spending example exercises this (a real provider is
  needed); it is not part of the unit suite.
- pure helpers (unit-tested, no server/tokens): ``parse_ws_frame`` (the frame protocol),
  ``atomic_write`` (checkpoint atomicity), and the exit-code contract.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import EXIT_DONE, EXIT_INVALID, EXIT_LIMIT, EXIT_REVIEW  # noqa: F401 (contract export)


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + ``os.replace`` (atomic, crash-safe checkpoint)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def parse_ws_frame(raw: str) -> dict:
    """Parse one WS frame from the bridge stream. Frame types: ``reply_chunk`` (has ``text``),
    ``tool_call`` (has ``tool``), ``done``, ``error`` (has ``code``)."""
    return json.loads(raw)


def collect_reply(frames: list[dict]) -> tuple[str, list[str], str | None, bool]:
    """Reduce a sequence of parsed frames to (reply_text, tools, error_code, got_done).

    Pure so a unit test can feed synthetic frames (no socket). Mirrors the driver's stream loop.
    """
    reply: list[str] = []
    tools: list[str] = []
    err: str | None = None
    got_done = False
    for f in frames:
        ty = f.get("type")
        if ty == "reply_chunk":
            reply.append(f.get("text", ""))
        elif ty == "tool_call":
            tools.append(f.get("tool"))
        elif ty == "done":
            got_done = True
            break
        elif ty == "error":
            err = f.get("code", "error")
            break
    if err is None and not got_done:
        err = "ws:incomplete"
    return "".join(reply), tools, err, got_done


def drive_ws(
    host: str,
    port: int,
    sid: str,
    message: str,
    *,
    timeout: float = 300.0,
    open_timeout: float = 30.0,
    token: str | None = None,
) -> tuple[str, list[str], str | None]:  # pragma: no cover - needs a live bridge / socket
    """Drive ONE turn over ``ws://{host}:{port}/stream/{sid}``; return (reply, tools, error).

    The single bounded recv loop reused by both :meth:`BridgeServer.drive_turn` and the Agent-Bob
    send-script (``agent_send.py``) — so there is one WS drive path, not two. Bounds the receive on
    ``timeout`` wall-clock; reduces frames with :func:`parse_ws_frame` + :func:`collect_reply`.

    When ``token`` is given, the WS is opened with the ``bearer, <token>`` **subprotocol** — the
    form the bridge's WS auth reads (``brain/bridge/server.py:_ws_subprotocol_token`` /
    ``_ws_accept_subprotocol``). Without a token the subprotocol is omitted (an ``auth_token=None``
    bridge — the sandbox default — needs none), so :meth:`BridgeServer.drive_turn` is unchanged.
    """
    from websockets.sync.client import connect

    base = f"{host}:{port}"
    subprotocols = ["bearer", token] if token else None
    frames: list[dict] = []
    t0 = time.time()
    try:
        with connect(
            f"ws://{base}/stream/{sid}", open_timeout=open_timeout, subprotocols=subprotocols
        ) as ws:
            ws.send(json.dumps({"message": message}))
            while time.time() - t0 < timeout:
                frames.append(parse_ws_frame(ws.recv()))
                if frames[-1].get("type") in ("done", "error"):
                    break
    except Exception as e:
        return "", [], f"ws:{type(e).__name__}"
    reply, tools, err, _ = collect_reply(frames)
    return reply, tools, err


class BridgeServer:
    """Start/stop the REAL bridge on a background uvicorn thread inside the sandbox.

    Only the token-spending example uses this (a real provider is required). Constructed with the
    persona_dir + port; ``start()`` blocks until the app is serving. NOT unit-tested (would need a
    live provider); the unit-tested surface is the pure helpers above.
    """

    def __init__(self, persona_dir: Path, port: int, *, host: str = "127.0.0.1") -> None:
        self.persona_dir = persona_dir
        self.port = port
        self.host = host
        self._server = None
        self._thread = None

    def start(self, *, ready_timeout: float = 60.0) -> None:  # pragma: no cover - needs live bridge
        import threading

        import uvicorn

        from brain.bridge.server import build_app

        app = build_app(
            self.persona_dir, tick_interval_s=1e9, silence_minutes=1e9,
            auth_token=None, allowed_origins=("null",),
        )
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if getattr(self._server, "started", False):
                return
            time.sleep(0.1)
        raise TimeoutError(f"bridge did not start on {self.host}:{self.port}")

    def stop(self) -> None:  # pragma: no cover - needs live bridge
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def drive_turn(
        self, sid: str, message: str, *, timeout: float = 300.0, open_timeout: float = 30.0
    ) -> tuple[str, list[str], str | None]:  # pragma: no cover - needs live bridge
        """Drive one turn over ``/stream/{sid}``; return (reply, tools, error).

        Delegates to the shared :func:`drive_ws` — the ONE bounded recv loop (the send-script uses
        the same helper), so there is a single WS drive path.
        """
        return drive_ws(
            self.host, self.port, sid, message, timeout=timeout, open_timeout=open_timeout
        )
