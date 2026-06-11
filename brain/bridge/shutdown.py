"""Cross-platform bridge shutdown controller.

The controller avoids using OS signals as the normal control plane. The
runner binds the live uvicorn.Server instance, and API/idle/CLI paths request
shutdown by setting ``server.should_exit``.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class _UvicornServerLike(Protocol):
    should_exit: bool


class BridgeShutdownController:
    """Thread-safe one-shot shutdown request handle."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: _UvicornServerLike | None = None
        self._requested = False
        self._reason: str | None = None

    @property
    def requested(self) -> bool:
        return self._requested

    @property
    def reason(self) -> str | None:
        return self._reason

    def bind_server(self, server: _UvicornServerLike) -> None:
        with self._lock:
            self._server = server
            if self._requested:
                server.should_exit = True

    def request(self, reason: str) -> bool:
        with self._lock:
            if self._server is None:
                logger.error(
                    "shutdown requested before uvicorn server was bound reason=%s", reason
                )
                return False
            if not self._requested:
                self._requested = True
                self._reason = reason
                logger.info("bridge shutdown requested reason=%s", reason)
            self._server.should_exit = True
            return True
