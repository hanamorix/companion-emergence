"""Calls to the persona's own bridge on the loopback address go DIRECT, never through a proxy (#475).

``urllib.request.urlopen``, httpx and a websockets client all read HTTP_PROXY / HTTPS_PROXY / ALL_PROXY from the environment. On a
machine with a proxy configured, a request to ``127.0.0.1`` would then be sent to the proxy, bearer token included, instead of to
the bridge. Everything here opens its own connection to the address it is given and never consults the environment:

* :func:`direct_request` uses ``http.client`` (one connection, no proxy handling at all);
* :func:`direct_ws_connect` passes ``proxy=None`` to the websockets sync client;
* the module-level httpx calls elsewhere pass ``trust_env=False``.
"""

from __future__ import annotations

import http.client
import inspect
import urllib.parse
from collections.abc import Mapping
from typing import Any


def direct_request(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    """One HTTP request straight to ``url``'s host and port; returns ``(status, body)``.

    A connection problem raises ``OSError`` (refused, reset, timed out); an HTTP error status is returned, not raised.
    """
    parts = urllib.parse.urlsplit(url)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=dict(headers or {}))
        response = conn.getresponse()
        return response.status, response.read()
    except http.client.HTTPException as exc:  # a malformed reply: report it like any connection problem
        raise OSError(f"bad reply from {parts.hostname}:{parts.port}: {exc}") from exc
    finally:
        conn.close()


def direct_ws_connect(url: str, **kwargs: Any):
    """``websockets.sync.client.connect`` with no proxy lookup (``proxy=None``) where the installed version has the option."""
    from websockets.sync.client import connect

    if "proxy" in inspect.signature(connect).parameters:
        kwargs.setdefault("proxy", None)
    return connect(url, **kwargs)
