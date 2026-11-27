"""The one call from another process to the running bridge's local web API (loopback, bearer token).

The memory server's `search_memories` tool and the vocabulary calls of `brain.emotion.vocabulary_access` reach the bridge through
`post_to_bridge`, so the address, the token, the direct (no proxy) connection and the three failure results exist once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from brain.bridge import state_file


def post_to_bridge(persona_dir: Path, route: str, body: dict[str, Any], *, timeout: float) -> Any:
    """POST `body` as JSON to `route` of the bridge serving `persona_dir`; the bridge's JSON answer, unchanged on a 200.

    The bridge's port and bearer token come from `state_file.read(persona_dir)` (the same auth `/health` uses). Outcomes:
      - bridge.json missing, no recorded port, connection refused or reset -> ``{"error": "bridge unreachable"}``.
      - the call takes longer than `timeout` seconds -> ``{"error": "bridge timeout"}`` (distinct from "unreachable").
      - a response that is not a 200 -> ``{"error": "bridge error <status>"}``.

    Any other exception (for example a malformed 200 body from ``resp.json()``) is not caught here; the caller decides.
    """
    state = state_file.read(persona_dir)
    if state is None or state.port is None:
        return {"error": "bridge unreachable"}
    headers = {"Authorization": f"Bearer {state.auth_token}"} if state.auth_token else {}
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{state.port}{route}",
            json=body,
            headers=headers,
            timeout=timeout,
            trust_env=False,  # the loopback bridge is never reached through a proxy (#475)
        )
    except httpx.TimeoutException:
        return {"error": "bridge timeout"}
    except httpx.HTTPError:
        return {"error": "bridge unreachable"}
    if resp.status_code == 200:
        return resp.json()
    return {"error": f"bridge error {resp.status_code}"}
