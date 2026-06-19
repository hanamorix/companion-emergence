"""Within-invocation read dedup. Keyed on realpath.casefold() so the same file
isn't re-emitted (and re-billed) repeatedly inside one reply. TTL-scoped so a
legitimate re-read in a later reply still returns content.

Scoping note: no per-turn correlation ID is available in the tool dispatch
path (dispatch() receives store/hebbian/persona_dir but no request/turn ID).
TTL fallback (_DEDUP_TTL_S=90) is used instead. Residual: a re-read >90s
later re-bills (rare, accepted); a re-read <90s in a new reply is wrongly
deduped (rare + low-harm since the dedup note points to prior content).
"""
from __future__ import annotations

import time

_DEDUP_TTL_S = 90.0
_seen: dict[str, float] = {}


def seen_recently(key: str, *, now: float | None = None) -> bool:
    now = time.monotonic() if now is None else now
    ts = _seen.get(key)
    return ts is not None and (now - ts) < _DEDUP_TTL_S


def mark(key: str, *, now: float | None = None) -> None:
    _seen[key] = time.monotonic() if now is None else now


def reset() -> None:
    _seen.clear()
