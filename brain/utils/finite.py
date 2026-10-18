"""Clamp non-finite floats at the producer of a record that a reader later parses with ``float()``.

A NaN or Infinity written as the JSON token ``NaN`` was readable by the old ``json.loads`` and by a ``float()`` reader.
The strict writer in ``brain.health.json_io`` refuses it, and ``nonfinite="null"`` would store ``null``, which a
``float()`` reader rejects. So a record whose readers need a number takes the clamp here and keeps the strict writer.

A clamp invents a value in a durable record, so each call that replaced anything logs ONE warning naming the field
paths and the original values, which is the only trace of where the non-finite number came from.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def _walk(value: Any, replacement: float | None, path: str, found: list[tuple[str, float]]) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        found.append((path or "<value>", value))
        return replacement
    if isinstance(value, dict):
        return {k: _walk(v, replacement, f"{path}.{k}" if path else str(k), found) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_walk(v, replacement, f"{path}[{i}]", found) for i, v in enumerate(value)]
    return value


def clamp_nonfinite(value: Any, replacement: float | None = 0.0, *, where: str = "") -> Any:
    """A copy of ``value`` with every NaN/Infinity float replaced by ``replacement`` (dicts, lists, tuples walked).

    Logs one WARNING per call that replaced a value: ``where`` (the caller's label), the dotted key paths and the
    original values. Nothing is logged when everything was finite.
    """
    found: list[tuple[str, float]] = []
    out = _walk(value, replacement, "", found)
    if found:
        shown = ", ".join(f"{p}={v!r}" for p, v in found[:10])
        more = f" (+{len(found) - 10} more)" if len(found) > 10 else ""
        logger.warning("non-finite value clamped to %r in %s: %s%s", replacement, where or "a record", shown, more)
    return out
