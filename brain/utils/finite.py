"""Clamp non-finite floats at the producer of a record that a reader later parses with ``float()``.

A NaN or Infinity written as the JSON token ``NaN`` was readable by the old ``json.loads`` and by a ``float()`` reader.
The strict writer in ``brain.health.json_io`` refuses it, and ``nonfinite="null"`` would store ``null``, which a
``float()`` reader rejects. So a record whose readers need a number takes the clamp here and keeps the strict writer.
"""

from __future__ import annotations

import math
from typing import Any


def clamp_nonfinite(value: Any, replacement: float = 0.0) -> Any:
    """A copy of ``value`` with every NaN/Infinity float replaced by ``replacement`` (dicts, lists, tuples walked)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else replacement
    if isinstance(value, dict):
        return {k: clamp_nonfinite(v, replacement) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clamp_nonfinite(v, replacement) for v in value]
    return value
