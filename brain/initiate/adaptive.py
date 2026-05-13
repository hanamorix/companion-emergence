"""Adaptive-D layer for v0.0.11 (Bundle C).

When `<persona_dir>/d_mode.json` is set to `"adaptive"`, D-reflection's
system message gets a calibration block prepended summarising the
companion's recent editorial track record. Default behaviour is
`"stateless"` — no calibration block, D behaves exactly as in v0.0.10.

Spec: docs/superpowers/specs/2026-05-13-v0.0.11-design.md
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


DMode = Literal["stateless", "adaptive"]
_VALID_MODES: frozenset[str] = frozenset({"stateless", "adaptive"})


def load_d_mode(persona_dir: Path) -> DMode:
    """Read `<persona_dir>/d_mode.json` and return the mode.

    Missing file, invalid JSON, non-dict JSON, or unknown mode values
    all fall back to `"stateless"` with a logged warning.
    """
    path = persona_dir / "d_mode.json"
    if not path.exists():
        return "stateless"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("d_mode.json read failed (%s); falling back to stateless", exc)
        return "stateless"
    if not isinstance(raw, dict):
        logger.warning("d_mode.json is not a JSON object; falling back to stateless")
        return "stateless"
    mode = raw.get("mode")
    if mode not in _VALID_MODES:
        logger.warning("d_mode.json has unknown mode %r; falling back to stateless", mode)
        return "stateless"
    return mode  # type: ignore[return-value]
