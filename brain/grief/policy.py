"""policy.py — module-level constants for the grief package.

All threshold/decay/scale values from the spec §3, §4, §5 live here so
callers and tests can reference a single source.

Spec: docs/superpowers/specs/2026-05-19-grief-design.md
"""

from __future__ import annotations

# §3 — intensity formula constants
DROP_SCALE: float = 7.0
ARC_CLOSE_SCALE: float = 7.0
RECALL_TOUCH_SCALE: float = 5.0
RECENCY_LIVED_DAYS_HALF_LIFE: float = 14.0  # exp-decay denominator

# §4 — threshold + debounce
THRESHOLD: float = 3.0
DEBOUNCE_HOURS: float = 2.0
RESIDUE_FACTOR: float = 0.5

# §5 — ambient block weight buckets
WEIGHT_HEAVY: float = 7.0
WEIGHT_MEDIUM: float = 3.0

# §5 — token budget for ambient grief block
BLOCK_TOKEN_CAP: int = 200

# §3 — memory_grief emotion half-life (also baked into emotion.vocabulary baseline)
DECAY_HALF_LIFE_DAYS: float = 30.0
