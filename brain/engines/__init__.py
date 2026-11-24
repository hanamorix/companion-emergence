"""Cognitive engines for companion-emergence.

Dreams consolidate associative patterns; heartbeat/reflex/research
follow in later weeks. Their week-4 designs (2026-04-23 and 2026-04-24) were never committed and are lost (#797).
"""

from brain.engines.dream import DreamEngine, DreamResult, NoSeedAvailable
from brain.engines.heartbeat import (
    HeartbeatConfig,
    HeartbeatEngine,
    HeartbeatResult,
    HeartbeatState,
)

__all__ = [
    "DreamEngine",
    "DreamResult",
    "HeartbeatConfig",
    "HeartbeatEngine",
    "HeartbeatResult",
    "HeartbeatState",
    "NoSeedAvailable",
]
