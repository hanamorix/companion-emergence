"""Cognitive engines for companion-emergence.

Dreams consolidate associative patterns; heartbeat/reflex/research
follow in later weeks. See:
docs/superpowers/specs/2026-04-23-week-4-dream-engine-design.md
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
