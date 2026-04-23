"""Heartbeat — event-driven orchestrator tick.

See docs/superpowers/specs/2026-04-23-week-4-heartbeat-engine-design.md.
Each `nell heartbeat` invocation applies decay, maybe-dreams (rate-limited
by config.dream_every_hours), and persists timing state. No daemon —
the hosting application (or CI) calls this on app open/close.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from brain.bridge.provider import LLMProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

EmitMemoryMode = Literal["always", "conditional", "never"]
_VALID_EMIT_MODES: tuple[str, ...] = ("always", "conditional", "never")


@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig:
        """Load config from JSON; return defaults if file missing or invalid."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()

        emit = data.get("emit_memory", "conditional")
        if emit not in _VALID_EMIT_MODES:
            emit = "conditional"

        try:
            return cls(
                dream_every_hours=float(data.get("dream_every_hours", 24.0)),
                decay_rate_per_tick=float(data.get("decay_rate_per_tick", 0.01)),
                gc_threshold=float(data.get("gc_threshold", 0.01)),
                emit_memory=emit,  # type: ignore[arg-type]
            )
        except (TypeError, ValueError):
            # Hand-edited config with wrong-type values (e.g. dream_every_hours={}
            # or dream_every_hours=[1,2]) should degrade to defaults rather than
            # crash the CLI with a traceback.
            return cls()

    def save(self, path: Path) -> None:
        """Write config JSON to path (non-atomic — config is user-edited)."""
        payload = {
            "dream_every_hours": self.dream_every_hours,
            "decay_rate_per_tick": self.decay_rate_per_tick,
            "gc_threshold": self.gc_threshold,
            "emit_memory": self.emit_memory,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class HeartbeatState:
    """Per-persona heartbeat state. Loaded from heartbeat_state.json."""

    last_tick_at: datetime
    last_dream_at: datetime
    last_research_at: datetime
    tick_count: int
    last_trigger: str

    @classmethod
    def load(cls, path: Path) -> HeartbeatState | None:
        """Load state; return None if the file is missing or corrupt.

        Returning None triggers the first-ever-tick defer path in the engine,
        which is the safest recovery from a hand-edited or truncated state
        file (user-facing crashes from a malformed JSON are worse UX than
        silently reinitialising).
        """
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                last_tick_at=_parse_iso_utc(data["last_tick_at"]),
                last_dream_at=_parse_iso_utc(data["last_dream_at"]),
                last_research_at=_parse_iso_utc(data["last_research_at"]),
                tick_count=int(data["tick_count"]),
                last_trigger=str(data["last_trigger"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @classmethod
    def fresh(cls, trigger: str) -> HeartbeatState:
        """Build an initial state with all timestamps = now and tick_count = 0."""
        now = datetime.now(UTC)
        return cls(
            last_tick_at=now,
            last_dream_at=now,
            last_research_at=now,
            tick_count=0,
            last_trigger=trigger,
        )

    def save(self, path: Path) -> None:
        """Atomic save via write-to-.new + os.replace."""
        payload = {
            "last_tick_at": _iso_utc(self.last_tick_at),
            "last_dream_at": _iso_utc(self.last_dream_at),
            "last_research_at": _iso_utc(self.last_research_at),
            "tick_count": self.tick_count,
            "last_trigger": self.last_trigger,
        }
        tmp = path.with_suffix(path.suffix + ".new")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX + Windows


@dataclass(frozen=True)
class HeartbeatResult:
    """Outcome of a single heartbeat tick."""

    trigger: str
    elapsed_seconds: float
    memories_decayed: int
    edges_pruned: int
    dream_id: str | None
    dream_gated_reason: str | None
    research_deferred: bool
    heartbeat_memory_id: str | None
    initialized: bool


@dataclass
class HeartbeatEngine:
    """Composes decay + dream + research into one orchestrator tick.

    run_tick() is implemented in Task 2; this task ships the scaffold only.
    """

    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider
    state_path: Path
    config_path: Path
    dream_log_path: Path
    heartbeat_log_path: Path

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> HeartbeatResult:
        """Run one heartbeat tick. Implemented in Task 2."""
        raise NotImplementedError("HeartbeatEngine.run_tick is implemented in T2")


def _iso_utc(dt: datetime) -> str:
    """ISO-8601 with Z suffix (matches Week 3.5 manifest format).

    Requires a tz-aware UTC datetime — a naive datetime would silently
    write a malformed stamp (no Z suffix, no offset) that doesn't parse
    back cleanly.
    """
    if dt.tzinfo is None:
        raise ValueError("_iso_utc requires a tz-aware datetime")
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 Z-suffix timestamp back to tz-aware datetime."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
