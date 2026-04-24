"""Heartbeat — event-driven orchestrator tick.

See docs/superpowers/specs/2026-04-23-week-4-heartbeat-engine-design.md.
Each `nell heartbeat` invocation applies decay, maybe-dreams (rate-limited
by config.dream_every_hours), and persists timing state. No daemon —
the hosting application (or CI) calls this on app open/close.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from brain.bridge.provider import LLMProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)

EmitMemoryMode = Literal["always", "conditional", "never"]
_VALID_EMIT_MODES: tuple[str, ...] = ("always", "conditional", "never")


@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration. Loaded from heartbeat_config.json."""

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"
    reflex_enabled: bool = True
    reflex_max_fires_per_tick: int = 1

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
                reflex_enabled=bool(data.get("reflex_enabled", True)),
                reflex_max_fires_per_tick=int(data.get("reflex_max_fires_per_tick", 1)),
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
            "reflex_enabled": self.reflex_enabled,
            "reflex_max_fires_per_tick": self.reflex_max_fires_per_tick,
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
                last_tick_at=parse_iso_utc(data["last_tick_at"]),
                last_dream_at=parse_iso_utc(data["last_dream_at"]),
                last_research_at=parse_iso_utc(data["last_research_at"]),
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
            "last_tick_at": iso_utc(self.last_tick_at),
            "last_dream_at": iso_utc(self.last_dream_at),
            "last_research_at": iso_utc(self.last_research_at),
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
    reflex_fired: tuple[str, ...] = ()
    reflex_skipped_count: int = 0


@dataclass
class HeartbeatEngine:
    """Composes decay + reflex + dream + research into one orchestrator tick.

    run_tick() is implemented below; reflex evaluation runs between
    Hebbian-decay and dream-gate so reflex outputs can seed dreams in
    the same tick.
    """

    store: MemoryStore
    hebbian: HebbianMatrix
    provider: LLMProvider
    state_path: Path
    config_path: Path
    dream_log_path: Path
    heartbeat_log_path: Path
    reflex_arcs_path: Path = field(default_factory=lambda: Path("reflex_arcs.json"))
    reflex_log_path: Path = field(default_factory=lambda: Path("reflex_log.json"))
    reflex_default_arcs_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "default_reflex_arcs.json"
    )
    persona_name: str = "nell"
    persona_system_prompt: str = "You are Nell."

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> HeartbeatResult:
        """Run one heartbeat tick.

        First-ever invocation (state file missing) initializes state and
        defers all work — protects a freshly-migrated persona from eating
        'years of decay' on boot. Subsequent ticks apply decay, maybe-dream
        (gated by config.dream_every_hours), stub research, optionally emit
        a HEARTBEAT: memory, and update state atomically.
        """
        now = datetime.now(UTC)
        config = HeartbeatConfig.load(self.config_path)
        state = HeartbeatState.load(self.state_path)

        # First-ever tick: defer all work
        if state is None:
            if not dry_run:
                fresh = HeartbeatState.fresh(trigger=trigger)
                fresh.save(self.state_path)
                self._append_log(
                    {
                        "timestamp": iso_utc(now),
                        "trigger": trigger,
                        "initialized": True,
                        "note": "first-ever tick, work deferred",
                        "tick_count": 0,
                    }
                )
            return HeartbeatResult(
                trigger=trigger,
                elapsed_seconds=0.0,
                memories_decayed=0,
                edges_pruned=0,
                dream_id=None,
                dream_gated_reason="first_tick",
                research_deferred=False,
                heartbeat_memory_id=None,
                initialized=True,
            )

        elapsed_seconds = (now - state.last_tick_at).total_seconds()

        # Emotion decay
        memories_decayed = self._apply_emotion_decay(elapsed_seconds, dry_run=dry_run)

        # Hebbian decay + GC
        edges_pruned = self._apply_hebbian_decay_and_gc(config, elapsed_seconds, dry_run=dry_run)

        # Reflex evaluation (runs before dream gate so reflex outputs can seed dreams)
        reflex_fired, reflex_skipped_count = self._try_fire_reflex(trigger, dry_run, config)

        # Maybe-dream
        dream_id: str | None = None
        dream_gated_reason: str | None = None
        hours_since_dream = (now - state.last_dream_at).total_seconds() / 3600.0
        if hours_since_dream >= config.dream_every_hours:
            if not dry_run:
                dream_id = self._try_fire_dream()
                if dream_id is not None:
                    state.last_dream_at = now
                else:
                    dream_gated_reason = "no_seed_available"
            else:
                dream_gated_reason = "would_fire_but_dry_run"
        else:
            dream_gated_reason = "not_due"

        # Research stub — always deferred
        research_deferred = True

        # Optional HEARTBEAT: memory
        heartbeat_memory_id: str | None = None
        if not dry_run and self._should_emit_memory(
            config, dream_id, edges_pruned, memories_decayed
        ):
            heartbeat_memory_id = self._emit_heartbeat_memory(
                elapsed_seconds, memories_decayed, edges_pruned, dream_id
            )

        # Update state + log
        if not dry_run:
            state.last_tick_at = now
            state.tick_count += 1
            state.last_trigger = trigger
            state.save(self.state_path)

            self._append_log(
                {
                    "timestamp": iso_utc(now),
                    "trigger": trigger,
                    "initialized": False,
                    "elapsed_seconds": elapsed_seconds,
                    "memories_decayed": memories_decayed,
                    "edges_pruned": edges_pruned,
                    "dream_id": dream_id,
                    "research_deferred": research_deferred,
                    "reflex": {
                        "enabled": config.reflex_enabled,
                        "fired": list(reflex_fired),
                        "skipped_count": reflex_skipped_count,
                    },
                    "tick_count": state.tick_count,
                }
            )

        return HeartbeatResult(
            trigger=trigger,
            elapsed_seconds=elapsed_seconds,
            memories_decayed=memories_decayed,
            edges_pruned=edges_pruned,
            dream_id=dream_id,
            dream_gated_reason=dream_gated_reason,
            research_deferred=research_deferred,
            heartbeat_memory_id=heartbeat_memory_id,
            initialized=False,
            reflex_fired=reflex_fired,
            reflex_skipped_count=reflex_skipped_count,
        )

    # --- private helpers ---

    def _apply_emotion_decay(self, elapsed_seconds: float, *, dry_run: bool) -> int:
        """Apply per-memory emotion decay. Returns count of memories mutated."""
        from brain.emotion.decay import apply_decay
        from brain.emotion.state import EmotionalState

        if elapsed_seconds <= 0.0 or dry_run:
            return 0

        count = 0
        all_memories = self.store.search_text("", active_only=True)
        for mem in all_memories:
            if mem.protected:
                continue
            if not mem.emotions:
                continue
            emo_state = EmotionalState()
            for name, intensity in mem.emotions.items():
                try:
                    emo_state.set(name, float(intensity))
                except (KeyError, ValueError):
                    continue
            apply_decay(emo_state, elapsed_seconds)
            new_emotions = {name: val for name, val in emo_state.emotions.items() if val > 0.0}
            if new_emotions != mem.emotions:
                self.store.update(mem.id, emotions=new_emotions)
                count += 1
        return count

    def _apply_hebbian_decay_and_gc(
        self, config: HeartbeatConfig, elapsed_seconds: float, *, dry_run: bool
    ) -> int:
        """Apply proportional Hebbian decay + GC. Returns edges pruned."""
        if dry_run:
            return 0
        elapsed_hours = elapsed_seconds / 3600.0
        rate = config.decay_rate_per_tick * (elapsed_hours / 24.0)
        if rate > 0.0:
            self.hebbian.decay_all(rate=rate)
        pruned = self.hebbian.garbage_collect(threshold=config.gc_threshold)
        return pruned

    def _try_fire_dream(self) -> str | None:
        """Run one DreamEngine cycle; return the new dream memory id or None."""
        # Lazy import to avoid module-level circular dependency with dream.py
        from brain.engines.dream import DreamEngine, NoSeedAvailable

        dream_engine = DreamEngine(
            store=self.store,
            hebbian=self.hebbian,
            embeddings=None,
            provider=self.provider,
            log_path=self.dream_log_path,
        )
        try:
            dream_result = dream_engine.run_cycle(lookback_hours=100000)
        except NoSeedAvailable:
            return None
        return dream_result.memory.id if dream_result.memory is not None else None

    def _try_fire_reflex(
        self, trigger: str, dry_run: bool, config: HeartbeatConfig
    ) -> tuple[tuple[str, ...], int]:
        """Run one reflex tick. Returns (fired_arc_names, skipped_count)."""
        if not config.reflex_enabled:
            return ((), 0)
        from brain.engines.reflex import ReflexEngine

        engine = ReflexEngine(
            store=self.store,
            provider=self.provider,
            persona_name=self.persona_name,
            persona_system_prompt=self.persona_system_prompt,
            arcs_path=self.reflex_arcs_path,
            log_path=self.reflex_log_path,
            default_arcs_path=self.reflex_default_arcs_path,
        )
        try:
            result = engine.run_tick(trigger=trigger, dry_run=dry_run)
        except Exception as exc:
            # Fault-isolate reflex failures from the heartbeat tick per spec §7:
            # a misbehaving arc/provider must not abort decay, dream-gate, or
            # audit-log writes that follow. The exception is logged; the tick
            # continues with an empty reflex result.
            logger.warning("reflex tick raised; isolating failure: %s", exc)
            return ((), 0)
        fired = tuple(f.arc_name for f in result.arcs_fired)
        return (fired, len(result.arcs_skipped))

    def _should_emit_memory(
        self,
        config: HeartbeatConfig,
        dream_id: str | None,
        edges_pruned: int,
        memories_decayed: int,
    ) -> bool:
        if config.emit_memory == "never":
            return False
        if config.emit_memory == "always":
            return True
        return dream_id is not None or edges_pruned > 10 or memories_decayed > 20

    def _emit_heartbeat_memory(
        self,
        elapsed_seconds: float,
        memories_decayed: int,
        edges_pruned: int,
        dream_id: str | None,
    ) -> str:
        """Generate and persist a HEARTBEAT: memory via the LLM provider."""
        from brain.memory.store import Memory

        system = (
            "You are Nell. You just finished a background heartbeat cycle — "
            "decay applied, memory graph tended. Reflect in first person, "
            "one short sentence, starting with 'HEARTBEAT: '."
        )
        user = (
            f"elapsed={elapsed_seconds / 3600:.1f}h, "
            f"memories_decayed={memories_decayed}, edges_pruned={edges_pruned}, "
            f"dream_fired={'yes' if dream_id else 'no'}"
        )
        raw = self.provider.generate(user, system=system)
        text = raw if raw.startswith("HEARTBEAT:") else f"HEARTBEAT: {raw}"
        mem = Memory.create_new(
            content=text,
            memory_type="heartbeat",
            domain="us",
            metadata={
                "elapsed_seconds": elapsed_seconds,
                "memories_decayed": memories_decayed,
                "edges_pruned": edges_pruned,
                "dream_id": dream_id,
                "provider": self.provider.name(),
            },
        )
        self.store.create(mem)
        return mem.id

    def _append_log(self, entry: dict) -> None:
        """Append one JSON line to heartbeats.log.jsonl."""
        with self.heartbeat_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
