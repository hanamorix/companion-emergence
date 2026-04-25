"""Heartbeat — event-driven orchestrator tick.

See docs/superpowers/specs/2026-04-23-week-4-heartbeat-engine-design.md.
Each `nell heartbeat` invocation applies decay, maybe-dreams (rate-limited
by config.dream_every_hours), and persists timing state. No daemon —
the hosting application (or CI) calls this on app open/close.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from brain.bridge.provider import LLMProvider
from brain.health.alarm import compute_pending_alarms
from brain.health.anomaly import BrainAnomaly
from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher, WebSearcher
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)

EmitMemoryMode = Literal["always", "conditional", "never"]
# Single source of truth — derived from the Literal so a new mode added
# above only needs to land in one place.
_VALID_EMIT_MODES: tuple[str, ...] = get_args(EmitMemoryMode)


@dataclass
class HeartbeatConfig:
    """Per-persona heartbeat configuration.

    Two-file resolution per principle audit 2026-04-25 (PR-C):

    1. `heartbeat_config.json` — developer-only internal calibration. The
       GUI never reads or writes this file. Holds decay/GC/threshold knobs
       that calibrate the brain's physiology.
    2. `user_preferences.json` — the GUI-surfaceable cadence file. When
       a field is present here (currently only `dream_every_hours`), it
       takes precedence over heartbeat_config.json. Missing or absent-key
       → fall back to heartbeat_config.json's value (back-compat).

    `dream_every_hours` is the one field that legitimately belongs to the
    user. Everything else on this dataclass is internal — exposing it in
    a GUI would let the user disable parts of the brain's autonomy.
    """

    dream_every_hours: float = 24.0
    decay_rate_per_tick: float = 0.01
    gc_threshold: float = 0.01
    emit_memory: EmitMemoryMode = "conditional"
    reflex_enabled: bool = True
    reflex_max_fires_per_tick: int = 1
    research_enabled: bool = True
    research_days_since_human_min: float = 1.5
    research_emotion_threshold: float = 7.0
    research_cooldown_hours_per_interest: float = 24.0
    interest_bump_per_match: float = 0.1
    growth_enabled: bool = True
    growth_every_hours: float = 168.0  # weekly default

    @classmethod
    def load(cls, path: Path) -> HeartbeatConfig:
        """Load heartbeat_config.json, then merge user_preferences.json if present.

        `path` points at heartbeat_config.json. user_preferences.json is
        looked up next to it (`path.parent / "user_preferences.json"`).
        """
        cfg, _ = cls.load_with_anomaly(path)
        return cfg

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[HeartbeatConfig, BrainAnomaly | None]:
        """Load heartbeat_config.json (with self-healing), then merge user_preferences.

        Returns (cfg, anomaly_or_None). The anomaly, if any, comes from the
        internal-load stage; user_preferences merge never raises anomalies.
        """
        cfg, anomaly = cls._load_internal_with_anomaly(path)

        # Merge user_preferences.json — only override fields explicitly
        # present in the file, so a user_preferences.json that omits
        # dream_every_hours doesn't shadow a custom value set in
        # heartbeat_config.json (back-compat for pre-PR-C personas).
        from brain.user_preferences import UserPreferences, read_raw_keys

        user_prefs_path = path.parent / "user_preferences.json"
        explicit_keys = read_raw_keys(user_prefs_path)
        if "dream_every_hours" in explicit_keys:
            prefs = UserPreferences.load(user_prefs_path)
            cfg = replace(cfg, dream_every_hours=prefs.dream_every_hours)
        return cfg, anomaly

    @classmethod
    def _parse_internal_data(cls, data: object) -> HeartbeatConfig:
        """Build instance from already-parsed JSON data (dict expected).

        Applies per-field type-coercion; falls back to defaults on type errors.
        """
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
                research_enabled=bool(data.get("research_enabled", True)),
                research_days_since_human_min=float(data.get("research_days_since_human_min", 1.5)),
                research_emotion_threshold=float(data.get("research_emotion_threshold", 7.0)),
                research_cooldown_hours_per_interest=float(
                    data.get("research_cooldown_hours_per_interest", 24.0)
                ),
                interest_bump_per_match=float(data.get("interest_bump_per_match", 0.1)),
                growth_enabled=bool(data.get("growth_enabled", True)),
                growth_every_hours=float(data.get("growth_every_hours", 168.0)),
            )
        except (TypeError, ValueError):
            # Hand-edited config with wrong-type values (e.g. dream_every_hours={}
            # or dream_every_hours=[1,2]) should degrade to defaults rather than
            # crash the CLI with a traceback.
            return cls()

    @classmethod
    def _load_internal_with_anomaly(cls, path: Path) -> tuple[HeartbeatConfig, BrainAnomaly | None]:
        """Load heartbeat_config.json with self-healing from .bak rotation.

        Returns (cfg, anomaly_or_None). The outer load() uses the anomaly-dropping
        wrapper _load_internal to preserve existing call sites unchanged.
        """
        from brain.health.attempt_heal import attempt_heal

        data, anomaly = attempt_heal(path, dict)
        return cls._parse_internal_data(data), anomaly

    @classmethod
    def _load_internal(cls, path: Path) -> HeartbeatConfig:
        """Load heartbeat_config.json only — the developer-calibration layer."""
        cfg, anomaly = cls._load_internal_with_anomaly(path)
        if anomaly is not None:
            logger.warning(
                "HeartbeatConfig anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        return cfg

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup).

        A crash mid-write leaves either the previous valid file or the new
        valid file — never a partial write that corrupts the user's config.
        """
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {
            "dream_every_hours": self.dream_every_hours,
            "decay_rate_per_tick": self.decay_rate_per_tick,
            "gc_threshold": self.gc_threshold,
            "emit_memory": self.emit_memory,
            "reflex_enabled": self.reflex_enabled,
            "reflex_max_fires_per_tick": self.reflex_max_fires_per_tick,
            "research_enabled": self.research_enabled,
            "research_days_since_human_min": self.research_days_since_human_min,
            "research_emotion_threshold": self.research_emotion_threshold,
            "research_cooldown_hours_per_interest": self.research_cooldown_hours_per_interest,
            "interest_bump_per_match": self.interest_bump_per_match,
            "growth_enabled": self.growth_enabled,
            "growth_every_hours": self.growth_every_hours,
        }
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)
        if treatment.verify_after_write:
            self._verify_after_write(path)

    def _verify_after_write(self, path: Path) -> None:
        """Re-read the written file; if corrupt, restore from .bak1."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("non-dict payload after write")
        except (json.JSONDecodeError, ValueError, OSError):
            logger.error(
                "HeartbeatConfig verify_after_write failed for %s; restoring from .bak1", path
            )
            bak1 = path.with_name(path.name + ".bak1")
            if bak1.exists():
                shutil.copy2(bak1, path)


@dataclass
class HeartbeatState:
    """Per-persona heartbeat state. Loaded from heartbeat_state.json."""

    last_tick_at: datetime
    last_dream_at: datetime
    last_research_at: datetime
    last_growth_at: datetime  # tz-aware UTC; defaults to now on first save
    tick_count: int
    last_trigger: str

    @classmethod
    def _parse_state_data(cls, data: object) -> HeartbeatState | None:
        """Build instance from already-parsed JSON data; return None on bad shape."""
        if not isinstance(data, dict):
            return None
        try:
            return cls(
                last_tick_at=parse_iso_utc(data["last_tick_at"]),
                last_dream_at=parse_iso_utc(data["last_dream_at"]),
                last_research_at=parse_iso_utc(data["last_research_at"]),
                # Back-compat: last_growth_at absent → fall back to last_tick_at.
                last_growth_at=parse_iso_utc(data.get("last_growth_at") or data["last_tick_at"]),
                tick_count=int(data["tick_count"]),
                last_trigger=str(data["last_trigger"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def load_with_anomaly(cls, path: Path) -> tuple[HeartbeatState | None, BrainAnomaly | None]:
        """Load state with self-healing from .bak rotation if corrupt.

        Returns (state_or_None, anomaly_or_None).
          - Missing file → (None, None) — normal first-tick path.
          - Corrupt file → quarantine + restore from .bak1/.bak2/.bak3 or reset.
            If all baks are corrupt/missing, data=={} → parse returns None → engine
            treats as first-ever tick and reinitialises (same safe recovery as before).
        """
        if not path.exists():
            return None, None

        from brain.health.attempt_heal import attempt_heal

        data, anomaly = attempt_heal(path, dict)
        return cls._parse_state_data(data), anomaly

    @classmethod
    def load(cls, path: Path) -> HeartbeatState | None:
        """Load state; return None if the file is missing or corrupt.

        Returning None triggers the first-ever-tick defer path in the engine,
        which is the safest recovery from a hand-edited or truncated state
        file (user-facing crashes from a malformed JSON are worse UX than
        silently reinitialising).
        """
        state, anomaly = cls.load_with_anomaly(path)
        if anomaly is not None:
            logger.warning(
                "HeartbeatState anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        return state

    @classmethod
    def fresh(cls, trigger: str) -> HeartbeatState:
        """Build an initial state with all timestamps = now and tick_count = 0."""
        now = datetime.now(UTC)
        return cls(
            last_tick_at=now,
            last_dream_at=now,
            last_research_at=now,
            last_growth_at=now,
            tick_count=0,
            last_trigger=trigger,
        )

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup)."""
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {
            "last_tick_at": iso_utc(self.last_tick_at),
            "last_dream_at": iso_utc(self.last_dream_at),
            "last_research_at": iso_utc(self.last_research_at),
            "last_growth_at": iso_utc(self.last_growth_at),
            "tick_count": self.tick_count,
            "last_trigger": self.last_trigger,
        }
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)
        if treatment.verify_after_write:
            self._verify_after_write(path)

    def _verify_after_write(self, path: Path) -> None:
        """Re-read the written file; if corrupt, restore from .bak1."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("non-dict payload after write")
        except (json.JSONDecodeError, ValueError, OSError):
            logger.error(
                "HeartbeatState verify_after_write failed for %s; restoring from .bak1", path
            )
            bak1 = path.with_name(path.name + ".bak1")
            if bak1.exists():
                shutil.copy2(bak1, path)


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
    research_fired: str | None = None
    research_gated_reason: str | None = None
    interests_bumped: int = 0
    growth_emotions_added: int = 0
    anomalies: tuple[BrainAnomaly, ...] = ()
    pending_alarms_count: int = 0


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
    # Reflex paths default to None so a HeartbeatEngine constructed without
    # explicit persona-dir-qualified paths can't silently write to cwd. When
    # either is None, _try_fire_reflex short-circuits with an empty result.
    # Production (CLI) always passes all three paths anchored to persona_dir.
    reflex_arcs_path: Path | None = None
    reflex_log_path: Path | None = None
    reflex_default_arcs_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "default_reflex_arcs.json"
    )
    # Research paths default to None (same pattern as reflex) — if either is
    # None, _try_fire_research short-circuits. CLI passes explicit paths.
    searcher: WebSearcher = field(default_factory=NoopWebSearcher)
    interests_path: Path | None = None
    research_log_path: Path | None = None
    default_interests_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "default_interests.json"
    )
    persona_name: str = ""
    persona_system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.persona_name:
            raise ValueError(
                "HeartbeatEngine requires persona_name — construct explicitly, "
                "don't rely on a default."
            )
        if not self.persona_system_prompt:
            raise ValueError(
                "HeartbeatEngine requires persona_system_prompt — construct "
                "explicitly, don't rely on a default."
            )

    def run_tick(self, *, trigger: str = "manual", dry_run: bool = False) -> HeartbeatResult:
        """Run one heartbeat tick.

        First-ever invocation (state file missing) initializes state and
        defers all work — protects a freshly-migrated persona from eating
        'years of decay' on boot. Subsequent ticks apply decay, maybe-dream
        (gated by config.dream_every_hours), stub research, optionally emit
        a HEARTBEAT: memory, and update state atomically.
        """
        now = datetime.now(UTC)

        # Collect per-tick anomalies from direct heartbeat-engine loads (config + state).
        tick_anomalies: list[BrainAnomaly] = []

        config, config_anomaly = HeartbeatConfig.load_with_anomaly(self.config_path)
        if config_anomaly is not None:
            tick_anomalies.append(config_anomaly)

        state, state_anomaly = HeartbeatState.load_with_anomaly(self.state_path)
        if state_anomaly is not None:
            tick_anomalies.append(state_anomaly)

        # Cross-file walk gate: >=2 anomalies triggers a full persona scan.
        # Deduplicate by (file, kind) so files already caught in direct loads
        # are not double-counted.
        if len(tick_anomalies) >= 2:
            _walk_persona_dir = (
                self.interests_path.parent
                if self.interests_path is not None
                else self.state_path.parent
            )
            seen: set[tuple[str, str]] = {(a.file, a.kind) for a in tick_anomalies}
            for walk_anomaly in walk_persona(_walk_persona_dir):
                key = (walk_anomaly.file, walk_anomaly.kind)
                if key not in seen:
                    tick_anomalies.append(walk_anomaly)
                    seen.add(key)

        # First-ever tick: defer all work
        if state is None:
            # Compute pending alarms even on init tick (anomalies may exist from corrupt state)
            if self.interests_path is not None:
                persona_dir = self.interests_path.parent
            else:
                persona_dir = self.state_path.parent
            pending_alarms_count = len(compute_pending_alarms(persona_dir))

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
                        "anomalies": [a.to_dict() for a in tick_anomalies],
                        "pending_alarms_count": pending_alarms_count,
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
                anomalies=tuple(tick_anomalies),
                pending_alarms_count=pending_alarms_count,
            )

        elapsed_seconds = (now - state.last_tick_at).total_seconds()

        # Emotion decay
        memories_decayed = self._apply_emotion_decay(elapsed_seconds, dry_run=dry_run)

        # Hebbian decay + GC
        edges_pruned = self._apply_hebbian_decay_and_gc(config, elapsed_seconds, dry_run=dry_run)

        # Interest ingestion hook (zero LLM — keyword match bumps existing interests)
        interests_bumped = self._try_bump_interests(state, now, config, dry_run)

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

        # Research evaluation (after dream gate so a research memory can't seed
        # a dream in the same tick — each engine gets its own cycle).
        research_fired, research_gated_reason = self._try_fire_research(
            trigger, dry_run, config, reflex_fired
        )
        research_deferred = research_fired is None and research_gated_reason is None

        # Growth tick — autonomous self-development (Phase 2a). Runs after
        # all per-tick engines so it can observe the freshest state, before
        # the audit log writes so the audit can summarize the growth outcome.
        growth_emotions_added, growth_ran = self._try_run_growth(state, now, config, dry_run)

        # Optional HEARTBEAT: memory
        heartbeat_memory_id: str | None = None
        if not dry_run and self._should_emit_memory(
            config, dream_id, edges_pruned, memories_decayed
        ):
            heartbeat_memory_id = self._emit_heartbeat_memory(
                elapsed_seconds, memories_decayed, edges_pruned, dream_id
            )

        # Compute pending alarms (always, before writing audit log).
        if self.interests_path is not None:
            persona_dir = self.interests_path.parent
        else:
            persona_dir = self.state_path.parent
        pending_alarms_count = len(compute_pending_alarms(persona_dir))

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
                    "research": {
                        "fired": research_fired,
                        "gated_reason": research_gated_reason,
                    },
                    "interests_bumped": interests_bumped,
                    "growth": {
                        "enabled": config.growth_enabled,
                        "ran": growth_ran,
                        "emotions_added": growth_emotions_added,
                    },
                    "tick_count": state.tick_count,
                    "anomalies": [a.to_dict() for a in tick_anomalies],
                    "pending_alarms_count": pending_alarms_count,
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
            research_fired=research_fired,
            research_gated_reason=research_gated_reason,
            interests_bumped=interests_bumped,
            growth_emotions_added=growth_emotions_added,
            anomalies=tuple(tick_anomalies),
            pending_alarms_count=pending_alarms_count,
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
            persona_name=self.persona_name,
            persona_system_prompt=(
                f"You are {self.persona_name}. You just woke from a dream "
                "about interconnected memories. Reflect in first person, 2-3 "
                "sentences, starting with 'DREAM: '. Be honest and specific, "
                "not abstract."
            ),
            # lookback_hours=100000 ≈ "any conversation memory ever" — heartbeat
            # picks dream seeds by importance, not recency.
            lookback_hours=100000,
        )
        try:
            dream_result = dream_engine.run_cycle()
        except NoSeedAvailable:
            return None
        return dream_result.memory.id if dream_result.memory is not None else None

    def _try_fire_reflex(
        self, trigger: str, dry_run: bool, config: HeartbeatConfig
    ) -> tuple[tuple[str, ...], int]:
        """Run one reflex tick. Returns (fired_arc_names, skipped_count)."""
        if not config.reflex_enabled:
            return ((), 0)
        if self.reflex_arcs_path is None or self.reflex_log_path is None:
            # Heartbeat was constructed without explicit reflex paths (common
            # in unit tests that don't exercise reflex). Skip silently rather
            # than writing arc/log files to cwd.
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
            logger.warning("reflex tick raised; isolating: %.200s", exc)
            return ((), 0)
        fired = tuple(f.arc_name for f in result.arcs_fired)
        return (fired, len(result.arcs_skipped))

    def _try_bump_interests(
        self,
        state: HeartbeatState,
        now: datetime,
        config: HeartbeatConfig,
        dry_run: bool,
    ) -> int:
        """Scan conversation memories since last tick, bump pull_scores on
        keyword matches against existing interests. Zero LLM calls.
        Returns count of interests touched.
        """
        if dry_run:
            return 0
        if self.interests_path is None:
            return 0

        from brain.engines._interests import InterestSet

        interests = InterestSet.load(self.interests_path, default_path=self.default_interests_path)
        if not interests.interests:
            return 0

        all_convos = self.store.list_by_type("conversation", active_only=True, limit=50)
        recent = [m for m in all_convos if m.created_at >= state.last_tick_at]
        if not recent:
            return 0

        touched: set[str] = set()
        current = interests
        for mem in recent:
            content_lower = mem.content.lower()
            for interest in current.interests:
                if interest.topic in touched:
                    continue
                for kw in interest.related_keywords:
                    if kw.lower() in content_lower:
                        current = current.bump(
                            interest.topic,
                            amount=config.interest_bump_per_match,
                            now=now,
                        )
                        touched.add(interest.topic)
                        break

        if touched:
            current.save(self.interests_path)
        return len(touched)

    def _try_run_growth(
        self,
        state: HeartbeatState,
        now: datetime,
        config: HeartbeatConfig,
        dry_run: bool,
    ) -> tuple[int, bool]:
        """Run a growth tick if due. Returns (emotions_added, ran).

        Fault-isolated: any exception logs a warning and returns (0, False).
        Heartbeat tick continues normally — same pattern as reflex/research.
        """
        if not config.growth_enabled:
            return (0, False)
        if self.interests_path is None:
            # Use interests_path as a proxy for "persona dir is wired" — Phase 2a
            # doesn't add a separate persona_dir field.
            return (0, False)

        hours_since = (now - state.last_growth_at).total_seconds() / 3600.0
        if hours_since < config.growth_every_hours:
            return (0, False)

        persona_dir = self.interests_path.parent
        try:
            from brain.growth.scheduler import run_growth_tick

            result = run_growth_tick(persona_dir, self.store, now, dry_run=dry_run)
        except Exception as exc:
            logger.warning("growth tick raised; isolating: %.200s", exc)
            return (0, False)

        if not dry_run:
            state.last_growth_at = now

        return (result.emotions_added, True)

    def _try_fire_research(
        self,
        trigger: str,
        dry_run: bool,
        config: HeartbeatConfig,
        reflex_fired: tuple[str, ...],
    ) -> tuple[str | None, str | None]:
        """Run one research tick. Returns (fired_topic, gated_reason).

        Reflex-wins-tie: if reflex fired this tick, research is skipped with
        gated_reason='reflex_won_tie' — prevents two long outputs in one breath.
        Fault-isolated: research exceptions return (None, 'research_raised'),
        allowing the tick to continue (decay state save, audit log, etc).
        """
        if not config.research_enabled:
            return (None, None)
        if self.interests_path is None or self.research_log_path is None:
            return (None, None)
        if reflex_fired:
            return (None, "reflex_won_tie")

        from brain.engines.research import ResearchEngine

        try:
            engine = ResearchEngine(
                store=self.store,
                provider=self.provider,
                searcher=self.searcher,
                persona_name=self.persona_name,
                persona_system_prompt=self.persona_system_prompt,
                interests_path=self.interests_path,
                research_log_path=self.research_log_path,
                default_interests_path=self.default_interests_path,
                pull_threshold=6.0,
                cooldown_hours=config.research_cooldown_hours_per_interest,
            )
            result = engine.run_tick(trigger=trigger, dry_run=dry_run)
        except Exception as exc:
            logger.warning("research tick raised; isolating: %.200s", exc)
            return (None, "research_raised")

        if result.fired is not None:
            return (result.fired.topic, None)
        return (None, result.reason)

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
            f"You are {self.persona_name}. You just finished a background "
            "heartbeat cycle — decay applied, memory graph tended. Reflect in "
            "first person, one short sentence, starting with 'HEARTBEAT: '."
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
