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
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, get_args

from brain import prompt_strings
from brain.bridge.cli_throttle import ThrottleDeferred
from brain.bridge.model_tier import (
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    build_tier_provider,
)
from brain.bridge.provider import LLMProvider
from brain.dev_constants import HEARTBEAT_DECAY_BATCH_BUDGET_S
from brain.engines.daemon_state import update_daemon_state
from brain.health.alarm import compute_pending_alarms
from brain.health.anomaly import BrainAnomaly
from brain.health.walker import walk_persona
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.search.base import NoopWebSearcher, WebSearcher
from brain.utils.file_lock import file_lock
from brain.utils.time import iso_utc, parse_iso_utc

# Rows fetched per `list_active_since` page inside one decay batch (S17/S45).
# Not a dev_constants entry: unlike HEARTBEAT_DECAY_BATCH_BUDGET_S, this value
# has no behavioural effect (it only bounds how many rows are fetched ahead of
# the per-row time-budget check) — a smaller/larger page changes SQL round
# trips, not what gets decayed or when a batch commits.
_DECAY_PAGE_SIZE = 1000

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [engines.heartbeat] (issue #129 stage 2c).
_TRY_FIRE_DREAM_SYSTEM_PROMPT_SEGMENTS = prompt_strings.register_segments(
    "engines.heartbeat.try_fire_dream_system_prompt_segments"
)
_EMIT_HEARTBEAT_SYSTEM_SEGMENTS = prompt_strings.register_segments(
    "engines.heartbeat.emit_heartbeat_system_segments"
)
_EMIT_HEARTBEAT_USER_SEGMENTS = prompt_strings.register_segments(
    "engines.heartbeat.emit_heartbeat_user_segments"
)

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


@dataclass(frozen=True)
class DecayCursor:
    """Keyset resume point for an in-progress/interrupted heartbeat decay
    pass (S1/S17/S33/C30). ``tick_at`` pins the elapsed-time basis a resumed
    pass must reuse (S17): recomputing elapsed from ``now`` instead would
    re-decay already-committed rows by a second delta. ``(created_at, id)``
    is the same keyset shape ``MemoryStore.list_active_since`` uses (S33)."""

    tick_at: datetime
    created_at: str
    id: str


@dataclass
class HeartbeatState:
    """Per-persona heartbeat state. Loaded from heartbeat_state.json."""

    last_tick_at: datetime
    last_dream_at: datetime
    last_research_at: datetime
    last_growth_at: datetime  # tz-aware UTC; defaults to now on first save
    tick_count: int
    last_trigger: str
    # Non-None only while a decay pass is mid-flight/interrupted (S1/S17):
    # set after every batch commit, cleared when the pass's rows run out.
    # `last_tick_at` only advances when this is None again (pass complete).
    decay_cursor: DecayCursor | None = None

    @classmethod
    def _parse_decay_cursor(cls, data: object) -> DecayCursor | None:
        """Parse the persisted `decay_cursor` sub-object.

        Deliberately does NOT independently swallow a malformed-but-present
        cursor into a bare `None` — stage-6 red-team BLOCKER: doing so let a
        cursor corrupted in isolation (the rest of the state file intact)
        silently fall back to "no cursor" while `last_tick_at` stayed at its
        OLD (pre-interrupted-pass) value, so the next tick started a FRESH
        pass with the SAME elapsed time a completed batch had already
        applied — a reproducible silent double-decay, contradicting S1's
        "no row decayed twice" outright. `None` here is returned ONLY for the
        genuinely-absent case (key missing, or explicit JSON `null`); a
        PRESENT-but-malformed cursor instead raises, which the caller
        (`_parse_state_data`'s own try/except) turns into the WHOLE state
        being treated as corrupt — the SAME `attempt_heal`/backup-rotation
        recovery every other malformed field on this dataclass already gets
        (see `load_with_anomaly`'s docstring: reinitializing on corruption is
        this class's existing, accepted recovery philosophy). Reinitializing
        resets `last_tick_at` too, so the next tick's elapsed time is 0 for
        already-decayed rows — safe, unlike silently keeping the stale
        `last_tick_at` with a dropped cursor."""
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("decay_cursor present but not an object")
        return DecayCursor(
            tick_at=parse_iso_utc(data["tick_at"]),
            created_at=str(data["created_at"]),
            id=str(data["id"]),
        )

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
                decay_cursor=cls._parse_decay_cursor(data.get("decay_cursor")),
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
            "decay_cursor": (
                {
                    "tick_at": iso_utc(self.decay_cursor.tick_at),
                    "created_at": self.decay_cursor.created_at,
                    "id": self.decay_cursor.id,
                }
                if self.decay_cursor is not None
                else None
            ),
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
    reflex_error: str | None = None
    research_fired: str | None = None
    research_gated_reason: str | None = None
    interests_bumped: int = 0
    growth_emotions_added: int = 0
    growth_error: str | None = None
    anomalies: tuple[BrainAnomaly, ...] = ()
    pending_alarms_count: int = 0
    # Non-None only when this call did no work at all because another
    # process/thread already holds the heartbeat's cross-process guard
    # (C30) — the pass was skipped, not run and found nothing to do.
    skipped_reason: str | None = None


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

    def run_tick(
        self,
        *,
        trigger: str = "manual",
        dry_run: bool = False,
        forced_resonance: float | None = None,
    ) -> HeartbeatResult:
        """Run one heartbeat tick.

        First-ever invocation (state file missing) initializes state and
        defers all work — protects a freshly-migrated persona from eating
        'years of decay' on boot. Subsequent ticks apply decay, maybe-dream
        (gated by config.dream_every_hours), stub research, optionally emit
        a HEARTBEAT: memory, and update state atomically.
        """
        now = datetime.now(UTC)

        config, config_anomaly = HeartbeatConfig.load_with_anomaly(self.config_path)

        # Cross-process guard (C30): only one heartbeat pass — bridge
        # supervisor, `nell heartbeat` CLI, or the shutdown close tick —
        # reads/decays/writes `heartbeat_state.json` at a time. This lives
        # INSIDE run_tick itself (not a caller-side check) so every caller
        # goes through it (all of cli.py's/server.py's/supervisor.py's
        # run_tick call sites). A losing pass skips entirely (logged)
        # rather than racing the winner.
        #
        # Stage-6 red-team MAJOR, fixed: `state` is loaded from disk ONLY
        # AFTER the lock is held (never before), and this whole function
        # body — including the first-ever-tick init branch, which also
        # WRITES `heartbeat_state.json` — runs under that same lock. Loading
        # state before attempting the lock (the original shape) let a caller
        # that read a stale snapshot, then acquired the lock only after a
        # DIFFERENT process's pass had already completed and released it,
        # overwrite that process's freshly-completed state with its own
        # stale one (a TOCTOU clobber) — reading under the lock closes it.
        with file_lock(self.state_path, blocking=False) as acquired:
            if not acquired:
                logger.info(
                    "heartbeat pass skipped: another process/thread holds "
                    "the heartbeat lock (C30); will retry next tick"
                )
                return HeartbeatResult(
                    trigger=trigger,
                    elapsed_seconds=0.0,
                    memories_decayed=0,
                    edges_pruned=0,
                    dream_id=None,
                    dream_gated_reason=None,
                    research_deferred=False,
                    heartbeat_memory_id=None,
                    initialized=False,
                    anomalies=(),
                    pending_alarms_count=0,
                    skipped_reason="heartbeat_locked",
                )

            tick_anomalies: list[BrainAnomaly] = []
            if config_anomaly is not None:
                tick_anomalies.append(config_anomaly)

            state, state_anomaly = HeartbeatState.load_with_anomaly(self.state_path)
            if state_anomaly is not None:
                tick_anomalies.append(state_anomaly)

            # Cross-file walk gate: >=2 anomalies triggers a full persona scan.
            # Deduplicate by (file, kind) so files already caught in direct
            # loads are not double-counted.
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
                # Compute pending alarms even on init tick (anomalies may
                # exist from corrupt state)
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

            # Resume check (S1/S17/S46): a saved decay cursor means the
            # PREVIOUS pass didn't finish (interrupted by a failure or a
            # crash) — this tick does ONLY the resumed decay, at the SAME
            # tick_at the interrupted pass used (S17), then stops: "then
            # (same tick) nothing else — a resumed pass completes the SAME
            # tick's work; the next fresh pass happens on the following
            # timer tick" (spec §5). Dream/research/hebbian/growth for
            # *this* invocation are simply not run; they run on the next
            # (fresh) tick instead.
            if state.decay_cursor is not None:
                return self._resume_decay_only(trigger, dry_run, state, tick_anomalies)

            return self._run_tick_body(
                now, trigger, dry_run, forced_resonance, config, state, tick_anomalies
            )

    def _resume_decay_only(
        self,
        trigger: str,
        dry_run: bool,
        state: HeartbeatState,
        tick_anomalies: list[BrainAnomaly],
    ) -> HeartbeatResult:
        """Finish an interrupted decay pass, and nothing else this tick.

        Called only when `state.decay_cursor` is already set on entry
        (a prior pass committed at least one batch, then failed/crashed
        before the pass as a whole completed). Reuses that cursor's own
        `tick_at` for elapsed-time math (S17) rather than `now`, so the
        resumed rows decay by the SAME delta an uninterrupted pass would
        have applied — never a second, additional delta.
        """
        assert state.decay_cursor is not None
        tick_at = state.decay_cursor.tick_at
        elapsed_seconds = (tick_at - state.last_tick_at).total_seconds()
        memories_decayed = self._apply_emotion_decay(
            elapsed_seconds, tick_at=tick_at, state=state, dry_run=dry_run
        )

        pending_alarms_count = 0
        if not dry_run:
            persona_dir = (
                self.interests_path.parent
                if self.interests_path is not None
                else self.state_path.parent
            )
            pending_alarms_count = len(compute_pending_alarms(persona_dir))
            # `_apply_emotion_decay` above always either raises (propagating
            # out of this call, leaving `state.decay_cursor` set from the
            # last successful batch) or returns with the pass fully
            # complete (`state.decay_cursor is None`) — a normal return
            # here always means "done", so `last_tick_at` advances to the
            # cursor's own tick_at (S17), not to a fresh `now`.
            state.last_tick_at = tick_at
            state.tick_count += 1
            state.last_trigger = trigger
            state.save(self.state_path)
            self._append_log(
                {
                    "timestamp": iso_utc(datetime.now(UTC)),
                    "trigger": trigger,
                    "initialized": False,
                    "resumed_decay_only": True,
                    "elapsed_seconds": elapsed_seconds,
                    "memories_decayed": memories_decayed,
                    "tick_count": state.tick_count,
                    "anomalies": [a.to_dict() for a in tick_anomalies],
                    "pending_alarms_count": pending_alarms_count,
                }
            )

        return HeartbeatResult(
            trigger=trigger,
            elapsed_seconds=elapsed_seconds,
            memories_decayed=memories_decayed,
            edges_pruned=0,
            dream_id=None,
            dream_gated_reason="resumed_decay_only",
            research_deferred=False,
            heartbeat_memory_id=None,
            initialized=False,
            anomalies=tuple(tick_anomalies),
            pending_alarms_count=pending_alarms_count,
        )

    def _run_tick_body(
        self,
        now: datetime,
        trigger: str,
        dry_run: bool,
        forced_resonance: float | None,
        config: HeartbeatConfig,
        state: HeartbeatState,
        tick_anomalies: list[BrainAnomaly],
    ) -> HeartbeatResult:
        """A fresh (non-resumed) tick: decay, then everything else, as before
        this increment — only the decay call itself changed (batched,
        cursor'd, S17/S24/S45). Entered only when `state.decay_cursor` was
        None on entry, i.e. no earlier pass is mid-flight."""
        elapsed_seconds = (now - state.last_tick_at).total_seconds()

        # Emotion decay
        memories_decayed = self._apply_emotion_decay(
            elapsed_seconds, tick_at=now, state=state, dry_run=dry_run
        )

        # Hebbian decay + GC
        edges_pruned = self._apply_hebbian_decay_and_gc(config, elapsed_seconds, dry_run=dry_run)

        # Interest ingestion hook (zero LLM — keyword match bumps existing interests)
        interests_bumped = self._try_bump_interests(state, now, config, dry_run)

        # Resolve persona_dir once — used for daemon_state writes later.
        if self.interests_path is not None:
            persona_dir = self.interests_path.parent
        else:
            persona_dir = self.state_path.parent

        # Consolidation gate — runs FIRST (before reflex/dream/research) so each
        # idle cycle consolidates the accumulated pending-candidate queue before
        # the generative engines produce the next cycle's candidates. Fault-
        # isolated: a gate failure must not abort the tick. TEMP (Root 2 stopgap).
        if not dry_run:
            try:
                from brain.engines.consolidation import run_consolidation

                run_consolidation(
                    self.store,
                    persona_dir=persona_dir,
                    provider=build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER),
                    hebbian=self.hebbian,
                )
            except Exception:  # noqa: BLE001
                logger.exception("consolidation gate raised; continuing tick")

        # Reflex evaluation (runs before dream gate so reflex outputs can seed dreams)
        reflex_fired, reflex_skipped_count, reflex_error = self._try_fire_reflex(
            trigger, dry_run, config
        )

        # Write daemon_state for any reflex fires (fault-isolated).
        if not dry_run and reflex_fired:
            self._write_daemon_state_for_reflex(persona_dir, reflex_fired)

        # Maybe-dream
        dream_id: str | None = None
        dream_gated_reason: str | None = None
        hours_since_dream = (now - state.last_dream_at).total_seconds() / 3600.0
        if hours_since_dream >= config.dream_every_hours:
            if not dry_run:
                dream_id = self._try_fire_dream()
                if dream_id is not None:
                    state.last_dream_at = now
                    # Write daemon_state for the dream that just fired (fault-isolated).
                    self._write_daemon_state_for_dream(persona_dir, dream_id)
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

        # Write daemon_state for research fire (fault-isolated).
        if not dry_run and research_fired is not None:
            self._write_daemon_state_for_research(persona_dir, research_fired)

        # Growth tick — autonomous self-development (Phase 2a). Runs after
        # all per-tick engines so it can observe the freshest state, before
        # the audit log writes so the audit can summarize the growth outcome.
        # Passing `tick_anomalies` as the collector lets growth-tick-internal
        # anomalies (e.g., vocab corruption discovered while reading current
        # vocabulary names) surface in the audit log alongside the engine's
        # own load anomalies.
        growth_emotions_added, growth_ran, growth_error = self._try_run_growth(
            state, now, config, dry_run, anomalies_collector=tick_anomalies
        )

        # Optional HEARTBEAT: memory
        heartbeat_memory_id: str | None = None
        if not dry_run and self._should_emit_memory(
            config, dream_id, edges_pruned, memories_decayed
        ):
            try:
                heartbeat_memory_id = self._emit_heartbeat_memory(
                    elapsed_seconds, memories_decayed, edges_pruned, dream_id, persona_dir
                )
            except ThrottleDeferred as exc:
                # #246: the memory is optional; the state save below is not.
                logger.info("heartbeat memory deferred this tick: %s", exc)

        # Compute pending alarms (always, before writing audit log).
        pending_alarms_count = len(compute_pending_alarms(persona_dir))

        # Write the heartbeat's own daemon_state entry (always, last in tick,
        # so it captures the full tick summary). Fault-isolated.
        if not dry_run:
            self._write_daemon_state_for_heartbeat(
                persona_dir,
                elapsed_seconds=elapsed_seconds,
                memories_decayed=memories_decayed,
                edges_pruned=edges_pruned,
                dream_id=dream_id,
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
                        "error": reflex_error,
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
                        "error": growth_error,
                    },
                    "tick_count": state.tick_count,
                    "anomalies": [a.to_dict() for a in tick_anomalies],
                    "pending_alarms_count": pending_alarms_count,
                }
            )

            # Emotion-spike initiate emitter (Phase 4.3). Runs after the
            # heartbeat log append so a spike emit can never roll back the
            # tick's audit trail. Fully fault-isolated inside the helper.
            current_resonance, current_vector = self._compute_current_resonance(forced_resonance)
            self._maybe_emit_emotion_spike(
                persona_dir=persona_dir,
                current_resonance=current_resonance,
                current_vector=current_vector,
                tick_count=state.tick_count,
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
            reflex_error=reflex_error,
            research_fired=research_fired,
            research_gated_reason=research_gated_reason,
            interests_bumped=interests_bumped,
            growth_emotions_added=growth_emotions_added,
            growth_error=growth_error,
            anomalies=tuple(tick_anomalies),
            pending_alarms_count=pending_alarms_count,
        )

    # --- private helpers ---

    def _apply_emotion_decay(
        self,
        elapsed_seconds: float,
        *,
        tick_at: datetime,
        state: HeartbeatState,
        dry_run: bool,
    ) -> int:
        """Apply per-memory emotion decay in time-budgeted batches (S17/S24/
        S45), resuming from `state.decay_cursor` if one is already set on
        entry. Returns the total count of memories mutated across every
        batch run by THIS call.

        One `MemoryStore.update_emotions_batch` transaction per batch
        (C10(b)); `state.decay_cursor` is persisted to disk after every
        batch commit (S17/S33) so a mid-pass crash or a write failure (e.g.
        "database is locked") leaves a cursor the next tick resumes from —
        no row already committed this pass is ever decayed a second time.

        A batch ending because the time budget tripped or a page ran short
        is normal — the loop below simply starts the next batch — it does
        NOT end the pass; only rows genuinely running out (`list_active_since`
        returns fewer than `limit` and every one of them was examined) ends
        it, at which point `state.decay_cursor` is cleared. The only way this
        call returns WITHOUT clearing the cursor is by raising (propagating
        to the caller, which does not catch it — see C9(a)'s "database is
        locked" fail-first): the pass is then genuinely interrupted, and
        `state.last_tick_at` is left untouched by the caller.
        """
        from brain.emotion.decay import apply_decay
        from brain.emotion.state import EmotionalState

        if dry_run or elapsed_seconds <= 0.0:
            return 0

        total = 0
        cursor: tuple[str, str] | None = (
            (state.decay_cursor.created_at, state.decay_cursor.id)
            if state.decay_cursor is not None
            else None
        )

        while True:
            rows = self.store.list_active_since(cursor, limit=_DECAY_PAGE_SIZE)
            if not rows:
                # Nothing left after the cursor: the pass is complete.
                state.decay_cursor = None
                state.save(self.state_path)
                break

            batch_start = time.monotonic()
            changed: list[tuple[str, dict[str, float]]] = []
            examined_all_fetched = True
            for mem in rows:
                cursor = (mem.created_at.isoformat(), mem.id)
                if not mem.protected and mem.emotions:
                    emo_state = EmotionalState()
                    for name, intensity in mem.emotions.items():
                        try:
                            emo_state.set(name, float(intensity))
                        except (KeyError, ValueError):
                            continue
                    apply_decay(emo_state, elapsed_seconds)
                    new_emotions = {
                        name: val for name, val in emo_state.emotions.items() if val > 0.0
                    }
                    # Write only rows whose values actually change (S18);
                    # unchanged/no-emotion/protected rows advance the
                    # cursor above without joining this batch's UPDATEs.
                    if new_emotions != mem.emotions:
                        changed.append((mem.id, new_emotions))
                # Budget checked after each row (S45), not each batch.
                if time.monotonic() - batch_start >= HEARTBEAT_DECAY_BATCH_BUDGET_S:
                    examined_all_fetched = mem is rows[-1]
                    break

            if changed:
                self.store.update_emotions_batch(changed)
                total += len(changed)

            # Persist the resume point after every batch commit (S17/S33),
            # even if this pass turns out to finish immediately after —
            # a crash between here and the next iteration's fetch still
            # resumes correctly (cursor already reflects everything just
            # written).
            state.decay_cursor = DecayCursor(
                tick_at=tick_at, created_at=cursor[0], id=cursor[1]
            )
            state.save(self.state_path)

            if len(rows) < _DECAY_PAGE_SIZE and examined_all_fetched:
                # Fewer rows than a full page came back AND we reached the
                # last one — no more work exists after this cursor.
                state.decay_cursor = None
                state.save(self.state_path)
                break

        return total

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
        """Run one DreamEngine cycle; return the new dream memory id or None.

        dream is `background-generative` tier (routed through the model-tier
        accessor, #154) — same model (`MODEL_MEDIUM`) `self.provider` already
        resolves to today, so this is a routing-only change, not a model
        change. Same applies to `_try_fire_reflex` and `_try_fire_research`
        below. See `brain/bridge/model_tier.py`'s module docstring."""
        # Lazy import to avoid module-level circular dependency with dream.py
        from brain.engines.dream import DreamEngine, NoSeedAvailable
        from brain.soul.store import SoulStore

        persona_dir = self.dream_log_path.parent
        soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
        dream_engine = DreamEngine(
            store=self.store,
            hebbian=self.hebbian,
            provider=build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
            log_path=self.dream_log_path,
            persona_dir=persona_dir,
            persona_name=self.persona_name,
            persona_system_prompt=(
                _TRY_FIRE_DREAM_SYSTEM_PROMPT_SEGMENTS[0]
                + self.persona_name
                + _TRY_FIRE_DREAM_SYSTEM_PROMPT_SEGMENTS[1]
            ),
            # lookback_hours=100000 ≈ "any conversation memory ever" — heartbeat
            # picks dream seeds by importance, not recency.
            lookback_hours=100000,
            soul_store=soul_store,
        )
        try:
            dream_result = dream_engine.run_cycle()
        except NoSeedAvailable:
            return None
        except ThrottleDeferred as exc:
            # #246: provider login expired (or slot denied) — the dream is skipped
            # quietly and, crucially, the tick still reaches state.save below.
            logger.info("dream deferred this tick: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            # #248: a failed dream must not abort the tick — an overdue dream is
            # retried every tick, and each abort skipped state.save so emotion
            # decay was re-applied over the whole growing window next tick.
            logger.warning("dream failed this tick; tick continues: %.200s", exc)
            logger.debug("dream failure traceback", exc_info=True)
            return None
        finally:
            soul_store.close()
        return dream_result.memory.id if dream_result.memory is not None else None

    def _try_fire_reflex(
        self, trigger: str, dry_run: bool, config: HeartbeatConfig
    ) -> tuple[tuple[str, ...], int, str | None]:
        """Run one reflex tick. Returns (fired_arc_names, skipped_count, error_reason)."""
        if not config.reflex_enabled:
            return ((), 0, None)
        if self.reflex_arcs_path is None or self.reflex_log_path is None:
            # Heartbeat was constructed without explicit reflex paths (common
            # in unit tests that don't exercise reflex). Skip silently rather
            # than writing arc/log files to cwd.
            return ((), 0, None)
        from brain.engines.reflex import ReflexEngine

        # Non-None past the guard above — safe to derive persona_dir here.
        persona_dir = self.reflex_arcs_path.parent
        engine = ReflexEngine(
            store=self.store,
            provider=build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
            persona_name=self.persona_name,
            persona_system_prompt=self.persona_system_prompt,
            arcs_path=self.reflex_arcs_path,
            log_path=self.reflex_log_path,
            default_arcs_path=self.reflex_default_arcs_path,
        )
        try:
            result = engine.run_tick(trigger=trigger, dry_run=dry_run)
        except ThrottleDeferred as exc:
            # #246: a deferred provider is not an engine crash — no reflex_error,
            # no WARNING; the success shape so the audit row reads "nothing fired".
            logger.info("reflex tick deferred: %s", exc)
            return ((), 0, None)
        except Exception as exc:
            # Fault-isolate reflex failures from the heartbeat tick per spec §7:
            # a misbehaving arc/provider must not abort decay, dream-gate, or
            # audit-log writes that follow. The exception is logged AND
            # surfaced through HeartbeatResult.reflex_error + the audit
            # JSON so operators tailing heartbeats.log.jsonl can tell
            # "nothing fired" apart from "the engine crashed."
            logger.warning("reflex tick raised; isolating: %.200s", exc)
            return ((), 0, f"{type(exc).__name__}: {exc}"[:200])
        fired = tuple(f.arc_name for f in result.arcs_fired)
        return (fired, len(result.arcs_skipped), None)

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
        from brain.utils.memory import list_conversation_memories

        interests = InterestSet.load(self.interests_path, default_path=self.default_interests_path)
        if not interests.interests:
            return 0

        all_convos = list_conversation_memories(self.store, active_only=True, limit=50)
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
        anomalies_collector: list[BrainAnomaly] | None = None,
    ) -> tuple[int, bool, str | None]:
        """Run a growth tick if due. Returns (emotions_added, ran, error_reason).

        Fault-isolated: any exception logs a warning and returns
        (0, False, "<class>: <msg>"). Heartbeat tick continues normally
        — same pattern as reflex/research. The error reason surfaces to
        HeartbeatResult.growth_error + the audit JSON so operators can
        tell "didn't run" apart from "crashed."

        `anomalies_collector` is forwarded to `run_growth_tick` so any
        anomaly produced inside growth (e.g., vocabulary file corruption
        detected by `_read_current_vocabulary_names`) surfaces in the
        heartbeat tick's audit log alongside engine-level anomalies.
        """
        if not config.growth_enabled:
            return (0, False, None)
        if self.interests_path is None:
            # Use interests_path as a proxy for "persona dir is wired" — Phase 2a
            # doesn't add a separate persona_dir field.
            return (0, False, None)

        hours_since = (now - state.last_growth_at).total_seconds() / 3600.0
        if hours_since < config.growth_every_hours:
            return (0, False, None)

        persona_dir = self.interests_path.parent
        try:
            from brain.growth.scheduler import run_growth_tick

            result = run_growth_tick(
                persona_dir,
                self.store,
                now,
                dry_run=dry_run,
                anomalies_collector=anomalies_collector,
            )
        except Exception as exc:
            logger.warning("growth tick raised; isolating: %.200s", exc)
            return (0, False, f"{type(exc).__name__}: {exc}"[:200])

        if not dry_run:
            state.last_growth_at = now

        return (result.emotions_added, True, None)

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

        # Non-None past the guard above — safe to derive persona_dir here.
        persona_dir = self.interests_path.parent
        try:
            engine = ResearchEngine(
                store=self.store,
                provider=build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
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
        except ThrottleDeferred as exc:
            logger.info("research tick deferred: %s", exc)  # #246
            return (None, "auth_deferred")
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
        persona_dir: Path,
    ) -> str:
        """Generate and persist a HEARTBEAT: memory via the LLM provider.

        `background-housekeeping` tier (Haiku) — a newly-decided tier
        assignment (#154), NOT a routing-only change: this used to reuse
        `self.provider` (Sonnet). A HEARTBEAT: memory is a cheap, templated
        one-liner (elapsed/decay/dream-fired stats), closer in kind to the
        other housekeeping ticks (session snapshot/finalize) than to the
        persona-voice generative sites (dream/reflex/research).
        """
        from brain.memory.store import Memory

        provider = build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING)
        sys_seg = _EMIT_HEARTBEAT_SYSTEM_SEGMENTS
        system = sys_seg[0] + self.persona_name + sys_seg[1]
        user_seg = _EMIT_HEARTBEAT_USER_SEGMENTS
        user = (
            user_seg[0] + f"{elapsed_seconds / 3600:.1f}" + user_seg[1]
            + str(memories_decayed) + user_seg[2] + str(edges_pruned) + user_seg[3]
            + ("yes" if dream_id else "no")
        )
        raw = provider.generate(user, system=system)
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
                "provider": provider.name(),
            },
        )
        from brain.memory.pending import route_write

        route_write(self.store, mem, source="heartbeat")
        return mem.id

    # --- daemon_state write helpers (all fault-isolated) ---

    def _write_daemon_state_for_dream(self, persona_dir: Path, dream_id: str) -> None:
        """Look up the just-fired dream memory and write a daemon_state fire entry.

        Peak emotion + intensity come from the dream memory's emotions dict.
        Falls back to 'curiosity'/'5' when the dict is missing or empty.
        """
        try:
            # The just-fired dream is a GATED candidate in the pending queue,
            # not a memories.db row yet — read it there (fall back to the store
            # in case it was promoted/bypassed). TEMP (Root 2 stopgap).
            from brain.memory.pending import PendingQueue

            mem = next(
                (
                    c
                    for c in PendingQueue(self.store.persona_dir).read_recent("dream", limit=10)
                    if c.id == dream_id
                ),
                None,
            )
            if mem is None:
                mem = self.store.get(dream_id)
            if mem is None:
                return
            dominant_emotion, intensity = self._peak_emotion(mem.emotions)
            theme = mem.content[:80]
            update_daemon_state(
                persona_dir,
                daemon_type="dream",
                dominant_emotion=dominant_emotion,
                intensity=intensity,
                theme=theme,
                summary=mem.content,
            )
        except Exception as exc:
            logger.warning("daemon_state write for dream failed; isolating: %.200s", exc)

    def _write_daemon_state_for_reflex(
        self, persona_dir: Path, fired_arcs: tuple[str, ...]
    ) -> None:
        """Write a daemon_state fire entry for the first fired reflex arc.

        Looks up the most-recently-created reflex memory (by store.list_by_type,
        limit=1, newest-first). Falls back to a compact synthetic summary if
        no reflex memory is found.
        """
        try:
            arc_name = fired_arcs[0] if fired_arcs else "reflex"
            # reflex_journal is a GATED type — the just-fired reflex memory is in
            # the pending queue, not memories.db. Read it there (fall back to the
            # store). TEMP (Root 2 stopgap).
            from brain.memory.pending import PendingQueue

            reflex_mems = PendingQueue(self.store.persona_dir).read_recent("reflex_journal", limit=1)
            if not reflex_mems:
                reflex_mems = self.store.list_by_type(
                    "reflex_journal", active_only=True, limit=1
                )
            if reflex_mems:
                mem = reflex_mems[0]
                dominant_emotion, intensity = self._peak_emotion(mem.emotions)
                theme = arc_name
                summary = mem.content
            else:
                # No memory found — write a minimal synthetic entry so the
                # daemon_state still records that reflex fired this tick.
                dominant_emotion = "curiosity"
                intensity = 5
                theme = arc_name
                summary = "reflex fired"
            update_daemon_state(
                persona_dir,
                daemon_type="reflex",
                dominant_emotion=dominant_emotion,
                intensity=intensity,
                theme=theme,
                summary=summary,
                trigger=arc_name,
            )
        except Exception as exc:
            logger.warning("daemon_state write for reflex failed; isolating: %.200s", exc)

    def _write_daemon_state_for_research(self, persona_dir: Path, research_topic: str) -> None:
        """Write a daemon_state fire entry for a research fire.

        Looks up the most-recently-created research memory (limit=1). Falls back
        to a synthetic entry keyed on the topic string if none is found.
        """
        try:
            # research is a GATED type — the just-fired research memory is in the
            # pending queue, not memories.db. Read it there (fall back to store).
            # TEMP (Root 2 stopgap).
            from brain.memory.pending import PendingQueue

            research_mems = PendingQueue(self.store.persona_dir).read_recent("research", limit=1)
            if not research_mems:
                research_mems = self.store.list_by_type("research", active_only=True, limit=1)
            if research_mems:
                mem = research_mems[0]
                dominant_emotion, intensity = self._peak_emotion(mem.emotions)
                theme = research_topic
                summary = mem.content
            else:
                dominant_emotion = "curiosity"
                intensity = 5
                theme = research_topic
                summary = f"research fired: {research_topic}"
            update_daemon_state(
                persona_dir,
                daemon_type="research",
                dominant_emotion=dominant_emotion,
                intensity=intensity,
                theme=theme,
                summary=summary,
            )
        except Exception as exc:
            logger.warning("daemon_state write for research failed; isolating: %.200s", exc)

    def _write_daemon_state_for_heartbeat(
        self,
        persona_dir: Path,
        *,
        elapsed_seconds: float,
        memories_decayed: int,
        edges_pruned: int,
        dream_id: str | None,
    ) -> None:
        """Write the heartbeat tick's own daemon_state entry.

        dominant_emotion + intensity come from the aggregate emotional state
        across all active conversation memories. Falls back to 'calm'/'3' when
        no conversation memories exist.
        """
        try:
            all_mems = self.store.list_active()
            combined_emotions: dict[str, float] = {}
            for m in all_mems:
                for name, val in m.emotions.items():
                    combined_emotions[name] = combined_emotions.get(name, 0.0) + val
            dominant_emotion, intensity = self._peak_emotion(combined_emotions, default=("calm", 3))
            theme = "heartbeat tick"
            summary = (
                f"elapsed={elapsed_seconds / 3600:.1f}h, "
                f"decays={memories_decayed}, edges_pruned={edges_pruned}, "
                f"dream_fired={'yes' if dream_id else 'no'}"
            )
            update_daemon_state(
                persona_dir,
                daemon_type="heartbeat",
                dominant_emotion=dominant_emotion,
                intensity=intensity,
                theme=theme,
                summary=summary,
            )
        except Exception as exc:
            logger.warning("daemon_state write for heartbeat failed; isolating: %.200s", exc)

    @staticmethod
    def _peak_emotion(
        emotions: dict[str, float],
        *,
        default: tuple[str, int] = ("curiosity", 5),
    ) -> tuple[str, int]:
        """Return (dominant_emotion, intensity_0_to_10) from an emotions dict.

        intensity is clamped to [0, 10] and rounded to the nearest integer.
        Falls back to `default` when the dict is empty.
        """
        if not emotions:
            return default
        name, val = max(emotions.items(), key=lambda kv: kv[1])
        return name, max(0, min(10, round(val)))

    # --- emotion-spike initiate emitter (Phase 4.3) ---

    def _compute_current_resonance(
        self, forced_resonance: float | None
    ) -> tuple[float, dict[str, float]]:
        """Return (resonance, aggregate_emotion_vector) for the current tick.

        Resonance is the peak intensity across all active memories' emotion
        dicts — same scalar used elsewhere as "dominant emotion intensity".
        When `forced_resonance` is given (test-only injection), the vector is
        a single-key dict so the snapshot still carries structured data.
        """
        if forced_resonance is not None:
            return float(forced_resonance), {"forced": float(forced_resonance)}

        combined: dict[str, float] = {}
        for mem in self.store.list_active():
            for name, val in mem.emotions.items():
                combined[name] = combined.get(name, 0.0) + float(val)
        if not combined:
            return 0.0, {}
        # Peak intensity — clamped to [0, 10] to match the project's
        # canonical 0..10 emotion scale (see _peak_emotion).
        peak = max(combined.values())
        peak = max(0.0, min(10.0, peak))
        return peak, combined

    def _update_rolling_baseline(self, current_resonance: float) -> tuple[float, float, float]:
        """Update the in-memory rolling-baseline window. Returns (mean, stdev, delta_sigma).

        Window: last 24 ticks (~6h at default cadence). Below 5 ticks we
        return zeros — no emission during warm-up. State is held in
        `self._resonance_window`, initialized lazily on first call so the
        dataclass definition stays unchanged.
        """
        import statistics

        if not hasattr(self, "_resonance_window"):
            self._resonance_window: list[float] = []
        self._resonance_window.append(current_resonance)
        self._resonance_window = self._resonance_window[-24:]
        if len(self._resonance_window) < 5:
            return 0.0, 0.0, 0.0
        mean = statistics.mean(self._resonance_window)
        stdev = statistics.pstdev(self._resonance_window) or 1.0
        delta_sigma = (current_resonance - mean) / stdev
        return mean, stdev, delta_sigma

    def _maybe_emit_emotion_spike(
        self,
        *,
        persona_dir: Path,
        current_resonance: float,
        current_vector: dict[str, float],
        tick_count: int,
    ) -> None:
        """If delta_sigma >= 1.5, emit an initiate candidate sourced from this spike.

        Fault-isolated: any failure (disk error, schema mismatch) is logged
        and swallowed so the heartbeat tick still completes normally.
        Delta-from-baseline is the architectural guard against the
        always-elevated-emotions firehose.
        """
        mean, stdev, delta_sigma = self._update_rolling_baseline(current_resonance)
        if delta_sigma >= 1.5:
            try:
                # Local import to keep initiate deps out of engines that don't
                # need them and to mirror the dream/crystallizer emit pattern.
                from brain.initiate.emit import emit_initiate_candidate
                from brain.initiate.schemas import EmotionalSnapshot, SemanticContext

                emit_initiate_candidate(
                    persona_dir,
                    kind="message",
                    source="emotion_spike",
                    source_id=f"emotion_{tick_count}",
                    emotional_snapshot=EmotionalSnapshot(
                        vector=dict(current_vector),
                        rolling_baseline_mean=mean,
                        rolling_baseline_stdev=stdev,
                        current_resonance=current_resonance,
                        delta_sigma=delta_sigma,
                    ),
                    semantic_context=SemanticContext(),
                )
            except Exception as exc:
                logger.warning("emotion spike initiate emit failed: %s", exc)
            return

        # Sub-initiate band: 0.5 <= delta_sigma < 1.5 lands in the draft space
        # rather than escalating to an outbound initiate. Quiet enough to note,
        # not loud enough to reach for Hana. (Phase 8.2.)
        if delta_sigma >= 0.5:
            try:
                from datetime import UTC, datetime

                from brain.initiate.draft import (
                    append_draft_fragment,
                    compose_draft_fragment,
                )

                # `background-housekeeping` tier (Haiku) — a newly-decided tier
                # assignment (#154), NOT routing-only: a quiet, observational
                # draft-space fragment is closer in kind to the housekeeping
                # ticks than to persona-voice generative content.
                body = compose_draft_fragment(
                    build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING),
                    source="emotion_spike",
                    source_id=f"emotion_{tick_count}",
                    linked_memory_excerpts=[],
                )
                append_draft_fragment(
                    persona_dir,
                    timestamp=datetime.now(UTC).isoformat(),
                    source="emotion_spike",
                    body=body,
                )
            except Exception as exc:
                logger.warning("emotion-spike draft fallback failed: %s", exc)

    def _append_log(self, entry: dict) -> None:
        """Append one JSON line to heartbeats.log.jsonl."""
        with self.heartbeat_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
