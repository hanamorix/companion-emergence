"""SP-7 supervisor thread — folded as non-daemon thread inside the bridge.

run_folded() is a synchronous loop running two cadences:

  * every ``tick_interval_s`` (default 60s): close stale sessions
  * every ``heartbeat_interval_s`` (default 900s = 15min): fire a
    heartbeat tick — memory decay, dream gate, reflex, growth, research

The heartbeat cadence keeps the autonomous brain alive while the user
is away. Without it, dreams / reflex / growth / research only fire when
the user manually runs ``nell heartbeat``, and memory decay never runs
in the background. The heartbeat engine has its own internal cadence
gates (e.g. memory decay every N hours, dreams every M hours) so
calling ``run_tick`` every 15 min is mostly cheap — only the
gate-passes do real work.

Lives in a separate thread so the async server stays responsive; uses
event_bus.publish (thread-safe) to fan out events to /events subscribers.

Non-daemon thread on purpose — SIGTERM must wait for the loop to exit
before process exit, so we don't kill mid-ingest or mid-heartbeat.

H-A hardening (2026-04-28): supervisor opens its OWN per-tick stores
inside its thread. Previously took store/hebbian/embeddings as kwargs,
which meant SQLite handles created on the main asyncio thread were used
from the supervisor thread — sqlite3 default mode raises ProgrammingError
on cross-thread use. Per-tick open/close means clean thread-local
ownership and no leaked connections.

OG reference: NellBrain/nell_supervisor.py:368-407 (run_folded).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.engines.heartbeat import HeartbeatResult
    from brain.memory.relevance_judge import RelevanceJudgeProvider

from brain import prompt_strings
from brain.attunement.backfill import (
    run_backfill as _attunement_run_backfill,
)
from brain.attunement.backfill import (
    run_supplementary_backfill as _attunement_run_supplementary_backfill,
)
from brain.attunement.backfill import (
    should_run_backfill as _attunement_should_run_backfill,
)
from brain.attunement.backfill import (
    should_run_supplementary_backfill as _attunement_should_run_supplementary_backfill,
)
from brain.bridge import central_cadence, cli_throttle, persisted_cadence
from brain.bridge.central_cadence import GatedJob, JobOutcome
from brain.bridge.events import EventBus
from brain.bridge.model_tier import (
    TIER_BACKGROUND_CLASSIFIER,
    TIER_BACKGROUND_GENERATIVE,
    TIER_BACKGROUND_HOUSEKEEPING,
    build_tier_provider,
)
from brain.bridge.provider import LLMProvider
from brain.chat import pass2_queue
from brain.chat.session import (
    has_prunable_empty_sessions,
    prune_empty_sessions,
    remove_session,
)
from brain.engines import interest_sweep
from brain.felt_time import FeltTime, TickContext
from brain.felt_time.lived_age import IntensityDrivers
from brain.forgetting import run_pass as forgetting_run_pass
from brain.health.log_rotation import (
    rotate_age_archive_yearly,
    rotate_rolling_size,
)
from brain.health.soul_candidate_repair import (
    run_soul_candidate_repair as _soul_candidate_repair_run,
)
from brain.health.soul_candidate_repair import (
    should_run_soul_candidate_repair as _soul_candidate_repair_should_run,
)
from brain.health.vocab_repair import (
    run_vocab_repair as _vocab_repair_run,
)
from brain.health.vocab_repair import (
    should_run_vocab_repair as _vocab_repair_should_run,
)
from brain.ingest.emotion_backfill import (
    has_emotion_backfill_work as _emotion_backfill_has_work,
)
from brain.ingest.emotion_backfill import (
    run_emotion_backfill as _emotion_backfill_run,
)
from brain.ingest.pipeline import (
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
from brain.initiate.review import _rest_state_from_energy, run_initiate_review_tick
from brain.initiate.user_pattern import compute_user_presence
from brain.memory.embedding_backfill import (
    delete_legacy_embeddings_db as _delete_legacy_embeddings_db,
)
from brain.memory.embedding_backfill import (
    has_embedding_backfill_work as _embedding_backfill_has_work_probe,
)
from brain.memory.embedding_backfill import (
    run_embedding_backfill_tick as _embedding_backfill_run_tick,
)
from brain.memory.embedding_matrix import build_embedding_matrix
from brain.memory.hebbian import HebbianMatrix
from brain.memory.judge_selftune import (
    JUDGE_TUNE_CADENCE_FILE,
    JUDGE_TUNE_INTERVAL_HOURS,
    _run_judge_selftune_tick,
)
from brain.memory.store import MemoryStore
from brain.narrative_memory import run_pass as narrative_memory_run_pass
from brain.persona_config import PersonaConfig
from brain.self_model import cadence as self_model_cadence
from brain.self_model import reconcile as sm_reconcile
from brain.self_model import state as self_model_state
from brain.self_model.articulate import articulate as sm_articulate
from brain.self_model.articulate import log_self_model_deferred
from brain.self_model.derived import compute_baseline, compute_derived
from brain.self_model.gap import compute_gap
from brain.self_model.resolve import (
    check_and_emit_resolution,
    increment_gaps_surfaced,
)
from brain.soul import cadence as soul_cadence

logger = logging.getLogger(__name__)

# Text externalized to prompt_strings.toml [bridge.supervisor] (issue #129 stage 2b).
_HEARTBEAT_TICK_SYSTEM_PROMPT_SEGMENTS = prompt_strings.register_segments(
    "bridge.supervisor.heartbeat_tick_system_prompt_segments"
)

# Backlog-aware soul-review drain: when candidates have piled up (e.g. after a
# model-call outage), clear up to this many per tick instead of the default 5,
# so a backlog clears in a couple of ticks rather than days.
_SOUL_BACKLOG_DRAIN_CAP = 25



def run_folded(
    stop_event: threading.Event,
    *,
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    tick_interval_s: float = 60.0,
    heartbeat_interval_s: float | None = 900.0,
    soul_review_interval_s: float | None = 6 * 3600.0,
    finalize_after_hours: float = 24.0,
    finalize_interval_s: float | None = 3600.0,
    log_rotation_interval_s: float | None = 3600.0,
    initiate_review_interval_s: float | None = 900.0,
    voice_reflection_interval_s: float | None = 86400.0,
    self_model_interval_s: float | None = 0.0,
    maker_enabled: bool = True,
    notes_enabled: bool = True,
    kindled_link_enabled: bool = True,
    compaction_interval_s: float | None = 86400.0,
    calibration_interval_s: float | None = 86400.0,
    interest_sweep_interval_s: float | None = interest_sweep.SWEEP_INTERVAL_HOURS * 3600.0,
    judge_selftune_interval_s: float | None = JUDGE_TUNE_INTERVAL_HOURS * 3600.0,
    clustering_interval_s: float | None = 6 * 3600.0,
    vocab_repair_interval_s: float | None = 6 * 3600.0,
    is_session_busy: Callable[[str], bool] | None = None,
    bridge_started_at: datetime | None = None,
) -> None:
    """Run supervisor + heartbeat + soul-review + finalize cadences until stop_event is set.

    Stores are opened per-tick inside this thread; never crosses thread
    boundaries with the asyncio main loop.

    ``heartbeat_interval_s=None`` disables the autonomous heartbeat
    cadence (used in tests + by callers that drive heartbeats
    externally). Default 900s (15 min) — heartbeat engine has internal
    gates so frequent ticks are mostly cheap.

    ``soul_review_interval_s=None`` disables the autonomous soul-review
    cadence. Default 6 hours — review_pending_candidates makes one LLM
    call per candidate (capped at 5/pass), so 6h pacing keeps the cost
    bounded while ensuring candidates don't sit forever waiting for the
    user to discover ``nell soul review``. The user-surface principle
    says soul review is physiology, not a CLI knob.

    ``finalize_interval_s=None`` disables the autonomous finalize
    cadence. Default 3600s (1 hour) with a 24h silence threshold — the
    sweep cadence is non-destructive (Task 3); finalize is the only path
    that deletes buffers + cursors and evicts from _SESSIONS, so it
    paces hourly because the threshold is days, not minutes.

    ``clustering_interval_s=None`` disables the autonomous memory-vector
    clustering cadence (Stage 5, #157, local semantic-retrieval build).
    Default 6h — same value as ``soul_review_interval_s``/the maintenance
    cadence, but its OWN persisted state file (mirrors the log-rotation/
    finalize/maintenance decoupling pattern: a slow, independent, idle-cadence
    batch pass, not chained to any other cadence's pacing). 6h is a
    documented default, not spec-pinned — the spec/plan intentionally leave
    cadence choice to the build (see ``hunts/semantic-retrieval/plan.md``
    Stage 5); chosen to match the other "whole-corpus batch pass" cadences in
    this file rather than the much tighter embedding-backfill tick (which
    chips a small per-row batch every base tick — clustering instead
    recomputes over the WHOLE cached vector set each firing, so it doesn't
    need or want that tight a cadence).

    ``calibration_interval_s=None`` disables the autonomous daily calibration
    cadence (F2a #250, spec Section 5) — a 4th sibling to
    ``compaction_interval_s``, mirroring its idle-gate + restart-safety shape
    (own persisted ``calibration_cadence.json``, daily fire at a lull via the
    central cadence function — no startup catch-up since ram-spike-fix INC-9).
    Default 86400s (daily), matching compaction — same
    rationale: rides existing, already-idle-gated infra. This increment
    (inc5) scopes the tick to retention pruning of ``calibration_log`` only
    (acceptance 5b); the judge-labeling pass (Section 6) and floor derivation
    (Section 7) are later increments — see ``_run_calibration_tick``'s
    docstring for the full scope note.

    ``judge_selftune_interval_s=None`` disables the autonomous weekly judge
    self-tune cadence (F2c inc2, spec Section 2) — wired structurally
    identically to ``interest_sweep_interval_s`` immediately above (own
    persisted ``judge_selftune_cadence.json``, own fault-isolated tick, NO
    startup catch-up). Default matches
    ``brain.memory.judge_selftune.JUDGE_TUNE_INTERVAL_HOURS`` (168h/weekly).
    This increment (inc2) is a SCAFFOLD ONLY: the cadence, the >handful
    gate, runtime RAM tier-detection and the cgroup-aware OOM-safety
    downgrade are built and wired; the actual tuning (knob-refit / LoRA /
    full fine-tune) is a later increment — see
    ``judge_selftune._run_judge_selftune_tick``'s docstring for the scope
    note.
    """
    logger.info(
        "supervisor folded persona=%s tick=%.2fs heartbeat=%s soul_review=%s finalize=%s",
        persona_dir.name,
        tick_interval_s,
        f"{heartbeat_interval_s:.0f}s" if heartbeat_interval_s is not None else "disabled",
        f"{soul_review_interval_s:.0f}s" if soul_review_interval_s is not None else "disabled",
        f"{finalize_interval_s:.0f}s" if finalize_interval_s is not None else "disabled",
    )
    # S84: "bridge start" for the empty-session prune's idle window when no
    # message has been seen in this process (see _build_gated_jobs). The
    # bridge passes the moment it captured BEFORE starting this thread (and
    # so before it serves /session/new): capturing it here, inside the
    # thread, would race the app-mount session (stage-6 S84 MAJOR). Every
    # production caller must pass it (today: server.py's lifespan only);
    # callers that pass nothing (tests, direct calls) get "now".
    if bridge_started_at is None:
        bridge_started_at = datetime.now(UTC)
    last_heartbeat_at = time.monotonic() if heartbeat_interval_s is not None else None
    # Non-gated cadences keep their own timing (S16): soul review (own
    # self-pacing cadence), log rotation, vocab repair, voice reflection.
    soul_cadence_state = (
        soul_cadence.load_cadence_state(persona_dir)
        if soul_review_interval_s is not None
        else None
    )
    _last_intensity_drivers: IntensityDrivers | None = None
    log_rotation_cadence_state = (
        persisted_cadence.load_cadence(persona_dir, "log_rotation_cadence.json")
        if log_rotation_interval_s is not None
        else None
    )
    voice_cadence_state = (
        persisted_cadence.load_cadence(persona_dir, "voice_reflection_cadence.json")
        if voice_reflection_interval_s is not None
        else None
    )
    vocab_repair_cadence_state = (
        persisted_cadence.load_cadence(persona_dir, "vocab_repair_cadence.json")
        if vocab_repair_interval_s is not None
        else None
    )

    # One-shot startup: run the attunement backfill if this is a first-launch
    # (≥10 user turns + no completed backfill_state.json). Wrapped in
    # try/except so a misbehaving backfill never crashes supervisor startup —
    # autonomous-behaviour recipe item 3 (defer cleanly, don't fail loudly).
    #
    # The two branches are mutually exclusive:
    #   - Fresh install → full backfill (should_run_backfill=True only when no
    #     completed state exists, so supplementary can never also fire)
    #   - Upgraded install → supplementary pass for new categories only
    #     (completed state exists at an older schema version)
    try:
        if _attunement_should_run_backfill(persona_dir):
            _attunement_run_backfill(persona_dir)
        elif _attunement_should_run_supplementary_backfill(persona_dir):
            _attunement_run_supplementary_backfill(persona_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("attunement backfill failed during startup: %s", exc)

    # ram-spike-fix INC-9 (S23/S34, C26): NO startup catch-up of any gated
    # job. The former one-shots here — catch-up compaction, the catch-up
    # calibration tick, the deploy-time floor recalibration and the emotion
    # backfill — are gated jobs of the central cadence function now; the
    # bridge starting is simply its first pass (a lull when chat is idle), so
    # an overdue job runs there and a not-due one does not. The one-shots
    # left below are not gated jobs (vocab repair keeps its own timing, S16;
    # the rest are provider-free repairs).

    # One-shot startup: repair already-stubbed emotion_vocabulary.json entries.
    # Step 1 bumps decay_half_life_days from the bad 1.0 → 14.0 (sync,
    # provider-free, so it always lands). Step 2 re-derives descriptions via
    # Haiku (fail-soft — placeholders kept if provider unavailable/fails).
    # Runs adjacent to emotion backfill; independent of it (separate try/except).
    try:
        _run_vocab_repair_tick(persona_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vocab repair failed during startup: %s", exc)

    # One-shot startup: repair stuck monologue soul candidates — context-free
    # fragments ("Ordinary trust") whose evidence sits in an unread memory.
    # Provider-free: backfills text from the existing placeholder memory or
    # expires the candidate. Adjacent to vocab_repair; independent try/except.
    try:
        if _soul_candidate_repair_should_run(persona_dir):
            from brain.memory.store import MemoryStore as _MemoryStore

            db_path = persona_dir / "memories.db"
            _store = _MemoryStore(str(db_path), integrity_check=False)
            try:
                _soul_candidate_repair_run(persona_dir, store=_store)
            finally:
                _store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("soul candidate repair failed during startup: %s", exc)

    # One-shot startup: clear a persisted self-model gap left by the pre-fix
    # (v0.0.36 total-mass-mean) derived read — the magnitude-354 artifact. The
    # next reflection tick recomputes honestly under the windowed-peak read; the
    # reset prevents a false "natural resolution" from the collapsing artifact.
    # Provider-free, state-file only; independent try/except.
    try:
        from brain.health.self_model_repair import (
            run_self_model_repair as _self_model_repair_run,
        )
        from brain.health.self_model_repair import (
            should_run_self_model_repair as _self_model_repair_should_run,
        )

        if _self_model_repair_should_run(persona_dir):
            _self_model_repair_run(persona_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("self-model repair failed during startup: %s", exc)

    # One-shot startup: run-once, fail-safe deletion of the legacy
    # embeddings.db file once every active row carries a current-model
    # embedding (F1 #259 increment 9, spec §5 / S9, invariant I9). No code
    # reads or writes embeddings.db anymore (increment 8's teardown) — on an
    # already-deployed persona it can only exist as a stale orphan left over
    # from before this column migration ran. Gated on the file's own
    # existence BEFORE opening a store — same "should_run" shape as every
    # other one-shot above, and it means a persona that has already
    # completed this migration (the overwhelming steady-state case once
    # this ships) never pays for a MemoryStore open here at all. Opens its
    # own short-lived MemoryStore handle (mirrors soul-candidate-repair
    # above) rather than reusing the main loop's per-tick store, since this
    # runs once before that loop starts. Fault-isolated (recipe item 3): the
    # function itself already catches delete errors per-file, and this
    # try/except is the outer safety net (e.g. a provider-load failure) so
    # a stuck stale file can never crash the bridge — worst case, it is
    # just re-checked (and left in place, or deleted) on the next startup.
    try:
        if (persona_dir / "embeddings.db").exists():
            _store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
            try:
                _delete_legacy_embeddings_db(persona_dir, _store)
            finally:
                _store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy embeddings.db deletion check failed during startup: %s", exc)

    def _maybe_run_heartbeat() -> None:
        """The heartbeat's own 15-minute timer (S16/S21) — not a gated job.
        Called at the top of every loop pass AND by the central cadence
        function between gated jobs, so a heartbeat that falls due while a
        job runs waits only for that job to return, and the next job waits
        for the whole heartbeat pass (single supervisor thread, S41/S65)."""
        nonlocal last_heartbeat_at, _last_intensity_drivers
        # Heartbeat cadence — independent of session-cleanup cadence.
        # Fault-isolated so a heartbeat failure can't take down the
        # session-cleanup loop or cascade into bridge shutdown.
        # INTENTIONALLY STILL MONOTONIC (defer #21 residual — NOT an oversight):
        # _heartbeat_and_felt_time consumes last_heartbeat_at to compute the
        # felt-time wall_s elapsed-since-last (line ~701), whose monotonic basis
        # is a deliberately-conservative bias (it underweights activity across a
        # system sleep rather than overweighting it — see that function's
        # docstring). The 15-min interval also fires within a typical session, so
        # the restart-reset bite is lowest of all cadences. Converting it would
        # need the felt-time elapsed reworked off a persisted last-fire; not worth
        # the risk on the highest-fan-out cadence. Persist this ONLY alongside a
        # felt-time-elapsed redesign.
        if (
            heartbeat_interval_s is not None
            and last_heartbeat_at is not None
            and time.monotonic() - last_heartbeat_at >= heartbeat_interval_s
        ):
            _heartbeat_attempt_result = _heartbeat_and_felt_time(
                persona_dir, provider, event_bus, last_heartbeat_at
            )
            # Stage-6 red-team MAJOR, fixed (ram-spike-fix INC-7, S21):
            # `_heartbeat_and_felt_time` now returns None WITHOUT doing
            # anything when a chat reply is in flight (the new start
            # check) — advancing `last_heartbeat_at` unconditionally in
            # that case would restart this whole 15-min countdown on a
            # tick that did no work, so a chat session busy enough to
            # almost always have a reply in flight near the 15-min mark
            # could defer the heartbeat far past its own interval,
            # indefinitely. Only advance on an ACTUAL attempt (non-None
            # return) — a skipped tick leaves `last_heartbeat_at`
            # untouched, so the very next supervisor loop pass (seconds
            # away, not 15 minutes) re-checks immediately instead of
            # waiting out a full fresh interval.
            if _heartbeat_attempt_result is not None:
                _last_intensity_drivers = _heartbeat_attempt_result
                last_heartbeat_at = time.monotonic()

    def _between_items() -> bool:
        """INC-10 between-items hook (spec §4, S14/S31/S41/S65): a pausable
        gated job's own loop calls this between items. It runs the SAME
        heartbeat hook `between_jobs` uses (so a heartbeat that falls due
        mid-job still runs only between items, never during one, S41/S65),
        then reports whether the job should pause (chat active again,
        S14/S43) — True means "stop here, save progress, return
        JobOutcome.PAUSED"; the caller does not advance its cadence."""
        try:
            _maybe_run_heartbeat()
        except Exception:
            logger.exception("supervisor between-items heartbeat hook raised")
        return not cli_throttle.is_chat_idle()

    # The gated jobs, in the S55 order (see brain/bridge/central_cadence.py).
    tick_stats = {"closed_sessions": 0, "pruned_empty_sessions": 0}
    # The per-tick shared MemoryStore (#132: one memories.db open per tick,
    # reused by the snapshot/prune job, the backfill probes and jobs, maker and
    # notes). Set for the duration of each loop iteration; None outside it or
    # when the open failed (a job then opens its own short-lived store).
    tick_ctx: dict[str, MemoryStore | None] = {"store": None}
    gated_jobs = _build_gated_jobs(
        persona_dir=persona_dir,
        provider=provider,
        event_bus=event_bus,
        is_session_busy=is_session_busy,
        finalize_after_hours=finalize_after_hours,
        finalize_interval_s=finalize_interval_s,
        initiate_review_interval_s=initiate_review_interval_s,
        maintenance_interval_s=soul_review_interval_s,
        self_model_interval_s=self_model_interval_s,
        compaction_interval_s=compaction_interval_s,
        calibration_interval_s=calibration_interval_s,
        interest_sweep_interval_s=interest_sweep_interval_s,
        judge_selftune_interval_s=judge_selftune_interval_s,
        clustering_interval_s=clustering_interval_s,
        intensity_drivers=lambda: _last_intensity_drivers,
        tick_stats=tick_stats,
        tick_ctx=tick_ctx,
        bridge_started_at=bridge_started_at,
        between_items=_between_items,
    )

    while not stop_event.is_set():
        tick_stats["closed_sessions"] = 0
        tick_stats["pruned_empty_sessions"] = 0
        # store is opened here (per-tick, this thread only — H-A hardening) and
        # shared by this iteration's snapshot/prune job, backfill probes/jobs and
        # the maker/notes ticks (#132: one memories.db open per tick). It is
        # closed once, in the `finally` below (search "per-tick store close"),
        # guaranteeing cleanup even if something in between raises uncaught.
        # Reset to None every iteration so a failed open can never fall through
        # to a stale/closed object from the previous one.
        store: MemoryStore | None = None
        try:
            try:
                store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
            except Exception:
                logger.exception("supervisor per-tick store open raised")
            tick_ctx["store"] = store

            # Heartbeat first (its own timer), fault-isolated so a heartbeat
            # failure can't take down the rest of the loop.
            try:
                _maybe_run_heartbeat()
            except Exception:
                logger.exception("supervisor heartbeat cadence raised")

            # The central cadence function (ram-spike-fix INC-9): every gated
            # job, idle-gated and in the fixed S55 order, with the heartbeat
            # hook between jobs. Fault-isolated per job inside, and as a whole
            # here.
            try:
                central_cadence.run_central_pass(
                    persona_dir, gated_jobs, between_jobs=_maybe_run_heartbeat
                )
            except Exception:
                logger.exception("supervisor central cadence pass raised")

            try:
                event_bus.publish(
                    {
                        "type": "supervisor_tick",
                        "closed_sessions": tick_stats["closed_sessions"],
                        "pruned_empty_sessions": tick_stats["pruned_empty_sessions"],
                        "next_tick_in_s": tick_interval_s,
                        "at": _now_iso(),
                    }
                )
            except Exception:
                logger.exception("supervisor tick event publish raised")

            # Non-gated work below keeps its own timing (S16) — unchanged by
            # INC-9 apart from moving out of the old per-job blocks.
            # Soul-review cadence — slowest of the three. Each pass is up to
            # 5 LLM calls (one per candidate). Fault-isolated so a model
            # outage doesn't take the supervisor down.
            # This tick, and voice-reflection/maker/notes below, each build their
            # OWN `background-generative` tier provider (#154) rather than reusing
            # the bare ambient `provider` — same model (`MODEL_MEDIUM`) it already
            # resolves to, so this is routing-only, not a model change.
            # Soul-review cadence — PERSISTED + self-pacing. Unlike the monotonic
            # timers, soul_review_state.json survives restart/sleep, so the 6h
            # interval can't be reset to zero by an app quit/reboot (the defect that
            # let candidates pile up). Self-paces by outcome: backlog → 30-min
            # catch-up; model failures (429) → escalating backoff; clean → 6h.
            if soul_review_interval_s is not None and soul_cadence.is_due(
                soul_cadence_state, now=datetime.now(UTC)
            ):
                model_failures = 0
                eligible_pending = 0
                try:
                    model_failures, eligible_pending = _run_soul_review_tick(
                        persona_dir,
                        build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
                        event_bus,
                    )
                except Exception:
                    logger.exception("supervisor soul-review tick raised")
                    model_failures = 1  # a crashed tick counts as a failure → backoff
                soul_cadence_state = soul_cadence.compute_next_state(
                    now=datetime.now(UTC),
                    model_failures=model_failures,
                    eligible_pending=eligible_pending,
                    normal_interval_s=soul_review_interval_s,
                    prev_failures=soul_cadence_state.consecutive_failures,
                )
                soul_cadence.save_cadence_state(persona_dir, soul_cadence_state)

            # Log-rotation cadence — hourly default. Bounded JSONL archives so
            # heartbeats/dreams/emotion_growth don't grow forever; yearly
            # archive for soul_audit (every decision must remain reachable).
            # Fault-isolated per-log inside the tick function.
            if log_rotation_cadence_state is not None and persisted_cadence.is_due(
                log_rotation_cadence_state, now=datetime.now(UTC)
            ):
                try:
                    _run_log_rotation_tick(persona_dir, event_bus)
                except Exception:
                    logger.exception("supervisor log-rotation tick raised")
                finally:
                    log_rotation_cadence_state = persisted_cadence.advance(
                        now=datetime.now(UTC), interval_s=log_rotation_interval_s
                    )
                    persisted_cadence.save_cadence(
                        persona_dir, "log_rotation_cadence.json", log_rotation_cadence_state
                    )

            # Vocab-repair cadence (#173) — 6h default. The startup pass above
            # only fires once per bridge start and can be throttle-deferred;
            # this retries the placeholder-describer (and the #174 variant
            # merge) on a persisted wall-clock cadence. Cheap when nothing is
            # pending: should_run is a file scan, no provider is built.
            if vocab_repair_cadence_state is not None and persisted_cadence.is_due(
                vocab_repair_cadence_state, now=datetime.now(UTC)
            ):
                try:
                    _run_vocab_repair_tick(persona_dir)
                except Exception:
                    logger.exception("supervisor vocab-repair tick raised")
                finally:
                    vocab_repair_cadence_state = persisted_cadence.advance(
                        now=datetime.now(UTC), interval_s=vocab_repair_interval_s
                    )
                    persisted_cadence.save_cadence(
                        persona_dir, "vocab_repair_cadence.json", vocab_repair_cadence_state
                    )

            # Voice-reflection cadence — daily by default. Gathers last 7 days
            # of crystallizations + dreams + message tones and may emit a
            # voice-edit candidate (gated by >=3 evidence items inside the
            # reflection tick itself). Fault-isolated.
            if voice_cadence_state is not None and persisted_cadence.is_due(
                voice_cadence_state, now=datetime.now(UTC)
            ):
                try:
                    _run_voice_reflection_tick(
                        persona_dir,
                        build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
                        event_bus,
                    )
                except Exception:
                    logger.exception("supervisor voice-reflection tick raised")
                finally:
                    voice_cadence_state = persisted_cadence.advance(
                        now=datetime.now(UTC), interval_s=voice_reflection_interval_s
                    )
                    persisted_cadence.save_cadence(
                        persona_dir, "voice_reflection_cadence.json", voice_cadence_state
                    )

            # Maker (autonomous making) tick — fail-isolated. The tick gates itself
            # internally on the persisted creative charge (maker_charge.json); this
            # block only switches the organ on/off. Reuses the per-tick `store`
            # opened at the top of this iteration (the charge readers need it)
            # instead of opening its own separate connection (#132).
            if maker_enabled and store is not None:  # default True; gate exists for tests/builds
                try:
                    _maybe_run_maker_tick(
                        persona_dir,
                        store=store,
                        provider=build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
                    )
                except Exception:
                    logger.exception("supervisor maker tick raised")
            elif maker_enabled:
                logger.debug("supervisor maker tick skipped: per-tick store unavailable this tick")

            # Notes (autonomous persona notes) tick — fail-isolated. The tick gates
            # itself internally on consent + away-time + cooldown + budget
            # (notes_state.json / notes_budget.json); this block only switches the
            # organ on/off. Reuses the per-tick `store` opened at the top of this
            # iteration (the `store` kwarg is accepted but currently unused by the
            # notes runner itself) instead of opening its own separate connection
            # (#132). Organ DoD — the producer fires on the live path.
            if notes_enabled and store is not None:  # default True; gate exists for tests/builds
                try:
                    _maybe_run_notes_tick(
                        persona_dir,
                        store=store,
                        provider=build_tier_provider(persona_dir, TIER_BACKGROUND_GENERATIVE),
                    )
                except Exception:
                    logger.exception("supervisor notes tick raised")
            elif notes_enabled:
                logger.debug("supervisor notes tick skipped: per-tick store unavailable this tick")
        finally:
            # per-tick store close
            tick_ctx["store"] = None
            if store is not None:
                try:
                    # Belt-and-suspenders; read BEFORE close. Every MemoryStore
                    # mutator used in this loop commits before returning (see
                    # store.py), so the shared connection should always be at
                    # rest here — this only logs if that invariant is ever
                    # violated. Guarded by the same try as the close below so a
                    # closed/invalid connection can't turn this safety check
                    # itself into an uncaught exception.
                    if store._conn.in_transaction:  # noqa: SLF001
                        logger.warning(
                            "supervisor per-tick store had an open transaction "
                            "at tick close (unexpected — every MemoryStore "
                            "mutator in this loop commits before returning); "
                            "closing anyway"
                        )
                    store.close()
                except Exception:
                    logger.exception("supervisor per-tick store close raised")

        # Kindled-link tick — fail-isolated. The tick self-paces via its own
        # persisted cadence (kindled_tick_cadence.json); this block only
        # switches the organ on/off. The REAL user gate is
        # config.kindled_link_enabled (default False) + config.kindled_relay_url
        # being set — checked inside _run_kindled_link_tick. ExitStack store
        # + httpx client ownership live inside that helper. Organ DoD — the
        # producer fires on the live path.
        if kindled_link_enabled:  # default True; coarse test/build switch
            try:
                _maybe_run_kindled_link_tick(persona_dir, provider=provider)
            except Exception:
                logger.exception("supervisor kindled-link tick raised")

        # Wait for the next tick or for stop_event, whichever comes first.
        stop_event.wait(timeout=tick_interval_s)
    logger.info("supervisor stopped persona=%s", persona_dir.name)


def _snapshot_has_work(persona_dir: Path) -> bool:
    """The session-snapshot half of the snapshot/prune job's "has work"
    probe (INC-9, S66): any active-conversation buffer that is a ghost (no
    readable turns — the sweep cleans it up) or holds turns past its
    extraction cursor (summary blocks excluded, as the snapshot itself
    excludes them). No age check: the job only runs on an idle pass, so
    every session's last turn is at least the lull old (S72/S83)."""
    from brain.ingest.buffer import (
        list_active_sessions,
        read_cursor,
        read_session,
        read_session_after,
    )

    for sid in list_active_sessions(persona_dir):
        try:
            if not read_session(persona_dir, sid):
                return True
            after = read_session_after(persona_dir, sid, read_cursor(persona_dir, sid))
            if any(t.get("speaker") != "summary" for t in after):
                return True
        except Exception:  # noqa: BLE001 — let the sweep's own per-session isolation handle it
            return True
    return False


def _build_gated_jobs(
    *,
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    is_session_busy: Callable[[str], bool] | None,
    finalize_after_hours: float,
    finalize_interval_s: float | None,
    initiate_review_interval_s: float | None,
    maintenance_interval_s: float | None,
    self_model_interval_s: float | None,
    compaction_interval_s: float | None,
    calibration_interval_s: float | None,
    interest_sweep_interval_s: float | None,
    judge_selftune_interval_s: float | None,
    clustering_interval_s: float | None,
    intensity_drivers: Callable[[], IntensityDrivers | None],
    tick_stats: dict[str, int],
    tick_ctx: dict[str, MemoryStore | None] | None = None,
    bridge_started_at: datetime | None = None,
    between_items: Callable[[], bool] | None = None,
) -> list[GatedJob]:
    """The job table of the central cadence function (INC-9, S16/S55/S70).

    One ``GatedJob`` per gated job; a job whose ``*_interval_s`` is None (the
    test/dev disable knob) is left out. Pass 2, session snapshot/prune,
    emotion backfill and embedding backfill have no interval: each runs at
    every idle pass while it has work (S53/S66). Deploy recalibration is due
    while the stored floor is stale (S70/S73). Self-model articulation keeps
    its own cadence (S29). Every other job is an interval job whose cadence
    file the central function owns.

    Each ``run`` closure looks its tick function up by module-global name at
    call time (so tests can monkeypatch them) and returns
    ``JobOutcome.SKIPPED`` when it lost the concurrency-slot race or deferred
    without doing the job's work (S20: no cadence advance).
    """
    jobs: list[GatedJob] = []
    started_at = bridge_started_at if bridge_started_at is not None else datetime.now(UTC)
    ctx: dict[str, MemoryStore | None] = tick_ctx if tick_ctx is not None else {"store": None}
    # INC-10 (S14/S41/S65): the between-items hook every pausable job's own
    # loop asks. Callers that don't wire one (older tests, direct unit
    # calls) get the bare idle check with no heartbeat hook — same
    # behavior pass2's should_pause had before this increment.
    _between_items: Callable[[], bool] = (
        between_items if between_items is not None else (lambda: not cli_throttle.is_chat_idle())
    )

    @contextmanager
    def _tick_store() -> Iterator[MemoryStore]:
        """The loop's per-tick shared store when there is one (#132), else a
        short-lived store of the job's own."""
        shared = ctx.get("store")
        if shared is not None:
            yield shared
            return
        own = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        try:
            yield own
        finally:
            own.close()

    # 1. pass 2 — S53: every idle pass while the saved queue is non-empty.
    def _pass2_run() -> JobOutcome:
        with cli_throttle.background_slot() as slot:
            if not slot:
                return JobOutcome.SKIPPED
            # S14 (per item): a message arriving mid-drain stops it at the next
            # item boundary; what's left stays saved for the next lull (S64).
            # INC-10: should_pause is the between-items hook (heartbeat + idle),
            # not a bare idle check, so a heartbeat due mid-drain still runs
            # between items rather than waiting for the whole drain (S41/S65).
            drained = pass2_queue.drain_all_locked(persona_dir, should_pause=_between_items)
        logger.info("pass-2 job: drained=%d", drained)
        # 0 drained with work queued = another process holds pass2_drain.lock
        # (S77): nothing ran, so report a skip (pass 2 has no cadence either way).
        if drained == 0:
            return JobOutcome.SKIPPED
        # INC-10 (C8): drained something but the saved queue (S64, the queue
        # file itself IS pass 2's saved progress) still has items — the
        # between-items hook stopped the drain for chat, not an empty queue.
        if pass2_queue.queue_length(persona_dir) > 0:
            return JobOutcome.PAUSED
        return JobOutcome.COMPLETED

    jobs.append(
        GatedJob(
            "pass2",
            run=_pass2_run,
            has_work=lambda: pass2_queue.queue_length(persona_dir) > 0,
        )
    )

    # 2. session snapshot / empty-session prune — S66/S72/S83.
    def _prune_age(now: datetime) -> float:
        # S72 / 2-plan §3.3a: prune only sessions that predate the CURRENT idle
        # window — never the lull value, never a constant. The window opens at
        # the last message. S84: when no message has been seen in this process
        # (S82 seeded nothing and none arrived, so time_since_last_message() is
        # +inf), the window is anchored at bridge start instead — an empty
        # session created before this bridge started is prunable, one created
        # after it (e.g. the app-mount session) is kept until a later start.
        # time_since_last_message() itself is left untouched (it is also
        # is_chat_idle's anchor, S82): the fallback lives only here.
        since = cli_throttle.time_since_last_message()
        if math.isinf(since):
            return max(0.0, (now - started_at).total_seconds())
        return since

    def _snapshot_prune_has_work() -> bool:
        if _snapshot_has_work(persona_dir):
            return True
        now = datetime.now(UTC)
        return has_prunable_empty_sessions(
            older_than_seconds=_prune_age(now), now=now, persona_name=persona_dir.name
        )

    def _snapshot_prune_run() -> JobOutcome:
        snapshot_paused_out: list[bool] = []
        with ExitStack() as stack:
            store = stack.enter_context(_tick_store())
            hebbian = HebbianMatrix(persona_dir / "hebbian.db", integrity_check=False)
            stack.callback(hebbian.close)
            # No per-session age threshold (silence 0): the job only runs on an
            # idle pass, so every session's last turn is at least the lull old.
            # Snapshot is NON-destructive — do NOT call remove_session here.
            reports = snapshot_stale_sessions(
                persona_dir,
                silence_minutes=0.0,
                store=store,
                hebbian=hebbian,
                provider=build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING),
                should_pause=_between_items,
                paused_out=snapshot_paused_out,
            )
        # Prune is indivisible (§4.4) — it always runs once per pass
        # regardless of whether the snapshot half paused above.
        prune_now = datetime.now(UTC)
        pruned = prune_empty_sessions(
            older_than_seconds=_prune_age(prune_now), now=prune_now, persona_name=persona_dir.name
        )
        tick_stats["closed_sessions"] += len(reports)
        tick_stats["pruned_empty_sessions"] += len(pruned)
        for r in reports:
            event_bus.publish(
                {
                    "type": "session_snapshot",
                    "session_id": r.session_id,
                    "extracted_since_cursor": r.extracted,
                    "committed": r.committed,
                    "enqueued": r.enqueued,
                    "deduped": r.deduped,
                    "soul_candidates": r.soul_candidates,
                    "errors": r.errors,
                    "at": _now_iso(),
                }
            )
        return JobOutcome.PAUSED if snapshot_paused_out else JobOutcome.COMPLETED

    jobs.append(
        GatedJob("session_snapshot_prune", run=_snapshot_prune_run, has_work=_snapshot_prune_has_work)
    )

    # 3. emotion backfill — S53/S66 (was a startup one-shot only).
    def _emotion_backfill_job_has_work() -> bool:
        with _tick_store() as store:
            return _emotion_backfill_has_work(persona_dir, store=store)

    def _emotion_backfill_job() -> JobOutcome:
        # The per-tick shared store (#132): one memories.db connection per tick.
        emotion_paused_out: list[bool] = []
        with _tick_store() as store:
            _emotion_backfill_run(
                persona_dir,
                provider=build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER),
                store=store,
                should_pause=_between_items,
                paused_out=emotion_paused_out,
            )
        # INC-10 (C8): PAUSED only when the between-items hook itself stopped
        # the pass (emotion_paused_out, set at that exact return site) — NOT
        # merely whenever status=="running", which ALSO covers the unrelated
        # zero-tagged-guard case (a systematic tagger failure leaves status
        # "running" too, but that is not a chat-idle pause and must not stop
        # the S55 sequence for the jobs after this one).
        return JobOutcome.PAUSED if emotion_paused_out else JobOutcome.COMPLETED

    jobs.append(
        GatedJob(
            "emotion_backfill",
            run=_emotion_backfill_job,
            has_work=_emotion_backfill_job_has_work,
        )
    )

    # 4. embedding backfill — S53/S66 (was every base tick behind a slot).
    def _embedding_backfill_has_work() -> bool:
        with _tick_store() as store:
            return _embedding_backfill_has_work_probe(store)

    def _embedding_backfill_job() -> JobOutcome:
        with cli_throttle.background_slot() as slot, _tick_store() as store:
            if not slot:
                return JobOutcome.SKIPPED
            result = _embedding_backfill_run_tick(persona_dir, store)
            logger.info(
                "embedding backfill tick: scanned=%d embedded=%d "
                "skipped_short=%d errors=%d batch_size=%d",
                result.scanned,
                result.embedded,
                result.skipped_short,
                result.errors,
                result.batch_size,
            )
        return JobOutcome.COMPLETED

    jobs.append(
        GatedJob(
            "embedding_backfill",
            run=_embedding_backfill_job,
            has_work=_embedding_backfill_has_work,
        )
    )

    # 5. maintenance — forgetting + narrative (+ the two cheap sweeps).
    if maintenance_interval_s is not None:

        def _maintenance_run() -> JobOutcome:
            with cli_throttle.background_slot() as slot:
                if not slot:
                    return JobOutcome.SKIPPED
                forgetting_progress: dict[str, bool] = {}
                try:
                    forgetting_run_pass(
                        persona_dir,
                        event_bus=event_bus,
                        intensity_drivers=intensity_drivers(),
                        should_pause=_between_items,
                        progress_out=forgetting_progress,
                    )
                except Exception:
                    logger.exception("supervisor forgetting pass raised")
                if forgetting_progress.get("paused"):
                    # INC-10 (C8): forgetting stopped between memories for
                    # chat — its own cursor (job_progress) already saved the
                    # resume point; skip narrative/the sweeps this pass so
                    # the whole maintenance job reports PAUSED (no cadence
                    # advance, S36) rather than silently completing them out
                    # of order relative to a still-mid-pass forgetting.
                    return JobOutcome.PAUSED
                # Narrative-memory arc-update runs AFTER forgetting so a memory
                # forgetting just dropped doesn't enter an arc born this tick.
                try:
                    _run_narrative_memory_pass(persona_dir, provider, event_bus)
                except Exception:
                    logger.exception("supervisor narrative-memory pass raised")
            # Expire stale pending file-write proposals (24h TTL).
            try:
                from brain.files import pending as _file_pending

                _file_pending.sweep_expired(persona_dir, now=datetime.now(UTC))
            except Exception:
                logger.exception("supervisor pending-write sweep raised")
            # Reap aged .lock.stale-* / .corrupt-* forensic residue (#176).
            try:
                from brain.health import sidecar_sweep as _sidecar_sweep

                _sidecar_sweep.sweep_stale_sidecars(persona_dir, now=datetime.now(UTC))
            except Exception:
                logger.exception("supervisor sidecar sweep raised")
            return JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "maintenance",
                run=_maintenance_run,
                cadence_file="maintenance_cadence.json",
                interval_s=maintenance_interval_s,
            )
        )

    # 6. interest sweep — weekly.
    if interest_sweep_interval_s is not None:

        def _interest_sweep_run() -> JobOutcome:
            with ExitStack() as stack, cli_throttle.background_slot() as slot:
                if not slot:
                    return JobOutcome.SKIPPED
                sweep_store = MemoryStore(persona_dir / "memories.db")
                stack.callback(sweep_store.close)
                interest_sweep.run_sweep_tick(
                    store=sweep_store,
                    provider=build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING),
                    interests_path=persona_dir / "interests.json",
                    default_interests_path=(
                        Path(__file__).resolve().parent.parent
                        / "engines"
                        / "default_interests.json"
                    ),
                    now=datetime.now(UTC),
                )
            return JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "interest_sweep",
                run=_interest_sweep_run,
                cadence_file=interest_sweep.SWEEP_CADENCE_FILE,
                interval_s=interest_sweep_interval_s,
            )
        )

    # 7. self-model articulation — keeps its own cadence (S29); due when that
    # cadence says so, with the gated-job missing/corrupt rule (S22/S69).
    if self_model_interval_s is not None:

        def _self_model_due() -> bool:
            now = datetime.now(UTC)
            state, created = self_model_cadence.load_or_init(persona_dir, now=now)
            return not created and self_model_cadence.is_due(state, now=now)

        def _self_model_run() -> JobOutcome:
            from brain.self_model.articulate import build_self_model_provider

            ran = _run_self_model_tick(
                persona_dir,
                provider=build_self_model_provider(persona_dir),
                event_bus=event_bus,
            )
            return JobOutcome.SKIPPED if ran is False else JobOutcome.COMPLETED

        jobs.append(
            GatedJob("self_model_articulation", run=_self_model_run, has_work=_self_model_due)
        )

    # 8. compaction — daily.
    if compaction_interval_s is not None:

        def _compaction_run() -> JobOutcome:
            from brain.chat.compaction import build_compaction_provider

            paused = _run_compaction_tick(
                persona_dir,
                build_compaction_provider(persona_dir),
                is_session_busy=is_session_busy,
                should_pause=_between_items,
            )
            return JobOutcome.PAUSED if paused else JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "compaction",
                run=_compaction_run,
                cadence_file="compaction_cadence.json",
                interval_s=compaction_interval_s,
            )
        )

    # 9. clustering — 6h.
    if clustering_interval_s is not None:
        jobs.append(
            GatedJob(
                "clustering",
                run=lambda: _run_clustering_tick(persona_dir),
                cadence_file="clustering_cadence.json",
                interval_s=clustering_interval_s,
            )
        )

    if calibration_interval_s is not None:
        # 10. deploy recalibration (the stale-floor refit, S70/S73) — due while
        # the stored floor is stale; immediately before daily calibration.
        jobs.append(
            GatedJob(
                "deploy_recalibration",
                run=lambda: _run_deploy_recalibration(persona_dir),
                has_work=lambda: _deploy_recalibration_due(persona_dir),
            )
        )

        # 11. daily calibration — calibration before self-tune (S43).
        def _calibration_run() -> JobOutcome:
            ran = _run_calibration_tick(
                persona_dir, is_session_busy=is_session_busy, should_pause=_between_items
            )
            if ran is None:
                return JobOutcome.PAUSED
            return JobOutcome.SKIPPED if ran is False else JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "daily_calibration",
                run=_calibration_run,
                cadence_file="calibration_cadence.json",
                interval_s=calibration_interval_s,
            )
        )

    # 12. weekly judge self-tune.
    if judge_selftune_interval_s is not None:

        def _selftune_run() -> JobOutcome:
            with ExitStack() as stack, cli_throttle.background_slot() as slot:
                if not slot:
                    return JobOutcome.SKIPPED
                selftune_store = MemoryStore(persona_dir / "memories.db")
                stack.callback(selftune_store.close)
                _run_judge_selftune_tick(
                    store=selftune_store,
                    now=datetime.now(UTC),
                    persona_dir=persona_dir,
                )
            return JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "weekly_selftune",
                run=_selftune_run,
                cadence_file=JUDGE_TUNE_CADENCE_FILE,
                interval_s=judge_selftune_interval_s,
            )
        )

    # 13. finalize — hourly sweep, 24h silence threshold.
    if finalize_interval_s is not None:

        def _finalize_run() -> JobOutcome:
            paused = _run_finalize_tick(
                persona_dir,
                build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING),
                event_bus,
                finalize_after_hours=finalize_after_hours,
                should_pause=_between_items,
            )
            return JobOutcome.PAUSED if paused else JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "finalize",
                run=_finalize_run,
                cadence_file="finalize_cadence.json",
                interval_s=finalize_interval_s,
            )
        )

    # 14. initiate review — 15 min.
    if initiate_review_interval_s is not None:

        def _initiate_review_run() -> JobOutcome:
            paused = _run_initiate_review_tick(
                persona_dir, provider, event_bus, should_pause=_between_items
            )
            return JobOutcome.PAUSED if paused else JobOutcome.COMPLETED

        jobs.append(
            GatedJob(
                "initiate_review",
                run=_initiate_review_run,
                cadence_file="initiate_review_cadence.json",
                interval_s=initiate_review_interval_s,
            )
        )

    return central_cadence.order_jobs(jobs)


def _run_vocab_repair_tick(persona_dir: Path) -> None:
    """Repair emotion_vocabulary.json if it has placeholder or variant-twin entries (#173/#174).

    Shared by the startup one-shot and the 6h cadence. Builds the Haiku-tier
    provider only when there is something to describe.
    """
    if not _vocab_repair_should_run(persona_dir):
        return
    from brain.memory.store import MemoryStore as _MemoryStore

    _store = _MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    try:
        _vocab_repair_run(
            persona_dir,
            store=_store,
            provider=build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER),
        )
    finally:
        _store.close()


def _run_maker_tick(persona_dir, *, store, provider):
    """Run one maker tick on the live supervisor path.

    Fires the real making closure (make_and_wire): the tick accumulates the
    charge and, when it crosses threshold under budget, makes + persists.
    """
    from brain.maker import run_maker_tick
    from brain.maker.making_runner import make_and_wire

    run_maker_tick(persona_dir, store=store, provider=provider, make_fn=make_and_wire)


def _maybe_run_maker_tick(persona_dir, *, store, provider):
    try:
        _run_maker_tick(persona_dir, store=store, provider=provider)
    except Exception:
        logger.exception("supervisor maker tick raised")


def _run_notes_tick(persona_dir, *, store, provider):
    """Run one notes tick on the live supervisor path.

    Loads persona config (consent + resolved folder), computes how long the user
    has been away from UserPresence, and runs the gated tick — which, when away +
    cooldown + budget clear, fires the real note-making closure
    (make_note_and_wire): compose from her interior, write the folder-bounded note.
    """
    from brain.initiate.user_pattern import compute_user_presence
    from brain.notes import run_notes_tick
    from brain.notes.runner import make_note_and_wire

    config = PersonaConfig.load(persona_dir / "persona_config.json")
    silence_hours = compute_user_presence(persona_dir).silence_days * 24
    run_notes_tick(
        persona_dir,
        config=config,
        provider=provider,
        silence_hours=silence_hours,
        make_fn=make_note_and_wire,
    )


def _maybe_run_notes_tick(persona_dir, *, store, provider):
    try:
        _run_notes_tick(persona_dir, store=store, provider=provider)
    except Exception:
        logger.exception("supervisor notes tick raised")


def _run_kindled_link_tick(persona_dir, *, provider, now=None):
    """Run one kindled-link tick on the live supervisor path.

    Loads config; returns immediately if kindled_relay_url is not set
    (feature not wired). Otherwise constructs identity, store, relay
    client and delegates to run_kindled_link_tick — which self-paces via
    its own persisted cadence (kindled_tick_cadence.json).

    The httpx.Client is built INSIDE this function (not at import time)
    so the supervisor never holds a live HTTP handle between ticks.

    Also opens the persona's MemoryStore (mirrors the maker/notes tick's
    per-tick store-ownership pattern — opened + closed inside this call,
    never held across ticks) and threads it into run_kindled_link_tick as
    mem_store= (§14 wire-back: forms a kindled_peer memory + moves capped
    emotion from the relationship reflection). Fail-soft: if the
    MemoryStore fails to open, the tick still runs with mem_store=None
    (no peer-memory/emotion this round; everything else unaffected).
    """
    import httpx

    from brain.kindled_link.identity import KindledIdentity
    from brain.kindled_link.relay_client import RelayClient
    from brain.kindled_link.store import KindledLinkStore, kindled_db_path
    from brain.kindled_link.tick import run_kindled_link_tick
    from brain.memory.store import MemoryStore

    now = now or datetime.now(UTC)
    config = PersonaConfig.load(persona_dir / "persona_config.json")

    if config.kindled_relay_url is None:
        return  # feature not wired — nothing to poll

    idn = KindledIdentity.load_or_create(persona_dir)

    with ExitStack() as _kl_stack:
        store = KindledLinkStore(kindled_db_path(persona_dir))
        _kl_stack.callback(store.close)
        mailbox = store.get_or_create_local_mailbox()
        http = httpx.Client(base_url=config.kindled_relay_url, timeout=10.0)
        _kl_stack.callback(http.close)
        relay = RelayClient(http, identity=idn, mailbox_id=mailbox)
        try:
            relay.register()
        except Exception:
            logger.warning(
                "supervisor kindled-link relay register failed — skipping tick this round"
            )
            return

        mem_store = None
        try:
            mem_store = MemoryStore(persona_dir / "memories.db")
            _kl_stack.callback(mem_store.close)
        except Exception:
            logger.warning(
                "supervisor kindled-link: MemoryStore open failed — "
                "peer-memory/emotion wire-back skipped this round", exc_info=True
            )

        run_kindled_link_tick(
            persona_dir,
            store=store,
            identity=idn,
            relay_client=relay,
            provider=provider,
            config=config,
            now=now,
            mem_store=mem_store,
        )


def _maybe_run_kindled_link_tick(persona_dir, *, provider, now=None):
    try:
        _run_kindled_link_tick(persona_dir, provider=provider, now=now)
    except Exception:
        logger.exception("supervisor kindled-link tick raised")


def _heartbeat_and_felt_time(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    last_heartbeat_at: float,
) -> IntensityDrivers | None:
    """Run a heartbeat tick then a felt-time tick, wiring real counters.

    Extracted so tests can call this directly and spy on _run_felt_time_tick
    without driving the full run_folded loop. Fault-isolated: heartbeat
    errors are caught and reflex_n defaults to 0; felt-time errors are
    caught and None is returned. The caller updates last_heartbeat_at.

    Start check (S21, C9(b)): does not start a pass — heartbeat OR
    felt-time — while a chat reply is being generated, checked once here;
    once a pass has started (this check passed) it always runs to
    completion even if a reply starts mid-pass (the heartbeat engine
    itself never re-checks this once inside `run_tick`). This is the ONLY
    caller of `run_tick` that carries this check — `nell heartbeat` (a
    separate process, cli.py) and the shutdown close tick (server.py) are
    not gated on it.
    """
    if cli_throttle.reply_in_flight():
        logger.debug(
            "supervisor heartbeat tick deferred: a chat reply is in flight (S21)"
        )
        return None

    from brain.felt_time.chat_log import count_chat_turns_since

    heartbeat_result = None
    try:
        heartbeat_result = _run_heartbeat_tick(persona_dir, provider, event_bus)
    except cli_throttle.ThrottleDeferred as exc:
        # #246 belt: every deferral should be caught inside the engine; one that
        # escapes is a wiring bug worth a WARNING (no traceback), never silence.
        logger.warning("supervisor heartbeat tick deferred at the belt: %s", exc)
    except Exception:
        logger.exception("supervisor heartbeat tick raised")
    try:
        mono_now = time.monotonic()
        wall_now = datetime.now(UTC)
        wall_s = mono_now - last_heartbeat_at
        # time.monotonic() does not advance during system sleep, but
        # datetime.now(UTC) does.  After a suspend, wall_s understates
        # the elapsed wall time, so since_dt lands in the "future" relative
        # to the last real activity and count_chat_turns_since undercounts.
        # This is a deliberately conservative bias: felt-time underweights
        # activity rather than overweighting it — consistent with the
        # documented monotonic-cadence behaviour across the supervisor.
        since_dt = wall_now - timedelta(seconds=wall_s)
        chat_n = count_chat_turns_since(persona_dir, since_dt.isoformat())
        reflex_n = len(heartbeat_result.reflex_fired) if heartbeat_result else 0
        return _run_felt_time_tick(
            persona_dir,
            wall_clock_s_since_last=wall_s,
            heartbeats_since_last=1,
            chat_turns_since_last=chat_n,
            reflex_firings_since_last=reflex_n,
        )
    except Exception:
        logger.exception("supervisor felt-time tick raised")
        return None


def _run_felt_time_tick(
    persona_dir: Path,
    wall_clock_s_since_last: float,
    heartbeats_since_last: int,
    chat_turns_since_last: int,
    reflex_firings_since_last: int,
) -> IntensityDrivers:
    """Fold one supervisor heartbeat cycle into felt-time state.

    drivers values are derived from existing body + emotion accessors;
    cold-start cases (no body state, no emotion vector) collapse all
    drivers to 0.0 so lived-age advances at baseline.

    Returns the computed IntensityDrivers so the caller can cache them for
    the next forgetting pass (arc pressure modulates the fade threshold).

    Fault-isolated upstream: caller wraps in try/except so a raise here
    cannot cascade into bridge shutdown or take down the heartbeat loop.
    """
    drivers = _derive_intensity_drivers(persona_dir, chat_turns_since_last, wall_clock_s_since_last)
    ft = FeltTime(persona_dir=persona_dir)
    ft.tick(
        TickContext(
            now_iso=datetime.now(UTC).isoformat(),
            heartbeats_in_tick=heartbeats_since_last,
            chat_turns_in_tick=chat_turns_since_last,
            reflex_firings_in_tick=reflex_firings_since_last,
            wall_clock_s_in_tick=wall_clock_s_since_last,
            drivers=drivers,
        )
    )
    from brain.felt_time.chat_log import append_chat_tick
    append_chat_tick(persona_dir, ts=datetime.now(UTC), turns=chat_turns_since_last)
    return drivers


def _derive_intensity_drivers(
    persona_dir: Path,
    chat_turns_in_tick: int,
    wall_clock_s_in_tick: float,
) -> IntensityDrivers:
    """Build IntensityDrivers from body + emotion state.

    Each driver clipped to [0, 1]; missing inputs collapse to 0.0
    so lived-age advances at baseline rate rather than crashing.

    Body strain is derived from exhaustion + energy via a full
    compute_body_state pass (opens its own MemoryStore per-call,
    thread-local, same pattern as _run_heartbeat_tick).

    Emotional intensity — computed as the max positive sigma-deviation
    across emotion channels (capped at 1.0), using aggregate_state over
    the 200 most recent active memories. Falls back to 0.0 on any error.

    Chat activity uses a fixed 6 turns/h baseline for the supervisor
    tick window. Phase 9.2 follow-up will tighten to a rolling baseline
    once weather_shift per-channel baselines are proven stable.
    """
    # Best-effort body strain + emotional intensity from exhaustion / energy.
    body_strain = 0.0
    emotional_intensity = 0.0
    try:
        from brain.body.state import compute_body_state
        from brain.body.words import count_words_in_session
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import MemoryStore, _row_to_memory
        from brain.utils.memory import days_since_human

        store = MemoryStore(persona_dir / "memories.db")
        try:
            rows = store._conn.execute(  # noqa: SLF001
                "SELECT * FROM memories "
                "WHERE active = 1 "
                "AND emotions_json IS NOT NULL "
                "AND emotions_json != '{}' "
                "ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
            memories = [_row_to_memory(row) for row in rows]
            emotion_state = aggregate_state(memories)
            _now = datetime.now(UTC)
            days = days_since_human(store, now=_now, persona_dir=persona_dir)
            words = count_words_in_session(
                store, persona_dir=persona_dir, session_hours=0.0, now=_now
            )
            body = compute_body_state(
                emotions=emotion_state.emotions,
                session_hours=0.0,
                words_written=words,
                days_since_contact=days,
                now=_now,
            )
            # exhaustion is int 0-9 (spec: max(0, 7-energy)); energy int 1-10.
            # Normalize each to [0, 1] then take the max (strain = worst axis).
            raw_exhaustion = float(body.exhaustion) / 9.0
            raw_energy_lack = 1.0 - float(body.energy) / 10.0
            body_strain = max(0.0, min(1.0, max(raw_exhaustion, raw_energy_lack)))
            # emotional_intensity: max positive sigma-deviation from per-channel baseline.
            # Channels with < 10 samples skipped (cold-start guard). 3σ ceiling clips to 1.0.
            from collections import defaultdict

            from brain.felt_time.weather_shift import update_baseline as _update_baseline_emo
            _channel_samples: dict[str, list] = defaultdict(list)
            for _mem in memories:
                if _mem.created_at is None:
                    continue
                for _ch, _val in _mem.emotions.items():
                    try:
                        _fval = float(_val)
                    except (TypeError, ValueError):
                        continue
                    if _fval > 0.0:
                        _channel_samples[_ch].append((_mem.created_at, _fval))
            _positive_devs: list[float] = []
            for _ch, _ch_samps in _channel_samples.items():
                if len(_ch_samps) < 10:
                    continue
                _bl = _update_baseline_emo(None, _ch_samps)
                if _bl.sigma <= 0.0:
                    continue
                _current_val = max(v for _, v in _ch_samps)  # max-pool mirrors aggregate_state behaviour
                _dev = (_current_val - _bl.mean) / max(_bl.sigma, 0.1)
                if _dev > 0.0:
                    _positive_devs.append(_dev)
            if _positive_devs:
                emotional_intensity = min(1.0, max(_positive_devs) / 3.0)
        finally:
            store.close()
    except Exception:
        logger.debug("supervisor felt-time: body strain read failed; using 0.0", exc_info=True)

    # Chat activity: rolling 7-day mean from chat_turns.log.jsonl.
    # Falls back to fixed 6 turns/h when log is absent or below cold-start threshold.
    from brain.felt_time.chat_log import load_recent_samples as _load_chat_samples
    from brain.felt_time.weather_shift import update_baseline as _update_baseline_chat
    _chat_samples = _load_chat_samples(persona_dir)
    if _chat_samples is not None:
        _chat_bl = _update_baseline_chat(None, _chat_samples)
        baseline_per_tick = max(0.1, _chat_bl.mean)
    else:
        baseline_per_tick = max(0.1, 6.0 * (wall_clock_s_in_tick / 3600.0))
    chat_activity = min(1.0, float(chat_turns_in_tick) / baseline_per_tick)

    # Narrative weight — a long, emotionally-heavy open arc makes time heavier.
    narrative_weight_val = 0.0
    try:
        from brain.felt_time.lived_age import (
            NARRATIVE_WEIGHT_HORIZON_HOURS,
        )
        from brain.felt_time.lived_age import (
            narrative_weight as _narrative_weight,
        )
        from brain.felt_time.state import load_or_recover as _load_felt_time
        from brain.narrative_memory.state import load_or_recover as _load_arcs

        felt_state, _ = _load_felt_time(persona_dir)
        arcs_state = _load_arcs(persona_dir)
        current_lived = felt_state.lived_age_hours
        arc_inputs = [
            (
                max(0.0, current_lived - arc.lived_age_at_open),
                arc.max_member_emotion_normalised,
            )
            for arc in arcs_state.open.values()
        ]
        narrative_weight_val = _narrative_weight(
            arc_inputs, horizon=NARRATIVE_WEIGHT_HORIZON_HOURS
        )
    except Exception:
        logger.debug(
            "supervisor felt-time: narrative weight read failed; using 0.0", exc_info=True
        )

    return IntensityDrivers(
        emotional_intensity=emotional_intensity,
        body_strain=body_strain,
        chat_activity=chat_activity,
        narrative_weight=narrative_weight_val,
    )


def _run_heartbeat_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> HeartbeatResult | None:
    """Build a HeartbeatEngine and run one tick. Publishes a result event.

    Returns the HeartbeatResult so the caller can read reflex_fired and
    wire it into the felt-time tick as reflex_firings_since_last.

    Constructs the engine per-tick (mirrors the per-tick store pattern)
    so SQLite handles + transient state stay thread-local. Reads the
    persona's PersonaConfig for searcher routing; PersonaConfig's
    allowlists heal invalid values to the default before we get here.
    """
    from brain.engines.heartbeat import HeartbeatEngine
    from brain.persona_config import PersonaConfig
    from brain.search.factory import get_searcher

    config = PersonaConfig.load(persona_dir / "persona_config.json")
    default_arcs_path = (
        Path(__file__).resolve().parent.parent / "engines" / "default_reflex_arcs.json"
    )
    default_interests_path = (
        Path(__file__).resolve().parent.parent / "engines" / "default_interests.json"
    )

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)

        # Vocabulary load mirrors the CLI heartbeat handler so growth
        # crystallizers see the same emotion universe the user does.
        try:
            from brain.emotion.persona_loader import load_persona_vocabulary

            load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        except Exception:
            logger.exception("supervisor heartbeat: vocabulary load skipped")

        searcher = get_searcher(config.searcher)

        engine = HeartbeatEngine(
            store=store,
            hebbian=hebbian,
            provider=provider,
            state_path=persona_dir / "heartbeat_state.json",
            config_path=persona_dir / "heartbeat_config.json",
            dream_log_path=persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
            reflex_arcs_path=persona_dir / "reflex_arcs.json",
            reflex_log_path=persona_dir / "reflex_log.json",
            reflex_default_arcs_path=default_arcs_path,
            searcher=searcher,
            interests_path=persona_dir / "interests.json",
            research_log_path=persona_dir / "research_log.json",
            default_interests_path=default_interests_path,
            persona_name=persona_dir.name,
            persona_system_prompt=(
                _HEARTBEAT_TICK_SYSTEM_PROMPT_SEGMENTS[0] + persona_dir.name
                + _HEARTBEAT_TICK_SYSTEM_PROMPT_SEGMENTS[1]
            ),
        )
        result = engine.run_tick(trigger="background", dry_run=False)

    event_bus.publish(
        {
            "type": "heartbeat_tick",
            "trigger": "background",
            "memories_decayed": result.memories_decayed,
            "edges_pruned": result.edges_pruned,
            "dream_id": result.dream_id,
            "reflex_fired": list(result.reflex_fired),
            "research_fired": result.research_fired,
            "growth_emotions_added": result.growth_emotions_added,
            "reflex_error": result.reflex_error,
            "growth_error": result.growth_error,
            "at": _now_iso(),
        }
    )
    return result


def _run_soul_review_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> tuple[int, int]:
    """Run one autonomous soul-review pass.

    Returns ``(model_failures, eligible_pending_after)`` so the caller can
    self-pace: failures (e.g. 429) trigger a backoff, a remaining backlog
    triggers a catch-up. Returns ``(0, 0)`` when there's nothing to act on.

    Skips silently (no store cost) when there are zero *eligible* candidates
    (auto_pending and not in defer-cooldown). On a backlog, raises the per-call
    cap to ``_SOUL_BACKLOG_DRAIN_CAP`` so a pile-up clears in a couple of ticks
    instead of days — bounded, so one tick can't run away on LLM cost.

    Mirrors the per-tick store-ownership pattern of [`_run_heartbeat_tick`]:
    opens MemoryStore + SoulStore inside this thread, closes via ExitStack.
    """
    from brain.soul.review import (
        DEFAULT_MAX_DECISIONS,
        count_eligible_pending,
        review_pending_candidates,
    )
    from brain.soul.store import SoulStore

    eligible_before = count_eligible_pending(persona_dir)
    if eligible_before == 0:
        # Nothing drainable — skip the pass and the open-store cost. No
        # failures, no backlog → caller schedules the normal interval.
        # NOTE: draft cursor is intentionally NOT advanced here — drafts
        # must wait for the next candidate-bearing tick so quiet ticks
        # don't silently skip un-consumed fragments.
        return 0, 0

    # ── Candidate-gated draft reading ────────────────────────────────────────
    # Runs only when there are actual candidates to review (past the early-
    # return above). Reads fragments since the last cursor position and passes
    # them to the review prompt as interior context.
    from brain.initiate.draft import (
        has_new_drafts_since,
        load_draft_review_cursor,
        read_drafts_since,
        save_draft_review_cursor,
    )

    cursor = load_draft_review_cursor(persona_dir)
    draft_fragments: list[str] = []
    if not cursor or has_new_drafts_since(persona_dir, cursor):
        frags = read_drafts_since(persona_dir, cursor or "0001-01-01T00:00:00")
        draft_fragments = [f"[{f.source}] {f.body}" for f in frags]

    # Backlog-aware drain: clear up to the cap per tick when candidates have
    # piled up, instead of the default 5.
    max_decisions = min(max(DEFAULT_MAX_DECISIONS, eligible_before), _SOUL_BACKLOG_DRAIN_CAP)

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        soul_store = SoulStore(str(persona_dir / "crystallizations.db"))
        stack.callback(soul_store.close)

        try:
            from brain.emotion.persona_loader import load_persona_vocabulary

            load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=store)
        except Exception:
            logger.exception("supervisor soul-review: vocabulary load skipped")

        report = review_pending_candidates(
            persona_dir,
            store=store,
            soul_store=soul_store,
            provider=provider,
            max_decisions=max_decisions,
            draft_fragments=draft_fragments if draft_fragments else None,
        )

    # Advance the draft cursor AFTER the review pass runs — never on early-
    # return, so a quiet tick doesn't silently skip un-consumed fragments.
    save_draft_review_cursor(persona_dir, datetime.now(UTC).isoformat())

    event_bus.publish(
        {
            "type": "soul_review_tick",
            "trigger": "background",
            "pending_at_start": report.pending_at_start,
            "examined": report.examined,
            "accepted": report.accepted,
            "rejected": report.rejected,
            "deferred": report.deferred,
            "parse_failures": report.parse_failures,
            "model_failures": report.model_failures,
            "at": _now_iso(),
        }
    )

    eligible_after = count_eligible_pending(persona_dir)
    return report.model_failures, eligible_after


# Minimum gap magnitude before the supervisor asks Haiku to articulate it.
# Mirrors brain/self_model/articulate.py::_GAP_THRESHOLD (which also gates
# internally) so the tick avoids a wasted call attempt below threshold.
_SELF_MODEL_ARTICULATE_THRESHOLD = 0.4


def _run_self_model_tick(
    persona_dir: Path,
    *,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> bool:
    """Run one autonomous self-model reflection pass — the whole organ.

    Returns True when the reflection ran (and its cadence was saved by
    outcome), False when it did not run (not due, or the pre-flight slot peek
    denied it — cadence untouched).

    Composes the self-model end-to-end on the live supervisor path
    (Organ Definition-of-Done — the producer fires here, not just in
    isolation):

      0. load the PERSISTED wall-clock cadence; if not due → return
         (the cadence survives restart/sleep, mirroring soul review's
         decoupling from the monotonic timers).
      0a. PRE-FLIGHT THROTTLE PEEK — a defer must cost NOTHING. Checked here,
         before any of the tick's own work, mirroring brain/maker/__init__.py's
         "Pre-flight throttle gate" comment exactly. cli_throttle.slot_available()
         is a read-only peek (never touches the shared semaphore); on denial we
         log and return WITHOUT calling _self_model_reflect at all and WITHOUT
         touching the cadence (ram-spike-fix INC-9, S20: a skip is not a run,
         so the reflection stays due) — steps 1-9 below never run, so a denied
         attempt has zero side effects (no gap computation, no budget
         consumption, no sustained_ticks/gaps_surfaced change, no
         self_model_state.json or cadence write).
      1. Read the recent active emotion-bearing memories.
      2. short = compute_derived(memories, body)       (recent normalized mean, short half-life)
      3. long  = compute_baseline(memories)            (baseline normalized mean, long half-life)
      4. gap   = compute_gap(short, long)              (bidirectional short − long, noise-floored)
      5. Track sustained_ticks vs the prior current_gap; bump the
         gaps_surfaced audit counter while a gap is active.
      6. If gap.magnitude >= threshold → articulate a note via Haiku. The real
         (authoritative) throttle acquire happens inside articulate() itself,
         only reachable once step 0a's peek has already granted — it is the
         guard for the rare race where chat resumes (or another accessor wins
         the shared slot) between the peek and this call, not the common-case
         check.
      7. check_and_emit_resolution(prior, new) → soul candidate + feed event
         when a sustained gap just resolved (either path).
      8. push_gap (displaced-gap helper preserves an unresolved displaced
         gap in history), save state.
      9. advance + save the cadence by outcome (clean / deferred / failure) so
         a crash backs off, a throttle deferral retries promptly at no cost,
         and a clean pass schedules the normal interval.

    Fail-isolated by the caller (run_folded's try/except), AND every I/O
    step here is locally fail-soft so a single bad read degrades to
    "no gap this tick" rather than a crash. On any exception the cadence
    is still advanced with a "failure" outcome so the tick backs off
    instead of busy-looping on a persistent error.

    Mirrors the per-tick store-ownership pattern of _run_soul_review_tick:
    opens MemoryStore inside this thread, closes via ExitStack.
    """
    now = datetime.now(UTC)

    # ── 0: persisted-cadence gate (NOT monotonic — survives restart/sleep) ──
    # ram-spike-fix INC-9 (S29): self-model articulation is a gated job of the
    # central cadence function (brain/bridge/central_cadence.py), which runs it
    # only on an idle pass; it STILL checks its own cadence here (6 h normal,
    # 30-min backlog re-run, failure backoff — brain/self_model/cadence.py)
    # before it actually runs. These own timers are to be absorbed later by the
    # expanded central cadence function (the planned version that takes over
    # everything that runs on a recurring cadence), at which point this
    # module-local cadence goes away.
    cadence_state = self_model_cadence.load(persona_dir)
    if not self_model_cadence.is_due(cadence_state, now=now):
        return False

    # ── 0a: pre-flight throttle peek (a defer must cost NOTHING) ────────────
    # ram-spike-fix INC-9 (S20, C22): a no-lull / slot-denied skip leaves the
    # cadence file UNCHANGED (it used to write a short "deferred" retry), so
    # the reflection is simply still due at the next idle pass.
    if not cli_throttle.slot_available():
        log_self_model_deferred(persona_dir)
        return False

    outcome = "clean"
    try:
        deferred = _self_model_reflect(
            persona_dir, provider=provider, event_bus=event_bus, now=now
        )
        if deferred:
            outcome = "deferred"
    except Exception:
        logger.exception("supervisor self-model reflect raised")
        outcome = "failure"

    cadence_state = self_model_cadence.compute_next_state(
        cadence_state, outcome=outcome, now=now
    )
    self_model_cadence.save(persona_dir, cadence_state)
    return True


def _self_model_reflect(
    persona_dir: Path,
    *,
    provider: LLMProvider,
    event_bus: EventBus | object,
    now: datetime,
) -> bool:
    """The reflection body — separated so cadence advance always runs.

    See _run_self_model_tick for the step-by-step contract. This function
    does steps 1-8; the caller owns the cadence gates (steps 0/0a) and
    advance (step 9) so a crash here still backs the cadence off.

    Returns True if this tick's articulation was deferred by the throttle
    (the rare peek-then-lost race — see step 6/articulate()'s docstring),
    False otherwise (gap below threshold, a note was produced, budget
    exhausted, or a provider exception — all pre-existing "no note" cases
    unrelated to the throttle). The caller uses this to pick the cadence
    outcome ("deferred" vs "clean").
    """
    from brain.body.state import compute_body_state
    from brain.body.words import count_words_in_session
    from brain.emotion.aggregate import aggregate_state
    from brain.memory.store import MemoryStore, _row_to_memory
    from brain.utils.memory import days_since_human

    # ── 1-3: read memories, body + short/long reads ─────────────────────────
    # Same query as _build_intensity_drivers: the 200 most recent active
    # emotion-bearing memories. aggregate_state (max-pool peak) still feeds the
    # body-state computation; the gap uses short/long normalized-mean reads.
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)

        rows = store._conn.execute(  # noqa: SLF001
            "SELECT * FROM memories "
            "WHERE active = 1 "
            "AND emotions_json IS NOT NULL "
            "AND emotions_json != '{}' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        memories = [_row_to_memory(row) for row in rows]

        declared = aggregate_state(memories)
        days = days_since_human(store, now=now, persona_dir=persona_dir)
        words = count_words_in_session(
            store, persona_dir=persona_dir, session_hours=0.0, now=now
        )
        body = compute_body_state(
            emotions=declared.emotions,
            session_hours=0.0,
            words_written=words,
            days_since_contact=days,
            now=now,
        )

    short = compute_derived(
        memories, body_energy=body.energy, body_exhaustion=body.exhaustion, now=now
    )
    long = compute_baseline(memories, now=now)
    new_gap = compute_gap(short, long)

    # ── 4: load prior state, track sustained ticks ──────────────────────────
    state, _recovered = self_model_state.load_or_recover(persona_dir)
    prior_gap = state.current_gap

    # ── 4a: R-B2 anti-oscillation — honour the reconcile cooldown (live) ─────
    # Drop any channel the prior gap put in cooldown (REGARDLESS of prior
    # status — the reconcile tool leaves prior acknowledged/dismissed) and
    # recompute magnitude, so a freshly-reconciled channel doesn't re-surface
    # within its window. Carry the non-expired cooldowns forward so they
    # survive this recompute and the status flip. Fail-soft.
    carried_cooldowns: dict[str, str] = {}
    if prior_gap is not None and prior_gap.channel_cooldowns:
        try:
            kept_channels: dict[str, float] = {}
            for ch, delta in new_gap.per_channel.items():
                if sm_reconcile.is_channel_in_cooldown(prior_gap, ch, now=now):
                    continue
                kept_channels[ch] = delta
            new_gap.per_channel = kept_channels
            new_gap.magnitude = sum(abs(v) for v in kept_channels.values())
            # Keep only cooldowns that have not yet expired.
            for ch, expiry in prior_gap.channel_cooldowns.items():
                if sm_reconcile.is_channel_in_cooldown(prior_gap, ch, now=now):
                    carried_cooldowns[ch] = expiry
        except Exception:  # noqa: BLE001 — never let cooldown bookkeeping crash the tick
            logger.exception("self_model: channel-cooldown filtering failed; ignoring")
            carried_cooldowns = {}

    deferred_by_throttle = False
    gap_active = new_gap.magnitude > 0.0 or new_gap.unnamed_pressure > 0.0
    if gap_active:
        # Carry sustained_ticks forward when this gap continues the prior one
        # (same set of channels, prior still open); otherwise it's a fresh gap.
        prior_open = prior_gap is not None and prior_gap.status == "open"
        same_channels = (
            prior_open
            and prior_gap is not None
            and set(prior_gap.per_channel) == set(new_gap.per_channel)
        )
        new_gap.sustained_ticks = (
            (prior_gap.sustained_ticks + 1) if same_channels and prior_gap else 1
        )
        new_gap.first_seen_ts = (
            prior_gap.first_seen_ts
            if same_channels and prior_gap and prior_gap.first_seen_ts
            else now.isoformat()
        )
        new_gap.last_seen_ts = now.isoformat()
        new_gap.status = "open"
        # Preserve any cooldowns: those the reconcile tool set on the prior gap
        # (carried_cooldowns, kept across status flips) plus the same-channel
        # carry-forward when this is a continuation of an open prior gap.
        new_gap.channel_cooldowns = dict(carried_cooldowns)
        if same_channels and prior_gap:
            new_gap.channel_cooldowns.update(prior_gap.channel_cooldowns)
        increment_gaps_surfaced(persona_dir)

        # ── 5: articulate above threshold (fail-soft inside) ─────────────────
        if new_gap.magnitude >= _SELF_MODEL_ARTICULATE_THRESHOLD:
            try:
                note = sm_articulate(new_gap, provider=provider, persona_dir=persona_dir)
            except cli_throttle.ThrottleDeferred:
                # Rare peek-then-lost race (the common case never reaches here —
                # it's short-circuited by _run_self_model_tick's own pre-flight
                # peek before this function is even called). Gap computation has
                # already run for this tick; only the note stays unset.
                note = None
                deferred_by_throttle = True
            if note:
                new_gap.note = note
            elif same_channels and prior_gap is not None and prior_gap.note:
                # same_channels also requires prior_gap.status == "open" (a
                # reconciled/dismissed prior gap's note must not carry forward
                # even on a channel-set match — see supervisor.py:~1400).
                new_gap.note = prior_gap.note

    # ── 6: resolution downstream (two paths) ────────────────────────────────
    # session_id is informational on the resolved soul candidate; the
    # supervisor has no live session, so label the source.
    check_and_emit_resolution(
        prior_gap, new_gap, persona_dir=persona_dir, session_id="self_model_tick"
    )

    # ── 7: persist state (displaced-gap helper) ─────────────────────────────
    if gap_active:
        new_state = self_model_state.push_gap(state, new_gap)
    else:
        # No active gap this tick. If a prior gap is still open, displace it
        # into history (push None is not supported, so move it manually) —
        # otherwise just persist the (unchanged) state so the file exists.
        if prior_gap is not None:
            new_history = list(state.gap_history)
            new_history.append(prior_gap)
            new_state = self_model_state.SelfModelState(
                current_gap=None, gap_history=new_history[-20:]
            )
        else:
            new_state = state
    self_model_state.save(persona_dir, new_state)

    if hasattr(event_bus, "publish"):
        event_bus.publish(
            {
                "type": "self_model_tick",
                "trigger": "background",
                "gap_active": gap_active,
                "magnitude": new_gap.magnitude if gap_active else 0.0,
                "sustained_ticks": new_gap.sustained_ticks if gap_active else 0,
                "at": _now_iso(),
            }
        )

    return deferred_by_throttle


def _run_narrative_memory_pass(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> None:
    """Soul-review-cadence wrapper around narrative_memory.run_pass.

    Opens per-call MemoryStore, HebbianMatrix (ExitStack — mirrors
    `_run_finalize_tick` ownership pattern), reads FeltTimeState,
    and builds the anchor-sweep + candidate-pool + salience + is_exempt
    closures against the real stores. Dispatches to the orchestrator.

    Runs AFTER forgetting_run_pass within the same soul-review cadence
    block so memories forgetting just dropped don't enter arcs born this
    same tick. Fault-isolated upstream.
    """
    # Local imports keep the module-load surface light — narrative_memory
    # is only exercised on the (slow) soul-review cadence.
    from brain.felt_time import FeltTime
    from brain.felt_time.anchors import scan_since as anchors_scan_since
    from brain.forgetting import _load_soul_linked_ids
    from brain.forgetting.policy import is_exempt as forgetting_is_exempt
    from brain.forgetting.salience import score as forgetting_salience
    from brain.health.jsonl_reader import iter_jsonl_skipping_corrupt

    # Map felt-time anchor type -> JSONL filename (matches
    # brain.felt_time.anchors._SOURCES; v1 skips weather_shift per spec).
    anchor_sources: dict[str, tuple[str, str]] = {
        "dream": ("dreams.log.jsonl", "summary"),
        "growth": ("growth.log.jsonl", "title"),
        "soul": ("soul.log.jsonl", "moment_label"),
    }

    class _AnchorAdapter:
        """Narrative-memory `_AnchorLike` view over a felt-time Anchor.

        Pulls seed_memory_ids + lived_age_hours from the raw JSONL entry the
        anchor's source_ref points at; falls back to empty tuple / 0.0 when
        absent. The orchestrator silently skips anchors with empty
        seed_memory_ids (see brain/narrative_memory/__init__.py:108).
        """

        def __init__(
            self,
            *,
            anchor_type: str,
            ref: str,
            label: str,
            ts_iso: str,
            lived_age_hours: float,
            seed_memory_ids: tuple[str, ...],
        ) -> None:
            self.type = anchor_type
            self.ref = ref
            self.label = label
            self.ts_iso = ts_iso
            self.lived_age_hours = lived_age_hours
            self.seed_memory_ids = seed_memory_ids

    def _extract_seed_memory_ids(entry: dict) -> tuple[str, ...]:
        """Best-effort: pluck a memory-id list from a JSONL anchor entry.

        Each anchor source uses a slightly different field name; we accept
        any of the known shapes. Returns empty tuple when no field matches,
        which causes the orchestrator to skip the anchor cleanly.
        """
        for key in ("seed_memory_ids", "linked_memory_ids", "evidence_memory_ids", "memory_ids"):
            val = entry.get(key)
            if isinstance(val, (list, tuple)) and val:
                return tuple(str(x) for x in val)
        single = entry.get("memory_id")
        if isinstance(single, str) and single:
            return (single,)
        return ()

    def _adapt_anchors(persona_dir: Path, last_pass_ts: str | None) -> list[_AnchorAdapter]:
        """Convert felt-time Anchors into narrative_memory _AnchorLike views.

        Iterates the underlying JSONL once per source so we can pluck
        per-entry seed_memory_ids + lived_age_hours alongside the matched
        anchor. v1 covers dream / growth / soul; weather_shift skipped.
        """
        felt_anchors = anchors_scan_since(persona_dir, last_pass_ts)
        # Index felt anchors by (filename, idx_1based) for fast match.
        wanted: dict[tuple[str, int], object] = {}
        for fa in felt_anchors:
            if fa.type not in anchor_sources:
                continue
            try:
                filename, _ = fa.source_ref.rsplit(":", 1)
                idx = int(_)
            except (ValueError, AttributeError):
                continue
            wanted[(filename, idx)] = fa

        adapted: list[_AnchorAdapter] = []
        for anchor_type, (filename, _label_key) in anchor_sources.items():
            path = persona_dir / filename
            if not path.exists():
                continue
            for entry_idx, entry in enumerate(iter_jsonl_skipping_corrupt(path), start=1):
                fa = wanted.get((filename, entry_idx))
                if fa is None:
                    continue
                seed_ids = _extract_seed_memory_ids(entry)
                if not seed_ids:
                    # Skip anchors we can't seed — orchestrator would too.
                    continue
                lived = entry.get("lived_age_hours")
                if not isinstance(lived, (int, float)):
                    lived = 0.0
                adapted.append(
                    _AnchorAdapter(
                        anchor_type=anchor_type,
                        ref=fa.source_ref,
                        label=fa.label,
                        ts_iso=fa.ts,
                        lived_age_hours=float(lived),
                        seed_memory_ids=seed_ids,
                    )
                )
        adapted.sort(key=lambda a: a.ts_iso)
        return adapted

    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        matrix = build_embedding_matrix(store.db_path)

        # FeltTime read — get_state() is cheap, doesn't tick.
        felt_time_state = FeltTime(persona_dir=persona_dir).get_state()

        # Soul-linked ids — best-effort, mirrors forgetting wrapper.
        crystallised_ids, under_review_ids = _load_soul_linked_ids(persona_dir)
        soul_linked = crystallised_ids | under_review_ids

        class _EmbeddingsByMemoryId:
            """Adapter exposing the narrative_memory EmbeddingsView protocol.

            F1 increment 2: pure read off the warm matrix, keyed by
            memory_id directly — no more store.get() (which would bump
            recall_count) + embeddings_cache.get_or_compute() compute-on-miss
            (approved flag-3 read-only behavior: an unembedded memory simply
            has no membership vector this pass, it is never embedded as a
            side effect of a membership check). Returns None on a miss (id
            unknown to the matrix, or not yet embedded) or if the matrix
            itself raises (defensive — the membership path falls back).
            """

            def get(self, memory_id: str):
                try:
                    return matrix.get(memory_id)
                except Exception:
                    return None

        embeddings_view = _EmbeddingsByMemoryId()

        def _candidate_pool(_persona_dir, *, opened_at_iso: str):
            return store.list_since_iso(opened_at_iso, include_fading=True)

        def _salience(memory, *, ctx=None):
            return forgetting_salience(
                memory,
                store=store,
                hebbian=hebbian,
                felt_time_state=felt_time_state,
                soul_linked_ids=soul_linked,
            )

        def _is_exempt(memory):
            return forgetting_is_exempt(
                memory,
                soul_crystallised_ids=crystallised_ids,
                under_review_ids=under_review_ids,
                now_lived_age_hours=felt_time_state.lived_age_hours,
            )

        narrative_memory_run_pass(
            persona_dir,
            event_bus=event_bus,
            anchor_sweep=_adapt_anchors,
            candidate_pool=_candidate_pool,
            salience_score=_salience,
            is_exempt=_is_exempt,
            hebbian=hebbian,
            embeddings=embeddings_view,
            felt_time_state=felt_time_state,
        )


def _run_compaction_tick(
    persona_dir: Path,
    provider: LLMProvider,
    *,
    is_session_busy: Callable[[str], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> bool:
    """Run the age-gated cascade on each active conversation, then check the weekly
    session-rollover (1c-B) — cascade-fold FIRST, then rollover, so a swap seeds from
    the just-updated tiers (M2). Per-session failures are logged and do not stop the
    sweep — autonomous-behaviour recipe: defer cleanly, don't fail loudly.

    **Idle-gate (owner ruling 2026-08-13):** compaction and the weekly rollover fire
    only at startup or during idle — never mid-exchange. A session with an in-flight
    request (``is_session_busy(sid)`` true) is SKIPPED this tick (both its cascade
    and its rollover) and retried on the next idle tick. This is a best-effort
    efficiency/UX belt (don't churn an actively-used session); it is NOT the
    race-safety mechanism. The resolve-persist race is closed structurally inside
    ``perform_rollover``, whose destructive section holds ``registry_lock()`` across
    its seed re-read → successor-pointer write, serializing it against a concurrent
    live-turn persist (see brain/chat/rollover.py + session.persist_turns_following_
    successor).

    Provider is COMPACTION_MODEL (haiku) via build_compaction_provider — cost
    stays off the chat model. Called only by the central cadence function's
    compaction job (daily default, persisted cadence per #21; ram-spike-fix
    INC-9 removed the startup catch-up call — an overdue cascade runs at the
    first lull, the bridge-start lull included).

    ``should_pause`` (INC-10, S14/S32/S41/S65): checked BETWEEN sessions —
    the S32 table's item unit for compaction is "one session"
    (supervisor.py:2125 historically; per-session buffer state is already
    persisted per fold, so stopping here loses no progress). Returns True
    when it stopped early for chat with sessions still unvisited this pass
    (C8: the caller must not advance ``compaction_cadence.json`` and must
    report ``JobOutcome.PAUSED``); resuming just re-lists active sessions and
    re-applies the same age-gated cascade, which is a no-op for a session
    this pass already cascaded (nothing new is old enough yet).
    """
    from brain.chat.compaction import (
        _ROLLOVER_QUIET_GAP,
        _WEEKLY_ROLLOVER_AGE,
        cascade_conversation,
    )
    from brain.chat.rollover import maybe_weekly_rollover
    from brain.ingest.buffer import list_active_sessions

    persona_name = persona_dir.name
    session_ids = list(list_active_sessions(persona_dir))
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)

        for i, session_id in enumerate(session_ids):
            now = datetime.now(UTC)
            # Idle-gate: skip a session with an in-flight request (owner ruling) —
            # its cascade AND its rollover defer to the next idle tick. Compaction
            # is thus startup/idle-only, never mid-exchange.
            if is_session_busy is not None and is_session_busy(session_id):
                continue
            try:
                cascade_conversation(persona_dir, session_id, provider=provider, now=now)
            except Exception:
                logger.exception("compaction tick: session=%s raised", session_id)
            # 1c-B weekly rollover (quiet-moment boundary). The is_session_busy belt
            # is passed through as a second guard (defer if a request slipped in
            # between the skip-check above and here).
            try:
                maybe_weekly_rollover(
                    persona_dir, session_id, persona_name,
                    weekly_age=_WEEKLY_ROLLOVER_AGE, quiet_gap=_ROLLOVER_QUIET_GAP,
                    now=now, provider=provider,
                    store=store, hebbian=hebbian,
                    is_session_busy=is_session_busy,
                )
            except Exception:
                logger.exception("weekly rollover: session=%s raised", session_id)

            if (
                should_pause is not None
                and i < len(session_ids) - 1
                and should_pause()
            ):
                logger.info("compaction tick: pausing between sessions for chat (INC-10)")
                return True
    return False


def _run_calibration_tick(
    persona_dir: Path,
    *,
    is_session_busy: Callable[[str], bool] | None = None,
    provider: LLMProvider | None = None,
    judge: RelevanceJudgeProvider | None = None,
    should_pause: Callable[[], bool] | None = None,
) -> bool | None:
    """F2a daily calibration tick (#250, spec Section 5) — 4th sibling cadence
    to compaction/clustering/vocab-repair, mirroring ``_run_compaction_tick``'s
    idle-gate + restart-safety shape: own persisted ``calibration_cadence.json``,
    daily fire via the central cadence function (no startup catch-up since
    ram-spike-fix INC-9). Returns False when it deferred for a busy session
    (a skip: the caller does not advance the cadence), True otherwise.

    **INC5 SCOPE:** retention pruning — deletes ``calibration_log`` rows
    outside the rolling ``day_bucket`` window via ``MemoryStore.
    prune_calibration_log`` (spec Section 5's "MUST before ship" pruning
    responsibility, acceptance 5b).

    **INC6 SCOPE:** the local-judge first pass + Haiku tie-break
    (``bge-reranker-v2-m3``, spec Section 6) — labels a SAMPLE of the rows
    pruning just left behind, via ``relevance_judge.label_calibration_
    sample``.

    **INC7 SCOPE:** floor derivation + persistence (spec Section 7), via
    ``floor_calibration.derive_and_persist_floor`` — runs against the
    CURRENT production reranker's model_id
    (``reranker.build_reranker_provider().model_id()``), reading whatever
    labeled ``calibration_log`` pairs the judge pass above has accumulated
    for the MOST RECENTLY COMPLETED DAY for that model_id (pre-flip
    revision Change 1 re-points this from a pooled multi-day read to a
    day-scoped one — see ``floor_calibration``'s module docstring and
    ``MemoryStore.labeled_calibration_pairs``; the EMA smoothing and
    bootstrap-CI stability gate this docstring used to describe here are
    REMOVED by that same revision, not merely superseded). F2a inc8 (#250
    §7/§8, this cutover) wired the derived floor into
    ``select_standouts`` (``brain/memory/semantic_recall.py``) — read via
    ``store.get_reranker_floor`` live rather than the deleted
    ``semantic_recall.RERANK_FLOOR`` constant. This tick's own job stays
    unchanged: derive and WRITE the floor to ``memories.db`` (I1).

    Pre-flip revision Change 2 removed the cross-increment step this
    docstring used to describe here: an ACCEPTED floor write no longer
    invalidates a cached fp16-vs-fp32 precision decision, because that
    self-check (and its cache) no longer exist — fp16 is now a pinned
    config default (``brain/memory/reranker.py``), not a runtime probe the
    floor could ever need to re-trigger. Wrapped in its OWN try/except
    (mirrors the judge-labeling step immediately above it) — a
    floor-derivation failure must not crash the tick or undo the
    prune/labeling steps that already completed.

    ``provider``: the Haiku tie-break's generation provider. ``None`` (the
    ``run_folded`` call sites' default) builds one via ``build_tier_provider
    (persona_dir, TIER_BACKGROUND_CLASSIFIER)`` inside this function —
    mirrors ``consolidation.run_consolidation``'s per-call construction
    (never the ambient chat provider) so Haiku classification always runs
    on the cheap tier regardless of what model the persona's own chat uses.

    ``judge``: test-injection point for ``relevance_judge.
    RelevanceJudgeProvider`` — production leaves this ``None`` so
    ``label_calibration_sample`` lazily builds the real torch-backed judge
    only when this tick actually has unlabeled rows to label (never at
    import time, never on the hot path).

    **Idle-gate:** unlike compaction (which skips only the BUSY session's own
    cascade/rollover, letting idle sessions' work proceed), this tick's work
    is corpus-global, not scoped to any one session — it reads/writes across
    the whole ``calibration_log`` table rather than per-session data. So the
    gate here is a single all-or-nothing check: if ANY active session
    currently has an in-flight request, the WHOLE tick defers to the next
    firing, rather than partially running while a live turn may still be
    calling ``store.log_calibration_sample`` (#250 inc4) against the same
    table.

    Fault-isolated by the caller (the central cadence function's per-job
    guard, mirroring ``_run_clustering_tick``) — this function itself
    does not swallow the PRUNE step's errors, so those are still visible in
    that wrapping try/except. The JUDGE-LABELING step is different: spec
    Section 6 requires a judge/torch/Haiku failure to never crash the tick
    or the bridge, so that step is wrapped in its OWN try/except HERE (not
    left to the caller) — a labeling failure must not undo or block the
    prune step that already completed successfully above it.
    """
    if is_session_busy is not None:
        from brain.ingest.buffer import list_active_sessions

        if any(is_session_busy(sid) for sid in list_active_sessions(persona_dir)):
            logger.info("calibration tick: deferred, a session is busy")
            # ram-spike-fix INC-9 (S20): a deferral is a skip, not a run — the
            # caller (the central cadence function) must NOT advance the cadence.
            return False

    with ExitStack() as stack:
        # integrity_check=False mirrors the sweep/maker/notes/vocab-repair/
        # clustering ticks in this file — a full PRAGMA integrity_check on
        # every construction is unwarranted for a background cadence tick.
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)

        pruned = store.prune_calibration_log()
        logger.info("calibration tick: pruned=%d calibration_log rows outside retention window", pruned)

        try:
            from brain.memory.relevance_judge import label_calibration_sample

            tiebreak_provider = (
                provider
                if provider is not None
                else build_tier_provider(persona_dir, TIER_BACKGROUND_CLASSIFIER)
            )
            # F2c inc7: serve this persona's ONE tuned judge when its weekly
            # self-tune has accepted one — always a plain checkpoint (a full
            # fine-tune or a merged LoRA week), named by the persona's single
            # `current` pointer (`resolve_current_checkpoint`). Only when the
            # caller injected no explicit `judge` (an injected test judge is
            # never clobbered). Resolving the pointer is a cheap filesystem
            # read; the (torch) judge itself is built LAZILY inside
            # label_calibration_sample, and only when there are rows to label.
            # Absent / unresolvable → base judge (I9).
            full_model_dir: str | None = None
            if judge is None:
                from brain.memory import judge_lora

                current = judge_lora.resolve_current_checkpoint(persona_dir)
                if current is not None:
                    full_model_dir = str(current)
            label_progress: dict[str, bool] = {}
            labeled = label_calibration_sample(
                store,
                provider=tiebreak_provider,
                judge=judge,
                full_model_dir=full_model_dir,
                should_pause=should_pause,
                progress_out=label_progress,
            )
            logger.info("calibration tick: labeled=%d calibration_log rows this pass", labeled)
        except Exception:  # noqa: BLE001 — judge/torch/Haiku failure must not crash the tick
            logger.exception("calibration tick: judge-labeling pass raised; continuing")
            label_progress = {}
        finally:
            # S11/S27 (inc5), S31 (inc10): the judge is built for this tick
            # alone (label_calibration_sample above, lazily and only if there
            # were rows to label) and never kept beyond it — release its RAM
            # whether or not the pass actually built one, whether it
            # succeeded/raised/PAUSED (release_judge() is a cheap no-op when
            # nothing was loaded). This covers BOTH the finish arm and the
            # INC-10 pause arm (C2): a between-items pause mid-labeling still
            # reaches this `finally` on the very next loop iteration's break.
            from brain.memory.relevance_judge import release_judge

            release_judge()

        if label_progress.get("paused"):
            # INC-10 (C8): the between-items hook stopped labeling with rows
            # still unlabeled — skip floor derivation (it reads the SAME
            # day's labeled pairs; run it once labeling actually finishes)
            # and report PAUSED so the central cadence function does not
            # advance calibration_cadence.json (S36) and stops the S55
            # sequence here (S43). The next lull re-fires this tick; already-
            # labeled rows are never re-sampled (label_calibration_sample's
            # own docstring), so no item 1..k is redone.
            logger.info("calibration tick: paused mid-labeling; floor derivation deferred")
            return None

        try:
            from brain.memory import floor_calibration
            from brain.memory.reranker import build_reranker_provider

            current_reranker_model_id = build_reranker_provider(store=store).model_id()
            outcome = floor_calibration.derive_and_persist_floor(store, current_reranker_model_id)
            logger.info(
                "calibration tick: floor derivation for %s -> accepted=%s floor=%s "
                "cold_start=%s held_for_data_starvation=%s sample_pairs=%d",
                current_reranker_model_id,
                outcome.accepted,
                outcome.floor,
                outcome.is_cold_start,
                outcome.held_for_data_starvation,
                outcome.sample_pairs,
            )
        except Exception:  # noqa: BLE001 — floor-derivation failure must not crash the tick
            logger.exception("calibration tick: floor-derivation pass raised; continuing")
    return True


# Pre-flip revision Change 1's §6 retry gate (see `_run_deploy_
# recalibration_check` below): a persisted-cadence file (mirrors every
# other `*_cadence.json` in this module), gating ONLY the no-persisted-row
# ramp case — a genuine raw-scale-row migration always fires immediately,
# ungated. 86400.0s (one day) mirrors the grain every other cadence in this
# file already uses, and reuses Change 1's own day-boundary rather than
# inventing a second one (spec's F2b §6 interaction note: "gate §6's retry
# on the same day-boundary the backstop uses").
_DEPLOY_RECAL_RETRY_CADENCE_FILE = "deploy_recalibration_retry_cadence.json"
_DEPLOY_RECAL_RETRY_INTERVAL_SECONDS = 86400.0


def _run_deploy_recalibration_check(persona_dir: Path) -> None:
    """F2b deploy-time ONE-TIME floor recalibration (spec §6, #276 inc3).

    F2b (§5) re-points F2a's daily calibration fit to compare NORMALIZED
    (anchor-corrected) scores; §5b additionally normalizes F2a's cold-start
    and bootstrap floor sources. But a floor row PERSISTED before F2b
    started producing normalized scores is still on the RAW scale — left
    alone, that stale row stays in effect until the next scheduled daily
    calibration tick happens to re-derive it, which can be a full day (or
    longer, on a corpus still in cold-start) after the deploy that flipped
    the score scale. This is a DIFFERENT trigger than that daily cadence
    (`_run_calibration_tick`, gated on `calibration_cadence.json`'s
    wall-clock `is_due`): this check fires on the SCALE TRANSITION itself,
    at startup, regardless of when the last daily tick ran or is next due.

    Mechanism: `MemoryStore.reranker_floor_is_stale` reads the PERSISTED
    `reranker_floor_calibration` row directly — stale means ABSENT or still
    stamped `score_scale != 'normalized'`. When stale, this runs F2a's
    already-built `floor_calibration.derive_and_persist_floor` ONCE,
    out-of-cycle (the SAME function the daily tick calls — no new
    calibration algorithm).

    PRE-FLIP REVISION CHANGE 1's §6 INTERACTION (build-time gate, spec's
    "F2b §6 interaction" note): Change 1 removes `derive_and_persist_
    floor`'s old cold-start branch, which used to unconditionally persist
    SOME row (bundled-pair fit) on every call — that write was what made
    `reranker_floor_is_stale` false again after the FIRST out-of-cycle
    pass, regardless of how little real data existed yet. Change 1's
    replacement backstop can instead write NOTHING (the no-prior-row edge
    case — a fresh deploy still short of `FLOOR_FIT_MIN_LABELED_PAIRS` on
    its most recent day), which leaves the row ABSENT — and an absent row
    reads as stale FOREVER until real data clears the threshold. Without a
    gate, THIS function would re-attempt `derive_and_persist_floor` (itself
    now a cheap no-op in that state) on every single bridge restart during
    the ramp, which is wasteful and noisy.

    The fix distinguishes the two genuinely different reasons a row can
    read stale, per the two properties this gate must satisfy simultaneously:
      - a row EXISTS but is still raw-scale (`get_persisted_reranker_floor`
        returns a row with `score_scale != 'normalized'`) — this IS the §6
        migration case and fires UNCONDITIONALLY on this restart, exactly
        as originally spec'd; it is never gated.
      - NO row exists at all (`get_persisted_reranker_floor` returns
        `None`) — this is either a true pre-Change-1 fresh install (no
        change in behavior: the very first attempt still fires
        immediately, so a genuinely fresh corpus that already has enough
        data gets its floor right away) OR Change 1's own ramp state
        (not enough data yet — this attempt will just no-op again). Only
        THIS no-row case is rate-limited, via `persisted_cadence` (the SAME
        module every other cadence in this file already uses): at most one
        attempt per `_DEPLOY_RECAL_RETRY_INTERVAL_SECONDS` (one day,
        mirroring the backstop's own day-boundary rather than inventing a
        second time-windowing scheme), tracked in its own small JSON side
        file — NOT in `memories.db` (no schema/table addition; this is
        cadence bookkeeping, not calibration data, exactly what `persisted_
        cadence.py` exists for). Once real data clears the threshold on
        some day, `derive_and_persist_floor` writes a real row and
        `reranker_floor_is_stale` goes False on the very next check
        regardless of this gate's state — self-healing, per the spec.

    Idempotent BY CONSTRUCTION once a row exists, not by a separate one-shot
    flag: an accepted `derive_and_persist_floor` call always stamps the
    fresh row `score_scale = CALIBRATION_SCORE_SCALE` (`MemoryStore.
    write_reranker_floor`'s default), so the very next call to this
    function — another startup, or interleaved with a normal daily tick —
    reads `reranker_floor_is_stale() == False` and returns immediately
    without re-deriving. This survives a real process restart (the marker
    lives in `memories.db`, not in-process state), unlike a boot-time flag
    — and the no-row gate above survives a real restart the same way (a
    JSON file under `persona_dir`, not in-process state).

    Run by the central cadence function immediately before the daily
    calibration job (INC-9) — but unlike that job (which reuses `_run_calibration_tick`'s prune + judge-label +
    cadence-scoped floor derivation), this is deliberately its OWN,
    narrower function: no pruning, no judge-labeling, no cadence-due check
    on the migration path — only the one scale-transition floor derivation,
    so a deploy recovers a scale-correct floor even when the daily cadence
    itself is disabled for a long window or has not yet come due.

    Fault-isolated by the CALLER (the central cadence function's per-job
    guard) — a failure constructing the store, the reranker provider, or
    running the derivation must never crash the supervisor; the stale row
    (or absence of one) is simply picked up again at a later lull.

    ram-spike-fix INC-9 (S70/S73, C38): this is no longer a startup one-shot.
    It is a gated job of the central cadence function
    (``brain/bridge/central_cadence.py``), run at the first lull (the
    bridge-start lull included) immediately before daily calibration: its
    "due" half is ``_deploy_recalibration_due`` and its "run" half is
    ``_run_deploy_recalibration``. This wrapper (due, then run) is kept for
    direct callers.
    """
    if _deploy_recalibration_due(persona_dir):
        _run_deploy_recalibration(persona_dir)


def _deploy_recalibration_due(persona_dir: Path) -> bool:
    """The deploy recalibration job's due predicate (INC-9, S70/S73, C38):
    the stored floor for the current reranker model is stale (absent, or
    still raw-scale) AND, when NO row exists at all, the existing daily
    no-row retry file (``_DEPLOY_RECAL_RETRY_CADENCE_FILE``) is due — today's
    daily retry, kept as is (a missing retry file is due, so a floor row that
    is absent runs at the first lull). A raw-scale row is always due. Read-
    only: writes nothing."""
    with ExitStack() as stack:
        # integrity_check=False mirrors every other background-tick store
        # open in this file (compaction/calibration/clustering ticks).
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)

        from brain.memory.reranker import build_reranker_provider

        current_reranker_model_id = build_reranker_provider(store=store).model_id()

        if not store.reranker_floor_is_stale(current_reranker_model_id):
            logger.debug(
                "deploy recalibration check: %s already normalized-scale, not due",
                current_reranker_model_id,
            )
            return False

        # Change 1's §6 gate (see _run_deploy_recalibration_check's docstring):
        # only the no-row case is rate-limited — an existing raw-scale row is
        # always due.
        if store.get_persisted_reranker_floor(current_reranker_model_id) is None:
            retry_cadence = persisted_cadence.load_cadence(
                persona_dir, _DEPLOY_RECAL_RETRY_CADENCE_FILE
            )
            if not persisted_cadence.is_due(retry_cadence, now=datetime.now(UTC)):
                logger.debug(
                    "deploy recalibration check: %s has no persisted floor yet (Change 1's "
                    "data-starvation ramp, or a genuinely fresh install) and was already "
                    "retried within the last %.0fs — not due; the daily calibration tick "
                    "stays the authoritative path once enough data accumulates",
                    current_reranker_model_id,
                    _DEPLOY_RECAL_RETRY_INTERVAL_SECONDS,
                )
                return False
        return True


def _run_deploy_recalibration(persona_dir: Path) -> None:
    """The deploy recalibration job's run half (INC-9): one out-of-cycle
    ``derive_and_persist_floor`` for the current reranker model. In the
    no-row case the daily retry file advances once the attempt returns or
    raises (S44), never before it (S36: a process death mid-attempt leaves
    it due). Re-checks staleness first, so a floor made fresh between the due
    check and this call is left alone."""
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)

        from brain.memory.reranker import build_reranker_provider

        current_reranker_model_id = build_reranker_provider(store=store).model_id()
        if not store.reranker_floor_is_stale(current_reranker_model_id):
            return
        no_row = store.get_persisted_reranker_floor(current_reranker_model_id) is None

        from brain.memory import floor_calibration

        try:
            outcome = floor_calibration.derive_and_persist_floor(
                store, current_reranker_model_id
            )
            logger.info(
                "deploy recalibration check: out-of-cycle floor derivation for %s -> "
                "accepted=%s floor=%s cold_start=%s held_for_data_starvation=%s "
                "sample_pairs=%d",
                current_reranker_model_id,
                outcome.accepted,
                outcome.floor,
                outcome.is_cold_start,
                outcome.held_for_data_starvation,
                outcome.sample_pairs,
            )
        finally:
            if no_row:
                persisted_cadence.save_cadence(
                    persona_dir,
                    _DEPLOY_RECAL_RETRY_CADENCE_FILE,
                    persisted_cadence.advance(
                        now=datetime.now(UTC), interval_s=_DEPLOY_RECAL_RETRY_INTERVAL_SECONDS
                    ),
                )


def _run_finalize_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    *,
    finalize_after_hours: float,
    should_pause: Callable[[], bool] | None = None,
) -> bool:
    """Run one finalize pass — per-tick stores, then drop registry entries
    for every session that was finalized.

    Mirrors the per-tick store ownership pattern of `_run_heartbeat_tick`:
    opens MemoryStore + HebbianMatrix inside this thread, closes them via
    ExitStack. The supervisor follows up by calling remove_session() for
    each finalized session — finalize itself doesn't touch the in-memory
    registry.

    ``should_pause`` (INC-10): forwarded to ``finalize_stale_sessions``,
    which checks it between finalized sessions (S32: item = one stale
    session). Returns True when it stopped early for chat.
    """
    paused_out: list[bool] = []
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)

        reports = finalize_stale_sessions(
            persona_dir,
            finalize_after_hours=finalize_after_hours,
            store=store,
            hebbian=hebbian,
            provider=provider,
            should_pause=should_pause,
            paused_out=paused_out,
        )

    for r in reports:
        remove_session(r.session_id)
        event_bus.publish(
            {
                "type": "session_finalized",
                "session_id": r.session_id,
                "committed": r.committed,
                "enqueued": r.enqueued,
                "deduped": r.deduped,
                "errors": r.errors,
                "at": _now_iso(),
            }
        )
    return bool(paused_out)


def _run_initiate_review_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
    *,
    should_pause: Callable[[], bool] | None = None,
) -> bool:
    """Build voice template + invoke run_initiate_review_tick.

    Mirrors _run_soul_review_tick's per-tick store-ownership pattern.
    Reads ``voice.md`` from the persona dir (empty string if absent)
    and ``initiate_review_cap_per_tick`` from PersonaConfig (default 3).
    Publishes an ``initiate_review_tick`` event on success.
    """
    voice_path = persona_dir / "voice.md"
    voice_template = voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""
    try:
        config = PersonaConfig.load(persona_dir / "persona_config.json")
        cap_per_tick = getattr(config, "initiate_review_cap_per_tick", 3) or 3
    except Exception:
        cap_per_tick = 3
    try:
        _user_presence = compute_user_presence(persona_dir)
    except Exception:
        logger.debug("_run_initiate_review_tick: compute_user_presence failed", exc_info=True)
        _user_presence = None
    # Soft rest gate: suppress recall-resonance when body energy is low.
    # Fail-open: a body-read error must never permanently silence her.
    is_rest_state = False
    try:
        from brain.body.state import compute_body_state
        from brain.body.words import count_words_in_session
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import MemoryStore, _row_to_memory
        from brain.utils.memory import days_since_human

        _store = MemoryStore(persona_dir / "memories.db")
        _now = datetime.now(UTC)
        _rows = _store._conn.execute(  # noqa: SLF001
            "SELECT * FROM memories "
            "WHERE active = 1 "
            "AND emotions_json IS NOT NULL "
            "AND emotions_json != '{}' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        _memories = [_row_to_memory(row) for row in _rows]
        _emotion_state = aggregate_state(_memories)
        _days = days_since_human(_store, now=_now, persona_dir=persona_dir)
        _words = count_words_in_session(
            _store, persona_dir=persona_dir, session_hours=0.0, now=_now
        )
        _body = compute_body_state(
            emotions=_emotion_state.emotions,
            session_hours=0.0,
            words_written=_words,
            days_since_contact=_days,
            now=_now,
        )
        is_rest_state = _rest_state_from_energy(_body.energy)
    except Exception:
        logger.debug(
            "body-energy rest gate read failed; not resting (fail-open)",
            exc_info=True,
        )
        is_rest_state = False  # fail-open: a body bug must never silence her permanently
    _paused_out: list[bool] = []
    run_initiate_review_tick(
        persona_dir,
        provider=provider,
        voice_template=voice_template,
        cap_per_tick=cap_per_tick,
        user_presence=_user_presence,
        is_rest_state=is_rest_state,
        should_pause=should_pause,
        paused_out=_paused_out,
    )
    event_bus.publish(
        {
            "type": "initiate_review_tick",
            "at": _now_iso(),
        }
    )
    return bool(_paused_out)


def _run_voice_reflection_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> None:
    """Gather inputs and invoke run_voice_reflection_tick.

    Reads the last 7 days of crystallizations (from SoulStore) and dream
    memories (from memories.db), each with a short text excerpt so the
    reflection can ground its evidence (#202). The message-tone stream that
    used to sit alongside them was a permanent ``[]`` placeholder; retired.
    Publishes a ``voice_reflection_tick`` event on success.
    """
    from brain.initiate.voice_reflection import run_voice_reflection_tick

    crystallizations = _read_recent_crystallizations(persona_dir, days=7)
    dreams = _read_recent_dreams(persona_dir, days=7)
    run_voice_reflection_tick(
        persona_dir,
        provider=provider,
        crystallizations=crystallizations,
        dreams=dreams,
        companion_name=persona_dir.name,
    )
    event_bus.publish(
        {
            "type": "voice_reflection_tick",
            "at": _now_iso(),
        }
    )


_VOICE_EVIDENCE_CHARS = 160


def _read_recent_crystallizations(persona_dir: Path, days: int) -> list[dict]:
    """Recent crystallizations as ``{"id", "ts", "text"}`` (#202: text, not just ids).

    ``text`` is the moment plus why it matters, capped. Failures swallowed —
    reflection still fires with whatever evidence exists.
    """
    from brain.soul.store import SoulStore

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        store = SoulStore(str(persona_dir / "crystallizations.db"))
        try:
            out: list[dict] = []
            for c in store.list_active():
                ts = c.crystallized_at.isoformat()
                if ts >= cutoff:
                    text = f"{c.moment} — {c.why_it_matters}".strip(" —")
                    out.append({"id": c.id, "ts": ts, "text": text[:_VOICE_EVIDENCE_CHARS]})
            return out
        finally:
            store.close()
    except Exception:
        return []


def _read_recent_dreams(persona_dir: Path, days: int) -> list[dict]:
    """Recent dream memories as ``{"id", "ts", "text"}`` (#202).

    Reads ``memory_type="dream"`` rows from memories.db rather than
    ``dreams.log.jsonl``: the log carries ids only (no text) and its
    ``timestamp`` field never matched the ``at``/``ts`` keys the old reader
    looked for, so dreams were silently absent from the evidence.
    """
    from brain.memory.store import MemoryStore

    cutoff = datetime.now(UTC) - timedelta(days=days)
    try:
        store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
        try:
            out: list[dict] = []
            for mem in store.list_by_type("dream", limit=50):
                if mem.created_at >= cutoff:
                    out.append(
                        {
                            "id": mem.id,
                            "ts": mem.created_at.isoformat(),
                            "text": mem.content[:_VOICE_EVIDENCE_CHARS],
                        }
                    )
            return out
        finally:
            store.close()
    except Exception:
        return []


_ROLLING_LOG_POLICIES: tuple[tuple[str, int], ...] = (
    ("heartbeats.log.jsonl", 3),
    ("dreams.log.jsonl", 5),
    ("emotion_growth.log.jsonl", 5),
    ("chat_usage.jsonl", 5),
    ("file_access.jsonl", 5),
    ("attunement_errors.jsonl", 5),
    ("gate_rejections.jsonl", 3),
    ("self_model_articulate_errors.jsonl", 5),
)
# This table is deliberately hand-maintained, NOT derived from the jsonl files
# present on disk. Several persona jsonl files are stores or queues, not logs —
# soul_candidates and initiate_candidates are review queues, forgotten_memories
# is the recovery graveyard, and adaptive-D reads every row of initiate_d_calls
# with no time filter. Rotating any of those silently destroys live state. Add
# entries here only after confirming the file has no reader, or a reader with a
# bounded window shorter than the retained history.
# Yearly-archive logs are kept forever — reader walks active + every archive
# so every decision / initiation event stays reachable.
_YEARLY_ARCHIVE_LOGS: tuple[tuple[str, str], ...] = (
    ("soul_audit.jsonl", "ts"),
    ("initiate_audit.jsonl", "ts"),
)
_DEFAULT_ROLLING_BYTES = 5 * 1024 * 1024  # 5 MB



def _run_log_rotation_tick(
    persona_dir: Path,
    event_bus: EventBus | object,
    *,
    rolling_size_bytes: int = _DEFAULT_ROLLING_BYTES,
    now: datetime | None = None,
) -> None:
    """Rotate JSONL audit logs per the baked policy table.

    Fault-isolated per-log: a failure in one rotation doesn't block the
    others. Each successful rotation publishes a structured
    ``log_rotation`` event.

    Args:
        persona_dir: persona root; logs live as immediate children.
        event_bus: target for ``log_rotation`` events.
        rolling_size_bytes: cap for rolling-size rotation (test override).
        now: current datetime (test override for yearly archive).
    """
    # Rolling-size logs.
    for log_name, keep in _ROLLING_LOG_POLICIES:
        log_path = persona_dir / log_name
        try:
            archive = rotate_rolling_size(log_path, max_bytes=rolling_size_bytes, archive_keep=keep)
        except Exception as exc:
            logger.exception("log rotation failed for %s: %s", log_name, exc)
            event_bus.publish(
                {
                    "type": "log_rotation",
                    "log": log_name,
                    "action": "failed",
                    "error": str(exc),
                    "at": _now_iso(),
                }
            )
            continue
        if archive is not None:
            event_bus.publish(
                {
                    "type": "log_rotation",
                    "log": log_name,
                    "action": "rotated",
                    "archive": archive.name,
                    "at": _now_iso(),
                }
            )

    # Yearly-archive logs (forever-keep): soul_audit + initiate_audit.
    for log_name, ts_field in _YEARLY_ARCHIVE_LOGS:
        audit_path = persona_dir / log_name
        try:
            archives = rotate_age_archive_yearly(audit_path, now=now, timestamp_field=ts_field)
        except Exception as exc:
            logger.exception("%s yearly rotation failed: %s", log_name, exc)
            event_bus.publish(
                {
                    "type": "log_rotation",
                    "log": log_name,
                    "action": "failed",
                    "error": str(exc),
                    "at": _now_iso(),
                }
            )
            continue
        for archive in archives:
            event_bus.publish(
                {
                    "type": "log_rotation",
                    "log": log_name,
                    "action": "archived",
                    "archive": archive.name,
                    "at": _now_iso(),
                }
            )


def _run_clustering_tick(persona_dir: Path) -> None:
    """Stage 5 (#157) — one full numpy k-means pass over the persona's
    currently-embedded vectors, off the message hot path (own persisted
    cadence — see ``clustering_interval_s`` on ``run_folded``, default 6h).

    F1 #259 increment 4: sources vectors from the warm ``EmbeddingMatrix``
    over ``memories.db`` (via ``run_clustering_pass(store)``) and writes
    ``cluster_id``/``cluster_model_id`` onto the ``memories`` row plus the
    ``cluster_centroids`` table — this tick no longer opens
    ``EmbeddingCache``/``MemoryClusterStore`` against ``embeddings.db`` and
    does not touch that file at all anymore.

    Opens its own ``MemoryStore`` (ExitStack — mirrors
    ``_run_log_rotation_tick``/``_run_narrative_memory_pass``'s per-call
    ownership pattern) since the per-tick handles opened earlier in
    ``run_folded``'s loop are already closed by the time this cadence block
    runs. Local import keeps the module-load surface light — clustering is
    only exercised on its own slow cadence, same rationale as narrative
    memory's local imports above.
    """
    from brain.memory.clustering import run_clustering_pass

    with ExitStack() as stack:
        # integrity_check=False mirrors the sweep/maker/notes/vocab-repair
        # ticks in this file (F1 #259 increment 7) — a full PRAGMA
        # integrity_check on every construction is unwarranted for a
        # background cadence tick; deep checks are health.py's job.
        store = MemoryStore(persona_dir / "memories.db", integrity_check=False)
        stack.callback(store.close)

        result = run_clustering_pass(store)
        logger.info(
            "clustering tick: ran=%s n_vectors=%d k=%d reason=%s",
            result.ran,
            result.n_vectors,
            result.k,
            result.reason,
        )


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
