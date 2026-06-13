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
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.engines.heartbeat import HeartbeatResult

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
from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider
from brain.chat.session import prune_empty_sessions, remove_session
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
    run_emotion_backfill as _emotion_backfill_run,
)
from brain.ingest.emotion_backfill import (
    should_run_emotion_backfill as _emotion_backfill_should_run,
)
from brain.ingest.pipeline import (
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
from brain.initiate.review import _rest_state_from_energy, run_initiate_review_tick
from brain.initiate.user_pattern import compute_user_presence
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.narrative_memory import run_pass as narrative_memory_run_pass
from brain.persona_config import PersonaConfig
from brain.soul import cadence as soul_cadence

logger = logging.getLogger(__name__)

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
    silence_minutes: float = 5.0,
    heartbeat_interval_s: float | None = 900.0,
    soul_review_interval_s: float | None = 6 * 3600.0,
    finalize_after_hours: float = 24.0,
    finalize_interval_s: float | None = 3600.0,
    log_rotation_interval_s: float | None = 3600.0,
    initiate_review_interval_s: float | None = 900.0,
    voice_reflection_interval_s: float | None = 86400.0,
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
    """
    logger.info(
        "supervisor folded persona=%s tick=%.2fs heartbeat=%s soul_review=%s finalize=%s",
        persona_dir.name,
        tick_interval_s,
        f"{heartbeat_interval_s:.0f}s" if heartbeat_interval_s is not None else "disabled",
        f"{soul_review_interval_s:.0f}s" if soul_review_interval_s is not None else "disabled",
        f"{finalize_interval_s:.0f}s" if finalize_interval_s is not None else "disabled",
    )
    last_heartbeat_at = time.monotonic() if heartbeat_interval_s is not None else None
    # Soul review is PERSISTED (survives restart/sleep); forgetting+narrative
    # stay on a monotonic timer (last_maintenance_at), decoupled so a soul-review
    # catch-up burst doesn't drag narrative's LLM calls onto the fast cadence.
    soul_cadence_state = (
        soul_cadence.load_cadence_state(persona_dir)
        if soul_review_interval_s is not None
        else None
    )
    last_maintenance_at = time.monotonic() if soul_review_interval_s is not None else None
    _last_intensity_drivers: IntensityDrivers | None = None
    last_finalize_at = time.monotonic() if finalize_interval_s is not None else None
    last_log_rotation_at = time.monotonic() if log_rotation_interval_s is not None else None
    last_initiate_review_at = time.monotonic() if initiate_review_interval_s is not None else None
    last_voice_reflection_at = time.monotonic() if voice_reflection_interval_s is not None else None

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

    # One-shot startup: re-tag emotion-less memories so existing personas
    # benefit from the A2 forward-only emotion seeding.  Independent of the
    # attunement backfill (separate if, not elif) — both can fire on the same
    # startup if needed.  Fault-isolated per autonomous-behaviour recipe item 3.
    try:
        if _emotion_backfill_should_run(persona_dir):
            _emotion_backfill_run(persona_dir, provider=provider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("emotion backfill failed during startup: %s", exc)

    # One-shot startup: repair already-stubbed emotion_vocabulary.json entries.
    # Step 1 bumps decay_half_life_days from the bad 1.0 → 14.0 (sync,
    # provider-free, so it always lands). Step 2 re-derives descriptions via
    # Haiku (fail-soft — placeholders kept if provider unavailable/fails).
    # Runs adjacent to emotion backfill; independent of it (separate try/except).
    try:
        if _vocab_repair_should_run(persona_dir):
            from brain.memory.store import MemoryStore as _MemoryStore

            db_path = persona_dir / "memories.db"
            _store = _MemoryStore(str(db_path), integrity_check=False)
            try:
                _vocab_repair_run(persona_dir, store=_store, provider=provider)
            finally:
                _store.close()
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

    while not stop_event.is_set():
        try:
            with ExitStack() as stack:
                store = MemoryStore(persona_dir / "memories.db")
                stack.callback(store.close)
                hebbian = HebbianMatrix(persona_dir / "hebbian.db")
                stack.callback(hebbian.close)
                embeddings = EmbeddingCache(
                    persona_dir / "embeddings.db",
                    FakeEmbeddingProvider(dim=256),
                )
                stack.callback(embeddings.close)

                reports = snapshot_stale_sessions(
                    persona_dir,
                    silence_minutes=silence_minutes,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                    embeddings=embeddings,
                )
                # Snapshot is NON-destructive — do NOT call remove_session
                # here. Session lifecycle is owned by finalize_stale_sessions
                # below and the explicit /sessions/close path.
                pruned_empty_sessions = prune_empty_sessions(
                    older_than_seconds=silence_minutes * 60.0,
                    persona_name=persona_dir.name,
                )

            # Publish events outside the with-block — events don't need stores.
            for r in reports:
                event_bus.publish(
                    {
                        "type": "session_snapshot",
                        "session_id": r.session_id,
                        "extracted_since_cursor": r.extracted,
                        "committed": r.committed,
                        "deduped": r.deduped,
                        "soul_candidates": r.soul_candidates,
                        "errors": r.errors,
                        "at": _now_iso(),
                    }
                )
            event_bus.publish(
                {
                    "type": "supervisor_tick",
                    "closed_sessions": len(reports),
                    "pruned_empty_sessions": len(pruned_empty_sessions),
                    "next_tick_in_s": tick_interval_s,
                    "at": _now_iso(),
                }
            )
        except Exception:
            logger.exception("supervisor tick raised")

        # Heartbeat cadence — independent of session-cleanup cadence.
        # Fault-isolated so a heartbeat failure can't take down the
        # session-cleanup loop or cascade into bridge shutdown.
        if (
            heartbeat_interval_s is not None
            and last_heartbeat_at is not None
            and time.monotonic() - last_heartbeat_at >= heartbeat_interval_s
        ):
            _last_intensity_drivers = _heartbeat_and_felt_time(
                persona_dir, provider, event_bus, last_heartbeat_at
            )
            last_heartbeat_at = time.monotonic()

        # Soul-review cadence — slowest of the three. Each pass is up to
        # 5 LLM calls (one per candidate). Fault-isolated so a model
        # outage doesn't take the supervisor down.
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
                    persona_dir, provider, event_bus
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

        # Maintenance cadence — forgetting + narrative, on the same interval but
        # a monotonic timer (decoupled from soul review above so a soul-review
        # catch-up burst doesn't run narrative's LLM calls every 30 min).
        # Decoupling is safe: forgetting already exempts under-review soul-linked
        # memories, so pass order relative to soul review doesn't matter.
        if (
            soul_review_interval_s is not None
            and last_maintenance_at is not None
            and time.monotonic() - last_maintenance_at >= soul_review_interval_s
        ):
            try:
                forgetting_run_pass(
                    persona_dir,
                    event_bus=event_bus,
                    intensity_drivers=_last_intensity_drivers,
                )
            except Exception:
                logger.exception("supervisor forgetting pass raised")
            # Narrative-memory arc-update runs AFTER forgetting so a memory
            # forgetting just dropped doesn't enter an arc born this tick.
            try:
                _run_narrative_memory_pass(persona_dir, provider, event_bus)
            except Exception:
                logger.exception("supervisor narrative-memory pass raised")
            last_maintenance_at = time.monotonic()

        # Finalize cadence — 24h silence (default) or explicit. Each pass
        # runs at most one final snapshot per stale session, then deletes
        # buffer + cursor + registry entry. Slow cadence (hourly default)
        # because the threshold is days, not minutes.
        if (
            finalize_interval_s is not None
            and last_finalize_at is not None
            and time.monotonic() - last_finalize_at >= finalize_interval_s
        ):
            try:
                _run_finalize_tick(
                    persona_dir,
                    provider,
                    event_bus,
                    finalize_after_hours=finalize_after_hours,
                )
            except Exception:
                logger.exception("supervisor finalize tick raised")
            last_finalize_at = time.monotonic()

        # Log-rotation cadence — hourly default. Bounded JSONL archives so
        # heartbeats/dreams/emotion_growth don't grow forever; yearly
        # archive for soul_audit (every decision must remain reachable).
        # Fault-isolated per-log inside the tick function.
        if (
            log_rotation_interval_s is not None
            and last_log_rotation_at is not None
            and time.monotonic() - last_log_rotation_at >= log_rotation_interval_s
        ):
            try:
                _run_log_rotation_tick(persona_dir, event_bus)
            except Exception:
                logger.exception("supervisor log-rotation tick raised")
            last_log_rotation_at = time.monotonic()

        # Initiate review cadence — mirrors soul_review. Per-pass cost cap
        # (3 candidates max). Fault-isolated.
        if (
            initiate_review_interval_s is not None
            and last_initiate_review_at is not None
            and time.monotonic() - last_initiate_review_at >= initiate_review_interval_s
        ):
            try:
                _run_initiate_review_tick(persona_dir, provider, event_bus)
            except Exception:
                logger.exception("supervisor initiate-review tick raised")
            last_initiate_review_at = time.monotonic()

        # Voice-reflection cadence — daily by default. Gathers last 7 days
        # of crystallizations + dreams + message tones and may emit a
        # voice-edit candidate (gated by >=3 evidence items inside the
        # reflection tick itself). Fault-isolated.
        if (
            voice_reflection_interval_s is not None
            and last_voice_reflection_at is not None
            and time.monotonic() - last_voice_reflection_at >= voice_reflection_interval_s
        ):
            try:
                _run_voice_reflection_tick(persona_dir, provider, event_bus)
            except Exception:
                logger.exception("supervisor voice-reflection tick raised")
            last_voice_reflection_at = time.monotonic()

        # Wait for the next tick or for stop_event, whichever comes first.
        stop_event.wait(timeout=tick_interval_s)
    logger.info("supervisor stopped persona=%s", persona_dir.name)


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
    """
    from brain.felt_time.chat_log import count_chat_turns_since

    heartbeat_result = None
    try:
        heartbeat_result = _run_heartbeat_tick(persona_dir, provider, event_bus)
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
            persona_system_prompt=f"You are {persona_dir.name}.",
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


def _run_narrative_memory_pass(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> None:
    """Soul-review-cadence wrapper around narrative_memory.run_pass.

    Opens per-call MemoryStore, HebbianMatrix, EmbeddingCache (ExitStack —
    mirrors `_run_finalize_tick` ownership pattern), reads FeltTimeState,
    and builds the anchor-sweep + candidate-pool + salience + is_exempt
    closures against the real stores. Dispatches to the orchestrator.

    Runs AFTER forgetting_run_pass within the same soul-review cadence
    block so memories forgetting just dropped don't enter arcs born this
    same tick. Fault-isolated upstream.
    """
    # Local imports keep the module-load surface light — narrative_memory
    # is only exercised on the (slow) soul-review cadence.
    import numpy as np

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
        embeddings_cache = EmbeddingCache(
            persona_dir / "embeddings.db",
            FakeEmbeddingProvider(dim=256),
        )
        stack.callback(embeddings_cache.close)

        # FeltTime read — get_state() is cheap, doesn't tick.
        felt_time_state = FeltTime(persona_dir=persona_dir).get_state()

        # Soul-linked ids — best-effort, mirrors forgetting wrapper.
        crystallised_ids, under_review_ids = _load_soul_linked_ids(persona_dir)
        soul_linked = crystallised_ids | under_review_ids

        class _EmbeddingsByMemoryId:
            """Adapter exposing the narrative_memory EmbeddingsView protocol.

            Maps memory_id -> content -> cached vector via store + embedding
            cache. Returns None when the memory is missing or the embedding
            provider raises (defensive — the membership path falls back).
            """

            def get(self, memory_id: str):
                try:
                    mem = store.get(memory_id)
                    if mem is None:
                        return None
                    vec = embeddings_cache.get_or_compute(mem.content)
                    return np.asarray(vec)
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


def _run_finalize_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
    *,
    finalize_after_hours: float,
) -> None:
    """Run one finalize pass — per-tick stores, then drop registry entries
    for every session that was finalized.

    Mirrors the per-tick store ownership pattern of `_run_heartbeat_tick`:
    opens MemoryStore + HebbianMatrix + EmbeddingCache inside this thread,
    closes them via ExitStack. The supervisor follows up by calling
    remove_session() for each finalized session — finalize itself doesn't
    touch the in-memory registry.
    """
    with ExitStack() as stack:
        store = MemoryStore(persona_dir / "memories.db")
        stack.callback(store.close)
        hebbian = HebbianMatrix(persona_dir / "hebbian.db")
        stack.callback(hebbian.close)
        embeddings = EmbeddingCache(
            persona_dir / "embeddings.db",
            FakeEmbeddingProvider(dim=256),
        )
        stack.callback(embeddings.close)

        reports = finalize_stale_sessions(
            persona_dir,
            finalize_after_hours=finalize_after_hours,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )

    for r in reports:
        remove_session(r.session_id)
        event_bus.publish(
            {
                "type": "session_finalized",
                "session_id": r.session_id,
                "committed": r.committed,
                "deduped": r.deduped,
                "errors": r.errors,
                "at": _now_iso(),
            }
        )


def _run_initiate_review_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> None:
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
    run_initiate_review_tick(
        persona_dir,
        provider=provider,
        voice_template=voice_template,
        cap_per_tick=cap_per_tick,
        user_presence=_user_presence,
        is_rest_state=is_rest_state,
    )
    event_bus.publish(
        {
            "type": "initiate_review_tick",
            "at": _now_iso(),
        }
    )


def _run_voice_reflection_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus | object,
) -> None:
    """Gather inputs and invoke run_voice_reflection_tick.

    Reads the last 7 days of crystallizations (from SoulStore), dreams
    (from ``dreams.log.jsonl``), and recent message tones (placeholder
    for v0.0.9 — empty list until the chat-turn tone schema lands).
    Publishes a ``voice_reflection_tick`` event on success.
    """
    from brain.initiate.voice_reflection import run_voice_reflection_tick

    crystallizations = _read_recent_crystallizations(persona_dir, days=7)
    dreams = _read_recent_dreams(persona_dir, days=7)
    recent_tones = _read_recent_message_tones(persona_dir, days=7)
    run_voice_reflection_tick(
        persona_dir,
        provider=provider,
        crystallizations=crystallizations,
        dreams=dreams,
        recent_tones=recent_tones,
        companion_name=persona_dir.name,
    )
    event_bus.publish(
        {
            "type": "voice_reflection_tick",
            "at": _now_iso(),
        }
    )


def _read_recent_crystallizations(persona_dir: Path, days: int) -> list[dict]:
    """Read recent crystallization summaries from SoulStore.

    Returns a list of ``{"id": ..., "ts": iso8601}`` dicts for
    crystallizations created within the last ``days`` days. Failures
    swallowed — reflection still fires with whatever evidence exists.
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
                    out.append({"id": c.id, "ts": ts})
            return out
        finally:
            store.close()
    except Exception:
        return []


def _read_recent_dreams(persona_dir: Path, days: int) -> list[dict]:
    """Read recent dream entries from ``dreams.log.jsonl``."""
    from brain.health.jsonl_reader import iter_jsonl_streaming

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    out: list[dict] = []
    try:
        for raw in iter_jsonl_streaming(persona_dir / "dreams.log.jsonl"):
            ts = raw.get("at") or raw.get("ts")
            if ts and ts >= cutoff:
                out.append({"id": raw.get("dream_id") or raw.get("id"), "ts": ts})
    except Exception:
        return []
    return out


def _read_recent_message_tones(persona_dir: Path, days: int) -> list[dict]:
    """Read recent Nell-authored chat turn tones — placeholder for v0.0.9.

    Real implementation requires schema on the chat-turn log we don't
    currently have. For v0.0.9, return an empty list; voice reflection
    still fires but with less material. Revisit when chat-turn tone
    tracking is added.
    """
    return []


# Per-log retention policies. Bake the cadence-tick policies here so the
# supervisor doesn't need a config file; defaults reflect Hana's 2026-05-11
# decisions (5 MB rolling cap; 3 archives for heartbeats, 5 for dreams +
# emotion_growth; yearly archive for soul_audit kept forever).
_ROLLING_LOG_POLICIES: tuple[tuple[str, int], ...] = (
    ("heartbeats.log.jsonl", 3),
    ("dreams.log.jsonl", 5),
    ("emotion_growth.log.jsonl", 5),
    ("chat_usage.jsonl", 5),
    ("file_access.jsonl", 5),
    ("attunement_errors.jsonl", 5),
)
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


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
