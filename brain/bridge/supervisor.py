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
from datetime import UTC, datetime
from pathlib import Path

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
from brain.ingest.pipeline import (
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
from brain.initiate.review import run_initiate_review_tick
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

logger = logging.getLogger(__name__)


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
    last_soul_review_at = time.monotonic() if soul_review_interval_s is not None else None
    last_finalize_at = time.monotonic() if finalize_interval_s is not None else None
    last_log_rotation_at = time.monotonic() if log_rotation_interval_s is not None else None
    last_initiate_review_at = time.monotonic() if initiate_review_interval_s is not None else None
    last_voice_reflection_at = time.monotonic() if voice_reflection_interval_s is not None else None
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
            try:
                _run_heartbeat_tick(persona_dir, provider, event_bus)
            except Exception:
                logger.exception("supervisor heartbeat tick raised")
            try:
                wall_s = time.monotonic() - last_heartbeat_at
                # heartbeats/chat_turns/reflex_firings counter sources are
                # placeholders (1/0/0) for v1 — Phase 9.2 (follow-up) tightens
                # these by reading actual engine counters once the right
                # accessor is identified.
                # TODO(v0.0.15): replace 0 chat_turns + 0 reflex_firings with
                # real counters from the event bus or engine state.
                _run_felt_time_tick(
                    persona_dir,
                    wall_clock_s_since_last=wall_s,
                    heartbeats_since_last=1,
                    chat_turns_since_last=0,
                    reflex_firings_since_last=0,
                )
            except Exception:
                logger.exception("supervisor felt-time tick raised")
            last_heartbeat_at = time.monotonic()

        # Soul-review cadence — slowest of the three. Each pass is up to
        # 5 LLM calls (one per candidate). Fault-isolated so a model
        # outage doesn't take the supervisor down.
        if (
            soul_review_interval_s is not None
            and last_soul_review_at is not None
            and time.monotonic() - last_soul_review_at >= soul_review_interval_s
        ):
            try:
                _run_soul_review_tick(persona_dir, provider, event_bus)
            except Exception:
                logger.exception("supervisor soul-review tick raised")
            try:
                forgetting_run_pass(persona_dir, event_bus=event_bus)
            except Exception:
                logger.exception("supervisor forgetting pass raised")
            last_soul_review_at = time.monotonic()

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


def _run_felt_time_tick(
    persona_dir: Path,
    wall_clock_s_since_last: float,
    heartbeats_since_last: int,
    chat_turns_since_last: int,
    reflex_firings_since_last: int,
) -> None:
    """Fold one supervisor heartbeat cycle into felt-time state.

    drivers values are derived from existing body + emotion accessors;
    cold-start cases (no body state, no emotion vector) collapse all
    drivers to 0.0 so lived-age advances at baseline.

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

    Emotional intensity — TODO(v0.0.15): wire to brain.emotion once a
    clean normalized accessor exists (current aggregate_state returns
    raw channel scores, not a single normalized deviation scalar).
    For now returns 0.0 (baseline lived-age rate; no false inflation).

    Chat activity uses a fixed 6 turns/h baseline for the supervisor
    tick window. Phase 9.2 follow-up will tighten to a rolling baseline
    once weather_shift per-channel baselines are proven stable.
    """
    # Best-effort body strain from exhaustion / energy.
    body_strain = 0.0
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
        finally:
            store.close()
    except Exception:
        logger.debug("supervisor felt-time: body strain read failed; using 0.0", exc_info=True)

    # Emotional intensity — best-effort, 0.0 fallback.
    # TODO(v0.0.15): wire to brain.emotion once a clean normalized accessor exists.
    emotional_intensity = 0.0

    # Chat activity normalized against a fixed 6 turns/h baseline.
    # TODO(v0.0.15): replace with rolling per-tick baseline from weather_shift.
    baseline_per_tick = max(0.1, 6.0 * (wall_clock_s_in_tick / 3600.0))
    chat_activity = min(1.0, float(chat_turns_in_tick) / baseline_per_tick)

    return IntensityDrivers(
        emotional_intensity=emotional_intensity,
        body_strain=body_strain,
        chat_activity=chat_activity,
    )


def _run_heartbeat_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> None:
    """Build a HeartbeatEngine and run one tick. Publishes a result event.

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


def _run_soul_review_tick(
    persona_dir: Path,
    provider: LLMProvider,
    event_bus: EventBus,
) -> None:
    """Run one autonomous soul-review pass.

    Gated upstream by the cadence interval (default 6h). Skips silently
    when there are zero pending candidates so an idle persona doesn't
    burn LLM calls for no work.

    Mirrors the per-tick store-ownership pattern of
    [`_run_heartbeat_tick`]: opens MemoryStore + SoulStore inside this
    thread, closes them via ExitStack. Per-call cap is the existing
    DEFAULT_MAX_DECISIONS=5 in `brain.soul.review`, so a backlog can't
    spike LLM cost on a single tick — it'll drain across multiple ticks.

    Spec basis: user-surface principle (soul review is physiology, not a
    CLI knob). The CLI command `nell soul review` still exists for
    operator/dev use; the autonomous trigger is what the user sees.
    """
    from brain.ingest.soul_queue import list_soul_candidates
    from brain.soul.review import review_pending_candidates
    from brain.soul.store import SoulStore

    pending = [
        rec for rec in list_soul_candidates(persona_dir) if rec.get("status") == "auto_pending"
    ]
    if not pending:
        # No candidates — skip the pass and don't pay the open-store cost.
        # Cadence-tracking still advances upstream so we don't re-check
        # every tick when the queue is empty.
        return

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
        )

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
            "at": _now_iso(),
        }
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
    Reads ``nell-voice.md`` from the persona dir (empty string if absent)
    and ``initiate_review_cap_per_tick`` from PersonaConfig (default 3).
    Publishes an ``initiate_review_tick`` event on success.
    """
    voice_path = persona_dir / "nell-voice.md"
    voice_template = voice_path.read_text(encoding="utf-8") if voice_path.exists() else ""
    try:
        config = PersonaConfig.load(persona_dir / "persona_config.json")
        cap_per_tick = getattr(config, "initiate_review_cap_per_tick", 3) or 3
    except Exception:
        cap_per_tick = 3
    run_initiate_review_tick(
        persona_dir,
        provider=provider,
        voice_template=voice_template,
        cap_per_tick=cap_per_tick,
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
    from datetime import timedelta

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
    from datetime import timedelta

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
