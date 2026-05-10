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
from brain.ingest.pipeline import close_stale_sessions
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

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
) -> None:
    """Run supervisor + heartbeat + soul-review cadences until stop_event is set.

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
    """
    logger.info(
        "supervisor folded persona=%s tick=%.2fs heartbeat=%s soul_review=%s",
        persona_dir.name,
        tick_interval_s,
        f"{heartbeat_interval_s:.0f}s" if heartbeat_interval_s is not None else "disabled",
        f"{soul_review_interval_s:.0f}s" if soul_review_interval_s is not None else "disabled",
    )
    last_heartbeat_at = (
        time.monotonic() if heartbeat_interval_s is not None else None
    )
    last_soul_review_at = (
        time.monotonic() if soul_review_interval_s is not None else None
    )
    while not stop_event.is_set():
        try:
            with ExitStack() as stack:
                store = MemoryStore(persona_dir / "memories.db")
                stack.callback(store.close)
                hebbian = HebbianMatrix(persona_dir / "hebbian.db")
                stack.callback(hebbian.close)
                embeddings = EmbeddingCache(
                    persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256),
                )
                stack.callback(embeddings.close)

                reports = close_stale_sessions(
                    persona_dir,
                    silence_minutes=silence_minutes,
                    store=store,
                    hebbian=hebbian,
                    provider=provider,
                    embeddings=embeddings,
                )
                closed_session_ids = [r.session_id for r in reports if r.errors == 0]
                for sid in closed_session_ids:
                    remove_session(sid)
                pruned_empty_sessions = prune_empty_sessions(
                    older_than_seconds=silence_minutes * 60.0,
                    persona_name=persona_dir.name,
                )

            # Publish events outside the with-block — events don't need stores.
            for r in reports:
                event_bus.publish(
                    {
                        "type": "session_closed",
                        "session_id": r.session_id,
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
            last_soul_review_at = time.monotonic()

        # Wait for the next tick or for stop_event, whichever comes first.
        stop_event.wait(timeout=tick_interval_s)
    logger.info("supervisor stopped persona=%s", persona_dir.name)


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

            load_persona_vocabulary(
                persona_dir / "emotion_vocabulary.json", store=store
            )
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
        rec for rec in list_soul_candidates(persona_dir)
        if rec.get("status") == "auto_pending"
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
            load_persona_vocabulary(
                persona_dir / "emotion_vocabulary.json", store=store
            )
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


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
