"""SP-7 supervisor thread — folded as non-daemon thread inside the bridge.

run_folded() is a synchronous loop: every `tick_interval_s` it calls
close_stale_sessions and publishes events. Lives in a separate thread so the
async server stays responsive; uses event_bus.publish (thread-safe) to fan
out events to /events subscribers.

Non-daemon thread on purpose — SIGTERM must wait for the loop to exit
before process exit, so we don't kill mid-ingest.

OG reference: NellBrain/nell_supervisor.py:368-407 (run_folded).
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.events import EventBus
from brain.bridge.provider import LLMProvider
from brain.ingest.pipeline import close_stale_sessions
from brain.memory.embeddings import EmbeddingCache
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)


def run_folded(
    stop_event: threading.Event,
    *,
    persona_dir: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    provider: LLMProvider,
    embeddings: EmbeddingCache,
    event_bus: EventBus,
    tick_interval_s: float = 60.0,
    silence_minutes: float = 5.0,
) -> None:
    """Run close_stale_sessions every tick_interval_s until stop_event is set."""
    logger.info("supervisor folded persona=%s tick=%.2fs", persona_dir.name, tick_interval_s)
    while not stop_event.is_set():
        try:
            reports = close_stale_sessions(
                persona_dir,
                silence_minutes=silence_minutes,
                store=store,
                hebbian=hebbian,
                provider=provider,
                embeddings=embeddings,
            )
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
                    "next_tick_in_s": tick_interval_s,
                    "at": _now_iso(),
                }
            )
        except Exception:
            logger.exception("supervisor tick raised")
        # Wait for the next tick or for stop_event, whichever comes first.
        # stop_event.wait() returns True as soon as the event is set, so
        # SIGTERM/teardown unblocks us immediately rather than waiting out
        # the rest of the interval.
        stop_event.wait(timeout=tick_interval_s)
    logger.info("supervisor stopped persona=%s", persona_dir.name)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
