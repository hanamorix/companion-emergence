"""brain.maker.making_runner — the discharge closure fired by run_maker_tick.

Holds the background throttle slot, runs the making act, persists it, and (Phase
3) wires the three feeds. Budget is already consumed by the tick before this
fires (Task 4)."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.bridge import cli_throttle as _cli_throttle
from brain.maker import maker as _maker
from brain.maker.charge import load_charge
from brain.maker.persist import persist_making
from brain.works.store import WorksStore

logger = logging.getLogger(__name__)


def _emotion_summary(store: Any) -> str:
    from brain.maker.sources import current_emotional_intensity
    peak = current_emotional_intensity(store)
    return f"peak intensity {peak:.1f}"


def make_and_wire(*, persona_dir: Path, store: Any, provider: Any,
                  now: datetime | None = None) -> None:
    now = now or datetime.now(UTC)
    charge = load_charge(persona_dir)
    sources = []  # provenance; refined in Phase 3 to name the actual contributors
    if charge.prior_soul_count:
        sources.append("a soul-thread that's been forming")
    sources.append("the weight of recent feeling")

    with _cli_throttle.background_slot() as slot:
        if not slot:
            logger.info("maker: throttle slot unavailable — deferring making")
            raise RuntimeError("throttle deferred")  # tick treats as fail-soft (partial charge)
        making = _maker.make(provider, charge_sources=sources,
                             emotion_summary=_emotion_summary(store))

    works = WorksStore(persona_dir / "works.db")
    try:
        wid = persist_making(persona_dir, works, making, charge_sources=sources, now=now)
    finally:
        works.close()
    logger.info("maker: made %r (%s) → %s", making.title, making.disposition, wid)

    # The three wire-backs of a completed making. For a non-discard making the
    # emotion delta is applied exactly as W8 applies reach_emotions: it is
    # vocab-filtered by making_emotion_delta() and seeded onto the act-memory
    # via Memory.create_new(emotions=...), so aggregate_state picks it up. A
    # discarded making moves nothing — empty-emotion act-memory only. The feed
    # wire-back (eventual_share → shared) is driven by flip_ready_shares from
    # run_maker_tick, not here.
    from brain.maker import wiring
    if making.disposition != "discard":
        delta = wiring.making_emotion_delta(making, dominant_source=sources[0] if sources else "soul")
        wiring.write_making_memory(store, making, emotions=delta)
    else:
        wiring.write_making_memory(store, making, emotions={})
