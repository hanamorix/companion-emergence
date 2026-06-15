"""brain.maker — autonomous making (Tier 2 #3 + #9 + #8).

run_maker_tick: integrate the charge from live signals each supervisor pass;
when it crosses threshold (cooldown clear, budget available) discharge into one
making via make_fn. make_fn is injected so the gate is testable in isolation;
the supervisor passes the real making closure (Task 9).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.maker import config as _cfg
from brain.maker.budget import consume_budget
from brain.maker.charge import accumulate, load_charge, save_charge
from brain.maker.sources import current_emotional_intensity, dreams_since, soul_pending_count

logger = logging.getLogger(__name__)


def run_maker_tick(
    persona_dir: Path,
    *,
    store: Any,
    provider: Any,
    now: datetime | None = None,
    make_fn: Callable[..., Any] | None = None,
    threshold: float = _cfg.DISCHARGE_THRESHOLD,
    cooldown_hours: float = _cfg.COOLDOWN_HOURS,
    daily_cap: int = _cfg.DAILY_CAP,
) -> None:
    now = now or datetime.now(tz=UTC)

    prior = load_charge(persona_dir)
    soul_now = soul_pending_count(persona_dir)
    soul_delta = max(0, soul_now - prior.prior_soul_count)
    last_tick = prior.last_tick_ts or now.isoformat()

    state = accumulate(
        persona_dir,
        emotional_intensity=current_emotional_intensity(store),
        soul_delta=soul_delta,
        dream_count=dreams_since(store, last_tick),
        now=now,
        w_emotion=_cfg.W_EMOTION,
        w_soul=_cfg.W_SOUL,
        w_dream=_cfg.W_DREAM,
        decay_per_hour=_cfg.DECAY_PER_HOUR,
    )
    state.prior_soul_count = soul_now
    save_charge(persona_dir, state)

    if state.charge < threshold:
        return
    # cooldown gate
    if state.last_fire_ts:
        try:
            elapsed = (now - datetime.fromisoformat(state.last_fire_ts)).total_seconds() / 3600.0
            if elapsed < cooldown_hours:
                return
        except ValueError:
            pass
    # cost cap
    if not consume_budget(persona_dir, now=now, cap=daily_cap):
        logger.info("maker: daily cap reached — deferring")
        return

    logger.info("maker: charge %.1f >= %.1f — discharging", state.charge, threshold)
    try:
        if make_fn is not None:
            make_fn(persona_dir=persona_dir, store=store, provider=provider, now=now)
        state.charge = 0.0
        state.last_fire_ts = now.isoformat()
    except Exception:
        logger.exception("maker: making failed — partial charge, cooldown engaged")
        state.charge = _cfg.FAILED_MAKE_CHARGE
        state.last_fire_ts = now.isoformat()  # no tight retry loop
    save_charge(persona_dir, state)
