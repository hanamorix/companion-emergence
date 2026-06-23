"""brain.notes — autonomous notes to an authorized folder (persona notes)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain.notes import config as _cfg
from brain.notes.state import consume_budget, load_notes_state, save_notes_state

logger = logging.getLogger(__name__)


def run_notes_tick(
    persona_dir: Path,
    *,
    config: Any,
    provider: Any,
    silence_hours: float,
    make_fn: Callable[..., Any] | None = None,
    now: datetime | None = None,
    away_hours: float = _cfg._NOTES_AWAY_HOURS,
    cooldown_hours: float = _cfg._NOTES_COOLDOWN_HOURS,
    daily_cap: int = _cfg._NOTES_DAILY_CAP,
) -> None:
    """Gate one notes opportunity. No-op unless notes are enabled with a folder,
    the user's been away ≥ away_hours, cooldown has elapsed, and budget remains.
    Fail-soft: if making raises, advance the cooldown anyway (no tight retry)."""
    if not getattr(config, "notes_enabled", False) or not getattr(config, "notes_folder", None):
        return
    if silence_hours < away_hours:
        return
    now = now or datetime.now(UTC)
    state = load_notes_state(persona_dir)
    if state.last_note_at:
        try:
            elapsed = (now - datetime.fromisoformat(state.last_note_at)).total_seconds() / 3600.0
            if elapsed < cooldown_hours:
                return
        except ValueError:
            pass
    # Pre-flight throttle gate — a defer must cost NOTHING. Check the slot BEFORE
    # consuming budget or advancing the cooldown, so a transient yield doesn't
    # burn the day's note and scream a traceback (Windows v0.0.38 report). The
    # runner's real background_slot() acquire stays the authoritative guard for
    # the rare race (handled by the ThrottleDeferred catch below).
    from brain.bridge.cli_throttle import ThrottleDeferred, slot_available
    if not slot_available():
        logger.info("notes: throttle slot unavailable — deferring (nothing spent)")
        return
    if not consume_budget(persona_dir, now=now, cap=daily_cap):
        return
    try:
        if make_fn is not None:
            make_fn(persona_dir=persona_dir, config=config, provider=provider, now=now)
        state.last_note_at = now.isoformat()
        save_notes_state(persona_dir, state)
    except ThrottleDeferred:
        # Lost the slot in the pre-flight→acquire race. Quiet no-op: do NOT
        # advance the cooldown — let the next tick retry once the slot frees.
        logger.info("notes: note deferred by throttle — retry next tick")
        return
    except Exception:
        logger.exception("notes: making a note failed")
        state.last_note_at = now.isoformat()  # advance to respect cooldown — no tight retry
        save_notes_state(persona_dir, state)
