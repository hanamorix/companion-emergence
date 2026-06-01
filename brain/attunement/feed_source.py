"""Attunement source for the inner-life feed.

Emits two event types:
- attunement_backfill: one-shot when backfill finishes (status='complete'
  in backfill_state.json)
- attunement_crystal: per learned pattern with crystallised_at set
  (Task 6's check_crystallisations is the producer)

Both render with soft-rose dot (#c89890) in the frontend TYPE_DOT lookup.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def build_attunement_entries(persona_dir: Path) -> list:
    """Return FeedEntry list for all attunement events from on-disk state.

    Reads:
    - attunement/backfill_state.json  → one attunement_backfill entry when complete
    - attunement/learned_patterns.jsonl → one attunement_crystal entry per
      pattern whose crystallised_at is set (first-cross only; subsequent
      confirmations don't change crystallised_at so they stay silent)

    Fault-isolated: any read error returns an empty list for that sub-source.
    """
    from brain.bridge.feed import TYPE_OPENER, FeedEntry

    entries: list[FeedEntry] = []

    # --- backfill_complete ---
    backfill_path = persona_dir / "attunement" / "backfill_state.json"
    if backfill_path.exists():
        try:
            state = json.loads(backfill_path.read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                entries.append(
                    FeedEntry(
                        type="attunement_backfill",
                        ts=state.get("started_at", ""),
                        opener=TYPE_OPENER["attunement_backfill"],
                        body="I spent some time thinking about you — patterns are settling.",
                        audit_id=None,
                    )
                )
        except (json.JSONDecodeError, ValueError, OSError):
            log.warning("attunement feed: could not read backfill_state.json — skipping")

    # --- crystallisation events ---
    try:
        from brain.attunement.store import read_learned_patterns

        for pattern in read_learned_patterns(persona_dir):
            if pattern.crystallised_at is None:
                continue
            entries.append(
                FeedEntry(
                    type="attunement_crystal",
                    ts=pattern.crystallised_at,
                    opener=TYPE_OPENER["attunement_crystal"],
                    body=pattern.description,
                    audit_id=None,
                )
            )
    except Exception:
        log.exception("attunement feed: reading learned patterns failed — skipping crystals")

    return entries
