"""One-time pronoun-setting nudge for personas upgraded past the
user-pronouns feature (spec 2026-06-11 §5).

The marker file IS the once-ever guarantee: written at most once, on the
first bridge startup where user_pronouns is unset. The feed source reads
the marker forever — the entry lives in feed history with its original ts
and ages out of the feed cap naturally. Setting pronouns later does not
delete the marker (history is history).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from brain.health.json_io import read_json, write_json

logger = logging.getLogger(__name__)
_MARKER = "pronoun_nudge.json"


def maybe_write_pronoun_nudge(persona_dir: Path, *, companion_name: str) -> bool:
    """Write the marker if pronouns unset and no marker exists. True if written."""
    marker = persona_dir / _MARKER
    if marker.exists():
        return False
    try:
        from brain.persona_config import PersonaConfig
        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        if cfg.user_pronouns is not None:
            return False
    except Exception:  # noqa: BLE001 — config trouble: don't nudge, don't crash startup
        return False
    # if_missing: the check and the create are one step; False also covers a failed write (retried next startup).
    return write_json(
        marker,
        {"ts": datetime.now(UTC).isoformat(), "companion_name": companion_name},
        indent=None,
        fsync=False,
        if_missing=True,
        on_error="log",
    )


def build_pronoun_nudge_entries(persona_dir: Path) -> list:
    """Feed source: one entry if the marker exists. Fault-isolated."""
    from brain.bridge.feed import TYPE_OPENER, FeedEntry
    data = read_json(persona_dir / _MARKER, default=lambda: None, expect=dict)
    if data is None:  # no marker, or one that cannot be read
        return []
    name = data.get("companion_name") or "your companion"
    return [
        FeedEntry(
            type="pronoun_nudge",
            ts=data.get("ts", ""),
            opener=TYPE_OPENER["pronoun_nudge"],
            body=(f"You can tell {name} your pronouns now — set them in the "
                  "Connection panel. Until then I'll keep the old default."),
            audit_id=None,
        )
    ]
