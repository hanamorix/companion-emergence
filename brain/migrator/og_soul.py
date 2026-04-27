"""Extract OG NellBrain soul crystallizations for migration into the new SoulStore.

OG file: data/nell_soul.json with shape:
  {"version": ..., "crystallizations": [...], "revoked": [...], "soul_truth": "...", "first_love": ...}

Each crystallization (per OG nell_brain.py:crystallize_soul):
  {
    "id": "<uuid>",
    "moment": "<text>",
    "love_type": "<one of LOVE_TYPES keys>",
    "who_or_what": "<text or null>",
    "why_it_matters": "<text>",
    "crystallized_at": "<iso8601>",
    "resonance": <int 1-10>,
    "permanent": true,
    # Optional revoke fields:
    "revoked_at": "<iso8601>",
    "revoked_reason": "<text>",
  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from brain.soul.crystallization import Crystallization
from brain.soul.love_types import LOVE_TYPES
from brain.utils.time import parse_iso_utc

logger = logging.getLogger(__name__)


def extract_crystallizations_from_og(
    og_data_dir: Path,
) -> tuple[list[Crystallization], list[dict]]:
    """Read nell_soul.json from the OG data dir; return (active_crystals, skipped).

    Skipped entries are dicts with shape {id, reason}. Reasons:
      - 'unknown_love_type'  (love_type not in LOVE_TYPES)
      - 'malformed'          (missing required field, bad timestamp, etc.)
      - 'revoked'            (entry has revoked_at — skip; the new framework
                              has its own revoke flow, we don't import history)

    Missing nell_soul.json file → ([], []) silently. Caller decides whether
    that's an anomaly.
    """
    soul_path = og_data_dir / "nell_soul.json"
    if not soul_path.exists():
        return [], []
    try:
        data = json.loads(soul_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("og_soul: nell_soul.json is corrupt JSON: %s", exc)
        return [], []

    if not isinstance(data, dict):
        return [], []

    crystals_raw = data.get("crystallizations") or []
    if not isinstance(crystals_raw, list):
        return [], []

    active: list[Crystallization] = []
    skipped: list[dict] = []

    for entry in crystals_raw:
        if not isinstance(entry, dict):
            skipped.append({"id": "?", "reason": "malformed"})
            continue

        entry_id = str(entry.get("id", "?"))

        # Skip already-revoked from OG side
        if entry.get("revoked_at"):
            skipped.append({"id": entry_id, "reason": "revoked"})
            continue

        love_type = str(entry.get("love_type", ""))
        if love_type not in LOVE_TYPES:
            skipped.append({"id": entry_id, "reason": "unknown_love_type", "love_type": love_type})
            continue

        try:
            crystallized_at = parse_iso_utc(entry["crystallized_at"])
            moment = str(entry["moment"])
            why_it_matters = str(entry["why_it_matters"])
            # who_or_what is optional — null/missing becomes empty string
            who_or_what = str(entry.get("who_or_what") or "")
            try:
                resonance = int(entry.get("resonance", 8))
            except (TypeError, ValueError):
                resonance = 8
            resonance = max(1, min(10, resonance))
        except (KeyError, ValueError, TypeError) as exc:
            skipped.append({"id": entry_id, "reason": f"malformed: {exc}"})
            continue

        active.append(
            Crystallization(
                id=entry_id,
                moment=moment,
                love_type=love_type,
                why_it_matters=why_it_matters,
                crystallized_at=crystallized_at,
                who_or_what=who_or_what,
                resonance=resonance,
                permanent=True,
            )
        )

    return active, skipped
