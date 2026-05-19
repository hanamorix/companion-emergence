"""recall.py — recall-touch detection + intensity per spec §3, §6.

handle_recall_touch is wired into:
  - chat.prompt._build_recall_block (per user turn)
  - tools.dispatch._dispatch_recall_forgotten (Nell's own tool calls)
  - narrative_memory.__init__ membership refresh (internal arc touch)

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §3 + §6
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.grief import breadcrumb, policy
from brain.memory.store import MemoryStore

log = logging.getLogger(__name__)


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def compute_touch_intensity(
    *,
    grave_emotion_max: float,
    salience_at_drop: float,
    lived_days_since_loss: float,
) -> float:
    """Recall-touch grief intensity per spec §3.

    intensity = clamp(grave_emotion_max * salience_at_drop * 5.0 * recency_factor)
    recency_factor = 0.5 ** (lived_days_since_loss / 14.0)
                 = exp(-ln(2) * lived_days_since_loss / 14.0)

    Half-life of 14 lived-days — a 14-day-old loss feels half as sharp as fresh,
    a 28-day-old loss a quarter as sharp, and so on.

    Args:
        grave_emotion_max: max emotion intensity on the lost memory, NORMALISED
            to [0, 1] (i.e., raw emotion / 10). Same scale as
            SalienceInputs.emotion and the same scale compute_drop_intensity
            expects.
        salience_at_drop: composite salience score at time of loss, [0, 1].
        lived_days_since_loss: lived-days elapsed since the memory entered
            the graveyard.

    Returns:
        Grief intensity in [0, 10].
    """
    d = max(lived_days_since_loss, 0.0)
    half_life = policy.RECENCY_LIVED_DAYS_HALF_LIFE
    recency = 0.5 ** (d / half_life)
    raw = grave_emotion_max * salience_at_drop * policy.RECALL_TOUCH_SCALE * recency
    return _clamp(raw)


def _lived_days_since_loss(
    *,
    entry: dict,
    lived_age_hours_now: float,
) -> float:
    """Compute lived-days since forgetting for a graveyard entry.

    Uses the stored lived_age_hours_at_forgetting field — already a
    felt-time stamp. No approximation needed.
    """
    at_forget = float(entry.get("lived_age_hours_at_forgetting") or 0.0)
    delta_hours = max(0.0, lived_age_hours_now - at_forget)
    return delta_hours / 24.0


def _dominant_lost_emotion(entry: dict) -> tuple[str, float] | None:
    """Return (emotion_name, intensity_0_to_10) for the dominant non-grief emotion
    in a graveyard entry's emotion_at_ingest, or None if empty.
    """
    emotions = entry.get("emotion_at_ingest") or {}
    candidates = [
        (name, float(val))
        for name, val in emotions.items()
        if name != "memory_grief" and isinstance(val, (int, float)) and float(val) > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: kv[1])


def handle_recall_touch(
    *,
    touched_ids: list[str],
    graveyard_entries: list[dict],
    persona_dir: Path,
    store: MemoryStore,
    lived_age_hours_now: float,
    triggering_arc_id: str | None = None,
) -> None:
    """For each touched_id that resolves to a graveyard entry, write a grief
    breadcrumb if intensity >= THRESHOLD and not debounced. Spec §6.

    Pure side-effect — no return value. Fault-isolated by callers (try/except
    wraps this call at every wiring site).
    """
    if not touched_ids or not graveyard_entries:
        return
    by_id = {e["memory_id"]: e for e in graveyard_entries if "memory_id" in e}

    for memory_id in touched_ids:
        entry = by_id.get(memory_id)
        if entry is None:
            continue  # active or fading hit — not lost

        salience_at_drop = float(entry.get("salience_at_drop") or 0.0)
        # Use raw emotion_at_ingest max (0-10 scale) as grave_emotion_max.
        # compute_touch_intensity is scale-agnostic; the caller passes what
        # makes physical sense. Passing the 0-10 value keeps the same
        # emotional weight as the live memory had at ingest.
        emotion_at_ingest = entry.get("emotion_at_ingest") or {}
        if emotion_at_ingest:
            emotion_max = max(
                float(v) for v in emotion_at_ingest.values()
                if isinstance(v, (int, float))
            )
        else:
            emotion_max = 0.0
        lived_days_since = _lived_days_since_loss(
            entry=entry, lived_age_hours_now=lived_age_hours_now
        )

        intensity = compute_touch_intensity(
            grave_emotion_max=emotion_max,
            salience_at_drop=salience_at_drop,
            lived_days_since_loss=lived_days_since,
        )
        if intensity < policy.THRESHOLD:
            continue
        if store.exists_recent_grief_touch(memory_id, hours=policy.DEBOUNCE_HOURS):
            continue

        residue = _dominant_lost_emotion(entry)
        breadcrumb.write_breadcrumb(
            store=store,
            intensity=intensity,
            subtype="recall_touch",
            referent_type="memory",
            referent_id=memory_id,
            content=breadcrumb.recall_touch_phrase(entry.get("summary") or ""),
            residue_emotion=residue,
            triggering_arc_id=triggering_arc_id,
        )
