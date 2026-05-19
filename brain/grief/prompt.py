"""prompt.py — ambient grief block per spec §5.

Replaces forgetting.prompt.render_fading_summary_block at the chat
prompt-builder call site. Names up to 2 specific losses (one memory,
one arc), with deterministic lived-time stamps + heavy/medium/light
weight labels.

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §5
"""

from __future__ import annotations

import math

from brain.grief import policy

_LOST_WINDOW_DAYS = 7


def weight_bucket(*, emotion_max_normalised: float) -> str:
    """Bucket emotion_at_ingest_max into heavy / medium / light per spec §5.

    Thresholds use 0-10 scale via WEIGHT_HEAVY (7.0) and WEIGHT_MEDIUM (3.0)
    re-scaled to 0-1 for normalised inputs.
    """
    scaled = emotion_max_normalised * 10.0
    if scaled >= policy.WEIGHT_HEAVY:
        return "heavy"
    if scaled >= policy.WEIGHT_MEDIUM:
        return "medium"
    return "light"


def _grave_rank(entry: dict, *, lived_age_hours_now: float) -> float:
    inputs = entry.get("salience_inputs_at_drop") or {}
    emotion = float(inputs.get("emotion") or 0.0)
    at_forget = float(entry.get("lived_age_hours_at_forgetting") or 0.0)
    lived_days_since = max(0.0, lived_age_hours_now - at_forget) / 24.0
    recency = math.exp(-lived_days_since / policy.RECENCY_LIVED_DAYS_HALF_LIFE)
    return emotion * recency


def pick_top_grave(*, entries: list[dict], lived_age_hours_now: float) -> dict | None:
    """Pick the highest-ranked graveyard entry whose weight bucket is heavy or medium.

    Returns None if no entry crosses the medium floor.
    """
    candidates = []
    for e in entries:
        inputs = e.get("salience_inputs_at_drop") or {}
        emotion_max = float(inputs.get("emotion") or 0.0)
        bucket = weight_bucket(emotion_max_normalised=emotion_max)
        if bucket == "light":
            continue
        candidates.append((e, _grave_rank(e, lived_age_hours_now=lived_age_hours_now)))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates[0][0]


def _arc_lived_days_since_close(
    *, closed_at_iso: str, now_iso: str, lived_age_rate: float
) -> float:
    from datetime import datetime

    closed_at = datetime.fromisoformat(closed_at_iso)
    now = datetime.fromisoformat(now_iso)
    wall_delta_hours = max(0.0, (now - closed_at).total_seconds() / 3600.0)
    lived_hours = wall_delta_hours * lived_age_rate
    return lived_hours / 24.0


def pick_top_closed_arc(
    *, arcs: list[dict], now_iso: str, lived_age_rate: float
) -> dict | None:
    """Pick the highest-ranked closed arc whose weight bucket is heavy or medium.

    Each arc dict must contain: id, title, closed_at_iso, max_member_emotion_normalised.
    """
    candidates = []
    for a in arcs:
        emotion_max = float(a.get("max_member_emotion_normalised") or 0.0)
        bucket = weight_bucket(emotion_max_normalised=emotion_max)
        if bucket == "light":
            continue
        lived_days = _arc_lived_days_since_close(
            closed_at_iso=str(a.get("closed_at_iso") or now_iso),
            now_iso=now_iso,
            lived_age_rate=lived_age_rate,
        )
        recency = math.exp(-lived_days / policy.RECENCY_LIVED_DAYS_HALF_LIFE)
        candidates.append((a, emotion_max * recency))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    return candidates[0][0]
