"""test_prompt.py — render_grief_block + helpers."""

from __future__ import annotations

from brain.grief import prompt


def test_weight_bucket_heavy() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.85) == "heavy"


def test_weight_bucket_medium() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.5) == "medium"


def test_weight_bucket_light() -> None:
    assert prompt.weight_bucket(emotion_max_normalised=0.2) == "light"


def test_weight_bucket_heavy_boundary() -> None:
    # exactly at WEIGHT_HEAVY normalised threshold (7.0 -> 0.7) -> heavy
    assert prompt.weight_bucket(emotion_max_normalised=0.7) == "heavy"


def test_weight_bucket_medium_boundary() -> None:
    # exactly at WEIGHT_MEDIUM normalised threshold (3.0 -> 0.3) -> medium
    assert prompt.weight_bucket(emotion_max_normalised=0.3) == "medium"


def _grave_entry(
    *,
    memory_id: str,
    summary: str,
    emotion_max: float,
    lived_age_at_forgetting: float,
    forgotten_at_iso: str = "2026-05-15T00:00:00+00:00",
) -> dict:
    return {
        "memory_id": memory_id,
        "summary": summary,
        "lived_age_hours_at_forgetting": lived_age_at_forgetting,
        "forgotten_at_iso": forgotten_at_iso,
        "salience_inputs_at_drop": {"emotion": emotion_max},
        "salience_at_drop": 0.4,
    }


def test_rank_grave_picks_heavy_recent_over_heavy_old() -> None:
    fresh = _grave_entry(
        memory_id="m1",
        summary="fresh heavy loss",
        emotion_max=0.9,
        lived_age_at_forgetting=24.0,
    )
    old = _grave_entry(
        memory_id="m2",
        summary="old heavy loss",
        emotion_max=0.9,
        lived_age_at_forgetting=0.0,
    )
    lived_now = 24.0  # m1 lost ~0 lived-days ago, m2 ~1 lived-day ago
    best = prompt.pick_top_grave(entries=[fresh, old], lived_age_hours_now=lived_now)
    assert best is not None and best["memory_id"] == "m1"


def test_rank_grave_skips_light() -> None:
    light = _grave_entry(
        memory_id="m1", summary="light", emotion_max=0.2, lived_age_at_forgetting=0.0
    )
    best = prompt.pick_top_grave(entries=[light], lived_age_hours_now=24.0)
    # Light losses don't surface in the block per spec §5.
    assert best is None


def test_rank_arc_picks_heavy_recent() -> None:
    fresh_arc = {
        "id": "a1",
        "title": "first cold week",
        "closed_at_iso": "2026-05-18T00:00:00+00:00",
        "max_member_emotion_normalised": 0.85,
    }
    old_arc = {
        "id": "a2",
        "title": "summer thread",
        "closed_at_iso": "2026-04-01T00:00:00+00:00",
        "max_member_emotion_normalised": 0.85,
    }
    best = prompt.pick_top_closed_arc(
        arcs=[fresh_arc, old_arc],
        now_iso="2026-05-19T00:00:00+00:00",
        lived_age_rate=1.0,
    )
    assert best is not None and best["id"] == "a1"
