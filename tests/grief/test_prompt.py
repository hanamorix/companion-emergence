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


def test_render_grief_block_only_memory() -> None:
    block = prompt._format_block(
        memory_phrase="the rooftop morning before (lost 2 lived-days ago, heavy)",
        arc_phrase=None,
        coda="3 have softened, 1 more lost in the last 7 days.",
    )
    assert block.startswith("memory · loss:")
    assert "the rooftop morning before" in block
    assert "3 have softened" in block


def test_render_grief_block_only_arc() -> None:
    block = prompt._format_block(
        memory_phrase=None,
        arc_phrase="the arc 'first cold week' (closed 5 lived-days ago, medium)",
        coda="0 have softened, 0 more lost in the last 7 days.",
    )
    assert "first cold week" in block


def test_render_grief_block_nothing_with_counts() -> None:
    block = prompt._format_block(
        memory_phrase=None,
        arc_phrase=None,
        coda="2 have softened, 0 more lost in the last 7 days.",
    )
    assert block == "memory · loss: still. 2 have softened, 0 more lost in the last 7 days."


def test_render_grief_block_nothing_no_counts() -> None:
    block = prompt._format_block(memory_phrase=None, arc_phrase=None, coda="")
    assert block == "memory · loss: still."


def test_render_grief_block_token_cap_truncates_first_step_shrinks_memory_phrase() -> None:
    """Cap forces step 1 (memory first_4 -> first_2) but arc survives."""
    block = prompt._format_block_with_budget(
        memory_summary_first_4="very very very long",  # 4 words
        memory_lost_days_ago=2,
        memory_weight="heavy",
        arc_name="short",
        arc_closed_days_ago=5,
        arc_weight="medium",
        coda="20 have softened, 17 more lost in the last 7 days.",
        # token_cap=42: full block ~43 tokens (just over), step-1 shrunk to ~41 tokens
        # (passes), step-2 not needed. If prefix or coda format changes, recompute
        # with prompt._count_tokens().
        token_cap=42,
    )
    # Step 1 must have fired: memory_phrase shrunk to first 2 words.
    assert "very very (lost 2" in block
    assert "very very very long" not in block
    # Arc must still be present (step 2 did NOT fire).
    assert "the arc 'short'" in block
    # Coda always stays.
    assert "17 more lost" in block


def test_render_grief_block_token_cap_truncates_second_step_drops_arc() -> None:
    """Cap forces step 2 (arc phrase dropped entirely)."""
    block = prompt._format_block_with_budget(
        memory_summary_first_4="very very very long",
        memory_lost_days_ago=2,
        memory_weight="heavy",
        arc_name="extremely lengthy elaborated arc name about one week",
        arc_closed_days_ago=5,
        arc_weight="medium",
        coda="20 have softened, 17 more lost in the last 7 days.",
        # token_cap=30: full block ~55 tokens, step-1 still ~53 tokens (over), step-2
        # drops arc to ~28 tokens (passes). Tightest cascade-step-2 cap.
        token_cap=30,
    )
    # Step 2 must have fired: arc dropped.
    assert "arc '" not in block
    # Coda always stays.
    assert "17 more lost" in block


def test_format_block_strips_leading_the_in_memory_phrase() -> None:
    """Memory summary starting with 'the' should not produce 'the the X'."""
    block = prompt._format_block_with_budget(
        memory_summary_first_4="the rooftop morning before",
        memory_lost_days_ago=2,
        memory_weight="heavy",
        arc_name=None,
        arc_closed_days_ago=None,
        arc_weight=None,
        coda="",
        token_cap=200,
    )
    assert "the the" not in block
    assert "the rooftop morning before" in block


def test_format_block_strips_leading_a_in_memory_phrase() -> None:
    block = prompt._format_block_with_budget(
        memory_summary_first_4="a quiet afternoon",
        memory_lost_days_ago=1,
        memory_weight="medium",
        arc_name=None,
        arc_closed_days_ago=None,
        arc_weight=None,
        coda="",
        token_cap=200,
    )
    assert "the a quiet afternoon" not in block
    assert "the quiet afternoon" in block


def test_format_block_strips_leading_an_in_memory_phrase() -> None:
    block = prompt._format_block_with_budget(
        memory_summary_first_4="an unusual evening",
        memory_lost_days_ago=1,
        memory_weight="medium",
        arc_name=None,
        arc_closed_days_ago=None,
        arc_weight=None,
        coda="",
        token_cap=200,
    )
    assert "the an " not in block
    assert "the unusual evening" in block


def test_count_recent_lost_counts_within_window() -> None:
    """Direct test of the private helper — counts entries with forgotten_at_iso
    inside the 7-lived-day window. Entries with None forgotten_at_iso don't count.
    """
    from datetime import UTC, datetime, timedelta

    recent = (datetime.now(UTC) - timedelta(days=2)).isoformat()
    old = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    entries = [
        {"forgotten_at_iso": recent},
        {"forgotten_at_iso": old},
        {"forgotten_at_iso": None},
    ]
    count = prompt._count_recent_lost(entries)
    assert count == 1
