"""Ambient narrative-arc block render."""
from __future__ import annotations

from pathlib import Path

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.prompt import render_current_arc_block
from brain.narrative_memory.state import ArcsState, save_state


def _arc(
    arc_id: str,
    title: str,
    last_extended_iso: str = "2026-05-19T10:00:00+00:00",
    member_count: int = 1,
    opened_iso: str = "2026-05-15T10:00:00+00:00",
) -> Arc:
    members = tuple(
        ArcMember(
            memory_id=f"mem_{i}",
            joined_at_iso=last_extended_iso,
            lived_age_at_join=412.0,
            salience_at_join=0.7,
        )
        for i in range(member_count)
    )
    return Arc(
        id=arc_id,
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref=f"dream_{arc_id}",
        seed_memory_ids=("mem_0",),
        title=title,
        opened_at_iso=opened_iso,
        lived_age_at_open=412.0,
        last_extended_at_iso=last_extended_iso,
        closed_at_iso=None,
        lived_age_at_close=None,
        members=members,
    )


def test_render_cold_start_no_arcs(tmp_path: Path):
    save_state(tmp_path, ArcsState())
    out = render_current_arc_block(tmp_path)
    assert "still forming" in out
    assert out.startswith("arcs")


def test_render_single_arc(tmp_path: Path):
    state = ArcsState(
        open={"arc_1": _arc("arc_1", "the boat one", member_count=8)},
        last_pass_ts_iso="2026-05-19T10:00:00+00:00",
    )
    save_state(tmp_path, state)
    out = render_current_arc_block(tmp_path)
    assert "the boat one" in out
    assert "current:" in out
    assert "8 memories" in out
    # No "also open" line for a single arc
    assert "also open" not in out


def test_render_two_arcs_shows_also_open(tmp_path: Path):
    state = ArcsState(
        open={
            "arc_1": _arc("arc_1", "the boat one", last_extended_iso="2026-05-19T11:00:00+00:00", member_count=8),
            "arc_2": _arc("arc_2", "the kitchen one", last_extended_iso="2026-05-19T10:00:00+00:00", member_count=12),
        },
        last_pass_ts_iso="2026-05-19T11:00:00+00:00",
    )
    save_state(tmp_path, state)
    out = render_current_arc_block(tmp_path)
    assert "current:" in out and "the boat one" in out
    assert "also open" in out
    assert "the kitchen one" in out
    assert "(12 memories)" in out


def test_render_more_than_two_arcs_caps_with_plus_n_more(tmp_path: Path):
    state = ArcsState(
        open={
            f"arc_{i}": _arc(
                f"arc_{i}",
                f"arc {i} title",
                last_extended_iso=f"2026-05-19T{10 + i:02d}:00:00+00:00",
                member_count=i + 2,
            )
            for i in range(5)
        },
        last_pass_ts_iso="2026-05-19T15:00:00+00:00",
    )
    save_state(tmp_path, state)
    out = render_current_arc_block(tmp_path)
    assert "current:" in out
    assert "also open" in out
    assert "+ 2 more" in out  # 5 total - 1 current - 2 listed = 2 more


def test_render_missing_state_file_returns_cold_start(tmp_path: Path):
    out = render_current_arc_block(tmp_path)
    assert "still forming" in out
