"""Arc + ArcMember dataclass shape tests."""
from __future__ import annotations

import dataclasses

import pytest

from brain.narrative_memory.arc import Arc, ArcMember


def test_arc_member_round_trip_and_frozen():
    member = ArcMember(
        memory_id="mem_20260519_abc12345",
        joined_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_join=412.7,
        salience_at_join=0.62,
    )
    assert member.memory_id == "mem_20260519_abc12345"
    assert member.lived_age_at_join == pytest.approx(412.7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        member.memory_id = "other"  # type: ignore[misc]


def test_arc_round_trip_and_frozen():
    member = ArcMember(
        memory_id="mem_seed",
        joined_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_join=412.7,
        salience_at_join=0.62,
    )
    arc = Arc(
        id="arc_20260519_a1b2c3d4",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_evt_42",
        seed_memory_ids=("mem_seed",),
        title="the boat one",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=412.7,
        last_extended_at_iso="2026-05-19T10:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(member,),
    )
    assert arc.state == "open"
    assert arc.members[0].memory_id == "mem_seed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        arc.state = "closed"  # type: ignore[misc]
