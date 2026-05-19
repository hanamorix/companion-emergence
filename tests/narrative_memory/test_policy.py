"""Policy thresholds + open/close gates."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.policy import (
    ARC_STALE_LIVED_HOURS,
    MAX_ARC_MEMBERS,
    should_close,
    should_open,
)


@dataclass
class FakeFeltTimeState:
    lived_age_hours: float


def _arc(
    arc_id: str = "arc_1",
    last_extended_iso: str = "2026-05-19T10:00:00+00:00",
    members_ids: tuple[str, ...] = ("mem_a",),
    lived_age_at_open: float = 412.0,
) -> Arc:
    members = tuple(
        ArcMember(
            memory_id=mid,
            joined_at_iso="2026-05-19T10:00:00+00:00",
            lived_age_at_join=lived_age_at_open,
            salience_at_join=0.7,
        )
        for mid in members_ids
    )
    return Arc(
        id=arc_id,
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_evt_1",
        seed_memory_ids=(members_ids[0],),
        title="test arc",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=lived_age_at_open,
        last_extended_at_iso=last_extended_iso,
        closed_at_iso=None,
        lived_age_at_close=None,
        members=members,
    )


def test_constants_match_spec():
    assert ARC_STALE_LIVED_HOURS == pytest.approx(72.0)
    assert MAX_ARC_MEMBERS == 50


def test_should_close_at_exactly_72_hours_lived():
    # last_extended at lived_age = 412.0; current felt = 484.0 → exactly 72 hours
    arc = _arc(lived_age_at_open=412.0)
    # Approximate last_extended lived-age by re-using lived_age_at_open
    # (in the v1 pass, last_extended lived-age isn't stored separately; the
    # orchestrator derives it from FeltTimeState at last extension. For this
    # unit test, we assert the staleness math handles boundary correctly via
    # the helper signature in policy.py: should_close(arc, *, lived_age_now,
    # last_extended_lived_age)).
    assert should_close(arc, lived_age_now=484.0, last_extended_lived_age=412.0) is True


def test_should_close_just_below_threshold():
    arc = _arc()
    assert should_close(arc, lived_age_now=483.99, last_extended_lived_age=412.0) is False


def test_should_close_with_missing_lived_age_falls_back_to_false():
    """When lived_age_now is None (no FeltTimeState yet), don't close."""
    arc = _arc()
    assert should_close(arc, lived_age_now=None, last_extended_lived_age=412.0) is False


def test_should_open_with_fresh_seed():
    open_arcs: dict[str, Arc] = {"arc_existing": _arc(members_ids=("mem_a", "mem_b"))}
    assert should_open(("mem_z",), open_arcs=open_arcs) is True


def test_should_open_with_already_member_seed():
    open_arcs: dict[str, Arc] = {"arc_existing": _arc(members_ids=("mem_a", "mem_b"))}
    assert should_open(("mem_a",), open_arcs=open_arcs) is False
