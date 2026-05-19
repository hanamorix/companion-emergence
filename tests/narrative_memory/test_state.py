"""ArcsState persistence + recovery tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.narrative_memory.arc import Arc, ArcMember
from brain.narrative_memory.state import (
    ArcsState,
    append_event,
    load_or_recover,
    save_state,
)


def _make_arc(arc_id: str = "arc_1", state: str = "open") -> Arc:
    return Arc(
        id=arc_id,
        state=state,
        seed_anchor_type="dream",
        seed_anchor_ref="dream_evt_1",
        seed_memory_ids=("mem_seed",),
        title="the boat one",
        opened_at_iso="2026-05-19T10:00:00+00:00",
        lived_age_at_open=412.7,
        last_extended_at_iso="2026-05-19T10:00:00+00:00",
        closed_at_iso=None if state == "open" else "2026-05-22T10:00:00+00:00",
        lived_age_at_close=None if state == "open" else 484.7,
        members=(
            ArcMember(
                memory_id="mem_seed",
                joined_at_iso="2026-05-19T10:00:00+00:00",
                lived_age_at_join=412.7,
                salience_at_join=0.7,
            ),
        ),
    )


def test_empty_state_defaults():
    s = ArcsState()
    assert s.open == {}
    assert s.recently_closed == []
    assert s.last_pass_ts_iso is None
    assert s.replayed is False
