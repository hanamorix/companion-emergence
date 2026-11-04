"""Pin for how arc close reads the stored emotions of an arc's members, before the readers move.

Arc close takes the stored maximum across the live members (normalised to 0..1) and names the emotion that
produced it, skipping ``memory_grief`` itself. Deleted with the old branch when the reader switch-over lands.
"""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

import pytest

from brain.forgetting import graveyard
from brain.memory.store import Memory, MemoryStore
from brain.narrative_memory import _compute_arc_close_emotion_inputs
from brain.narrative_memory.arc import Arc, ArcMember


@pytest.mark.no_base_failure(
    reason=(
        "pins the old arc close branch (stored maximum, memory_grief skipped); "
        "passes on the current code by design and is deleted with the branch at phase C"
    )
)
def test_arc_close_reads_the_stored_maximum_and_skips_memory_grief(tmp_path: Path):
    memory_id = str(uuid.uuid4())
    stamp = "2026-05-19T10:00:00+00:00"
    arc = Arc(
        id="arc_1",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_arc_1",
        seed_memory_ids=(memory_id,),
        title="an arc",
        opened_at_iso="2026-05-15T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso=stamp,
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(
            ArcMember(memory_id=memory_id, joined_at_iso=stamp, lived_age_at_join=412.0, salience_at_join=0.7),
        ),
    )
    store = MemoryStore(tmp_path / "memories.db")
    try:
        memory = Memory.create_new(
            content="a member",
            memory_type="episodic",
            domain="us",
            emotions={"joy": 6.0, "love": 8.0, "memory_grief": 9.5},
        )
        store.create(dataclasses.replace(memory, id=memory_id))

        result = _compute_arc_close_emotion_inputs(arc=arc, store=store, persona_dir=tmp_path)

        assert result == (0.8, ("love", 8.0))
        assert graveyard.read_all(tmp_path) == []
    finally:
        store.close()


def test_arc_close_reads_the_faded_values_when_the_engine_fades(tmp_path: Path, emotion_engine_stub):
    """B2 phase B: the live-member read takes `faded_value`; the graveyard proxy is used as stored."""
    memory_id = str(uuid.uuid4())
    stamp = "2026-05-19T10:00:00+00:00"
    arc = Arc(
        id="arc_2",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_arc_2",
        seed_memory_ids=(memory_id,),
        title="an arc",
        opened_at_iso="2026-05-15T10:00:00+00:00",
        lived_age_at_open=412.0,
        last_extended_at_iso=stamp,
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(
            ArcMember(memory_id=memory_id, joined_at_iso=stamp, lived_age_at_join=412.0, salience_at_join=0.7),
        ),
    )
    store = MemoryStore(tmp_path / "memories.db")
    try:
        memory = Memory.create_new(
            content="a member", memory_type="episodic", domain="us", emotions={"joy": 6.0, "love": 8.0}
        )
        store.create(dataclasses.replace(memory, id=memory_id))
        emotion_engine_stub.faded = {(memory_id, "joy"): 5.0, (memory_id, "love"): 2.0}
        assert _compute_arc_close_emotion_inputs(arc=arc, store=store, persona_dir=tmp_path) == (0.5, ("joy", 5.0))
    finally:
        store.close()
