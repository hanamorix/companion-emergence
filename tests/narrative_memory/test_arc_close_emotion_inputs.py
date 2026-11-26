"""Arc close reads the engine's faded values of an arc's members (normalised to 0..1) and names the emotion that
produced the maximum, skipping ``memory_grief`` itself; the graveyard proxy of a forgotten member is used as stored.
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

_MOVED = (
    "moved from tests/narrative_memory/test_old_branch_arc_close.py (deleted with the old reader branches); "
    "the behaviour it checks exists on the base, so it passes there by design"
)


def _arc_with_one_member(memory_id: str, arc_id: str) -> Arc:
    stamp = "2026-05-19T10:00:00+00:00"
    return Arc(
        id=arc_id,
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref=f"dream_{arc_id}",
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


@pytest.mark.no_base_failure(reason=_MOVED)
def test_arc_close_takes_the_maximum_and_skips_memory_grief(tmp_path: Path, emotion_engine_stub):
    """With no faded entry the stub returns the stored value, so this checks the maximum and the skip."""
    memory_id = str(uuid.uuid4())
    arc = _arc_with_one_member(memory_id, "arc_1")
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


@pytest.mark.no_base_failure(reason=_MOVED)
def test_arc_close_reads_the_faded_values_when_the_engine_fades(tmp_path: Path, emotion_engine_stub):
    """The live-member read takes `faded_value`; the graveyard proxy is used as stored."""
    memory_id = str(uuid.uuid4())
    arc = _arc_with_one_member(memory_id, "arc_2")
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
