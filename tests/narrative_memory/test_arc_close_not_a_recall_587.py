"""#587: closing an arc reads each live member's emotions, which shows the kindled nothing, so it is not a recall."""

from __future__ import annotations

import dataclasses
import uuid
from pathlib import Path

from brain.memory.store import Memory, MemoryStore
from brain.narrative_memory import _compute_arc_close_emotion_inputs
from brain.narrative_memory.arc import Arc, ArcMember


def test_arc_close_does_not_tick_its_members(tmp_path: Path):
    memory_id = str(uuid.uuid4())
    stamp = "2026-05-19T10:00:00+00:00"
    arc = Arc(
        id="arc_587",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream_arc_587",
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
        memory = Memory.create_new(content="a member", memory_type="episodic", domain="us", emotions={"love": 8.0})
        store.create(dataclasses.replace(memory, id=memory_id))

        _compute_arc_close_emotion_inputs(arc=arc, store=store, persona_dir=tmp_path)

        row = store.get(memory_id, bump=False)
        assert (row.recall_count, row.last_accessed_at) == (0.0, None)
    finally:
        store.close()
