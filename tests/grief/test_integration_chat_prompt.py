"""test_integration_chat_prompt.py — chat prompt block swap + recall-touch wiring."""

from __future__ import annotations

import json
from pathlib import Path


def test_chat_prompt_uses_grief_block(tmp_path: Path) -> None:
    """The prompt's loss block now reads as grief, even with empty inputs."""
    from brain.chat import prompt as chat_prompt
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.memory.store import MemoryStore

    persist_felt_time(FeltTimeState(lived_age_hours=24.0), tmp_path)
    store = MemoryStore(tmp_path / "memories.db")

    rendered = chat_prompt._build_fading_summary_block(tmp_path, store)
    assert rendered.startswith("memory · loss:")


def test_chat_prompt_recall_writes_grief_breadcrumb_on_lost_hit(tmp_path: Path) -> None:
    """A user message that resolves a graveyard hit triggers a grief breadcrumb."""
    from brain.chat import prompt as chat_prompt
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.forgetting import graveyard as gv
    from brain.forgetting.salience import SalienceInputs
    from brain.memory.store import Memory, MemoryStore

    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)
    store = MemoryStore(tmp_path / "memories.db")

    m = Memory.create_new(
        content="the rooftop morning before the cold rain hit",
        memory_type="episodic",
        domain="memory",
        emotions={"joy": 8.5},
    )
    object.__setattr__(m, "id", "mem-rooftop")
    gv.append(
        tmp_path,
        memory=m,
        salience_at_drop=0.6,
        inputs=SalienceInputs(emotion=0.85, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.1),
        lived_age_hours=24.0,
        reason="test-seed",
    )

    # _build_recall_block(store, user_input, *, persona_dir=None, limit=5, max_chars=140)
    chat_prompt._build_recall_block(
        store,
        "we were talking about the rooftop morning",
        persona_dir=tmp_path,
        limit=5,
    )

    rows = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(rows) == 1
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "mem-rooftop"
    assert meta["grief_subtype"] == "recall_touch"
