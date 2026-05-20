"""test_integration_tools_dispatch.py — recall_forgotten tool writes grief breadcrumbs."""

from __future__ import annotations

import json
from pathlib import Path


def test_recall_forgotten_tool_writes_grief_breadcrumb(tmp_path: Path) -> None:
    """When Nell calls the recall_forgotten tool and it resolves a graveyard hit,
    a grief breadcrumb should be written via handle_recall_touch.
    """
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.forgetting import graveyard as gv
    from brain.forgetting.salience import SalienceInputs
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore
    from brain.tools.dispatch import dispatch

    # Seed felt-time at 48 lived hours.
    persist_felt_time(FeltTimeState(lived_age_hours=48.0), tmp_path)

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(tmp_path / "hebbian.db")

    # Plant a graveyard entry for mem-lake.
    m = Memory.create_new(
        content="the late summer afternoon at the lake we drove to",
        memory_type="episodic",
        domain="memory",
        emotions={"joy": 9.0},
    )
    object.__setattr__(m, "id", "mem-lake")
    gv.append(
        tmp_path,
        memory=m,
        salience_at_drop=0.55,
        inputs=SalienceInputs(emotion=0.9, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.1),
        lived_age_hours=24.0,
        reason="test-seed",
    )

    # Call the recall_forgotten tool via the dispatcher.
    # dispatch(name, arguments, *, store, hebbian, persona_dir)
    # Use "lake" — a substring present in the graveyard summary.
    dispatch(
        "recall_forgotten",
        {"query": "lake"},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    rows = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(rows) == 1, "expected one grief breadcrumb from recall_forgotten dispatch"
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "mem-lake"
    assert meta["grief_subtype"] == "recall_touch"
