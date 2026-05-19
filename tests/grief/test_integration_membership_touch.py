"""test_integration_membership_touch.py — membership refresh writes grief on lost member."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def test_membership_refresh_writes_grief_when_member_is_lost(tmp_path: Path) -> None:
    """An open arc whose member resolves to a graveyard entry should produce
    a recall-touch grief breadcrumb attributed to both the lost memory AND
    the still-open arc.
    """
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.forgetting import graveyard as gv
    from brain.forgetting.salience import SalienceInputs
    from brain.memory.store import Memory, MemoryStore
    from brain.narrative_memory import run_pass
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, save_state

    # felt_time at lived-age 80 hours. The arc was last extended very
    # recently (so it won't close due to staleness — we want the OPEN-arc
    # membership path to fire).
    persist_felt_time(FeltTimeState(lived_age_hours=80.0), tmp_path)
    store = MemoryStore(tmp_path / "memories.db")

    # Plant a graveyard entry for mem-lost — this is the member that "was"
    # in the arc but has been forgotten.
    m = Memory.create_new(
        content="the afternoon by the river when the rain held off",
        memory_type="episodic",
        domain="memory",
        emotions={"sorrow": 8.0},
    )
    object.__setattr__(m, "id", "mem-lost")
    gv.append(
        tmp_path,
        memory=m,
        salience_at_drop=0.6,
        inputs=SalienceInputs(emotion=0.8, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.0),
        lived_age_hours=24.0,
        reason="test-seed",
    )

    # An open arc with mem-lost as a member (memory no longer in MemoryStore).
    # last_extended very recent so the arc itself stays open.
    arc = Arc(
        id="arc-touch",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream-1",
        seed_memory_ids=("mem-lost",),
        title="river thread",
        opened_at_iso="2026-05-15T00:00:00+00:00",
        lived_age_at_open=50.0,
        last_extended_at_iso="2026-05-19T00:00:00+00:00",  # very recent — won't go stale
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(
            ArcMember(
                memory_id="mem-lost",
                joined_at_iso="2026-05-15T00:00:00+00:00",
                lived_age_at_join=50.0,
                salience_at_join=0.7,
            ),
        ),
    )
    save_state(tmp_path, ArcsState(open={"arc-touch": arc}, recently_closed=[]))

    class _NullBus:
        def publish(self, _evt: Any) -> None:
            pass

    class _StubHebbian:
        def weight(self, a: str, b: str) -> float:
            return 0.0

    class _StubEmbeddings:
        def get(self, memory_id: str) -> np.ndarray | None:
            return None

    class _FakeFeltTime:
        lived_age_hours: float = 80.0

    run_pass(
        tmp_path,
        event_bus=_NullBus(),
        anchor_sweep=lambda persona_dir, last_pass_ts: [],
        candidate_pool=lambda persona_dir, opened_at_iso: [],
        salience_score=lambda memory, ctx: 0.5,
        is_exempt=lambda memory: False,
        hebbian=_StubHebbian(),
        embeddings=_StubEmbeddings(),
        felt_time_state=_FakeFeltTime(),
    )

    rows = store._conn.execute(
        "SELECT metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(rows) >= 1, "membership refresh should write a grief breadcrumb on the lost member"

    # At least one breadcrumb should reference mem-lost as the referent + arc-touch as triggering_arc_id.
    found = False
    for row in rows:
        meta = json.loads(row["metadata_json"])
        if (
            meta.get("grief_referent_id") == "mem-lost"
            and meta.get("grief_subtype") == "recall_touch"
            and meta.get("triggering_arc_id") == "arc-touch"
        ):
            found = True
            break
    assert found, "expected a recall_touch breadcrumb attributed to mem-lost + arc-touch"
