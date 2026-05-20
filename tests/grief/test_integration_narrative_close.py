"""test_integration_narrative_close.py — arc-close grief breadcrumb."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def test_arc_close_writes_grief_breadcrumb_when_emotion_heavy(tmp_path: Path) -> None:
    """A heavy-emotion arc that closes due to staleness writes a grief breadcrumb."""
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.memory.store import Memory, MemoryStore
    from brain.narrative_memory import run_pass
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, save_state

    # Seed felt-time at 1000 lived hours.
    persist_felt_time(
        FeltTimeState(lived_age_hours=1000.0),
        tmp_path,
    )

    # Seed a memory whose emotion is heavy.
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="cold rain on a cold night",
        memory_type="episodic",
        domain="memory",
        emotions={"sorrow": 8.5},
    )
    object.__setattr__(m, "id", "mem-heavy")
    store.create(m)
    store.close()

    # Build an open arc with that single member, last extended at lived-age
    # 100, which is now 900 lived-hours stale (well past 72h close threshold).
    arc = Arc(
        id="arc-test",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream-1",
        seed_memory_ids=("mem-heavy",),
        title="cold rain thread",
        opened_at_iso="2026-05-10T00:00:00+00:00",
        lived_age_at_open=100.0,
        last_extended_at_iso="2026-05-10T00:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(
            ArcMember(
                memory_id="mem-heavy",
                joined_at_iso="2026-05-10T00:00:00+00:00",
                lived_age_at_join=100.0,
                salience_at_join=0.7,
            ),
        ),
    )
    save_state(tmp_path, ArcsState(open={"arc-test": arc}, recently_closed=[]))

    class _NullBus:
        def publish(self, _evt: Any) -> None:
            pass

    class _FakeFeltTime:
        lived_age_hours: float = 1000.0

    class _StubHebbian:
        def weight(self, a: str, b: str) -> float:
            return 0.0

    class _StubEmbeddings:
        def get(self, memory_id: str) -> np.ndarray | None:
            return None

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

    store = MemoryStore(tmp_path / "memories.db")
    grief_rows = store._conn.execute(
        "SELECT content, emotions_json, metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    assert len(grief_rows) == 1, "expected exactly one arc-close grief breadcrumb"
    meta = json.loads(grief_rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "arc-test"
    assert meta["grief_subtype"] == "arc_close"
    content_str = (
        grief_rows[0]["content"]
        if isinstance(grief_rows[0]["content"], str)
        else grief_rows[0]["content"].decode()
    )
    assert content_str.startswith("the arc 'cold rain thread'")
    em = json.loads(grief_rows[0]["emotions_json"])
    assert em["memory_grief"] >= 3.0
    # Residue: sorrow at 8.5 × 0.5 = 4.25
    assert em.get("sorrow", 0.0) == pytest.approx(4.25, abs=0.01)
    store.close()


def test_arc_close_grieves_when_all_members_forgotten(tmp_path: Path) -> None:
    """An arc closes after all its members have already been forgotten — the
    graveyard-proxy fallback must reconstruct the original emotion at the
    same scale as a live lookup, so the arc grieves at full intensity.
    """
    from brain.felt_time.state import FeltTimeState
    from brain.felt_time.state import persist as persist_felt_time
    from brain.forgetting import graveyard as gv
    from brain.forgetting.salience import SalienceInputs
    from brain.memory.store import Memory, MemoryStore
    from brain.narrative_memory import run_pass
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, save_state

    persist_felt_time(
        FeltTimeState(lived_age_hours=1000.0),
        tmp_path,
    )
    # Create DB file without any live memories
    store = MemoryStore(tmp_path / "memories.db")
    store.close()

    # Plant a graveyard entry for the only arc member — memory is GONE from
    # the live store, only graveyard remains.
    m = Memory.create_new(
        content="the night we walked through the cold rain",
        memory_type="episodic",
        domain="memory",
        emotions={"sorrow": 8.5},
    )
    object.__setattr__(m, "id", "mem-gone")
    gv.append(
        tmp_path,
        memory=m,
        salience_at_drop=0.08,  # below LOST_THRESHOLD — typical at drop
        inputs=SalienceInputs(emotion=0.85, hebbian=0.0, recall=0.0, soul=0.0, freshness=0.0),
        lived_age_hours=24.0,
        reason="test-seed",
    )

    # An open arc whose only member is the now-forgotten memory. last_extended
    # very old so it will close due to staleness.
    arc = Arc(
        id="arc-forgotten",
        state="open",
        seed_anchor_type="dream",
        seed_anchor_ref="dream-1",
        seed_memory_ids=("mem-gone",),
        title="cold rain memory",
        opened_at_iso="2026-05-10T00:00:00+00:00",
        lived_age_at_open=100.0,
        last_extended_at_iso="2026-05-10T00:00:00+00:00",
        closed_at_iso=None,
        lived_age_at_close=None,
        members=(
            ArcMember(
                memory_id="mem-gone",
                joined_at_iso="2026-05-10T00:00:00+00:00",
                lived_age_at_join=100.0,
                salience_at_join=0.7,
            ),
        ),
    )
    save_state(tmp_path, ArcsState(open={"arc-forgotten": arc}, recently_closed=[]))

    class _NullBus:
        def publish(self, _evt: Any) -> None:
            pass

    class _FakeFeltTime:
        lived_age_hours: float = 1000.0

    class _StubHebbian:
        def weight(self, a: str, b: str) -> float:
            return 0.0

    class _StubEmbeddings:
        def get(self, memory_id: str) -> np.ndarray | None:
            return None

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

    store = MemoryStore(tmp_path / "memories.db")
    all_grief_rows = store._conn.execute(
        "SELECT content, emotions_json, metadata_json FROM memories WHERE memory_type='grief_event'"
    ).fetchall()
    arc_close_rows = [
        r
        for r in all_grief_rows
        if json.loads(r["metadata_json"]).get("grief_subtype") == "arc_close"
    ]
    # Expect ONE arc-close breadcrumb (not zero — the bug was that all-forgotten arcs never grieved)
    assert len(arc_close_rows) == 1, (
        "expected an arc-close grief breadcrumb even though all members are in the graveyard; "
        f"all grief rows: {[(r['content'], r['metadata_json']) for r in all_grief_rows]}"
    )
    meta = json.loads(arc_close_rows[0]["metadata_json"])
    assert meta["grief_referent_id"] == "arc-forgotten"
    assert meta["grief_subtype"] == "arc_close"
    em = json.loads(arc_close_rows[0]["emotions_json"])
    # emotion_norm 0.85 -> proxy 8.5 -> normalised 0.85 -> intensity 0.85 × 7.0 = 5.95
    assert em["memory_grief"] == pytest.approx(5.95, abs=0.05)
    # Residue: dominant non-grief emotion was sorrow=8.5; residue = 8.5 * 0.5 = 4.25
    assert em.get("sorrow", 0.0) == pytest.approx(4.25, abs=0.05)
    store.close()
