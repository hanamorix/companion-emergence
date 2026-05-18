"""tests/forgetting/test_prompt.py — TDD for brain.forgetting.prompt.

Per plan Phase 8 Task 8.1 Step 1.
"""

from __future__ import annotations

from pathlib import Path

from brain.forgetting import graveyard
from brain.forgetting.prompt import render_fading_summary_block
from brain.forgetting.salience import SalienceInputs
from brain.memory.store import Memory, MemoryStore


def _make_memory(*, content: str = "x", state: str = "active") -> Memory:
    m = Memory.create_new(content=content, memory_type="episodic", domain="chat", emotions={})
    object.__setattr__(m, "state", state)
    return m


def test_render_empty_week_says_nothing_softened(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    blob = render_fading_summary_block(tmp_path, store)
    assert "nothing has softened" in blob.lower()
    store.close()


def test_render_with_mixed_states(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    # 2 fading
    for i in range(2):
        m = Memory.create_new(
            content=f"item {i}", memory_type="episodic", domain="chat", emotions={}
        )
        store.create(m)
        store.fade(m.id, summary=f"item {i} summary")
    # 1 lost (graveyard entry within last 7 days)
    lost_m = _make_memory(content="lost item")
    graveyard.append(
        tmp_path,
        memory=lost_m,
        salience_at_drop=0.05,
        inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
        lived_age_hours=100.0,
        reason="x",
    )

    blob = render_fading_summary_block(tmp_path, store)
    assert "2" in blob  # 2 softened
    assert "1" in blob  # 1 lost
    assert "softened" in blob.lower()
    assert "lost" in blob.lower()
    store.close()


def test_render_stays_under_token_budget(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    # 50 fading + 50 lost — enough to verify counts render and stay compact
    for i in range(50):
        m = Memory.create_new(
            content=f"item {i}", memory_type="episodic", domain="chat", emotions={}
        )
        store.create(m)
        store.fade(m.id, summary=f"s{i}")
        graveyard.append(
            tmp_path,
            memory=m,
            salience_at_drop=0.05,
            inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
            lived_age_hours=100.0,
            reason="x",
        )
    blob = render_fading_summary_block(tmp_path, store)
    # 1 token ≈ 4 chars. 120-token budget → ≤480 chars.
    assert len(blob) <= 480
    store.close()
