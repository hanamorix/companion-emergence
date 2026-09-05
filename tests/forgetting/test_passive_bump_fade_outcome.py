# tests/forgetting/test_passive_bump_fade_outcome.py
"""G20 (recall-reinforcement) — the passive-path analog of the W5 promotion
gate: a memory that is passively surface-bumped for many turns and then
abandoned still fades / is not permanently rescued by the accumulated
recall_count alone.

Why this can't false-green: the recall arm's weight in the salience blend
(``DEFAULT_WEIGHTS["recall"] == 0.20``) is capped BELOW ``FADE_THRESHOLD``
(0.25, ``brain/forgetting/policy.py``) — even a fully clamped recall_count
(``recall_count/10``, capped at 1.0) can, at most, contribute 0.20 to
salience. So a memory with no emotion, no hebbian edges, no soul link and
stale freshness cannot be held above the fade line by recall_count alone,
no matter how many times it is passively surfaced. This test drives the
REAL passive-recall bump path (``_build_recall_block`` ->
``store.bump_recall``) rather than asserting that arithmetic fact directly,
so it exercises the actual code, not just the formula.

The existing W5 test (test_corecall_decay_balance.py) drives
search_memories/rank_memories, which are bump-free by design and so cannot
catch a regression in the passive-bump path — this test closes that gap
(stage-3 CH8 gate-4 addition).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brain.chat.prompt import _build_recall_block
from brain.felt_time.state import FeltTimeState
from brain.felt_time.state import persist as persist_felt_time
from brain.forgetting import run_pass
from brain.memory.store import Memory, MemoryStore


def test_passively_bumped_memory_still_fades_after_abandonment(tmp_path):
    """A memory passively surface-bumped for many turns, then abandoned,
    still fades/is lost on the normal forgetting schedule — the passive
    recall_count reinforcement does not create a permanent survival
    attractor."""
    persist_felt_time(
        FeltTimeState(lived_age_hours=200.0, last_tick_ts="2026-05-18T00:00:00+00:00"),
        tmp_path,
    )

    # Old enough (well past the 30-day wall-clock recent-buffer AND the
    # 90-day freshness horizon) and low-salience (no emotion, no soul link)
    # so recall_count is the only arm this test exercises.
    store = MemoryStore(tmp_path / "memories.db")
    m = Memory.create_new(
        content="orchid signal beacon in the old workshop",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    m.created_at = datetime.now(UTC) - timedelta(days=200)
    store.create(m)
    store.close()

    # --- Phase 1: passively surface-bump the memory for many turns via the
    # REAL _build_recall_block assembly path (it is the sole match each
    # turn, so N=1 -> top amount ~0.8 every time), well past the recall
    # arm's clamp (recall_count/10, capped at 1.0). ---
    store = MemoryStore(tmp_path / "memories.db")
    for _ in range(20):
        block = _build_recall_block(store, "orchid signal beacon", persona_dir=tmp_path)
        assert block.strip() != ""
    row = store._conn.execute(
        "SELECT recall_count FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    rc = row["recall_count"]
    assert rc >= 10.0, "recall_count should be well past the salience clamp before abandonment"
    store.close()

    # --- Phase 2: ABANDON — stop surfacing it (no further calls). ---

    # --- Phase 3: run forgetting; the abandoned memory must still fade/lose,
    # even though recall_count is maxed. ---
    faded_or_lost = 0
    for _ in range(4):
        summary = run_pass(tmp_path, event_bus=MagicMock())
        faded_or_lost += summary["faded"] + summary["lost"]

    assert faded_or_lost >= 1, (
        "a passively surface-bumped memory must still fade/lose once abandoned — "
        "the recall_count reinforcement is not a permanent survival attractor "
        "(the recall arm's weight, 0.20, is capped below FADE_THRESHOLD, 0.25)"
    )

    store = MemoryStore(tmp_path / "memories.db")
    result_row = store._conn.execute(
        "SELECT state FROM memories WHERE id = ?", (m.id,)
    ).fetchone()
    store.close()

    if result_row is None:
        from brain.forgetting import graveyard

        entries = graveyard.read_all(tmp_path)
        assert any(e["memory_id"] == m.id for e in entries), (
            "memory row is gone but no graveyard entry — unexpected deletion path"
        )
    else:
        assert result_row["state"] in ("fading", "lost"), (
            f"memory should have decayed off 'active', got state={result_row['state']!r}"
        )
