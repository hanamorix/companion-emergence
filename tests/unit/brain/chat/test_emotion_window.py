"""#90 — the felt-state window must not be consumed by emotionless memories.

Measured on the real persona before writing this: 48 of the 50 most recent
memories were `heartbeat` rows, none carrying emotion, so `aggregate_state`
over that window returned `{}` — her injected felt state was empty while
`love: 10.0` and `belonging: 10.0` sat just outside it.

heartbeat writes ~50-75 emotionless memories a day and is 49% of the store, so
the window fills with them within a day of quiet.

The fix is semantically neutral: an emotionless memory is a proven no-op in
aggregate_state (it contributes nothing to a max-pool), so excluding it cannot
change the aggregation — it only stops the window being spent on rows that
cannot contribute.
"""
from __future__ import annotations

from pathlib import Path

from brain.memory.store import Memory, MemoryStore


def _seed(store: MemoryStore, *, emotionless: int) -> None:
    """One older emotive memory, then `emotionless` newer inert ones on top."""
    store.create(Memory.create_new(
        content="Something that mattered.", memory_type="fact", domain="us",
        # A BODY emotion, so it renders by name in the block (BODY_EMOTION_NAMES).
        emotions={"comfort_seeking": 7.0, "love": 9.0},
    ))
    for i in range(emotionless):
        store.create(Memory.create_new(
            content=f"HEARTBEAT: tended the graph. {i}",
            memory_type="heartbeat", domain="us",
        ))


def test_felt_state_survives_a_window_full_of_heartbeats(tmp_path: Path):
    """The real-world case: a day of quiet buries the emotion under heartbeats."""
    from brain.chat.prompt import _build_body_block

    store = MemoryStore(db_path=":memory:")
    try:
        _seed(store, emotionless=60)  # more than the 50-row window
        block = _build_body_block(store, tmp_path)
    finally:
        store.close()

    assert block, "body block should render"
    assert "body emotions:" in block, (
        f"her felt state was buried by emotionless memories — block was:\n{block}"
    )
    assert "comfort_seeking" in block, block


def test_get_body_state_tool_sees_the_same_emotions(tmp_path: Path):
    """The tool she calls herself must agree with the block she is given.

    Its comment claimed it matched _build_emotion_summary's window; it did not —
    that function had already been filtered and this had not, so the two
    disagreed about her own emotional state.
    """
    from brain.memory.hebbian import HebbianMatrix
    from brain.tools.impls.get_body_state import get_body_state

    store = MemoryStore(db_path=":memory:")
    try:
        _seed(store, emotionless=60)
        heb = HebbianMatrix(db_path=":memory:")
        try:
            out = get_body_state(store=store, hebbian=heb, persona_dir=tmp_path)
        finally:
            heb.close()
    finally:
        store.close()

    body_emotions = out.get("body_emotions") or {}
    assert body_emotions.get("comfort_seeking", 0) > 0, (
        f"the tool reports no felt state while the block does — out={out}"
    )
