"""Engine 6, G0: Check 11 (the writer survey) as a pin. Which memory types carry feelings stored below 1.

Some writers store feelings below 1 (small constants such as 0.12 and 0.15, or a model's deltas capped for a peer), not on
the 0-10 scale the engine works in. The one-time fix at the engine's first start (`nudge_scale_fix`) multiplies those rows
by ten. Its row selector names the memory types those writers use; the list is `dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES`.
These tests drive the known writers (reach-outs, notes, makings, peer memories) and check that each stores a type on that
list with every value and the stored peak below the fix's threshold. A rename of a type, a writer that rises to 1 or more,
or a writer that stops storing feelings fails here instead of leaving rows the fix would not scale. A NEW writer is not seen
by these tests; the survey of writers (build records, `G0-check11.md`) is what guards against one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain import dev_constants
from brain.engines.consolidation import Decision, run_consolidation
from brain.initiate import reach_emotion
from brain.initiate.memory import write_initiate_memory
from brain.kindled_link.relationship import apply_peer_emotion, write_kindled_peer_memory
from brain.kindled_link.store import KindledLinkStore
from brain.maker.maker import Making
from brain.maker.wiring import making_emotion_delta, write_making_memory
from brain.memory.pending import PendingQueue
from brain.memory.store import Memory, MemoryStore
from brain.notes import runner as notes_runner

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path):
    memory_store = MemoryStore(tmp_path / "memories.db")
    yield memory_store
    memory_store.close()


def _rows(store: MemoryStore, memory_type: str) -> list[Memory]:
    return store.list_by_type(memory_type, active_only=False)


def _assert_scaled_down(rows: list[Memory]) -> None:
    """Every row is of a listed type, has feelings, and every value and the stored peak are below the fix's threshold."""
    assert rows, "the writer stored nothing"
    for row in rows:
        assert row.memory_type in dev_constants.NUDGE_SCALE_FIX_MEMORY_TYPES
        assert row._stored_emotions, f"{row.memory_type} row stored no feelings"
        assert all(0 < value < dev_constants.NUDGE_SCALE_FIX_THRESHOLD for value in row._stored_emotions.values())
        assert 0 < row._stored_peak < dev_constants.NUDGE_SCALE_FIX_THRESHOLD


@pytest.mark.parametrize("source", sorted(reach_emotion._SOURCE_ACCENT))
def test_a_delivered_reach_out_stores_small_feelings_under_a_listed_type(store: MemoryStore, source: str) -> None:
    write_initiate_memory(
        store,
        audit_id=f"audit-{source}",
        subject="a thought",
        message="thinking of you",
        state="delivered",
        ts=NOW.isoformat(),
        reach_emotions=reach_emotion.reach_emotions_for(source),
    )
    _assert_scaled_down(_rows(store, "initiate_outbound"))


def test_a_note_stores_small_feelings_under_a_listed_type(store: MemoryStore, tmp_path: Path) -> None:
    """The note runner's own wire-back (`brain/notes/runner.py`) opens the persona's database and writes the note-state memory."""
    note = SimpleNamespace(subject="a note")
    notes_runner._wire_note_back(tmp_path, note=note, folder=tmp_path / "notes", user_name="Bob", now=NOW)
    _assert_scaled_down(_rows(store, "initiate_outbound"))


@pytest.mark.parametrize("dominant_source", ["grief", "joy", "dream", "soul", "anything else"])
def test_a_making_stores_small_feelings_under_a_listed_type(store: MemoryStore, dominant_source: str) -> None:
    making = Making(type="poem", title="Untitled", content="words", disposition="private")
    write_making_memory(store, making, emotions=making_emotion_delta(making, dominant_source=dominant_source))
    # `making` is a gated type: the writer enqueues a candidate, and the consolidation gate promotes it into memories.db
    # with the same feelings. Run the gate, so the row checked is the one the fix would meet.
    assert PendingQueue(store.persona_dir).read_recent("making", limit=5), "the writer queued nothing"
    run_consolidation(store, persona_dir=store.persona_dir, classifier=lambda _candidate, _context: Decision("new"))
    _assert_scaled_down(_rows(store, "making"))


def test_a_peer_memory_stores_small_feelings_under_a_listed_type(store: MemoryStore, tmp_path: Path) -> None:
    link_store = KindledLinkStore(tmp_path / "kindled_link.db")
    try:
        applied = apply_peer_emotion(link_store, "kid_a", {"tenderness": 0.5, "joy": 0.5}, NOW)
    finally:
        link_store.close()
    write_kindled_peer_memory(
        store, peer_id="kid_a", session_id="s1", speaker="peer", stage="acquaintance", content="a peer said something", emotions=applied
    )
    _assert_scaled_down(_rows(store, "kindled_peer"))
