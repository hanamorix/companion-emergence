"""Tests for W7 — emotion seeding on bulk-extracted memories.

Ensures the bulk conversation extractor seeds emotions on each ExtractedItem,
those emotions are clamped/filtered correctly, and the committed memories
carry non-empty emotion vectors that the emotion aggregate can read.
"""

from __future__ import annotations

from brain.ingest.types import ExtractedItem


def test_extracteditem_carries_and_clamps_emotions():
    it = ExtractedItem(text="hi", label="feeling", importance=5,
                       emotions={"longing": 12.0, "peace": -1.0, "bogus": 3.0})
    it.normalize(valid_emotions={"longing", "peace"})
    assert it.emotions["longing"] == 10.0     # clamped to <=10
    assert "peace" not in it.emotions          # <=0 dropped
    assert "bogus" not in it.emotions          # unregistered dropped


def test_parse_extraction_reads_emotions():
    """Extractor parse path passes emotions through into ExtractedItem."""
    import json
    from unittest.mock import MagicMock

    from brain.ingest.extract import extract_items_with_status

    fake_provider = MagicMock()
    fake_provider.generate.return_value = json.dumps([
        {"text": "x", "label": "feeling", "importance": 6, "emotions": {"longing": 7}}
    ])

    outcome = extract_items_with_status(
        "user: hello",
        provider=fake_provider,
        max_retries=0,
    )
    assert len(outcome.items) == 1
    item = outcome.items[0]
    assert item.emotions == {"longing": 7.0}


def test_commit_item_passes_emotions_to_memory():
    """commit_item forwards item.emotions into the committed Memory."""
    from brain.ingest.commit import commit_item
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")
    item = ExtractedItem(
        text="she seemed wistful today",
        label="observation",
        importance=6,
        emotions={"longing": 7.0, "nostalgia": 5.0},
    )
    mem_id = commit_item(item, session_id="sess_w7", store=store, hebbian=hebbian)

    assert mem_id is not None
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.emotions == {"longing": 7.0, "nostalgia": 5.0}


def test_close_session_commits_memories_with_emotions_and_aggregate_is_non_empty(
    tmp_path,
):
    """Integration: close_session with a provider emitting emotions → committed
    memories carry emotions → aggregate_state is non-empty."""
    from unittest.mock import patch

    from brain.emotion.aggregate import aggregate_state
    from brain.ingest.buffer import ingest_turn
    from brain.ingest.extract import ExtractionOutcome
    from brain.ingest.pipeline import close_session
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hebbian = HebbianMatrix(":memory:")

    # Write one conversation turn so the buffer exists.
    ingest_turn(
        tmp_path,
        {"session_id": "sess_integ", "speaker": "user", "text": "I miss you so much"},
    )

    # Fake extractor returns an item with baseline emotions (longing + love both
    # exist in the baseline vocabulary, so they pass the registered-vocab gate).
    # Use only baseline vocab names (loneliness + love are both baseline).
    extracted_items = [
        ExtractedItem(
            text="user misses the assistant deeply",
            label="feeling",
            importance=7,
            emotions={"loneliness": 8.0, "love": 6.0},
        )
    ]

    with patch(
        "brain.ingest.pipeline.extract_items_with_status",
        return_value=ExtractionOutcome(items=extracted_items),
    ):
        report = close_session(
            tmp_path,
            "sess_integ",
            store=store,
            hebbian=hebbian,
            provider=None,  # provider not reached — patched above
        )

    assert report.committed >= 1
    assert len(report.memory_ids) >= 1

    # At least one committed memory carries non-empty emotions.
    mems = [store.get(mid) for mid in report.memory_ids]
    assert any(m is not None and m.emotions for m in mems), (
        "Expected at least one committed memory with non-empty emotions"
    )

    # aggregate_state over committed memories returns a non-empty state.
    active_mems = store.list_active()
    state = aggregate_state(active_mems)
    # loneliness is a baseline vocabulary name — it should appear in the aggregated state.
    assert state.emotions.get("loneliness", 0.0) > 0.0, (
        f"Expected loneliness > 0 in aggregate; got {state.emotions}"
    )
