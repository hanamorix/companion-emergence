"""Tests for brain.growth.crystallizers.vocabulary."""

from __future__ import annotations

from brain.growth.crystallizers.vocabulary import crystallize_vocabulary
from brain.memory.store import Memory, MemoryStore


def _mem(content: str, emotions: dict[str, float], *, domain: str = "us") -> Memory:
    return Memory.create_new(
        content=content,
        memory_type="conversation",
        domain=domain,
        emotions=emotions,
        tags=[],
        importance=8.0,
    )


def test_crystallizer_returns_empty_when_there_is_not_enough_evidence() -> None:
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("one tender complicated moment", {"love": 8, "grief": 7}))

        result = crystallize_vocabulary(store, current_vocabulary_names=set(), eng=None)

        assert result == []
    finally:
        store.close()


def test_crystallizer_proposes_recurring_unnamed_emotional_configuration() -> None:
    store = MemoryStore(":memory:")
    try:
        first = _mem("love and grief braided through the promise", {"love": 9, "grief": 8})
        second = _mem("another love-grief evening", {"love": 8, "grief": 7})
        third = _mem("grief softened by love", {"grief": 9, "love": 7})
        store.create(first)
        store.create(second)
        store.create(third)
        store.create(_mem("plain joy", {"joy": 9}))

        result = crystallize_vocabulary(store, current_vocabulary_names={"love", "grief", "joy"}, eng=None)

        assert len(result) == 1
        proposal = result[0]
        assert proposal.name == "grief_love_blend"  # #174: canonical sorted spelling
        assert "love" in proposal.description
        assert "grief" in proposal.description
        assert proposal.evidence_memory_ids == (first.id, second.id, third.id)
        assert proposal.score >= 0.7
        assert proposal.relational_context == "recurring emotional configuration in us memories"

    finally:
        store.close()


def test_crystallizer_evidence_keeps_write_order_when_times_tie() -> None:
    """Windows' coarse clock gives memories made in a row the same created_at; evidence is still oldest first."""
    from datetime import UTC, datetime

    store = MemoryStore(":memory:")
    try:
        same = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        written = []
        for content in ("love and grief braided", "another love-grief evening", "grief softened by love"):
            memory = _mem(content, {"love": 9, "grief": 8})
            memory.created_at = same
            store.create(memory)
            written.append(memory.id)

        result = crystallize_vocabulary(store, current_vocabulary_names={"love", "grief"}, eng=None)

        assert [p.evidence_memory_ids for p in result] == [tuple(written)]
    finally:
        store.close()


def test_crystallizer_skips_configuration_when_name_already_exists() -> None:
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("love and grief braided through the promise", {"love": 9, "grief": 8}))
        store.create(_mem("another love-grief evening", {"love": 8, "grief": 7}))
        store.create(_mem("grief softened by love", {"grief": 9, "love": 7}))

        result = crystallize_vocabulary(
            store,
            current_vocabulary_names={"love", "grief", "love_grief_blend"},
            eng=None,
        )

        assert result == []
    finally:
        store.close()


def test_crystallizer_treats_reversed_blend_name_as_existing() -> None:
    """#174: grief_love_blend and love_grief_blend are the same configuration."""
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("love and grief braided", {"love": 9, "grief": 8}))
        store.create(_mem("another love-grief evening", {"love": 8, "grief": 7}))
        store.create(_mem("grief softened by love", {"love": 9, "grief": 7}))

        result = crystallize_vocabulary(
            store, current_vocabulary_names={"love", "grief", "grief_love_blend"}, eng=None
        )

        assert result == []
    finally:
        store.close()


def test_crystallizer_mints_blend_name_in_canonical_sorted_order() -> None:
    """#174: never mint love_grief_blend — the canonical spelling sorts its parts."""
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("love and grief braided", {"love": 9, "grief": 8}))
        store.create(_mem("another love-grief evening", {"love": 8, "grief": 7}))
        store.create(_mem("grief softened by love", {"love": 9, "grief": 7}))

        result = crystallize_vocabulary(store, current_vocabulary_names={"love", "grief"}, eng=None)

        assert [p.name for p in result] == ["grief_love_blend"]
    finally:
        store.close()


def test_crystallizer_treats_hyphen_and_case_variants_as_existing() -> None:
    """#174 reopen: Grief-Love_blend already names this configuration."""
    store = MemoryStore(":memory:")
    try:
        store.create(_mem("love and grief braided", {"love": 9, "grief": 8}))
        store.create(_mem("another love-grief evening", {"love": 8, "grief": 7}))
        store.create(_mem("grief softened by love", {"love": 9, "grief": 7}))

        result = crystallize_vocabulary(
            store, current_vocabulary_names={"love", "grief", "Love-Grief_blend"}, eng=None
        )

        assert result == []
    finally:
        store.close()


def _half_life_of_blend(emotions: dict[str, float]) -> float:
    store = MemoryStore(":memory:")
    try:
        for n in range(3):
            store.create(_mem(f"blend moment {n}", dict(emotions)))
        (proposal,) = crystallize_vocabulary(store, current_vocabulary_names=set(), eng=None)
        return proposal.decay_half_life_days
    finally:
        store.close()


def test_crystallizer_scans_only_the_newest_window(monkeypatch) -> None:
    """A cluster older than the scan window forms no proposal; one inside it does."""
    from datetime import UTC, datetime, timedelta

    from brain.growth.crystallizers import vocabulary as crystallizer

    monkeypatch.setattr(crystallizer, "_SCAN_LIMIT", 3)
    store = MemoryStore(":memory:")
    try:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        # Oldest: a grief/love cluster, then the three newest: a joy/awe cluster.
        for n, emotions in enumerate([{"grief": 8, "love": 7}] * 3 + [{"joy": 8, "awe": 7}] * 3):
            m = _mem(f"moment {n}", dict(emotions))
            m.created_at = base + timedelta(days=n)
            store.create(m)

        result = crystallize_vocabulary(store, current_vocabulary_names=set(), eng=None)

        assert [p.name for p in result] == ["awe_joy_blend"]  # canonical sorted spelling

        monkeypatch.setattr(crystallizer, "_SCAN_LIMIT", 2)
        assert crystallize_vocabulary(store, current_vocabulary_names=set(), eng=None) == []
    finally:
        store.close()


def test_crystallized_blend_half_life_is_parent_mean() -> None:
    assert _half_life_of_blend({"joy": 8, "grief": 7}) == (3.0 + 60.0) / 2


def test_crystallizer_skips_a_refused_blend_name_and_proposes_the_next_cluster() -> None:
    """A top cluster whose blend name the gate refuses must not block every other blend each tick."""
    store = MemoryStore(":memory:")
    try:
        for n in range(4):
            store.create(_mem(f"digit moment {n}", {"grief2": 9, "love": 8}))
        for n in range(3):
            store.create(_mem(f"bright moment {n}", {"joy": 8, "awe": 7}))

        result = crystallize_vocabulary(store, current_vocabulary_names=set(), eng=None)

        assert [p.name for p in result] == ["awe_joy_blend"]
    finally:
        store.close()


def test_crystallized_blend_parent_uses_its_registered_half_life() -> None:
    """Without an engine, a registered persona-extension parent lends its own half-life, not the default."""
    from brain.emotion import vocabulary
    from brain.emotion.vocabulary import Emotion

    vocabulary.register(
        Emotion(name="ember_feel", description="d", category="persona_extension", decay_half_life_days=40.0)
    )
    assert _half_life_of_blend({"ember_feel": 8, "joy": 7}) == (40.0 + 3.0) / 2


def test_crystallized_blend_with_love_uses_its_dev_half_life() -> None:
    assert _half_life_of_blend({"love": 8, "joy": 7}) == (730.0 + 3.0) / 2
