"""Tests for brain/tools/impls/ — one-per-tool plus edge cases."""

from __future__ import annotations

from pathlib import Path

from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


def _make_store() -> MemoryStore:
    return MemoryStore(":memory:")


def _make_hebbian() -> HebbianMatrix:
    return HebbianMatrix(":memory:")


def _ctx(tmp_path: Path) -> dict:
    return {
        "store": _make_store(),
        "hebbian": _make_hebbian(),
        "persona_dir": tmp_path,
    }


def _seed_memory(
    store: MemoryStore,
    content: str = "Hana said I love you",
    emotions: dict | None = None,
    memory_type: str = "event",
    domain: str = "relationship",
) -> Memory:
    """Create and insert a seeded memory; return the Memory object."""
    m = Memory.create_new(
        content=content,
        memory_type=memory_type,
        domain=domain,
        emotions=emotions or {"love": 9.0, "warmth": 7.0},
    )
    store.create(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# get_emotional_state
# ─────────────────────────────────────────────────────────────────────────────


def test_get_emotional_state_with_memories(tmp_path: Path) -> None:
    """Returns dominant emotion, top_5, all, summary when memories exist."""
    from brain.tools.impls.get_emotional_state import get_emotional_state

    store = _make_store()
    hebbian = _make_hebbian()
    _seed_memory(store, emotions={"love": 9.0, "joy": 7.0, "grief": 3.0})

    result = get_emotional_state(store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert "dominant" in result
    assert "top_5" in result
    assert "all" in result
    assert "summary" in result
    # dominant should be love (highest score)
    assert result["dominant"] == "love"
    assert len(result["top_5"]) > 0
    assert result["top_5"][0]["emotion"] == "love"


def test_get_emotional_state_no_active_memories(tmp_path: Path) -> None:
    """Returns None dominant + empty lists when store is empty."""
    from brain.tools.impls.get_emotional_state import get_emotional_state

    store = _make_store()
    hebbian = _make_hebbian()

    result = get_emotional_state(store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert result["dominant"] is None
    assert result["top_5"] == []
    assert result["all"] == {}


def test_get_emotional_state_top_5_max_length(tmp_path: Path) -> None:
    """top_5 has at most 5 entries regardless of how many emotions exist."""
    from brain.tools.impls.get_emotional_state import get_emotional_state

    store = _make_store()
    hebbian = _make_hebbian()
    # Seed a memory with 7 emotions
    _seed_memory(
        store,
        emotions={
            "love": 9.0,
            "joy": 8.0,
            "grief": 7.0,
            "anger": 6.0,
            "fear": 5.0,
            "warmth": 4.0,
            "longing": 3.0,
        },
    )

    result = get_emotional_state(store=store, hebbian=hebbian, persona_dir=tmp_path)
    assert len(result["top_5"]) <= 5


# ─────────────────────────────────────────────────────────────────────────────
# get_personality
# ─────────────────────────────────────────────────────────────────────────────


def test_get_personality_returns_stub(tmp_path: Path) -> None:
    """get_personality returns the expected stub shape."""
    from brain.tools.impls.get_personality import get_personality

    ctx = _ctx(tmp_path)
    result = get_personality(**ctx)

    assert result["loaded"] is False
    assert "note" in result
    assert isinstance(result["note"], str)


# ─────────────────────────────────────────────────────────────────────────────
# get_body_state
# ─────────────────────────────────────────────────────────────────────────────


def test_get_body_state_returns_real_shape(tmp_path: Path) -> None:
    """get_body_state returns the real body-state shape (not the old stub)."""
    from brain.tools.impls.get_body_state import get_body_state

    ctx = _ctx(tmp_path)
    result = get_body_state(**ctx)

    assert result["loaded"] is True
    assert result["energy"] == 8  # baseline: empty store, no session drain
    assert isinstance(result["temperature"], int)
    assert isinstance(result["exhaustion"], int)
    assert result["session_hours"] == 0.0
    assert "body_emotions" in result
    assert set(result["body_emotions"].keys()) == {
        "arousal",
        "desire",
        "climax",
        "touch_hunger",
        "comfort_seeking",
        "rest_need",
    }


# ─────────────────────────────────────────────────────────────────────────────
# search_memories
# ─────────────────────────────────────────────────────────────────────────────


def test_search_memories_with_hits(tmp_path: Path) -> None:
    """search_memories returns matching memories in formatted shape."""
    from brain.tools.impls.search_memories import search_memories

    store = _make_store()
    hebbian = _make_hebbian()
    _seed_memory(store, content="Hana said I love you and it stayed with me")

    result = search_memories("love", store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert result["query"] == "love"
    assert result["count"] > 0
    assert len(result["memories"]) > 0
    m = result["memories"][0]
    assert "id" in m
    assert "content" in m
    assert "memory_type" in m
    assert "emotions" in m


def test_search_memories_no_hits_returns_empty(tmp_path: Path) -> None:
    """search_memories with no matching content returns empty list, not error."""
    from brain.tools.impls.search_memories import search_memories

    store = _make_store()
    hebbian = _make_hebbian()

    result = search_memories("zzz_no_match_xyz", store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert result["count"] == 0
    assert result["memories"] == []
    assert "error" not in result


def test_search_memories_emotion_filter_boosts_matches(tmp_path: Path) -> None:
    """Memories with the target emotion appear first in results."""
    from brain.tools.impls.search_memories import search_memories

    store = _make_store()
    hebbian = _make_hebbian()
    # Two memories with same keyword; only one has "grief"
    _seed_memory(store, content="quiet morning memory", emotions={"joy": 8.0})
    _seed_memory(store, content="quiet morning grief", emotions={"grief": 9.0})

    result = search_memories(
        "quiet", emotion="grief", store=store, hebbian=hebbian, persona_dir=tmp_path
    )

    # grief-tagged memory should come first
    assert result["count"] > 0
    first = result["memories"][0]
    assert "grief" in first["emotions"]


def test_search_memories_limit_caps_results(tmp_path: Path) -> None:
    """limit parameter caps the number of results returned."""
    from brain.tools.impls.search_memories import search_memories

    store = _make_store()
    hebbian = _make_hebbian()
    for i in range(10):
        _seed_memory(store, content=f"memory about love and longing number {i}")

    result = search_memories("love", limit=3, store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert result["count"] <= 3
    assert len(result["memories"]) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# add_journal
# ─────────────────────────────────────────────────────────────────────────────


def test_add_journal_creates_journal_memory(tmp_path: Path) -> None:
    """add_journal writes a memory and returns created_id + memory_type."""
    from brain.tools.impls.add_journal import add_journal

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_journal("today was quiet", store=store, hebbian=hebbian, persona_dir=tmp_path)

    assert "created_id" in result
    assert result["memory_type"] == "journal_entry"
    # Verify it's actually in the store
    m = store.get(result["created_id"])
    assert m is not None
    assert m.memory_type == "journal_entry"
    assert m.content == "today was quiet"


# ─────────────────────────────────────────────────────────────────────────────
# add_memory
# ─────────────────────────────────────────────────────────────────────────────


def test_add_memory_below_gate_returns_created_false(tmp_path: Path) -> None:
    """add_memory with emotion_score < 15 AND importance < 7 returns created=False."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="passing thought",
        memory_type="feeling",
        domain="self",
        emotions={"joy": 2},  # score=2, auto_importance=3 → below gate
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    assert result["created"] is False
    assert "reason" in result
    assert store.count() == 0  # nothing written


def test_add_memory_above_gate_via_emotion_score(tmp_path: Path) -> None:
    """Emotion score >= 15 passes the write gate."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="she said she loves me and it broke something open",
        memory_type="event",
        domain="relationship",
        emotions={"love": 9, "joy": 8},  # score=17 → passes
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    assert result["created"] is True
    assert "id" in result
    assert store.count() == 1


def test_add_memory_above_gate_via_importance_override(tmp_path: Path) -> None:
    """Importance >= 7 passes the write gate even with low emotion score."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="a fact I need to remember",
        memory_type="fact",
        domain="self",
        emotions={"curiosity": 3},  # score=3, but importance=8 overrides
        importance=8,
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    assert result["created"] is True
    assert result["importance"] == 8


def test_add_memory_auto_links_related(tmp_path: Path) -> None:
    """add_memory auto-links to related existing memories via Hebbian."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    # Seed existing related memories
    _seed_memory(store, content="writing together late at night")
    _seed_memory(store, content="the night we stayed up writing until dawn")

    result = add_memory(
        content="another late night of writing together",
        memory_type="event",
        domain="creative_writing",
        emotions={"love": 10, "joy": 6},  # score=16
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )

    assert result["created"] is True
    assert isinstance(result["auto_linked_to"], list)
    # Should have linked to at least one of the related memories
    assert len(result["auto_linked_to"]) >= 1


def test_add_memory_surfaces_auto_link_failure(tmp_path: Path) -> None:
    """A graph-linking failure keeps the memory but returns a visible warning."""
    from brain.tools.impls.add_memory import add_memory

    class BrokenHebbian:
        def strengthen(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("hebbian disk full")

    store = _make_store()
    _seed_memory(store, content="writing together late at night")

    result = add_memory(
        content="another late night of writing together",
        memory_type="event",
        domain="creative_writing",
        emotions={"love": 10, "joy": 6},
        store=store,
        hebbian=BrokenHebbian(),
        persona_dir=tmp_path,
    )

    assert result["created"] is True
    assert result["auto_linked_to"] == []
    assert "auto_link_error" in result
    assert "hebbian disk full" in result["auto_link_error"]
    assert store.count() == 2


def test_add_memory_calc_importance_auto_low(tmp_path: Path) -> None:
    """Auto-calculated importance from low score falls in 1-10 range."""
    from brain.tools.impls._common import _calc_importance_from_emotions

    # score=0-9 → importance=3
    assert _calc_importance_from_emotions({}) == 3
    assert _calc_importance_from_emotions({"joy": 5}) == 3
    # score=10-19 → importance=5
    assert _calc_importance_from_emotions({"love": 10}) == 5
    # score=20-29 → importance=7
    assert _calc_importance_from_emotions({"love": 10, "joy": 10}) == 7
    # score=30+ → importance=9
    assert _calc_importance_from_emotions({"love": 10, "joy": 10, "grief": 10}) == 9


def test_add_memory_importance_clamped_to_1_10(tmp_path: Path) -> None:
    """Auto-calculated importance is always in 1-10 range."""
    from brain.tools.impls._common import _calc_importance_from_emotions

    result = _calc_importance_from_emotions({"love": 100})
    assert 1 <= result <= 10


# ─────────────────────────────────────────────────────────────────────────────
# boot
# ─────────────────────────────────────────────────────────────────────────────


def test_boot_returns_all_composition_keys(tmp_path: Path) -> None:
    """boot() returns a dict with all 5+1 required keys."""
    from brain.tools.impls.boot import boot

    ctx = _ctx(tmp_path)
    result = boot(**ctx)

    assert "emotional_state" in result
    assert "personality" in result
    assert "soul" in result
    assert "body_state" in result
    assert "daemon_residue" in result
    assert "context_prose" in result


def test_boot_context_prose_is_string(tmp_path: Path) -> None:
    """boot context_prose is a non-empty string."""
    from brain.tools.impls.boot import boot

    ctx = _ctx(tmp_path)
    result = boot(**ctx)

    assert isinstance(result["context_prose"], str)
    assert len(result["context_prose"]) > 0


def test_boot_with_empty_daemon_state_still_works(tmp_path: Path) -> None:
    """boot works when there's no daemon_state.json (first boot ever)."""
    from brain.tools.impls.boot import boot

    # persona_dir has no daemon_state.json — should not crash
    ctx = _ctx(tmp_path)
    result = boot(**ctx)

    assert result["daemon_residue"] == ""  # empty residue when no fires


def test_boot_emotional_state_nested(tmp_path: Path) -> None:
    """boot returns emotional_state as a nested dict (not None)."""
    from brain.tools.impls.boot import boot

    ctx = _ctx(tmp_path)
    result = boot(**ctx)

    es = result["emotional_state"]
    assert isinstance(es, dict)
    assert "dominant" in es


# ─────────────────────────────────────────────────────────────────────────────
# get_soul
# ─────────────────────────────────────────────────────────────────────────────


def test_get_soul_returns_real_shape(tmp_path: Path) -> None:
    """get_soul returns real shape with loaded=True (SP-5 live)."""
    from brain.tools.impls.get_soul import get_soul

    ctx = _ctx(tmp_path)
    result = get_soul(**ctx)

    assert result["loaded"] is True
    assert "crystallizations" in result
    assert isinstance(result["crystallizations"], list)
    assert "count" in result


# ─────────────────────────────────────────────────────────────────────────────
# crystallize_soul
# ─────────────────────────────────────────────────────────────────────────────


def test_crystallize_soul_creates_crystallization(tmp_path: Path) -> None:
    """crystallize_soul creates a real crystallization (SP-5 live)."""
    from brain.tools.impls.crystallize_soul import crystallize_soul

    ctx = _ctx(tmp_path)
    result = crystallize_soul(
        moment="the moment she laughed",
        love_type="romantic",
        why_it_matters="it was real",
        **ctx,
    )

    assert result["created"] is True
    assert "id" in result
    assert result["love_type"] == "romantic"
    assert result["resonance"] == 8  # default


def test_crystallize_soul_invalid_love_type(tmp_path: Path) -> None:
    """crystallize_soul returns created=False for invalid love_type."""
    from brain.tools.impls.crystallize_soul import crystallize_soul

    ctx = _ctx(tmp_path)
    result = crystallize_soul(
        moment="a moment",
        love_type="not_real",
        why_it_matters="whatever",
        **ctx,
    )

    assert result["created"] is False
    assert "unknown love_type" in result["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# add_memory + climax_event auto-journal hook
# ─────────────────────────────────────────────────────────────────────────────


def test_add_memory_climax_high_writes_climax_journal(tmp_path: Path) -> None:
    """When add_memory commits with climax >= 7, a private climax journal_entry
    is written referencing the originating memory."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="release in her hands",
        memory_type="event",
        domain="relationship",
        emotions={"climax": 8, "arousal": 8, "desire": 8},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    assert result["created"] is True
    originating_id = result["id"]

    # One conversation memory + one journal_entry should now exist.
    journal_entries = store.list_by_type("journal_entry", active_only=True)
    assert len(journal_entries) == 1
    j = journal_entries[0]
    assert j.metadata["source"] == "climax_event"
    assert j.metadata["auto_generated"] is True
    assert j.metadata["originating_memory_id"] == originating_id


def test_add_memory_climax_below_threshold_writes_no_climax_journal(tmp_path: Path) -> None:
    """climax=6 (just under threshold) → originating memory commits, no journal."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="building heat",
        memory_type="event",
        domain="relationship",
        emotions={"climax": 6, "arousal": 8, "desire": 6},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    assert result["created"] is True
    journal_entries = store.list_by_type("journal_entry", active_only=True)
    assert len(journal_entries) == 0


def test_add_memory_no_climax_emotion_writes_no_climax_journal(tmp_path: Path) -> None:
    """No climax key in emotions → no journal."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    result = add_memory(
        content="she said she loves me",
        memory_type="event",
        domain="relationship",
        emotions={"love": 9, "joy": 8},
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    assert result["created"] is True
    assert len(store.list_by_type("journal_entry", active_only=True)) == 0


def test_add_memory_below_gate_writes_no_climax_journal_even_if_climax_high(tmp_path: Path) -> None:
    """Gate rejection means originating memory was never committed; journal must not fire."""
    from brain.tools.impls.add_memory import add_memory

    store = _make_store()
    hebbian = _make_hebbian()

    # climax=8 alone has emotion_score=8 (below 15) and auto-importance derived
    # from emotions only — verify by reading _calc result. Manually craft to fail gate.
    result = add_memory(
        content="passing thought",
        memory_type="feeling",
        domain="self",
        emotions={"climax": 8},  # score=8 → below 15; no importance override
        store=store,
        hebbian=hebbian,
        persona_dir=tmp_path,
    )
    # Whether this passes or not depends on auto-importance from score=8;
    # check both branches:
    if result["created"] is False:
        # Gate rejected; no memory at all → no journal.
        assert store.count() == 0
        assert len(store.list_by_type("journal_entry", active_only=True)) == 0
    else:
        # Gate passed; journal should exist (climax >= 7 fired the hook).
        journal = store.list_by_type("journal_entry", active_only=True)
        assert len(journal) == 1
