"""Tests for P3 retention rework, Change 1 — trustworthy importance assignment.

Covers C1.1-C1.5 from changes/p3-retention/1.5-criteria.md. Each test
documents how the PRE-CHANGE code would have violated the criterion (the
oracle must be shown able to fail).

Naming: synthetic user = Bob, persona = Canary, model = Claude. No Phoebe.
"""

from __future__ import annotations

import re

import pytest

from brain.memory.store import Memory, clamp_importance

# --------------------------------------------------------------------------- C1.1
# Fixed-constant producers: reflex, add_journal, research, initiate_outbound,
# making, kindled_peer must all emit importance > 1.0 (floor above the
# deflated ~0.0-0.3 range that `emotions={}` / a tiny emotion vector produced
# pre-change). Verified by constructing each via its producer path.


def test_c1_1_reflex_engine_fire_produces_nonzero_importance(tmp_path):
    """End-to-end through the real reflex.py `_fire` code path (not
    reimplemented). reflex_output is a gated type, so the written memory
    lands in the pending queue (route_write), not memories.db directly."""
    from datetime import UTC, datetime

    from brain.engines.reflex import ReflexArc, ReflexEngine
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    class _FakeProvider:
        def generate(self, prompt, *, system=None):
            return "a reflexive thought"

        def name(self):
            return "fake"

    store = MemoryStore(tmp_path / "memories.db")
    arc = ReflexArc(
        name="test_arc",
        description="test",
        trigger={"joy": 5.0},
        days_since_human_min=0.0,
        cooldown_hours=0.0,
        action="write",
        output_memory_type="reflex_output",
        prompt_template="hi",
    )
    engine = ReflexEngine(
        store=store,
        provider=_FakeProvider(),
        persona_name="Canary",
        persona_system_prompt="",
        arcs_path=tmp_path / "arcs.json",
        log_path=tmp_path / "reflex_log.jsonl",
        default_arcs_path=tmp_path / "default_arcs.json",
    )
    engine._fire(arc, {"joy": 5.0}, 1.0, [], datetime.now(UTC))
    mems = PendingQueue(tmp_path).read_recent("reflex_output", limit=10)
    assert mems
    # Fail-test: pre-change this producer passed emotions={} with no explicit
    # importance -> Memory.create_new deflated importance to 0.0.
    assert mems[0].importance > 1.0
    store.close()


def test_c1_1_reflex_journal_shaped_arc_matches_add_journal_floor(tmp_path):
    """A journal-shaped reflex arc output (output_memory_type='journal_entry')
    gets the same importance floor as add_journal's direct-write path (6.0),
    not the plain reflex floor (4.0)."""
    from datetime import UTC, datetime

    from brain.engines.reflex import ReflexArc, ReflexEngine
    from brain.memory.store import MemoryStore

    class _FakeProvider:
        def generate(self, prompt, *, system=None):
            return "a journal-shaped reflexive thought"

        def name(self):
            return "fake"

    store = MemoryStore(tmp_path / "memories.db")
    arc = ReflexArc(
        name="journal_arc",
        description="test",
        trigger={"joy": 5.0},
        days_since_human_min=0.0,
        cooldown_hours=0.0,
        action="write",
        output_memory_type="journal_entry",
        prompt_template="hi",
    )
    engine = ReflexEngine(
        store=store,
        provider=_FakeProvider(),
        persona_name="Canary",
        persona_system_prompt="",
        arcs_path=tmp_path / "arcs.json",
        log_path=tmp_path / "reflex_log.jsonl",
        default_arcs_path=tmp_path / "default_arcs.json",
    )
    # journal_entry bypasses the gate (route_write) -> lands directly in
    # memories.db, unlike the plain reflex_output case above.
    engine._fire(arc, {"joy": 5.0}, 1.0, [], datetime.now(UTC))
    mems = store.list_by_type("journal_entry", active_only=False)
    assert mems
    assert mems[0].importance == pytest.approx(6.0)
    store.close()


def test_c1_1_add_journal_above_floor(tmp_path):
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore
    from brain.tools.impls.add_journal import add_journal

    store = MemoryStore(tmp_path / "memories.db")
    hebbian = HebbianMatrix(":memory:")
    add_journal("a private thought", store=store, hebbian=hebbian, persona_dir=tmp_path)
    mems = store.list_by_type("journal_entry", active_only=False)
    assert mems
    # Fail-test: pre-change emotions={} with no explicit importance -> 0.0.
    assert mems[0].importance > 1.0
    store.close()
    hebbian.close()


def test_c1_1_research_memory_above_floor():
    from brain.engines.research import _create_research_memory

    mem = _create_research_memory(
        content="notes on a topic",
        interest=type(
            "I", (), {"id": "i1", "topic": "topic", "scope": "general"}
        )(),
        web_results=[],
        web_used=False,
        trigger="scheduled",
        provider_name="fake",
        searcher_name=None,
    )
    # Fail-test: pre-change emotions={} with no explicit importance -> 0.0.
    assert mem.importance > 1.0


def test_c1_1_initiate_outbound_above_floor(tmp_path):
    from brain.initiate.memory import write_initiate_memory
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    write_initiate_memory(
        store,
        audit_id="a1",
        subject="checking in",
        message="hey, thinking of you",
        state="warm",
        ts="2026-01-01T00:00:00+00:00",
    )
    mems = store.list_by_type("initiate_outbound", active_only=False)
    assert mems
    # Fail-test: pre-change the /10.0 default on a small/absent emotion
    # vector deflated importance to <= 0.025.
    assert mems[0].importance > 1.0
    store.close()


def test_c1_1_making_memory_above_floor(tmp_path):
    from brain.maker.wiring import Making, write_making_memory
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    making = Making(
        type="poem",
        title="a small poem",
        content="roses are red",
        disposition="private",
    )
    write_making_memory(store, making, emotions={"joy": 0.2})
    # "making" is a gated type: route_write enqueues it, it does not land in
    # memories.db directly.
    mems = PendingQueue(tmp_path).read_recent("making", limit=10)
    assert mems
    # Fail-test: pre-change the /10.0 default on a tiny emotion vector
    # deflated importance to <= 0.03.
    assert mems[0].importance > 1.0
    store.close()


def test_c1_1_kindled_peer_memory_above_floor(tmp_path):
    from brain.kindled_link.relationship import write_kindled_peer_memory
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    write_kindled_peer_memory(
        store,
        peer_id="peer1",
        session_id="s1",
        speaker="peer",
        stage="early",
        content="a peer said something",
    )
    mems = store.list_by_type("kindled_peer", active_only=False)
    assert mems
    # Fail-test: pre-change the /10.0 default on an absent/small emotion
    # vector deflated importance to near-0.
    assert mems[0].importance > 1.0
    store.close()


def test_c1_1_grief_event_intensity_proportional():
    """grief_event is intensity-PROPORTIONAL (finding #6), not a flat floor:
    importance == clamp(intensity, 0, 10) across a low and a high intensity.
    Fail-test: pre-change importance == intensity/10.0 (deflated)."""
    from brain.grief.breadcrumb import write_breadcrumb
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")

    low_id = write_breadcrumb(
        store=store,
        intensity=0.5,
        subtype="drop",
        referent_type="memory",
        referent_id="ref1",
        content="the memory of something is gone",
        residue_emotion=None,
    )
    high_id = write_breadcrumb(
        store=store,
        intensity=8.0,
        subtype="drop",
        referent_type="memory",
        referent_id="ref2",
        content="the memory of something else is gone",
        residue_emotion=None,
    )
    low = store.get(low_id, bump=False)
    high = store.get(high_id, bump=False)
    assert low.importance == pytest.approx(0.5)
    assert high.importance == pytest.approx(8.0)
    # Fail-test: the deflated pre-change formula would give 0.05 / 0.8.
    assert low.importance != pytest.approx(0.05)
    assert high.importance != pytest.approx(0.8)
    store.close()


# --------------------------------------------------------------------------- C1.2
def test_c1_2_monologue_trace_importance_varies_with_signal():
    """monologue_trace importance is no longer a flat 0.3 — it varies with
    the aggregate's peak emotion intensity across distinct inputs.
    Fail-test: pre-change _TRACE_IMPORTANCE == 0.3 for every input."""
    from brain.monologue.trace import _trace_importance

    flat = _trace_importance({})
    charged = _trace_importance({"joy": 9.0})
    mild = _trace_importance({"joy": 2.0})
    assert flat != charged
    assert mild != charged
    assert flat == pytest.approx(0.2)  # floor, not 0.3
    assert charged > mild > flat


# --------------------------------------------------------------------------- C1.3
def test_c1_3a_dream_from_saturated_aggregate_clamped():
    """A dream constructed from a saturated (sum > 10) emotion aggregate via
    create_new yields importance <= 10.0. Fail-test: pre-change (no clamp in
    create_new) a saturated aggregate produced importance > 10.0."""
    saturated_emotions = {"love": 60.0, "awe": 50.0, "joy": 40.0}  # sum = 150
    mem = Memory.create_new(
        content="a vivid dream",
        memory_type="dream",
        domain="us",
        emotions=saturated_emotions,
    )
    assert mem.importance <= 10.0
    # Confirms the clamp actually engaged (would be 9.0 unclamped == still <=10,
    # so use a case that clearly overflows without the clamp).
    assert mem.importance == pytest.approx(10.0)


def test_c1_3b_migrator_oversized_importance_clamped():
    """An OG memory with a numeric-but-oversized importance (bypasses
    create_new) is also clamped. Fail-test: pre-change transform.py passed
    50.0 through unclamped."""
    from brain.migrator.transform import transform_memory

    og = {
        "id": "og-1",
        "content": "an old memory",
        "created_at": "2020-01-01T00:00:00+00:00",
        "importance": 50.0,
    }
    mem, skipped = transform_memory(og)
    assert skipped is None
    assert mem is not None
    assert mem.importance <= 10.0
    assert mem.importance == pytest.approx(10.0)


def test_c1_3_clamp_importance_helper_boundaries():
    assert clamp_importance(-5.0) == 0.0
    assert clamp_importance(15.0) == 10.0
    assert clamp_importance(4.2) == pytest.approx(4.2)


# --------------------------------------------------------------------------- C1.4
def test_c1_4_migration_og0_backfill_nonzero():
    """An OG memory with missing importance backfills to a nonzero value
    (not hard 0.0). Fail-test: pre-change transform.py always set 0.0 here."""
    from brain.migrator.transform import transform_memory

    # No "importance" key at all, and no emotions -> flat moderate default.
    og_no_emotions = {
        "id": "og-2",
        "content": "an old memory with no signal",
        "created_at": "2020-01-01T00:00:00+00:00",
    }
    mem, skipped = transform_memory(og_no_emotions)
    assert skipped is None
    assert mem.importance == pytest.approx(4.0)
    assert mem.importance != 0.0

    # Malformed importance (a string) + present emotions -> bucket backfill.
    og_with_emotions = {
        "id": "og-3",
        "content": "an old emotional memory",
        "created_at": "2020-01-01T00:00:00+00:00",
        "importance": "high",  # malformed -> ignored
        "emotions": {"joy": 25.0},  # score 25 -> bucket 7
    }
    mem2, skipped2 = transform_memory(og_with_emotions)
    assert skipped2 is None
    assert mem2.importance == pytest.approx(7.0)
    assert mem2.importance != 0.0


# --------------------------------------------------------------------------- C1.5
_HIGH_END_ANCHOR_PATTERNS = [
    re.compile(r"0\.9-1\.0"),  # chat/extractor.py _SYSTEM_PROMPT anchor
]


def _em_dash_free(text: str) -> bool:
    """Normalized zero-em-dash check (U+2014)."""
    return "—" not in text


def test_c1_5_system_prompt_anchor_present_and_no_em_dash():
    from brain.chat.extractor import _SYSTEM_PROMPT

    assert "0.9-1.0" in _SYSTEM_PROMPT
    assert "pivotal" in _SYSTEM_PROMPT
    assert _em_dash_free(_SYSTEM_PROMPT)


def test_c1_5_extraction_prompts_anchor_present_and_no_em_dash():
    from brain.ingest.extract import EXTRACTION_PROMPT_LEGACY, EXTRACTION_PROMPT_NAMED

    for prompt in (EXTRACTION_PROMPT_LEGACY, EXTRACTION_PROMPT_NAMED):
        assert "rubric" in prompt
        assert "pivotal" in prompt
        assert _em_dash_free(prompt)


def test_c1_5_em_dash_check_flags_pre_change_strings():
    """Self-test (per criteria + plan): the SAME assertion, run against a
    reconstruction of the PRE-CHANGE strings (which contained em-dashes),
    must FAIL — proving the oracle can actually flag a violation."""
    pre_change_system_prompt = (
        "Identify what surfaced that should affect their memory, emotional "
        "state, or growth — and what you noticed they should have done "
        "differently."
    )
    pre_change_extraction_named_fragment = (
        "{assistant_name} is the assistant — the AI persona."
    )
    assert not _em_dash_free(pre_change_system_prompt)
    assert not _em_dash_free(pre_change_extraction_named_fragment)
