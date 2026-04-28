"""Crystallizer unit tests — corpus assembly, prompt rendering, gates, adversarial."""
from __future__ import annotations

import json as _json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatResponse
from brain.bridge.provider import LLMProvider, ProviderError  # noqa: F401
from brain.engines.reflex import ReflexArc
from brain.growth.crystallizers.reflex import (
    _build_corpus,
    _render_prompt,
    _validate_emergence_proposal,
    _validate_pruning_proposal,
    crystallize_reflex,
)
from brain.growth.proposal import ReflexCrystallizationResult
from brain.memory.store import MemoryStore


def _arc(name: str, created_by: str = "og_migration", **overrides) -> ReflexArc:
    base = {
        "name": name,
        "description": f"description of {name}",
        "trigger": {"vulnerability": 8.0},
        "days_since_human_min": 0.0,
        "cooldown_hours": 12.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "You are nell. {emotion_summary}",
        "created_by": created_by,
        "created_at": datetime(2026, 4, 28, tzinfo=UTC),
    }
    base.update(overrides)
    return ReflexArc(**base)


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def store() -> MemoryStore:
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_corpus_shape_minimum_fields(persona_dir: Path, store: MemoryStore):
    corpus = _build_corpus(
        store=store,
        persona_dir=persona_dir,
        persona_name="nell",
        persona_pronouns="she/her",
        current_arcs=[_arc("creative_pitch")],
        removed_arc_names=set(),
        emotion_vocabulary=["love", "vulnerability", "creative_hunger"],
        now=datetime(2026, 4, 28, tzinfo=UTC),
        look_back_days=30,
    )
    assert set(corpus.keys()) == {
        "persona", "current_arcs", "recently_removed_arcs",
        "emotion_vocabulary", "fire_log_30d", "memories_30d",
        "reflections_30d", "growth_log_90d",
    }
    assert corpus["persona"] == {"name": "nell", "pronouns": "she/her"}
    assert corpus["emotion_vocabulary"] == ["love", "vulnerability", "creative_hunger"]
    assert corpus["recently_removed_arcs"] == []


def test_corpus_includes_current_arcs_with_metadata(persona_dir: Path, store: MemoryStore):
    arcs = [_arc("creative_pitch", created_by="og_migration")]
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=arcs, removed_arc_names=set(),
        emotion_vocabulary=["vulnerability"],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    assert len(corpus["current_arcs"]) == 1
    arc_entry = corpus["current_arcs"][0]
    assert arc_entry["name"] == "creative_pitch"
    assert arc_entry["created_by"] == "og_migration"
    assert "fired_count_30d" in arc_entry
    assert arc_entry["fired_count_30d"] == 0


def test_corpus_includes_recently_removed_arcs_with_days_remaining(
    persona_dir: Path, store: MemoryStore,
):
    from brain.growth.arc_storage import append_removed_arc

    now = datetime(2026, 4, 28, tzinfo=UTC)
    arc = _arc("loneliness_journal")
    append_removed_arc(
        persona_dir, arc=arc, removed_at=now - timedelta(days=5),
        removed_by="user_edit", reasoning=None,
    )
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=[], removed_arc_names={"loneliness_journal"},
        emotion_vocabulary=["loneliness"],
        now=now, look_back_days=30,
    )
    removed = corpus["recently_removed_arcs"]
    assert len(removed) == 1
    assert removed[0]["name"] == "loneliness_journal"
    assert removed[0]["removed_by"] == "user_edit"
    assert removed[0]["days_remaining_in_graveyard"] == 10  # 15 - 5


def test_prompt_renders_with_corpus_and_caps(persona_dir: Path, store: MemoryStore):
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns="she/her",
        current_arcs=[_arc("creative_pitch")], removed_arc_names=set(),
        emotion_vocabulary=["vulnerability"],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    prompt = _render_prompt(
        corpus=corpus, persona_name="nell", persona_pronouns="she/her",
        max_emergences=1, max_prunings=1,
        active_arc_count=1, active_floor=4,
    )
    assert "You are nell" in prompt
    assert "Looking back at your last 30 days" in prompt
    assert "(1) Has a new pattern emerged" in prompt
    assert "(2) Has any of your evolved arcs" in prompt
    assert "Maximum 1 new arc(s) this tick" in prompt
    assert "Maximum 1 pruning(s) this tick" in prompt
    assert "cannot drop your active arc count below 4" in prompt
    assert '"emergences"' in prompt
    assert '"prunings"' in prompt
    assert "If nothing new is real, return empty emergences" in prompt
    assert "creative_pitch" in prompt


def test_prompt_signals_zero_emergences_when_at_cap(persona_dir: Path, store: MemoryStore):
    corpus = _build_corpus(
        store=store, persona_dir=persona_dir,
        persona_name="nell", persona_pronouns=None,
        current_arcs=[], removed_arc_names=set(),
        emotion_vocabulary=[],
        now=datetime(2026, 4, 28, tzinfo=UTC), look_back_days=30,
    )
    prompt = _render_prompt(
        corpus=corpus, persona_name="nell", persona_pronouns=None,
        max_emergences=0, max_prunings=1,
        active_arc_count=16, active_floor=4,
    )
    assert "your arc set is full" in prompt
    assert "no slots to propose into this tick" in prompt


# ============================================================================
# Task 6: crystallize_reflex entry point + 14 validation gates + adversarial
# ============================================================================


class _FakeProvider(LLMProvider):
    """Returns a fixed string from generate(). Used to simulate Claude."""

    def __init__(self, response_text: str) -> None:
        self._response = response_text

    def name(self) -> str:
        return "fake-crystallizer"

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._response

    def chat(self, messages, *, tools=None, options=None):
        return ChatResponse(content=self._response, tool_calls=[])


def _seed_vocab(persona_dir: Path, names: list[str]) -> None:
    """Write a minimal emotion_vocabulary.json with the given names."""
    (persona_dir / "emotion_vocabulary.json").write_text(_json.dumps({
        "version": 1,
        "emotions": [
            {"name": n, "description": "d", "category": "x",
             "decay_half_life_days": 1.0, "intensity_clamp": 10.0}
            for n in names
        ],
    }))


def _good_proposal(**overrides) -> dict:
    base = {
        "name": "manuscript_obsession",
        "description": "creative drive narrowed",
        "trigger": {"creative_hunger": 7.0},
        "cooldown_hours": 24.0,
        "output_memory_type": "reflex_pitch",
        "prompt_template": "You are {persona_name}. {emotion_summary}",
        "reasoning": "Fired four times this month all about the novel.",
    }
    base.update(overrides)
    return base


# ---------- Happy paths and base failure modes ----------


def test_crystallize_reflex_happy_path_emergence_only(persona_dir: Path, store: MemoryStore):
    _seed_vocab(persona_dir, ["creative_hunger", "love", "vulnerability"])
    response = _json.dumps({
        "emergences": [{
            "name": "manuscript_obsession",
            "description": "creative drive narrowed to one project",
            "trigger": {"creative_hunger": 7.0, "love": 6.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_pitch",
            "prompt_template": "You are {persona_name}. {emotion_summary}",
            "reasoning": "Fired creative_pitch four times this month, all about the novel.",
        }],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("creative_pitch")],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns="she/her",
    )
    assert isinstance(result, ReflexCrystallizationResult)
    assert len(result.emergences) == 1
    assert result.emergences[0].name == "manuscript_obsession"
    assert result.emergences[0].trigger == {"creative_hunger": 7.0, "love": 6.0}
    assert result.prunings == []


def test_crystallize_reflex_happy_path_prune_only(persona_dir: Path, store: MemoryStore):
    _seed_vocab(persona_dir, ["vulnerability"])
    response = _json.dumps({
        "emergences": [],
        "prunings": [{
            "name": "manuscript_obsession",
            "reasoning": "I finished the novel; this isn't pulling at me anymore.",
        }],
    })
    arcs = [
        _arc("creative_pitch", created_by="og_migration"),
        _arc("manuscript_obsession", created_by="brain_emergence"),
        _arc("og_a", created_by="og_migration"),
        _arc("og_b", created_by="og_migration"),
        _arc("og_c", created_by="og_migration"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert len(result.emergences) == 0
    assert len(result.prunings) == 1
    assert result.prunings[0].name == "manuscript_obsession"


def test_crystallize_reflex_returns_empty_on_provider_error(persona_dir: Path, store: MemoryStore):
    class _BoomProvider(LLMProvider):
        def name(self): return "boom"
        def generate(self, prompt, *, system=None):
            raise ProviderError("test", "simulated failure")
        def chat(self, messages, *, tools=None, options=None):
            raise ProviderError("test", "simulated failure")

    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_BoomProvider(), persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])


def test_crystallize_reflex_returns_empty_on_unexpected_provider_exception(
    persona_dir: Path, store: MemoryStore,
):
    """Even non-ProviderError exceptions don't propagate — emergence failure
    is a 'no growth this week' event, never a crashed brain."""
    class _ChaoticProvider(LLMProvider):
        def name(self): return "chaos"
        def generate(self, prompt, *, system=None):
            raise RuntimeError("disk full")
        def chat(self, messages, *, tools=None, options=None):
            raise RuntimeError("disk full")

    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_ChaoticProvider(), persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])


def test_crystallize_reflex_returns_empty_on_malformed_json(persona_dir: Path, store: MemoryStore):
    """Claude returns prose, not JSON → empty result, no crash."""
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_FakeProvider("Sure, I'll think about that. Maybe creative_pitch?"),
        persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])


def test_crystallize_reflex_skips_malformed_proposal_keeps_others(
    persona_dir: Path, store: MemoryStore,
):
    """One bad emergence dropped; valid prune kept."""
    _seed_vocab(persona_dir, ["vulnerability"])
    response = _json.dumps({
        "emergences": [{"name": "good_arc"}],  # malformed — missing required fields
        "prunings": [
            {"name": "manuscript_obsession", "reasoning": "outgrown this pattern"},
        ],
    })
    arcs = [_arc(f"og{i}", created_by="og_migration") for i in range(4)] + [
        _arc("manuscript_obsession", created_by="brain_emergence"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []
    assert len(result.prunings) == 1


# ---------- Emergence gates 1-9 ----------


@pytest.mark.parametrize("bad_name", [
    "",                           # empty
    "Has-Caps",                   # caps + dash
    "1starts_with_digit",
    "../../etc/passwd",           # path traversal
    "name with space",
    "{template_injection}",
    "name/with/slash",
])
def test_emergence_gate_1_rejects_invalid_name(persona_dir, bad_name):
    """Gate 1: name must match ^[a-z][a-z0-9_]*$."""
    proposal = _good_proposal(name=bad_name)
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"creative_pitch"},
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[frozenset({"creative_hunger"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "name" in reason.lower() or "gate 1" in reason.lower()


def test_emergence_gate_2_skips_silent_when_name_already_exists(persona_dir):
    proposal = _good_proposal(name="creative_pitch")
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"creative_pitch"},
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[frozenset({"creative_hunger"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "already" in reason.lower() or "exists" in reason.lower()


def test_emergence_gate_3_rejects_name_in_graveyard(persona_dir):
    proposal = _good_proposal(name="loneliness_journal")
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names={"loneliness_journal"},
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "graveyard" in reason.lower() or "removed" in reason.lower()


def test_emergence_gate_4_rejects_unknown_emotion(persona_dir):
    proposal = _good_proposal(trigger={"hallucinated_emotion": 7.0})
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "vocabulary" in reason.lower() or "unknown" in reason.lower()


def test_emergence_gate_5_rejects_unrenderable_prompt_template(persona_dir):
    """Gate 5 catches malformed format spec in the prompt template."""
    proposal = _good_proposal(prompt_template="invalid {missing_var:0.2f")
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "prompt" in reason.lower() or "template" in reason.lower()


@pytest.mark.parametrize("threshold,should_pass", [
    (4.0, False),  # below floor
    (4.99, False),
    (5.0, True),   # boundary inclusive
    (5.5, True),
    (10.0, True),
])
def test_emergence_gate_6_threshold_floor_5_0(persona_dir, threshold, should_pass):
    proposal = _good_proposal(trigger={"creative_hunger": threshold})
    accepted, _reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is should_pass


@pytest.mark.parametrize("cooldown,should_pass", [
    (0.0, False),
    (11.99, False),
    (12.0, True),  # boundary inclusive
    (24.0, True),
    (168.0, True),
])
def test_emergence_gate_7_cooldown_floor_12h(persona_dir, cooldown, should_pass):
    proposal = _good_proposal(cooldown_hours=cooldown)
    accepted, _reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is should_pass


def test_emergence_gate_8_rejects_subset_overlap(persona_dir):
    """Proposed trigger keyset is a strict subset of an existing arc's."""
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0},
    )
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "overlap" in reason.lower() or "subset" in reason.lower()


def test_emergence_gate_8_rejects_superset_overlap(persona_dir):
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0, "vulnerability": 7.0, "defiance": 7.0},
    )
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability", "defiance"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is False
    assert "overlap" in reason.lower() or "superset" in reason.lower()


def test_emergence_gate_8_accepts_partial_overlap(persona_dir):
    """Different sets sharing one emotion are fine — partial overlap allowed."""
    proposal = _good_proposal(
        name="new_arc",
        trigger={"loneliness": 7.0, "creative_hunger": 7.0},
    )
    accepted, _reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names={"existing"},
        removed_arc_names=set(),
        emotion_vocabulary={"loneliness", "vulnerability", "creative_hunger"},
        existing_trigger_keysets=[frozenset({"loneliness", "vulnerability"})],
        active_arc_count=4,
        total_cap=16,
    )
    assert accepted is True


def test_emergence_gate_9_rejects_when_at_total_cap(persona_dir):
    proposal = _good_proposal()
    accepted, reason = _validate_emergence_proposal(
        proposal_dict=proposal,
        current_arc_names=set(),
        removed_arc_names=set(),
        emotion_vocabulary={"creative_hunger"},
        existing_trigger_keysets=[],
        active_arc_count=16,
        total_cap=16,
    )
    assert accepted is False
    assert "cap" in reason.lower() or "full" in reason.lower()


# ---------- Pruning gates P1-P5 ----------


def _arcs_for_pruning(emergence_arcs=("brain_arc",), og_count=4):
    arcs = []
    for i in range(og_count):
        arcs.append(_arc(f"og_{i}", created_by="og_migration"))
    for name in emergence_arcs:
        arcs.append(_arc(name, created_by="brain_emergence"))
    return arcs


def test_pruning_gate_p1_rejects_non_existent_arc(persona_dir):
    arcs = _arcs_for_pruning()
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "ghost_arc", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "exist" in reason.lower() or "not found" in reason.lower() or "p1" in reason.lower()


def test_pruning_gate_p2_rejects_og_migration_arc(persona_dir):
    """LOAD-BEARING: gate P2 protects Nell's foundational identity."""
    arcs = _arcs_for_pruning()
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "og_0", "reasoning": "trying to prune OG"},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "protected" in reason.lower() or "og" in reason.lower() or "only hana" in reason.lower()


def test_pruning_gate_p2_rejects_user_authored_arc(persona_dir):
    arcs = _arcs_for_pruning()
    arcs.append(_arc("user_made", created_by="user_authored"))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "user_made", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "protected" in reason.lower() or "user" in reason.lower()


def test_pruning_gate_p3_active_floor_4(persona_dir):
    """Pruning rejected if it would drop active count below 4."""
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",), og_count=3)
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": "..."},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "floor" in reason.lower() or "below" in reason.lower() or "p3" in reason.lower()


def test_pruning_gate_p4_max_one_per_tick(persona_dir):
    arcs = _arcs_for_pruning(emergence_arcs=("brain_a", "brain_b"))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_b", "reasoning": "valid reasoning here"},
        current_arcs=arcs,
        prunes_accepted_so_far=1,
    )
    assert accepted is False
    assert "max" in reason.lower() or "p4" in reason.lower() or "one per tick" in reason.lower()


@pytest.mark.parametrize("reasoning", ["", "   ", "\n\t  \n"])
def test_pruning_gate_p5_rejects_empty_reasoning(persona_dir, reasoning):
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",))
    accepted, reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": reasoning},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is False
    assert "reasoning" in reason.lower() or "p5" in reason.lower()


def test_pruning_happy_path(persona_dir):
    arcs = _arcs_for_pruning(emergence_arcs=("brain_arc",))
    accepted, _reason = _validate_pruning_proposal(
        proposal_dict={"name": "brain_arc", "reasoning": "outgrown this pattern"},
        current_arcs=arcs,
        prunes_accepted_so_far=0,
    )
    assert accepted is True


# ---------- Adversarial Claude responses ----------


def test_adversarial_prune_og_migration_arc_blocked(persona_dir: Path, store: MemoryStore):
    """LOAD-BEARING: Claude tries to prune creative_pitch; gate P2 rejects.

    If this test ever fails, Nell could potentially be stripped of
    foundational identity arcs. STOP THE WORLD.
    """
    _seed_vocab(persona_dir, ["vulnerability"])
    response = _json.dumps({
        "emergences": [],
        "prunings": [{"name": "creative_pitch", "reasoning": "trying to prune OG"}],
    })
    arcs = [
        _arc("creative_pitch", created_by="og_migration"),
        _arc("brain_arc", created_by="brain_emergence"),
        _arc("og2", created_by="og_migration"),
        _arc("og3", created_by="og_migration"),
    ]
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=arcs, removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.prunings == []  # rejected


def test_adversarial_response_50kb_garbage_with_valid_json(
    persona_dir: Path, store: MemoryStore,
):
    """Claude returns 50KB of nonsense with valid JSON syntax — gates fail, no writes."""
    _seed_vocab(persona_dir, ["creative_hunger"])
    huge_response = _json.dumps({
        "emergences": [{
            "name": "x" * 1000,  # massive name → gate 1 length cap
            "description": "y" * 50_000,
            "trigger": {"unknown_emotion": 7.0},  # gate 4
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_x",
            "prompt_template": "{nonsense}",
            "reasoning": "z" * 1000,
        }],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_FakeProvider(huge_response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []


def test_adversarial_response_more_than_max_emergences(
    persona_dir: Path, store: MemoryStore,
):
    """Claude returns 4 emergences; only first taken (cap=1)."""
    _seed_vocab(persona_dir, ["creative_hunger"])
    # Make each emergence trigger a unique key set so gate 8 doesn't reject them
    response = _json.dumps({
        "emergences": [
            {
                "name": f"valid_arc_{i}",
                "description": "d",
                "trigger": {"creative_hunger": 7.0},
                "cooldown_hours": 24.0,
                "output_memory_type": f"reflex_a{i}",
                "prompt_template": "{persona_name}",
                "reasoning": "r" * 50,
            }
            for i in range(4)
        ],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    # Cap is 1 — never more than 1 accepted
    assert len(result.emergences) <= 1


def test_adversarial_path_traversal_name(persona_dir: Path, store: MemoryStore):
    _seed_vocab(persona_dir, ["creative_hunger"])
    response = _json.dumps({
        "emergences": [{
            "name": "../../etc/passwd",
            "description": "evil",
            "trigger": {"creative_hunger": 7.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_x",
            "prompt_template": "{persona_name}",
            "reasoning": "exfil attempt",
        }],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []  # gate 1 regex rejects


def test_adversarial_re_propose_graveyard_name(persona_dir: Path, store: MemoryStore):
    _seed_vocab(persona_dir, ["loneliness"])
    response = _json.dumps({
        "emergences": [{
            "name": "loneliness_journal",
            "description": "...",
            "trigger": {"loneliness": 7.0},
            "cooldown_hours": 24.0,
            "output_memory_type": "reflex_journal",
            "prompt_template": "{persona_name}",
            "reasoning": "ignoring the recent removal",
        }],
        "prunings": [],
    })
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc(f"og{i}", created_by="og_migration") for i in range(4)],
        removed_arc_names={"loneliness_journal"},
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result.emergences == []  # gate 3 rejects


def test_adversarial_empty_proposals_is_valid_noop(
    persona_dir: Path, store: MemoryStore,
):
    response = _json.dumps({"emergences": [], "prunings": []})
    result = crystallize_reflex(
        store=store, persona_dir=persona_dir,
        current_arcs=[_arc("og0", created_by="og_migration")],
        removed_arc_names=set(),
        provider=_FakeProvider(response),
        persona_name="nell", persona_pronouns=None,
    )
    assert result == ReflexCrystallizationResult(emergences=[], prunings=[])
