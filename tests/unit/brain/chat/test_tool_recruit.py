import pytest

from brain.chat.salience import SalienceSignal, assess_salience
from brain.chat.tool_recruit import REFLEXIVE_CORE, select_tools
from brain.tools import NELL_TOOL_NAMES


def test_trivial_turn_gets_only_core():
    allowed = select_tools(assess_salience("ok"))
    assert set(allowed) == set(REFLEXIVE_CORE)
    assert "read_file" not in allowed
    assert "recall_forgotten" not in allowed


def test_low_salience_question_can_still_search():
    """Regression: the Phoebe dog-name failure (v0.0.33 spec, Track 1).

    'what's my dog's name?' scores ~0.19 salience with no recruitment flags;
    pre-v0.0.33 the model could not search at all on such turns.
    search_memories is reflexive-core now — always in hand.
    """
    signal = assess_salience("what's my dog's name?")
    assert signal.score < 0.999  # stays a non-maximal turn — guards the premise
    allowed = select_tools(signal)
    assert "search_memories" in allowed
    # Heavy memory tools stay salience-gated.
    assert "recall_forgotten" not in allowed
    assert "add_memory" not in allowed


def test_past_reference_recruits_memory():
    allowed = select_tools(assess_salience("remember when we discussed the manuscript?"))
    assert "search_memories" in allowed and "recall_forgotten" in allowed


def test_file_mention_recruits_file_tools():
    allowed = select_tools(assess_salience("read ~/Desktop/notes.txt"))
    assert "read_file" in allowed and "list_directory" in allowed


def test_core_always_includes_reach_and_monologue():
    allowed = select_tools(assess_salience("ok"))
    assert "reach_for_capability" in allowed and "record_monologue" in allowed


def test_maximal_signal_recruits_everything():
    allowed = set(select_tools(SalienceSignal.maximal()))
    assert allowed == set(NELL_TOOL_NAMES)


def test_select_tools_returns_base_order():
    """Result must be a subsequence of NELL_TOOL_NAMES in the same relative order.

    This pins the set-refactor: assembling membership via a set must not
    scramble the output order — base order is what the LLM sees and must
    be stable. Use a mixed signal (past reference + file path) so several
    tiers are recruited and the ordering check is non-trivial.
    """
    signal = assess_salience("remember ~/notes.txt from last week?")
    allowed = select_tools(signal)
    # Verify it is a (strict or non-strict) subsequence of NELL_TOOL_NAMES.
    base_iter = iter(NELL_TOOL_NAMES)
    for name in allowed:
        found = any(b == name for b in base_iter)
        assert found, f"{name!r} not found in remaining base order"
    # Also ensure the list is not empty and contains expected tools.
    assert "search_memories" in allowed
    assert "read_file" in allowed


@pytest.mark.xfail(
    strict=True,
    reason="Deferred D1 (v0.0.33 spec §Deferred / project_companion_emergence_deferred.md): "
    "memory-shaped statements without '?' or past-cues do not recruit heavy memory "
    "tools. If this XPASSes, the heuristics were broadened — update the ledger.",
)
def test_deferred_d1_memoryish_statement_recruits_heavy_memory_tools():
    allowed = select_tools(assess_salience("tell me about my dog"))
    assert "recall_forgotten" in allowed


def test_deferred_d1_reach_valve_always_in_hand():
    """Pin (D1 live-signal): reach_for_capability stays reflexive-core — its
    dispatches in tool_invocations.log.jsonl are the usage data that tells us
    whether always-on search_memories is sufficient. Ledger:
    project_companion_emergence_deferred.md."""
    assert "reach_for_capability" in REFLEXIVE_CORE
    assert "reach_for_capability" in select_tools(assess_salience("ok"))
