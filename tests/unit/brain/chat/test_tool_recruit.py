from brain.chat.salience import SalienceSignal, assess_salience
from brain.chat.tool_recruit import REFLEXIVE_CORE, select_tools


def test_trivial_turn_gets_only_core():
    allowed = select_tools(assess_salience("ok"))
    assert set(allowed) == set(REFLEXIVE_CORE)
    assert "search_memories" not in allowed and "read_file" not in allowed


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
    from brain.tools import NELL_TOOL_NAMES
    assert allowed == set(NELL_TOOL_NAMES)
