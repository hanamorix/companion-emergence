"""Tests for research_session parser."""
from brain.engines.research_session import parse_session_output


def _reply(notes="facts here", memory="I dug in today.", verdict="continue"):
    return f"NOTES:\n{notes}\n\nMEMORY:\n{memory}\n\nVERDICT:\n{verdict}\n"


def test_parses_three_sections():
    out = parse_session_output(_reply())
    assert out.notes == "facts here" and out.memory == "I dug in today."
    assert out.verdict == "continue" and not out.degraded and out.spawn_topics == ()


def test_close_verdict():
    assert parse_session_output(_reply(verdict="close")).verdict == "close"


def test_spawn_verdict_parses_and_caps_two():
    out = parse_session_output(_reply(verdict="spawn: topic a; topic b; topic c"))
    assert out.verdict == "continue"
    assert out.spawn_topics == ("topic a", "topic b")


def test_missing_markers_degrades_to_notes_only():
    out = parse_session_output("just prose, no markers at all")
    assert out.degraded and out.memory is None and out.verdict == "continue"
    assert out.notes == "just prose, no markers at all"


def test_case_insensitive_markers():
    out = parse_session_output("notes:\nn\n\nMemory:\nm\n\nverdict:\nclose")
    assert out.notes == "n" and out.memory == "m" and out.verdict == "close"


def test_unknown_verdict_treated_as_continue():
    assert parse_session_output(_reply(verdict="maybe??")).verdict == "continue"


def test_empty_memory_section_skips_memory():
    assert parse_session_output(_reply(memory="")).memory is None
