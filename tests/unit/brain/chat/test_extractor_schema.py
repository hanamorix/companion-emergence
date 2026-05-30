"""Schema validation tests for the pass-2 extractor output."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from brain.chat.extractor import (
    CrystallisationCandidate,
    EmotionDelta,  # noqa: F401 — placeholder class; imported for public-API surface check
    ExtractorOutput,
    MemoryWrite,
    ReflexAuditEntry,
)


def test_empty_output_valid():
    out = ExtractorOutput()
    assert out.memory_writes == []
    assert out.emotion_delta == {}
    assert out.crystallisation == []
    assert out.reflex_audit == []


def test_memory_write_requires_episode_and_salience():
    mw = MemoryWrite(episode="searched for Loopy, nothing surfaced", salience=0.4)
    assert mw.episode.startswith("searched")
    assert mw.salience == 0.4


def test_memory_write_salience_clamped_0_to_1():
    with pytest.raises(ValidationError):
        MemoryWrite(episode="x", salience=1.5)
    with pytest.raises(ValidationError):
        MemoryWrite(episode="x", salience=-0.1)


def test_emotion_delta_accepts_dict_of_str_to_float():
    out = ExtractorOutput(emotion_delta={"curious": 0.1, "warm": -0.05})
    assert out.emotion_delta["curious"] == 0.1


def test_emotion_delta_rejects_out_of_range_magnitude():
    """Per-call deltas are bounded so a malformed extractor can't slam the vector."""
    with pytest.raises(ValidationError):
        ExtractorOutput(emotion_delta={"curious": 2.0})
    with pytest.raises(ValidationError):
        ExtractorOutput(emotion_delta={"curious": -2.0})


def test_crystallisation_candidate_requires_theme_and_evidence():
    cc = CrystallisationCandidate(theme="late-night writing rhythm", evidence="recurred in turns 14, 19, 23")
    assert cc.theme.startswith("late-night")


def test_reflex_audit_entry_requires_tool_and_reason():
    ra = ReflexAuditEntry(tool="search_memories", reason="user referenced Loopy as known")
    assert ra.tool == "search_memories"


def test_full_output_round_trip():
    """End-to-end: dump to JSON-able dict, re-validate."""
    out = ExtractorOutput(
        memory_writes=[MemoryWrite(episode="x", salience=0.5)],
        emotion_delta={"warm": 0.1},
        crystallisation=[CrystallisationCandidate(theme="t", evidence="e")],
        reflex_audit=[ReflexAuditEntry(tool="search_memories", reason="r")],
    )
    dumped = out.model_dump()
    ExtractorOutput.model_validate(dumped)


def test_memory_write_rejects_whitespace_only_episode():
    with pytest.raises(ValidationError):
        MemoryWrite(episode="   ", salience=0.5)


def test_crystallisation_rejects_whitespace_only_theme():
    with pytest.raises(ValidationError):
        CrystallisationCandidate(theme="  ", evidence="something")


def test_reflex_audit_rejects_whitespace_only_tool():
    with pytest.raises(ValidationError):
        ReflexAuditEntry(tool="   ", reason="something")


def test_emotion_delta_rejects_empty_channel_name():
    with pytest.raises(ValidationError):
        ExtractorOutput(emotion_delta={"": 0.5})


def test_emotion_delta_rejects_whitespace_channel_name():
    with pytest.raises(ValidationError):
        ExtractorOutput(emotion_delta={"  ": 0.5})
