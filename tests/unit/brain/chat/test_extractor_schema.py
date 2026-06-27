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


# ---------------------------------------------------------------------------
# B5 — extractor-scored importance field on CrystallisationCandidate
# ---------------------------------------------------------------------------


def test_crystallisation_candidate_importance_default_is_8():
    """importance defaults to 8 (== DEFAULT_SOUL_THRESHOLD) for back-compat."""
    cc = CrystallisationCandidate(theme="a theme", evidence="some evidence here")
    assert cc.importance == 8


def test_crystallisation_candidate_importance_explicit():
    """importance accepts valid 1-10 values."""
    cc = CrystallisationCandidate(theme="a theme", evidence="some evidence", importance=6)
    assert cc.importance == 6


def test_crystallisation_candidate_importance_out_of_range_rejected():
    """importance=0 and importance=11 are rejected by Pydantic (ge=1, le=10)."""
    with pytest.raises(ValidationError):
        CrystallisationCandidate(theme="t", evidence="e", importance=0)
    with pytest.raises(ValidationError):
        CrystallisationCandidate(theme="t", evidence="e", importance=11)


def test_system_prompt_includes_formative_hint():
    """_SYSTEM_PROMPT must include 'how formative' for the importance field."""
    from brain.chat.extractor import _SYSTEM_PROMPT

    assert "how formative" in _SYSTEM_PROMPT, (
        f"_SYSTEM_PROMPT should contain 'how formative' for the importance field; "
        f"current prompt:\n{_SYSTEM_PROMPT}"
    )


# ── Over-length truncation (live bug 2026-06-27): the extractor LLM occasionally
#    returns a field longer than its cap. Previously Field(max_length=...) raised
#    string_too_long, and because the WHOLE ExtractorOutput is model_validate()'d
#    at once, ONE over-length nested field dropped the entire turn's extraction.
#    Fix: mode="before" validators truncate to the cap instead of raising.

def test_reflex_audit_reason_truncates_instead_of_raising():
    ra = ReflexAuditEntry(tool="search_memories", reason="A" * 400)
    assert len(ra.reason) == 300
    ra2 = ReflexAuditEntry(tool="t" * 100, reason="ok")
    assert len(ra2.tool) == 64


def test_crystallisation_candidate_fields_truncate():
    cc = CrystallisationCandidate(theme="T" * 300, evidence="E" * 600)
    assert len(cc.theme) == 200
    assert len(cc.evidence) == 500


def test_extractor_output_model_validate_survives_overlong_nested_field():
    # The real path (extractor.py:228 model_validate) must NOT drop the whole
    # extraction because one nested reason/evidence overflowed.
    data = {
        "memory_writes": [{"episode": "something happened", "salience": 0.5}],
        "crystallisation": [{"theme": "T" * 250, "evidence": "E" * 700}],
        "reflex_audit": [{"tool": "search_memories", "reason": "R" * 500}],
    }
    out = ExtractorOutput.model_validate(data)
    assert len(out.memory_writes) == 1
    assert len(out.crystallisation[0].theme) == 200
    assert len(out.crystallisation[0].evidence) == 500
    assert len(out.reflex_audit[0].reason) == 300
