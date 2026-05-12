"""Tests for brain.initiate.voice_reflection — daily voice-edit reflection tick."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from brain.initiate.emit import read_candidates
from brain.initiate.voice_reflection import run_voice_reflection_tick


def _evidence_provider(emit: bool = True) -> MagicMock:
    """Provider that returns a structured proposal or a 'no edit needed' response."""
    if emit:
        canned = json.dumps({
            "should_propose": True,
            "diff": "- old line\n+ new line",
            "old_text": "old line",
            "new_text": "new line",
            "rationale": "the old wording felt too clipped",
            "evidence": ["dream_a", "cryst_b", "tone_c"],
        })
    else:
        canned = json.dumps({"should_propose": False, "reason": "no coherent pattern"})
    provider = MagicMock(complete=MagicMock(return_value=canned))
    return provider


def test_voice_reflection_emits_candidate_when_evidence_strong(
    tmp_path: Path,
) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("old voice template\nold line\n")
    run_voice_reflection_tick(
        persona_dir,
        provider=_evidence_provider(emit=True),
        crystallizations=[{"id": "c1", "ts": "2026-05-08T00:00:00+00:00"}],
        dreams=[{"id": "d1", "ts": "2026-05-09T00:00:00+00:00"}],
        recent_tones=[{"id": "t1", "ts": "2026-05-10T00:00:00+00:00"}],
    )
    candidates = read_candidates(persona_dir)
    assert len(candidates) == 1
    assert candidates[0].kind == "voice_edit_proposal"
    assert candidates[0].proposal is not None
    assert candidates[0].proposal["old_text"] == "old line"


def test_voice_reflection_skips_when_evidence_thin(tmp_path: Path) -> None:
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    run_voice_reflection_tick(
        persona_dir,
        provider=_evidence_provider(emit=False),
        crystallizations=[],
        dreams=[],
        recent_tones=[],
    )
    assert read_candidates(persona_dir) == []


def test_voice_reflection_requires_at_least_3_evidence_pieces(
    tmp_path: Path,
) -> None:
    """If the LLM tries to propose with <3 evidence, reject."""
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    (persona_dir / "nell-voice.md").write_text("voice\n")
    canned = json.dumps({
        "should_propose": True,
        "diff": "- a\n+ b",
        "old_text": "a",
        "new_text": "b",
        "rationale": "x",
        "evidence": ["only_one"],
    })
    provider = MagicMock(complete=MagicMock(return_value=canned))
    run_voice_reflection_tick(
        persona_dir,
        provider=provider,
        crystallizations=[],
        dreams=[],
        recent_tones=[],
    )
    assert read_candidates(persona_dir) == []
