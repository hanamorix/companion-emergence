"""Tests for the v0.0.10 new emitter gates and emitters."""
from __future__ import annotations

from pathlib import Path

from brain.initiate.new_sources import load_gate_thresholds


def test_load_gate_thresholds_returns_defaults_when_no_file(tmp_path: Path):
    persona = tmp_path / "fresh"
    persona.mkdir()
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.70
    assert t.reflex_flinch_intensity_min == 0.60
    assert t.research_maturity_min == 0.75
    assert t.research_topic_overlap_min == 0.30
    assert t.research_freshness_minutes == 30


def test_load_gate_thresholds_overrides_from_persona_file(tmp_path: Path):
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text(
        '{"reflex_confidence_min": 0.5}'
    )
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.5
    # Unset fields use defaults.
    assert t.reflex_flinch_intensity_min == 0.60
