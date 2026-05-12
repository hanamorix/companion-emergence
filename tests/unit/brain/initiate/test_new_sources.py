"""Tests for the v0.0.10 new emitter gates and emitters."""
from __future__ import annotations

import json
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


def test_load_gate_thresholds_non_dict_json_returns_defaults(tmp_path: Path):
    """A gate_thresholds.json file that contains valid JSON but isn't a dict
    falls back to defaults rather than crashing."""
    persona = tmp_path / "p"
    persona.mkdir()
    (persona / "gate_thresholds.json").write_text("[]")
    t = load_gate_thresholds(persona)
    assert t.reflex_confidence_min == 0.70  # default


def test_write_gate_rejection_appends_jsonl(tmp_path: Path):
    from datetime import UTC, datetime

    from brain.initiate.new_sources import write_gate_rejection

    persona = tmp_path / "p"
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
        source="reflex_firing",
        source_id="r1",
        gate_name="confidence_min",
        threshold_value=0.70,
        observed_value=0.5,
    )
    write_gate_rejection(
        persona,
        ts=datetime(2026, 5, 12, 10, 1, 0, tzinfo=UTC),
        source="research_completion",
        source_id="t9",
        gate_name="topic_overlap_min",
        threshold_value=0.30,
        observed_value=0.10,
    )
    path = persona / "gate_rejections.jsonl"
    rows = [json.loads(line) for line in path.read_text().strip().split("\n")]
    assert len(rows) == 2
    assert rows[0]["source"] == "reflex_firing"
    assert rows[0]["gate_name"] == "confidence_min"
    assert rows[0]["observed_value"] == 0.5
    assert rows[1]["source"] == "research_completion"
