"""brain.creative.dna — load/save creative_dna with default fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.creative.dna import (
    load_creative_dna,
    save_creative_dna,
)


def test_load_returns_default_when_missing(tmp_path: Path):
    """No file → loads framework default + persists it to persona dir."""
    dna = load_creative_dna(tmp_path)
    assert dna["version"] == 1
    assert dna["core_voice"]  # non-empty
    assert dna["tendencies"]["active"] == []
    assert dna["tendencies"]["emerging"] == []
    assert dna["tendencies"]["fading"] == []
    # Default was persisted to the persona dir
    assert (tmp_path / "creative_dna.json").exists()


def test_load_returns_existing_file(tmp_path: Path):
    custom = {
        "version": 1,
        "core_voice": "literary, sensory-dense",
        "strengths": ["power dynamics"],
        "tendencies": {
            "active": [
                {
                    "name": "ending on physical action",
                    "added_at": "2026-04-21T00:00:00Z",
                    "reasoning": "imported",
                    "evidence_memory_ids": [],
                },
            ],
            "emerging": [],
            "fading": [],
        },
        "influences": ["clarice lispector"],
        "avoid": [],
    }
    (tmp_path / "creative_dna.json").write_text(json.dumps(custom))
    loaded = load_creative_dna(tmp_path)
    assert loaded["core_voice"] == "literary, sensory-dense"
    assert loaded["tendencies"]["active"][0]["name"] == "ending on physical action"


def test_save_writes_atomic_and_round_trips(tmp_path: Path):
    dna = {
        "version": 1,
        "core_voice": "test voice",
        "strengths": [],
        "tendencies": {"active": [], "emerging": [], "fading": []},
        "influences": [],
        "avoid": [],
    }
    save_creative_dna(tmp_path, dna)
    loaded = load_creative_dna(tmp_path)
    assert loaded == dna


def test_load_corrupt_falls_back_to_default(tmp_path: Path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    (tmp_path / "creative_dna.json").write_text("not valid json")
    dna = load_creative_dna(tmp_path)
    # Default values, not corrupt content
    assert dna["version"] == 1
    assert dna["core_voice"]  # non-empty
    # Anomaly was logged
    assert "creative_dna" in caplog.text.lower() or "anomaly" in caplog.text.lower()


def test_save_rejects_malformed_schema(tmp_path: Path):
    """save_creative_dna validates schema before writing — bad input raises."""
    with pytest.raises(ValueError):
        save_creative_dna(tmp_path, {"not": "a valid creative_dna shape"})


def test_save_rejects_tendencies_missing_buckets(tmp_path: Path):
    """tendencies must have all three buckets (active, emerging, fading)."""
    bad = {
        "version": 1,
        "core_voice": "v",
        "strengths": [],
        "tendencies": {"active": []},  # missing emerging + fading
        "influences": [],
        "avoid": [],
    }
    with pytest.raises(ValueError):
        save_creative_dna(tmp_path, bad)
