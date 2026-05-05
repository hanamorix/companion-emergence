"""Tests for brain.migrator.og_reflex_log — schema migration for OG nell_reflex_log.json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.migrator.og_reflex_log import migrate_reflex_log


def _write_og_log(og_data_dir: Path, payload: dict) -> None:
    og_data_dir.mkdir(parents=True, exist_ok=True)
    (og_data_dir / "nell_reflex_log.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_migrated(persona_dir: Path) -> dict:
    path = persona_dir / "reflex_log.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def test_migrates_fires_with_version_wrapper(tmp_path: Path) -> None:
    """OG file with 2 fires → output has {version: 1, fires: [...]} with 2 entries."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_log(og_data, {
        "fires": [
            {
                "arc": "creative_pitch",
                "fired_at": "2026-03-31T11:30:52.498666+00:00",
                "trigger_state": {"creative_hunger": 9},
            },
            {
                "arc": "gift_creation",
                "fired_at": "2026-03-31T11:31:02.122702+00:00",
                "trigger_state": {"love": 9, "creative_hunger": 9},
            },
        ],
    })

    migrated = migrate_reflex_log(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 2

    output = _load_migrated(persona_dir)
    assert output["version"] == 1
    assert len(output["fires"]) == 2
    fire = output["fires"][0]
    assert fire["arc"] == "creative_pitch"
    assert fire["fired_at"] == "2026-03-31T11:30:52.498666+00:00"
    assert fire["trigger_state"] == {"creative_hunger": 9}
    assert fire["output_memory_id"] is None  # OG didn't have this concept


def test_drops_og_only_fields(tmp_path: Path) -> None:
    """OG fire with output_preview + output_type + days_since_human + description →
    none of those keys appear in the output."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_log(og_data, {
        "fires": [{
            "arc": "creative_pitch",
            "fired_at": "2026-03-31T11:30:52.498666+00:00",
            "trigger_state": {"creative_hunger": 9},
            "days_since_human": 0.0,
            "output_type": "gifts",
            "output_preview": "Listen to this, babe...",
            "description": "creative hunger overwhelmed",
        }],
    })

    migrate_reflex_log(og_data_dir=og_data, persona_dir=persona_dir)

    output = _load_migrated(persona_dir)
    fire = output["fires"][0]
    # OG-only fields dropped
    assert "days_since_human" not in fire
    assert "output_type" not in fire
    assert "output_preview" not in fire
    assert "description" not in fire
    # New schema kept
    assert fire["arc"] == "creative_pitch"
    assert fire["fired_at"] == "2026-03-31T11:30:52.498666+00:00"
    assert fire["trigger_state"] == {"creative_hunger": 9}
    assert fire["output_memory_id"] is None


def test_skips_malformed_fire_entries(tmp_path: Path) -> None:
    """fires array with one valid + one missing arc + one missing fired_at →
    only valid one in output; count = 1."""
    og_data = tmp_path / "og_data"
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    _write_og_log(og_data, {
        "fires": [
            {
                "arc": "creative_pitch",
                "fired_at": "2026-03-31T11:30:52.498666+00:00",
                "trigger_state": {"creative_hunger": 9},
            },
            {
                # missing 'arc'
                "fired_at": "2026-03-31T11:31:00+00:00",
                "trigger_state": {"love": 9},
            },
            {
                "arc": "gift_creation",
                # missing 'fired_at'
                "trigger_state": {"love": 9},
            },
        ],
    })

    migrated = migrate_reflex_log(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 1

    output = _load_migrated(persona_dir)
    assert len(output["fires"]) == 1
    assert output["fires"][0]["arc"] == "creative_pitch"


def test_returns_zero_when_og_file_missing(tmp_path: Path) -> None:
    """og_data_dir without nell_reflex_log.json → returns 0; no output written."""
    og_data = tmp_path / "og_data"
    og_data.mkdir()
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()

    migrated = migrate_reflex_log(og_data_dir=og_data, persona_dir=persona_dir)
    assert migrated == 0
    assert not (persona_dir / "reflex_log.json").exists()
