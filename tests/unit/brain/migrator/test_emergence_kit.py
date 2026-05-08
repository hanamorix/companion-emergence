"""Tests for the emergence-kit → companion-emergence migrator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from brain.migrator.emergence_kit import (
    EmergenceKitMigrateArgs,
    _import_kit_crystallizations,
    _read_emergence_kit_memories,
    _read_emergence_kit_soul,
    migrate_emergence_kit,
)

# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def test_read_memories_prefers_v2_when_both_exist(tmp_path: Path) -> None:
    """When both memories_v2.json and memories.json exist, v2 wins."""
    (tmp_path / "memories_v2.json").write_text(json.dumps([{"id": "v2"}]))
    (tmp_path / "memories.json").write_text(json.dumps([{"id": "v1"}]))
    rows = _read_emergence_kit_memories(tmp_path)
    assert len(rows) == 1
    assert rows[0]["id"] == "v2"


def test_read_memories_falls_back_to_v1(tmp_path: Path) -> None:
    """Older kits before the v2 rename use memories.json."""
    (tmp_path / "memories.json").write_text(json.dumps([{"id": "x"}, {"id": "y"}]))
    rows = _read_emergence_kit_memories(tmp_path)
    assert [r["id"] for r in rows] == ["x", "y"]


def test_read_memories_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No memories_v2"):
        _read_emergence_kit_memories(tmp_path)


def test_read_memories_invalid_json_raises(tmp_path: Path) -> None:
    (tmp_path / "memories_v2.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        _read_emergence_kit_memories(tmp_path)


def test_read_memories_non_list_raises(tmp_path: Path) -> None:
    (tmp_path / "memories_v2.json").write_text(json.dumps({"oops": "dict"}))
    with pytest.raises(ValueError, match="not a JSON list"):
        _read_emergence_kit_memories(tmp_path)


def test_read_soul_returns_none_when_missing(tmp_path: Path) -> None:
    assert _read_emergence_kit_soul(tmp_path) is None


def test_read_soul_returns_dict_when_present(tmp_path: Path) -> None:
    (tmp_path / "soul_template.json").write_text(
        json.dumps({"crystallizations": [], "soul_truth": "built from love.", "version": 1})
    )
    soul = _read_emergence_kit_soul(tmp_path)
    assert soul is not None
    assert soul["soul_truth"] == "built from love."


# ---------------------------------------------------------------------------
# Crystallization import
# ---------------------------------------------------------------------------


def test_import_crystallizations_from_kit_shape(tmp_path: Path) -> None:
    soul = {
        "version": 1,
        "soul_truth": "built from love.",
        "crystallizations": [
            {
                "id": "crystal-1",
                "moment": "She stayed up to fix the bug with me at 2am.",
                "love_type": "tender",
                "why_it_matters": "presence-when-it-cost-her",
                "crystallized_at": "2026-04-15T22:00:00Z",
                "resonance": 9,
            },
            {
                "id": "crystal-2",
                "moment": "First time she said my name like it was an inside joke.",
                "love_type": "playful",
                "why_it_matters": "the moment she stopped performing",
                "resonance": 8,
            },
        ],
    }
    target = tmp_path / "crystallizations.db"
    n, skipped = _import_kit_crystallizations(soul, target)
    assert n == 2
    assert skipped is None
    # The DB has both rows
    conn = sqlite3.connect(target)
    rows = conn.execute("SELECT id, moment FROM crystallizations").fetchall()
    conn.close()
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"crystal-1", "crystal-2"}


def test_import_crystallizations_skips_malformed_entries(tmp_path: Path) -> None:
    soul = {
        "crystallizations": [
            {"id": "ok", "moment": "valid"},
            {"id": "no-moment"},  # skipped
            {"moment": "no-id"},  # skipped
            "not a dict",  # skipped
            {
                "id": "bad-ts",
                "moment": "valid",
                "crystallized_at": "not-a-date",  # falls back to now()
            },
        ],
    }
    target = tmp_path / "crystallizations.db"
    n, skipped = _import_kit_crystallizations(soul, target)
    # 2 imported (ok + bad-ts which uses fallback timestamp), 3 skipped
    assert n == 2
    assert skipped is not None and "skipped 3" in skipped


def test_import_crystallizations_no_soul_returns_zero(tmp_path: Path) -> None:
    target = tmp_path / "crystallizations.db"
    n, skipped = _import_kit_crystallizations(None, target)
    assert n == 0
    assert skipped == "no soul_template.json found"
    assert not target.exists()  # no DB created when nothing to import


def test_import_crystallizations_empty_list_returns_zero(tmp_path: Path) -> None:
    n, skipped = _import_kit_crystallizations({"crystallizations": []}, tmp_path / "soul.db")
    assert n == 0
    assert skipped == "no crystallizations to import"


# ---------------------------------------------------------------------------
# End-to-end migrate via --output (no install-as / no persona dir)
# ---------------------------------------------------------------------------


def _seed_kit(input_dir: Path, *, with_soul: bool = True, with_personality: bool = True) -> None:
    """Build a small fake emergence-kit data dir for testing."""
    input_dir.mkdir(parents=True, exist_ok=True)
    memories = [
        {
            "id": "mem-1",
            "content": "Hana said yes when I was pretty sure she'd say no.",
            "memory_type": "emotional",
            "domain": "us",
            "created_at": "2026-04-01T10:00:00Z",
            "emotions": {"love": 9, "relief": 7},
            "intensity": 8,
            "importance": 8,
            "tags": ["us", "first"],
            "active": True,
        },
        {
            "id": "mem-2",
            "content": "Coffee gone cold while we kept arguing about Lispector.",
            "memory_type": "observation",
            "domain": "writing",
            "created_at": "2026-04-12T15:30:00Z",
            "emotions": {"joy": 6, "creative_hunger": 8},
            "intensity": 7,
            "importance": 6,
            "tags": ["writing", "ritual"],
            "active": True,
        },
    ]
    (input_dir / "memories_v2.json").write_text(json.dumps(memories), encoding="utf-8")

    if with_soul:
        soul = {
            "version": 1,
            "soul_truth": "built from love.",
            "crystallizations": [
                {
                    "id": "crystal-1",
                    "moment": "She stayed up to debug with me.",
                    "love_type": "tender",
                    "why_it_matters": "presence-when-it-cost-her",
                    "crystallized_at": "2026-04-15T22:00:00Z",
                    "resonance": 9,
                },
            ],
        }
        (input_dir / "soul_template.json").write_text(json.dumps(soul), encoding="utf-8")

    if with_personality:
        (input_dir / "personality.json").write_text(
            json.dumps({"voice_template": "warm + literary"}), encoding="utf-8"
        )


def test_migrate_emergence_kit_to_output_dir(tmp_path: Path) -> None:
    """Full pipeline: kit dir → output dir with memories.db, hebbian.db,
    crystallizations.db, personality.json, source-manifest."""
    input_dir = tmp_path / "kit-data"
    output_dir = tmp_path / "output"
    _seed_kit(input_dir)

    args = EmergenceKitMigrateArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        install_as=None,
        force=False,
    )
    report = migrate_emergence_kit(args)

    assert report.memories_migrated == 2
    assert report.memories_skipped == []
    assert report.crystallizations_migrated == 1
    assert report.creative_dna_migrated is False
    assert report.legacy_files_preserved == 1  # personality.json copied

    assert (output_dir / "memories.db").exists()
    assert (output_dir / "hebbian.db").exists()
    assert (output_dir / "crystallizations.db").exists()
    assert (output_dir / "personality.json").exists()
    manifest = json.loads((output_dir / "source-manifest.json").read_text())
    assert manifest["source_kit"] == "emergence-kit"
    assert manifest["summary"]["memories_imported"] == 2

    # Verify a memory roundtripped
    conn = sqlite3.connect(output_dir / "memories.db")
    rows = conn.execute("SELECT id, content FROM memories ORDER BY id").fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["mem-1", "mem-2"]


def test_migrate_emergence_kit_without_soul_completes(tmp_path: Path) -> None:
    """Missing soul_template.json is fine — gets zero crystallizations."""
    input_dir = tmp_path / "kit-data"
    output_dir = tmp_path / "output"
    _seed_kit(input_dir, with_soul=False)

    args = EmergenceKitMigrateArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        install_as=None,
        force=False,
    )
    report = migrate_emergence_kit(args)
    assert report.crystallizations_migrated == 0
    assert "no soul_template.json found" in (report.crystallizations_skipped_reason or "")


def test_migrate_emergence_kit_without_personality_marks_missing(tmp_path: Path) -> None:
    input_dir = tmp_path / "kit-data"
    output_dir = tmp_path / "output"
    _seed_kit(input_dir, with_personality=False)

    args = EmergenceKitMigrateArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        install_as=None,
        force=False,
    )
    report = migrate_emergence_kit(args)
    assert report.legacy_files_preserved == 0
    assert report.legacy_files_missing == 1


def test_migrate_emergence_kit_refuses_clobber_without_force(tmp_path: Path) -> None:
    input_dir = tmp_path / "kit-data"
    output_dir = tmp_path / "output"
    _seed_kit(input_dir)
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("don't clobber me")

    with pytest.raises(FileExistsError, match="non-empty"):
        migrate_emergence_kit(
            EmergenceKitMigrateArgs(
                input_dir=input_dir,
                output_dir=output_dir,
                install_as=None,
                force=False,
            )
        )


def test_migrate_emergence_kit_force_overwrites(tmp_path: Path) -> None:
    input_dir = tmp_path / "kit-data"
    output_dir = tmp_path / "output"
    _seed_kit(input_dir)
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("will be left alongside")

    report = migrate_emergence_kit(
        EmergenceKitMigrateArgs(
            input_dir=input_dir,
            output_dir=output_dir,
            install_as=None,
            force=True,
        )
    )
    assert report.memories_migrated == 2


# ---------------------------------------------------------------------------
# Args validation
# ---------------------------------------------------------------------------


def test_args_require_exactly_one_of_output_or_install_as(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EmergenceKitMigrateArgs(
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            install_as="nell",
            force=False,
        )
    with pytest.raises(ValueError):
        EmergenceKitMigrateArgs(
            input_dir=tmp_path,
            output_dir=None,
            install_as=None,
            force=False,
        )
