import json
import sqlite3
from datetime import datetime
from pathlib import Path

from brain.migrator.companion_emergence import (
    CompanionEmergenceMigrateArgs, migrate_companion_emergence)
from brain.migrator.emergence_kit import EmergenceKitMigrateArgs, migrate_emergence_kit
from brain.migrator.og import FileManifest
from brain.migrator.report import write_source_manifest


def _make_ce_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    conn = sqlite3.connect(src / "memories.db")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.commit()
    conn.close()
    (src / "persona_config.json").write_text(json.dumps({"user_name": "Z"}))
    (src / "felt_time_state.json").write_text(json.dumps({"lived_age_hours": 42.0}))
    return src


def test_ce_manifest_records_source_lived_age(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    src = _make_ce_source(tmp_path)
    migrate_companion_emergence(CompanionEmergenceMigrateArgs(
        input_dir=src, install_as="Phoebe", force=False))
    manifest = json.loads(
        (tmp_path / "home" / "personas" / "Phoebe" / "source-manifest.json").read_text())
    assert manifest["lived_age_hours_at_migration"] == 42.0
    assert "migrated_at_utc" in manifest
    datetime.fromisoformat(manifest["migrated_at_utc"].replace("Z", "+00:00"))


def test_ce_manifest_lived_age_defaults_zero_when_no_felt_time(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    src = tmp_path / "src2"; src.mkdir()
    conn = sqlite3.connect(src / "memories.db")
    conn.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT)")
    conn.commit(); conn.close()
    (src / "persona_config.json").write_text(json.dumps({"user_name": "Z"}))
    migrate_companion_emergence(CompanionEmergenceMigrateArgs(
        input_dir=src, install_as="Nova", force=False))
    manifest = json.loads(
        (tmp_path / "home" / "personas" / "Nova" / "source-manifest.json").read_text())
    assert manifest["lived_age_hours_at_migration"] == 0.0


def test_kit_manifest_records_grace_fields(tmp_path: Path) -> None:
    """emergence-kit manifest must include migrated_at_utc and lived_age_hours_at_migration=0.0."""
    input_dir = tmp_path / "kit"
    input_dir.mkdir()
    (input_dir / "memories_v2.json").write_text(json.dumps([]))
    output_dir = tmp_path / "out"
    args = EmergenceKitMigrateArgs(
        input_dir=input_dir,
        output_dir=output_dir,
        install_as=None,
        force=False,
    )
    migrate_emergence_kit(args)
    manifest = json.loads((output_dir / "source-manifest.json").read_text())
    assert manifest["lived_age_hours_at_migration"] == 0.0
    assert "migrated_at_utc" in manifest
    datetime.fromisoformat(manifest["migrated_at_utc"].replace("Z", "+00:00"))


def test_write_source_manifest_records_grace_fields(tmp_path: Path) -> None:
    """write_source_manifest must include migrated_at_utc and lived_age_hours_at_migration."""
    out = tmp_path / "source-manifest.json"
    write_source_manifest(out, [], lived_age_hours_at_migration=0.0)
    data = json.loads(out.read_text())
    assert "migrated_at_utc" in data
    assert data["lived_age_hours_at_migration"] == 0.0
    datetime.fromisoformat(data["migrated_at_utc"].replace("Z", "+00:00"))
