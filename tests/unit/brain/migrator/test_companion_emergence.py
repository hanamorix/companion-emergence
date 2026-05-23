import json
import sqlite3
from pathlib import Path

import pytest

from brain.migrator.companion_emergence import (
    CompanionEmergenceMigrateArgs,
    migrate_companion_emergence,
    preflight_companion_emergence,
)


def _make_v012_persona(root: Path, name: str = "phoebe") -> Path:
    """Build a minimal but valid v0.0.12-shaped persona dir."""
    persona_dir = root / name
    persona_dir.mkdir(parents=True)
    conn = sqlite3.connect(persona_dir / "memories.db")
    conn.execute("""CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, importance INT,
                                            memory_type TEXT, domain TEXT, created_at TEXT,
                                            emotions TEXT, tags TEXT, active INT)""")
    conn.execute("INSERT INTO memories VALUES ('m1','hello',7,'fact','test','2026-01-01T00:00:00Z','{}','[]',1)")
    conn.commit()
    conn.close()
    sqlite3.connect(persona_dir / "hebbian.db").close()
    sqlite3.connect(persona_dir / "crystallizations.db").close()
    (persona_dir / "persona_config.json").write_text(json.dumps({
        "persona_name": name, "user_name": "zero",
        "voice_template": "nell-example", "provider": "claude-cli", "model": "sonnet",
    }))
    return persona_dir


def test_preflight_happy_path(tmp_path):
    src = _make_v012_persona(tmp_path)
    result = preflight_companion_emergence(src)
    assert result["ok"] is True
    assert result["persona_name"] == "phoebe"
    assert result["imported_user_name"] == "zero"
    assert result["imported_voice_template"] == "nell-example"
    assert result["memory_count"] == 1
    assert result["crystallization_count"] == 0
    assert result["hebbian_edge_count"] == 0
    assert result["source_size_bytes"] > 0
    assert result["errors"] == []
    assert result["warnings"] == []


def test_preflight_missing_dir(tmp_path):
    r = preflight_companion_emergence(tmp_path / "nope")
    assert r["ok"] is False
    assert any(e["code"] == "input_missing" for e in r["errors"])


def test_preflight_not_a_dir(tmp_path):
    f = tmp_path / "thing.txt"
    f.write_text("x")
    r = preflight_companion_emergence(f)
    assert any(e["code"] == "input_not_dir" for e in r["errors"])


def test_preflight_no_memories_db(tmp_path):
    bad = tmp_path / "empty"
    bad.mkdir()
    r = preflight_companion_emergence(bad)
    assert any(e["code"] == "no_memories_db" for e in r["errors"])


def test_preflight_pointed_at_parent(tmp_path):
    _make_v012_persona(tmp_path, "phoebe")
    _make_v012_persona(tmp_path, "nell")
    r = preflight_companion_emergence(tmp_path)
    errs = [e for e in r["errors"] if e["code"] == "pointed_at_parent"]
    assert errs
    assert set(errs[0]["detail"]["suggested_subdirs"]) == {"phoebe", "nell"}


def test_preflight_missing_persona_config(tmp_path):
    src = _make_v012_persona(tmp_path)
    (src / "persona_config.json").unlink()
    r = preflight_companion_emergence(src)
    assert any(e["code"] == "no_persona_config" for e in r["errors"])


def test_preflight_bad_persona_config_json(tmp_path):
    src = _make_v012_persona(tmp_path)
    (src / "persona_config.json").write_text("not json {")
    r = preflight_companion_emergence(src)
    assert any(e["code"] == "bad_persona_config" for e in r["errors"])


def test_migrate_copies_dir_and_writes_app_config(tmp_path, monkeypatch):
    src = _make_v012_persona(tmp_path / "src", name="phoebe")
    kindled_home = tmp_path / "home"
    monkeypatch.setenv("KINDLED_HOME", str(kindled_home))

    args = CompanionEmergenceMigrateArgs(
        input_dir=src, install_as="phoebe", force=False,
    )
    report = migrate_companion_emergence(args)

    target = kindled_home / "personas" / "phoebe"
    assert target.is_dir()
    assert (target / "memories.db").is_file()
    assert (target / "persona_config.json").is_file()
    assert (target / "source-manifest.json").is_file()

    app_cfg = json.loads((kindled_home / "app_config.json").read_text())
    assert app_cfg["selected_persona"] == "phoebe"

    assert report.memories_migrated == 1
    assert report.bytes_copied > 0
    assert report.source_kind == "companion-emergence"


def test_migrate_refuses_without_force_on_existing_target(tmp_path, monkeypatch):
    src = _make_v012_persona(tmp_path / "src", name="phoebe")
    kindled_home = tmp_path / "home"
    monkeypatch.setenv("KINDLED_HOME", str(kindled_home))
    (kindled_home / "personas" / "phoebe").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        migrate_companion_emergence(CompanionEmergenceMigrateArgs(
            input_dir=src, install_as="phoebe", force=False,
        ))


def test_migrate_force_backs_up_then_replaces(tmp_path, monkeypatch):
    src = _make_v012_persona(tmp_path / "src", name="phoebe")
    kindled_home = tmp_path / "home"
    monkeypatch.setenv("KINDLED_HOME", str(kindled_home))
    target = kindled_home / "personas" / "phoebe"
    target.mkdir(parents=True)
    (target / "marker.txt").write_text("old")

    migrate_companion_emergence(CompanionEmergenceMigrateArgs(
        input_dir=src, install_as="phoebe", force=True,
    ))

    assert (target / "memories.db").is_file()
    assert not (target / "marker.txt").exists()
    backups = list(target.parent.glob("phoebe.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "marker.txt").read_text() == "old"
