import json
import sqlite3
from pathlib import Path

from brain import cli


def _make_v012_persona(root: Path, name: str = "phoebe") -> Path:
    persona_dir = root / name
    persona_dir.mkdir(parents=True)
    conn = sqlite3.connect(persona_dir / "memories.db")
    conn.execute("""CREATE TABLE memories (id TEXT PRIMARY KEY, content TEXT, importance INT,
                                            memory_type TEXT, domain TEXT, created_at TEXT,
                                            emotions TEXT, tags TEXT, active INT)""")
    conn.execute("INSERT INTO memories VALUES ('m1','x',5,'fact','t','2026-01-01T00:00:00Z','{}','[]',1)")
    conn.commit()
    conn.close()
    sqlite3.connect(persona_dir / "hebbian.db").close()
    sqlite3.connect(persona_dir / "crystallizations.db").close()
    (persona_dir / "persona_config.json").write_text(json.dumps({
        "persona_name": name, "user_name": "zero", "voice_template": "skip",
        "provider": "claude-cli", "model": "sonnet",
    }))
    return persona_dir


def test_migrate_companion_emergence_roundtrip(tmp_path, monkeypatch, capsys):
    src = _make_v012_persona(tmp_path / "src")
    home = tmp_path / "home"
    monkeypatch.setenv("KINDLED_HOME", str(home))
    rc = cli.main([
        "migrate", "--source", "companion-emergence",
        "--input", str(src), "--install-as", "phoebe", "--json",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["kind"] == "MigrationReport"
    assert payload["source_kind"] == "companion-emergence"
    assert payload["memories_migrated"] == 1
    assert (home / "personas" / "phoebe" / "memories.db").is_file()
    assert json.loads((home / "app_config.json").read_text())["selected_persona"] == "phoebe"


def test_migrate_preflight_only(tmp_path, monkeypatch, capsys):
    src = _make_v012_persona(tmp_path / "src")
    home = tmp_path / "home"
    monkeypatch.setenv("KINDLED_HOME", str(home))
    rc = cli.main([
        "migrate", "--source", "companion-emergence",
        "--input", str(src), "--preflight",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["persona_name"] == "phoebe"
    assert not (home / "personas" / "phoebe").exists()
