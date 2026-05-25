import json
from datetime import UTC, datetime

from brain.memory.store import Memory, MemoryStore
from brain.memory.hebbian import HebbianMatrix
from brain.recovery.cli import run_recover_cli, RecoverArgs


def _persona_and_source(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    from brain.paths import get_persona_dir
    p = get_persona_dir("Phoebe"); p.mkdir(parents=True)
    s = MemoryStore(p / "memories.db")
    s.create(Memory(id="S", content="s", memory_type="conversation", domain="us",
                    created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    s.close()
    HebbianMatrix(p / "hebbian.db").close()
    src = tmp_path / "src"; src.mkdir()
    ss = MemoryStore(src / "memories.db")
    ss.create(Memory(id="S", content="s", memory_type="conversation", domain="us",
                     created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    ss.create(Memory(id="V", content="v", memory_type="conversation", domain="us",
                     created_at=datetime(2026, 4, 1, tzinfo=UTC)))
    ss.close()
    return p, src


def test_cli_dry_run_emits_json(tmp_path, monkeypatch, capsys):
    p, src = _persona_and_source(tmp_path, monkeypatch)
    rc = run_recover_cli(RecoverArgs(persona="Phoebe", source_dir=src,
                                     force=True, dry_run=True, json_out=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["kind"] == "RecoveryReport"
    assert payload["dry_run"] is True


def test_cli_unknown_persona_exits(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    import pytest
    with pytest.raises(SystemExit):
        run_recover_cli(RecoverArgs(persona="Nope", source_dir=None,
                                    force=True, dry_run=True, json_out=True))


def test_build_parser_registers_recover_subcommand():
    import argparse
    from brain.recovery.cli import build_parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_parser(subparsers)
    args = parser.parse_args(["recover", "--persona", "X", "--dry-run", "--json"])
    assert args.persona == "X"
    assert args.dry_run is True
    assert args.json_out is True
