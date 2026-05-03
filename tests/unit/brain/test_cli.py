"""Tests for brain.cli entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain import cli
from brain.bridge.state_file import BridgeState
from brain.bridge.state_file import write as write_bridge_state
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`nell --version` prints version and exits with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "companion-emergence" in captured.out


def test_no_args_prints_help_and_exits_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`nell` with no args shows help and returns 1."""
    result = cli.main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


STUB_COMMANDS = [
    "supervisor",
    "rest",
    "memory",
    "works",
]


@pytest.mark.parametrize("name", STUB_COMMANDS)
def test_stub_subcommand_runs_and_reports_not_implemented(
    capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """Every stub subcommand exits non-zero and prints 'not implemented yet'."""
    result = cli.main([name])
    assert result == 2
    captured = capsys.readouterr()
    assert "not implemented" in captured.out.lower()
    assert name in captured.out


def test_stub_subcommand_help_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each stub subcommand supports --help without crashing."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["supervisor", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "supervisor" in captured.out.lower()


def _make_persona(tmp_path: Path, name: str = "nell") -> Path:
    home = tmp_path / "home"
    persona_dir = home / "personas" / name
    persona_dir.mkdir(parents=True)
    return persona_dir


def test_status_reports_missing_persona_without_creating_files(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell status` reports a missing persona and returns non-zero."""
    home = tmp_path / "home"
    monkeypatch.setenv("NELLBRAIN_HOME", str(home))

    result = cli.main(["status", "--persona", "nell"])

    assert result == 1
    assert not (home / "personas" / "nell").exists()
    captured = capsys.readouterr()
    assert "companion-emergence" in captured.out
    assert "persona: nell" in captured.out
    assert "persona_exists: no" in captured.out
    assert "bridge: not running" in captured.out


def test_status_reports_persona_config_memory_and_bridge_state(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`nell status` summarizes local persona health without hitting providers."""
    persona_dir = _make_persona(tmp_path)
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "home"))
    PersonaConfig(provider="fake", searcher="noop", mcp_audit_log_level="metadata").save(
        persona_dir / "persona_config.json"
    )
    store = MemoryStore(persona_dir / "memories.db")
    store.close()
    write_bridge_state(
        persona_dir,
        BridgeState(
            persona="nell",
            pid=12345,
            port=8765,
            started_at="2026-05-03T12:00:00Z",
            stopped_at=None,
            shutdown_clean=False,
            client_origin="cli",
            auth_token="secret-token",
        ),
    )
    monkeypatch.setattr(cli.state_file, "pid_is_alive", lambda pid: pid == 12345)

    result = cli.main(["status", "--persona", "nell"])

    assert result == 0
    captured = capsys.readouterr()
    assert "persona: nell" in captured.out
    assert "persona_exists: yes" in captured.out
    assert "provider: fake" in captured.out
    assert "searcher: noop" in captured.out
    assert "mcp_audit_log_level: metadata" in captured.out
    assert "memories_active: 0" in captured.out
    assert "bridge: running" in captured.out
    assert "pid: 12345" in captured.out
    assert "port: 8765" in captured.out
    assert "secret-token" not in captured.out
