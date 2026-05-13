"""Tests for brain.cli entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("brain.initiate")

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


# ---------- nell bridge alias has been removed (audit 2026-05-07) ----------


def test_nell_bridge_alias_removed(capsys: pytest.CaptureFixture[str]) -> None:
    """`nell bridge X` no longer exists — the alias was removed in the
    2026-05-07 audit cycle. `nell supervisor` is the only lifecycle
    surface."""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["bridge", "start", "--persona", "nell"])
    # argparse exits with 2 on unknown subcommand
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err.lower() or "argument command" in captured.err.lower()


# ---------- dream handler — friendly NoSeedAvailable (audit 2026-05-10 P2) ----------


def test_dream_handler_friendly_message_when_no_seed(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh persona with no conversation memories: `nell dream --dry-run`
    used to print a NoSeedAvailable traceback. Now it prints a friendly
    "Dream skipped: ..." message and exits 0."""
    persona_dir = _make_persona(tmp_path, "freshie")
    monkeypatch.setenv("NELLBRAIN_HOME", str(tmp_path / "home"))
    PersonaConfig(provider="fake", searcher="noop", mcp_audit_log_level="metadata").save(
        persona_dir / "persona_config.json"
    )

    result = cli.main(["dream", "--persona", "freshie", "--provider", "fake", "--dry-run"])

    assert result == 0, "fresh-persona dream must exit cleanly"
    captured = capsys.readouterr()
    assert "Dream skipped" in captured.out
    # The traceback would include `NoSeedAvailable` or `Traceback`.
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert "NoSeedAvailable" not in captured.out
