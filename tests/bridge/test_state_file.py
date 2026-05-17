"""Bridge state file — bridge.json schema, atomic writes, recovery predicates."""

from __future__ import annotations

import os
from pathlib import Path

from brain.bridge import state_file


def test_round_trip_preserves_all_fields(persona_dir: Path):
    state = state_file.BridgeState(
        persona="test-persona",
        pid=os.getpid(),
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, state)
    read_back = state_file.read(persona_dir)
    assert read_back == state


def test_write_protects_bridge_state_and_backup_files(persona_dir: Path):
    state = state_file.BridgeState(
        persona="test-persona",
        pid=os.getpid(),
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
        auth_token="secret-token",
    )
    state_file.write(persona_dir, state)
    state_file.write(persona_dir, state)

    if os.name == "posix":
        assert oct(persona_dir.stat().st_mode & 0o777) == "0o700"
        assert oct((persona_dir / "bridge.json").stat().st_mode & 0o777) == "0o600"
        assert oct((persona_dir / "bridge.json.bak1").stat().st_mode & 0o777) == "0o600"
    else:
        assert (persona_dir / "bridge.json").exists()
        assert (persona_dir / "bridge.json.bak1").exists()


def test_read_returns_none_when_missing(persona_dir: Path):
    assert state_file.read(persona_dir) is None


def test_pid_is_alive_true_for_self():
    assert state_file.pid_is_alive(os.getpid()) is True


def test_pid_is_alive_false_for_dead_pid():
    # A pid that's almost certainly not in use.
    assert state_file.pid_is_alive(999_999) is False


def test_dirty_shutdown_predicate(persona_dir: Path):
    """shutdown_clean: false + dead pid => recovery needed."""
    s = state_file.BridgeState(
        persona="test-persona",
        pid=999_999,  # dead
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.recovery_needed(persona_dir) is True


def test_clean_shutdown_predicate(persona_dir: Path):
    """shutdown_clean: true => recovery NOT needed even if pid was set."""
    s = state_file.BridgeState(
        persona="test-persona",
        pid=None,
        port=None,
        started_at="2026-04-28T10:15:00Z",
        stopped_at="2026-04-28T10:30:00Z",
        shutdown_clean=True,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.recovery_needed(persona_dir) is False


def test_running_predicate_with_live_pid(persona_dir: Path):
    s = state_file.BridgeState(
        persona="test-persona",
        pid=os.getpid(),
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.is_running(persona_dir) is True


def test_running_predicate_with_dead_pid(persona_dir: Path):
    s = state_file.BridgeState(
        persona="test-persona",
        pid=999_999,
        port=51234,
        started_at="2026-04-28T10:15:00Z",
        stopped_at=None,
        shutdown_clean=False,
        client_origin="cli",
    )
    state_file.write(persona_dir, s)
    assert state_file.is_running(persona_dir) is False
