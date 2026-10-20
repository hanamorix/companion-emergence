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


# ---- a reader process and the owning writer on bridge.json (the heal-on-read question) ----
# state_file.read has always healed (it used attempt_heal, with no cross-process lock); under json_io the heal takes
# <name>.heal.lock, the same lock the owner's backup-chain write takes, and re-validates under it. These pin that a
# reader can never clobber or interleave with the owner's write, and that a healthy file is only ever read.


def _state(pid: int) -> state_file.BridgeState:
    return state_file.BridgeState(
        persona="p", pid=pid, port=51000 + pid, started_at="2026-10-19T00:00:00Z", stopped_at=None,
        shutdown_clean=False, client_origin="cli",
    )


def _tree(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file() and not p.name.endswith(".lock")}


def test_a_reader_of_a_healthy_bridge_json_leaves_every_file_byte_identical(persona_dir: Path):
    for pid in (1, 2, 3):
        state_file.write(persona_dir, _state(pid))
    before = _tree(persona_dir)
    for _ in range(3):
        assert state_file.read(persona_dir) == _state(3)
    assert _tree(persona_dir) == before  # no heal, no quarantine, no rewrite of the file or its .bak chain


def test_reading_a_corrupt_bridge_json_restores_it_from_the_previous_save(persona_dir: Path):
    """The contract the CLI and the daemon rely on (unchanged by the migration): a damaged file heals on read."""
    state_file.write(persona_dir, _state(1))
    state_file.write(persona_dir, _state(2))
    (persona_dir / "bridge.json").write_text("{nope")
    assert state_file.read(persona_dir) == _state(1)  # .bak1 holds the previous save
    assert state_file.read(persona_dir) == _state(1)  # and the live file is whole again


def test_a_healing_reader_never_undoes_or_interleaves_with_the_owners_writes(persona_dir: Path):
    import threading

    state_file.write(persona_dir, _state(1))
    state_file.write(persona_dir, _state(2))
    (persona_dir / "bridge.json").write_text("{nope")
    last = 40
    errors: list[BaseException] = []
    seen: list[int | None] = []
    barrier = threading.Barrier(4)

    def owner() -> None:
        try:
            barrier.wait()
            for pid in range(3, last + 1):
                state_file.write(persona_dir, _state(pid))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(60):
                s = state_file.read(persona_dir)
                seen.append(None if s is None else s.pid)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=owner)] + [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert state_file.read(persona_dir) == _state(last)  # the owner's last write stands: no heal reinstalled an older bak
    assert all(pid is not None and pid in range(1, last + 1) for pid in seen)  # every read saw a whole state
    left = [p.name for p in persona_dir.iterdir() if ".tmp" in p.name or p.name.endswith(".new")]
    assert not left, left
