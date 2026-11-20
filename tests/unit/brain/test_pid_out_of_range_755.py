"""#755: a pid too large for the OS (read from a damaged or hand-edited file) names no live process.

`os.kill` raises `OverflowError` on POSIX and ctypes raises `ArgumentError` on Windows for such a pid; neither is an
`OSError`, so the liveness checks crashed their callers instead of reading the record as stale.
"""

from __future__ import annotations

import sys

# The first number past each platform's pid range (a POSIX pid_t is a signed C int; a Windows pid is an unsigned
# 32-bit value), and one far past any.
_FIRST_PAST = (1 << 32) if sys.platform == "win32" else (1 << 31)
_OUT_OF_RANGE = (_FIRST_PAST, 10**20)


def test_pid_is_alive_says_an_out_of_range_pid_is_not_alive():
    from brain.bridge.state_file import pid_is_alive

    for pid in _OUT_OF_RANGE:
        assert pid_is_alive(pid) is False, pid


def test_token_of_reads_an_out_of_range_pid_as_gone_without_asking_the_os(monkeypatch):
    """Windows read the ctypes error as UNKNOWN (counted alive: never reclaimed); macOS cut the number to a C int and read
    whatever process that names. Neither platform's reader may be reached."""
    from brain import process_identity as pi

    def reached(pid):
        raise AssertionError(f"the OS reader was asked about {pid}")

    for reader in ("_linux_token", "_windows_token", "_macos_token"):
        monkeypatch.setattr(pi, reader, reached)
    for pid in _OUT_OF_RANGE:
        assert pi.token_of(pid) == pi.GONE, pid


def test_a_migrate_lock_holding_an_out_of_range_pid_is_taken_over(tmp_path):
    """The issue's first case: the check sat outside the reader's try, so `nell migrate` crashed and the stale lock was
    never cleared. Both lock formats: the older bare pid, and pid + time + holder token."""
    import os

    from brain.migrator.cli import _acquire_migrate_lock, _migrate_lock_path, _release_migrate_lock

    for i, body in enumerate(("99999999999999999999\n", f"{1 << 40}\n2026-01-01T00:00:00+00:00\nwin:{1 << 40}:1\n")):
        lock = _migrate_lock_path(tmp_path / f"nell{i}")
        lock.write_text(body, encoding="utf-8")
        fd = _acquire_migrate_lock(lock)
        try:
            assert lock.read_text(encoding="utf-8").splitlines()[0] == str(os.getpid())
        finally:
            _release_migrate_lock(lock, fd)


def test_a_bridge_record_holding_an_out_of_range_pid_reads_as_not_running(tmp_path):
    """bridge.json's pid feeds start, stop, kill and recovery decisions: a damaged one is a stale record, not a crash.
    Both a record with an identity token and an older record without one."""
    from brain.bridge import state_file
    from brain.recovery.cli import _bridge_is_live

    for i, token in enumerate((f"linux:boot:{1 << 40}:7", None)):
        persona = tmp_path / f"p{i}"
        persona.mkdir()
        state_file.write(persona, state_file.BridgeState(
            persona=persona.name, pid=10**20, port=51234, started_at="2026-04-28T10:15:00Z", stopped_at=None,
            shutdown_clean=False, client_origin="cli", pid_token=token, identity="token" if token else None,
        ))
        assert state_file.liveness(persona) == state_file.GONE
        assert state_file.is_running(persona) is False
        assert _bridge_is_live(persona) is False


def test_a_held_bridge_lock_naming_an_out_of_range_pid_names_no_holder(tmp_path):
    """Review of #755: the kill helper passed the lock file's pid to `psutil.pid_exists`, which raises OverflowError on
    it, so `nell stop --kill` crashed on a hand-edited lock instead of killing."""
    import os

    from brain.bridge import daemon, kill

    persona = tmp_path / "nell"
    persona.mkdir()
    fd = daemon.acquire_lock(persona)
    assert fd is not None
    try:
        edit = os.open(str(persona / daemon.LOCKFILE), os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(edit, b"99999999999999999999")
        finally:
            os.close(edit)
        assert daemon.lock_holder(persona) is None
        assert kill._other_live_bridge(persona, target_pid=os.getpid()) is False
    finally:
        daemon.release_lock(persona, fd)
