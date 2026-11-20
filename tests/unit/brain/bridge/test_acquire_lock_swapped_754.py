"""#754: a start that waited out a migrate's folder swap locks the persona's lock file, not the backup's.

A forced migrate install holds the start lock across two renames, so the locked file moves into the backup folder. A
starter that opened `<persona>/bridge.json.lock` before the swap gets that file's lock once the migrate lets go: it
would hold the backup's lock while running the new persona, and a later start could lock the new file, giving two
bridges. `acquire_lock` checks the locked file is still the one at the path, and locks again if not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows can't rename a folder with a file open")


def test_a_lock_taken_on_a_file_swapped_out_meanwhile_is_taken_again_on_the_file_at_the_path(tmp_path, monkeypatch):
    import fcntl

    from brain.bridge import daemon

    persona = tmp_path / "personas" / "nell"
    persona.mkdir(parents=True)
    backup = persona.with_name("nell.backup-x")
    swaps: list[str] = []

    def flock(fd, op):
        if not swaps:  # the migrate's swap happens while this starter waits for the lock
            os.rename(persona, backup)
            persona.mkdir()
            swaps.append("swapped")
        return fcntl.flock(fd, op)

    fake = SimpleNamespace(flock=flock, LOCK_EX=fcntl.LOCK_EX, LOCK_NB=fcntl.LOCK_NB, LOCK_UN=fcntl.LOCK_UN)
    monkeypatch.setattr(daemon, "fcntl", fake)
    fd = daemon.acquire_lock(persona)
    monkeypatch.setattr(daemon, "fcntl", fcntl)
    assert fd is not None and swaps
    try:
        here = os.stat(persona / daemon.LOCKFILE)
        locked = os.fstat(fd)
        assert (locked.st_dev, locked.st_ino) == (here.st_dev, here.st_ino), "locked the backup's file"
        assert daemon.acquire_lock(persona) is None  # the persona's own lock is held
        other = daemon.acquire_lock(backup)
        assert other is not None  # the backup's file was let go
        daemon.release_lock(backup, other)
    finally:
        daemon.release_lock(persona, fd)


def test_a_filesystem_whose_inodes_never_match_still_gets_the_lock(tmp_path, monkeypatch):
    """Review of #754: on a filesystem with unstable inode numbers the check never matches. The retry is bounded and
    the lock is kept, as before the check existed, instead of recursing until RecursionError."""
    from brain.bridge import daemon

    persona = tmp_path / "nell"
    persona.mkdir()
    calls: list[int] = []

    def never(fd, path):
        calls.append(fd)
        return False

    monkeypatch.setattr(daemon, "_is_file_at", never)
    fd = daemon.acquire_lock(persona)
    monkeypatch.undo()
    assert fd is not None
    try:
        assert 1 < len(calls) <= 5
        assert daemon.acquire_lock(persona) is None  # held
    finally:
        daemon.release_lock(persona, fd)


@pytest.mark.no_base_failure(
    reason="guards the new inode check: main has no check, so it keeps the lock there too; this pins that the check "
    "doesn't leak a locked fd or raise when it can't inspect the file"
)
def test_a_lock_file_that_cannot_be_inspected_keeps_the_lock(tmp_path, monkeypatch):
    """Review of #754: an unexpected error from inspecting the path must not escape with the fd still open and locked;
    the check can't tell, so the lock is kept, as before the check existed."""
    from brain.bridge import daemon

    persona = tmp_path / "nell"
    persona.mkdir()
    lock = persona / daemon.LOCKFILE
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if Path(path) == lock:
            raise PermissionError(13, "Permission denied", str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(daemon.os, "stat", stat)
    fd = daemon.acquire_lock(persona)
    monkeypatch.undo()
    assert fd is not None
    try:
        assert daemon.acquire_lock(persona) is None  # held
    finally:
        daemon.release_lock(persona, fd)
