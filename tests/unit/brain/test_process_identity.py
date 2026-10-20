"""Process identity token (C20, C46): fixed at creation, GONE for dead / zombie processes, UNKNOWN when unreadable,
and unchanged by a system clock step."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from brain import process_identity as pi

linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux /proc reader")


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


def test_a_live_child_has_a_stable_token():
    p = _sleeper()
    try:
        a = pi.token_of(p.pid)
        b = pi.token_of(p.pid)
        assert pi.is_token(a)
        assert a == b
        assert pi.matches(p.pid, a) == "alive"
    finally:
        p.kill()
        p.wait()


def test_another_process_has_another_token_and_a_dead_pid_is_gone():
    p = _sleeper()
    q = _sleeper()
    try:
        tp, tq = pi.token_of(p.pid), pi.token_of(q.pid)
        assert tp != tq
        assert pi.matches(q.pid, tp) == pi.GONE  # q is alive, but it is not the recorded process
    finally:
        for proc in (p, q):
            proc.kill()
            proc.wait()
    assert pi.token_of(p.pid) == pi.GONE
    assert pi.matches(p.pid, tp) == pi.GONE


@linux_only
def test_an_exited_but_unreaped_child_is_gone():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 10
    while time.time() < deadline:
        with open(f"/proc/{p.pid}/stat") as fh:
            if fh.read().rsplit(")", 1)[1].split()[0] == "Z":
                break
        time.sleep(0.02)
    try:
        assert (
            pi.token_of(p.pid) == pi.GONE
        )  # a zombie still has /proc/<pid>/stat and its start time
    finally:
        p.wait()


@linux_only
def test_an_unreadable_process_is_unknown(monkeypatch):
    from pathlib import Path

    real = Path.read_bytes

    def deny(self):
        if str(self).startswith("/proc/") and str(self).endswith("/stat"):
            raise PermissionError("denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", deny)
    assert pi.token_of(os.getpid()) == pi.UNKNOWN
    assert pi.matches(os.getpid(), "linux:x:1") == pi.UNKNOWN


@linux_only
def test_an_unreadable_boot_id_is_unknown(monkeypatch):
    monkeypatch.setattr(pi, "_linux_boot_id", lambda: None)
    assert pi.token_of(os.getpid()) == pi.UNKNOWN


def test_nonpositive_and_none_pids_are_gone():
    assert pi.token_of(None) == pi.GONE
    assert pi.token_of(0) == pi.GONE
    assert pi.token_of(-5) == pi.GONE


@linux_only
def test_the_token_does_not_change_under_a_clock_step(monkeypatch):
    """C46: the token uses only creation-time fields; stepping the wall clock and the boot-time source between the
    write and the read leaves it equal. The same patch moves psutil's public create_time() (the fail-demo)."""
    import psutil
    import psutil._pslinux as pl

    p = _sleeper()
    try:
        before = pi.token_of(p.pid)
        ct_before = psutil.Process(p.pid).create_time()
        real_bt = pl.boot_time
        monkeypatch.setattr(time, "time", lambda: 4_000_000_000.0)
        monkeypatch.setattr(pl, "boot_time", lambda: real_bt() + 3600.0)
        after = pi.token_of(p.pid)
        assert after == before
        ct_after = psutil.Process(p.pid).create_time()
        assert ct_after != ct_before, (
            "fail-demo: psutil's create_time() follows the boot-time source"
        )
    finally:
        p.kill()
        p.wait()


def test_own_token_is_cached_and_matches_this_process():
    tok = pi.own_token()
    if sys.platform.startswith("linux"):
        assert pi.is_token(tok)
        assert pi.matches(os.getpid(), tok) == "alive"
    assert pi.own_token() == tok


def test_windows_reader_reads_the_error_saved_by_ctypes_not_a_clobbered_one(monkeypatch):
    """Review S13: OpenProcess failing with ERROR_INVALID_PARAMETER (no such process) reads as GONE even when a later
    call has overwritten the thread's last-error value: the reader uses the value ctypes saved (use_last_error)."""
    import ctypes

    from brain import process_identity as pi

    class _Kernel32:
        def OpenProcess(self, *_a):  # noqa: N802 - Win32 name
            return 0

        def GetLastError(self):  # noqa: N802 - Win32 name
            return 0  # clobbered by something that ran after OpenProcess

    k32 = _Kernel32()
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: k32, raising=False)
    monkeypatch.setattr(ctypes, "windll", type("W", (), {"kernel32": k32})(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 87, raising=False)
    assert pi._windows_token(4242) == pi.GONE
