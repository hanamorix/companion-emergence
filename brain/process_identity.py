"""Process identity that survives pid reuse and system clock steps (#456, #472).

A pid alone is not an identity: after a process dies the OS hands the same number to an unrelated
process. ``token_of(pid)`` returns a string fixed when the process was created, so "the process
recorded here is still that process" is ``token_of(pid) == recorded``.

Per OS, the token is built from a value the kernel stores at process creation and never recomputes
from the current wall clock, so stepping the clock (NTP, a VM resume, a manual change) cannot change it:

* Linux: ``linux:<boot id>:<pid>:<start ticks>`` from ``/proc/sys/kernel/random/boot_id`` and field 22 of
  ``/proc/<pid>/stat`` (clock ticks since boot).
* Windows: ``win:<pid>:<creation FILETIME>`` from ``GetProcessTimes``.
* macOS: ``mac:<pid>:<tv_sec>.<tv_usec>`` from ``kinfo_proc.kp_proc.p_starttime``.

The pid is part of the token because two processes can start within one clock tick (10 ms on Linux).

psutil's public ``create_time()`` is not used: on Linux it adds the *current* boot time, and on macOS
it adjusts for clock changes, so it can move under a clock step.

``token_of`` has three outcomes: a token (the process runs), ``GONE`` (no such process, or a zombie /
exited process), ``UNKNOWN`` (it could not be read). What UNKNOWN means is the caller's decision: a
wait keeps waiting, a kill treats it as "not this process".
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Any

GONE = "gone"
UNKNOWN = "unknown"

_OWN: str | None = None


def _linux_boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip() or None
    except OSError:
        return None


def boot_id() -> str | None:
    """The Linux boot id, or None (other platforms, or unreadable)."""
    if sys.platform.startswith("linux"):
        return _linux_boot_id()
    return None


def _linux_token(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes().decode("ascii", "replace")
    except (FileNotFoundError, ProcessLookupError, NotADirectoryError):
        return GONE
    except OSError:
        return UNKNOWN
    close = raw.rfind(")")
    if close < 0:
        return UNKNOWN
    rest = raw[close + 1 :].split()
    if len(rest) < 20:
        return UNKNOWN
    state, starttime = rest[0], rest[19]
    if state in ("Z", "X", "x"):
        return GONE
    bid = _linux_boot_id()
    if bid is None:
        return UNKNOWN
    return f"linux:{bid}:{pid}:{starttime}"


def _windows_token(pid: int) -> str:  # pragma: no cover - exercised on the Windows CI leg
    import ctypes
    from ctypes import wintypes

    # use_last_error (review S13): ctypes saves the error right after each call, so nothing that runs later can
    # overwrite it before it is read.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        err = ctypes.get_last_error()  # type: ignore[attr-defined]
        return GONE if err == 87 else UNKNOWN  # ERROR_INVALID_PARAMETER: no such process
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return UNKNOWN
        if code.value != 259:  # STILL_ACTIVE
            return GONE
        created, exited, kern, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kern),
            ctypes.byref(user),
        ):
            return UNKNOWN
        return f"win:{pid}:{(created.dwHighDateTime << 32) | created.dwLowDateTime}"
    finally:
        kernel32.CloseHandle(handle)


def _macos_token(pid: int) -> str:  # pragma: no cover - exercised on the macOS CI leg
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
        size = ctypes.c_size_t(0)
        if libc.sysctl(mib, 4, None, ctypes.byref(size), None, 0) != 0:
            return UNKNOWN
        if size.value == 0:
            return GONE
        buf = ctypes.create_string_buffer(size.value)
        if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) != 0:
            return UNKNOWN
        if size.value == 0:
            return GONE
        # struct kinfo_proc { struct extern_proc kp_proc; ... }: kp_proc starts with the p_un union whose
        # __p_starttime is a struct timeval {int64 tv_sec; int32 tv_usec}. p_stat is a char at offset 36
        # (after p_un (16), p_vmspace (8), p_sigacts (8), p_flag (4)).
        tv_sec = int.from_bytes(buf.raw[0:8], "little", signed=True)
        tv_usec = int.from_bytes(buf.raw[8:12], "little", signed=True)
        p_stat = buf.raw[36]
        if p_stat == 5:  # SZOMB
            return GONE
        return f"mac:{pid}:{tv_sec}.{tv_usec:06d}"
    except (OSError, AttributeError, ValueError, IndexError):
        return UNKNOWN


def token_of(pid: int | None) -> str:
    """The identity token of ``pid``, or ``GONE`` / ``UNKNOWN``."""
    if pid is None or pid <= 0:
        return GONE
    try:
        if sys.platform.startswith("linux"):
            return _linux_token(pid)
        if sys.platform == "win32":
            return _windows_token(pid)
        if sys.platform == "darwin":
            return _macos_token(pid)
    except Exception:  # noqa: BLE001 - a reader fault is "could not read", never a crash
        return UNKNOWN
    return UNKNOWN


def is_token(value: str | None) -> bool:
    return bool(value) and value not in (GONE, UNKNOWN)


def is_pid_number(value: int) -> bool:
    """Can ``value`` be a process id on this platform? A POSIX ``pid_t`` is a signed C int and a Windows process id is an
    unsigned 32-bit value (both sizes from ``struct``, no literal); 0 and below name no process. A larger number can name no
    process either, and the OS calls that take a pid refuse it with ``OverflowError`` (``os.kill``), which no liveness check
    expects: a pid read from a record or a file name goes through here first."""
    if sys.platform == "win32":
        limit = 1 << (8 * struct.calcsize("I"))
    else:
        limit = 1 << (8 * struct.calcsize("i") - 1)
    return 0 < value < limit


def own_pid() -> int:
    """This process's pid: the one place code outside this module gets it (engine_files makes no raw ``os`` call)."""
    return os.getpid()


def own_token() -> str | None:
    """This process's token, or None when it cannot be read here.

    The first answer stands for the life of the process, a failed read included: a bridge that recorded no token
    (runner.py, identity "unavailable") must keep matching its own record in ``is_own_identity`` (``state_file.update``,
    the engine's progress record).
    """
    global _OWN
    if _OWN is None:
        tok = token_of(own_pid())
        _OWN = tok if is_token(tok) else ""
    return _OWN or None


def matches(pid: int | None, recorded: str | None) -> str:
    """``"alive"`` when ``pid`` runs and carries ``recorded``; ``GONE`` when it does not (dead, zombie, or
    another process); ``UNKNOWN`` when it cannot be read. ``recorded`` must be a real token."""
    tok = token_of(pid)
    if tok == UNKNOWN:
        return UNKNOWN
    if tok == GONE or tok != recorded:
        return GONE
    return "alive"


def holder_alive(pid: int | None, recorded: str | None) -> bool:
    """Is the holder recorded in a lock or marker file (#472) still alive?

    With a recorded token: alive unless the pid is GONE or carries another token; an UNKNOWN reading counts as alive
    (never reap a holder that may still run). Without one (a file in the older format): the pid-only check, as before.
    """
    if pid is None or pid <= 0:
        return False
    if is_token(recorded):
        tok = token_of(pid)
        if tok == UNKNOWN:
            return True
        return tok == recorded
    from brain.bridge.state_file import pid_is_alive

    return pid_is_alive(pid)


def names_this_bridge(record: dict[str, Any] | None, pid: int | None, token: str | None) -> bool:
    """True when ``record`` names exactly ``pid`` + ``token`` (C43 a: anything else is ignored).

    A comparison against an expected identity only, no liveness (that is ``holder_alive``). The bridge's watcher files use it
    for "is this record the bridge's": a record with no token never matches. A process asking whether a record is its OWN uses
    ``is_own_identity``, which lets a missing token match a missing token.
    """
    if not isinstance(record, dict) or pid is None or not token:
        return False
    return record.get("pid") == pid and record.get("pid_token") == token


def is_own_identity(pid: int | None, token: str | None) -> bool:
    """Do ``pid`` and ``token`` name this process: the pid is this process's and the token equals ``own_token()``?

    Both tokens None count as equal: a process that cannot read its own token records none, and that record is still
    its own (the engine's progress record, bridge.json). A real token against a missing one, or the reverse, does not match.
    A bool is not a pid.
    """
    if isinstance(pid, bool) or pid != own_pid():
        return False
    return token == own_token()


def token_digest(token: str) -> str:
    """A filename-safe 12-hex digest of a token (the width of the save-temp tag's existing random part)."""
    import hashlib

    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:12]
