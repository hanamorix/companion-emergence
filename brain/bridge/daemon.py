"""SP-7 bridge daemon orchestration — process spawn, stop, status, recovery.

Public surface used by CLI handlers in brain.cli:
  cmd_start(args)    -> int
  cmd_run(args)      -> int
  cmd_stop(args)     -> int
  cmd_restart(args)  -> int
  cmd_status(args)   -> int
  cmd_tail(args)     -> int
  cmd_tail_log(args) -> int

Internal:
  run_recovery_if_needed(persona_dir) — snapshot orphan buffers (non-destructive)
    if previous bridge exited dirty.
  spawn_detached(persona_dir, idle_shutdown_seconds, client_origin, log_path) -> pid
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import httpx

from brain.bridge import state_file
from brain.bridge.model_tier import TIER_BACKGROUND_HOUSEKEEPING, build_tier_provider
from brain.ingest.pipeline import snapshot_stale_sessions
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

logger = logging.getLogger(__name__)

LOCKFILE = "bridge.json.lock"

# S50/S56: real OS-level lock (like brain.utils.file_lock's sidecar pattern),
# held for the bridge's whole process life and released by the OS itself on
# crash or reboot — no pid-alive / age / health-probe staleness guessing.
_IS_WINDOWS = sys.platform.startswith("win")

if _IS_WINDOWS:
    # msvcrt is a Windows-only stdlib module; importing on POSIX would fail
    # at module load (mirrors brain.utils.file_lock's guard).
    import msvcrt
else:
    import fcntl

# Windows only: the byte range msvcrt.locking() locks/unlocks. Windows
# mandatory byte-range locking blocks ANY overlapping I/O from another
# handle — including a plain read — not just writes (unlike POSIX flock,
# which is advisory and never blocks a read). A far, fixed offset well
# beyond the file's real (tiny) content means a normal whole-file read of
# that real content (which ends long before this offset) never overlaps
# the locked range, so the pid at offset 0 stays plainly readable while
# the lock is held (C18c) — no special "read only this sub-region" contract
# for readers, and the pid sits at the SAME offset on both platforms.
# msvcrt.locking locks a real byte range via LockFileEx even when nothing
# has ever been written there; it does not require file content to exist
# at that offset, and an os.ftruncate() to a smaller size afterward does
# not release or move the lock (the lock is independent of current EOF).
_WINDOWS_LOCK_OFFSET = 1 << 20  # 1 MiB


def run_recovery_if_needed(persona_dir: Path) -> int | None:
    """If previous bridge exited dirty, drain orphan buffers.

    Returns:
        None — recovery was not needed (clean previous shutdown or fresh start)
        int  — recovery ran; value is the count of drained sessions (may be 0)
    """
    if not state_file.recovery_needed(persona_dir):
        return None
    prev = state_file.read(persona_dir)
    logger.warning(
        "previous bridge exited dirty (pid=%s started_at=%s) — running recovery",
        prev.pid if prev else "?",
        prev.started_at if prev else "?",
    )
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    provider = build_tier_provider(persona_dir, TIER_BACKGROUND_HOUSEKEEPING)
    try:
        reports = snapshot_stale_sessions(
            persona_dir,
            silence_minutes=0,
            store=store,
            hebbian=hebbian,
            provider=provider,
        )
        return len(reports)
    finally:
        store.close()
        hebbian.close()


def acquire_lock(persona_dir: Path) -> int | None:
    """Take the bridge's OS-level lock for the whole process life.

    Same pattern as brain.utils.file_lock: open with O_CREAT (never O_EXCL —
    the file itself is never recreated, only its lock contended for) and take
    a non-blocking flock/msvcrt lock on that fd. A crash or reboot releases
    the OS lock instantly, so there is no pid-alive / age / health-probe
    staleness logic left to guess with (S50) — the lock IS the liveness
    signal.

    Returns the open fd (the caller must keep it open for the lock's
    duration and pass it to release_lock) on success, or None if another
    live process holds the lock.

    The pid is written into the file for information only (S50) — never
    read back to make a decision here. It sits at offset 0 on BOTH
    platforms: on Windows the locked byte range is at a far fixed offset
    (`_WINDOWS_LOCK_OFFSET`, see its own comment) well past the file's real
    content, so a plain whole-file read never overlaps it; on POSIX flock
    is advisory over the whole file regardless, so a reader was never
    blocked there either way.
    """
    path = persona_dir / LOCKFILE
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR)
    try:
        if _IS_WINDOWS:
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None

    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, str(os.getpid()).encode())
    return fd


def release_lock(persona_dir: Path, fd: int) -> None:
    """Unlock and close the fd. Never unlinks the file (S56) — unlinking

    would let a second process create a NEW file under the same name and
    lock THAT one, so two processes could each believe they hold "the"
    bridge lock on two different inodes."""
    try:
        if _IS_WINDOWS:
            os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _kill_if_alive(pid: int, sig: int) -> bool:
    """``os.kill(pid, sig)``, tolerating "already dead" identically on both
    platforms. Returns True if the pid was alive and got signaled, False if
    it was already dead.

    POSIX: a dead pid raises ``ProcessLookupError``. Windows: ``os.kill``'s
    ``TerminateProcess``-based implementation instead raises a plain
    ``OSError`` with ``winerror == 87`` (ERROR_INVALID_PARAMETER) for an
    already-exited pid — a documented CPython-on-Windows quirk, NOT
    ``ProcessLookupError`` — so it must be checked for explicitly rather
    than assumed covered by the POSIX exception type. (CI 2026-09-26:
    windows-latest hit exactly this, unhandled, inside cmd_start's
    readiness-timeout orphan-kill path.)
    """
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        if _IS_WINDOWS and getattr(exc, "winerror", None) == 87:
            return False
        raise


def _path_eq(p1: str, p2: str) -> bool:
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)


def bridge_python() -> tuple[str, dict[str, str] | None]:
    """The interpreter (and env, or None to inherit) to spawn a bridge child with,
    such that the spawned process's pid IS the runner's own ``os.getpid()``.

    Windows venv: ``sys.executable`` (``.venv\\Scripts\\python.exe``) is the
    venv *redirector* (``venvlauncher.exe``, used by both ``python -m venv``
    and uv), which starts the real base interpreter as a SECOND process. So
    ``Popen(...).pid`` is the redirector's pid, while the runner writes its
    own (different) ``os.getpid()`` into bridge.json. ``cmd_start``'s
    readiness check (``s.pid == pid``) could then never match: a healthy
    bridge was reported as failed after the full 50s wait and its
    redirector killed. Same fix as the stdlib's own
    ``multiprocessing.popen_spawn_win32``: launch ``sys._base_executable``
    directly and pass the venv via ``__PYVENV_LAUNCHER__``, exactly what the
    redirector itself would have done, minus the extra process.

    Everywhere else (POSIX, where a venv python is a symlink/copy with no
    redirector; or a non-venv Windows runtime such as the bundled
    python-build-standalone one) this is plain ``sys.executable``, env
    inherited.
    """
    base = getattr(sys, "_base_executable", None)
    if _IS_WINDOWS and base and not _path_eq(sys.executable, base):
        env = os.environ.copy()
        env["__PYVENV_LAUNCHER__"] = sys.executable
        return base, env
    return sys.executable, None


def spawn_detached(
    persona_dir: Path,
    idle_shutdown_seconds: float | None,
    client_origin: str,
    log_path: Path,
) -> int:
    """Spawn the bridge server in a detached process. Returns child pid —
    the runner's own pid on every OS (see ``bridge_python``)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab")  # noqa: SIM115
    python, env = bridge_python()
    popen_extra: dict[str, object] = {} if env is None else {"env": env}
    cmd = [
        python,
        "-P",  # -m would put the caller's cwd (maybe a checkout's brain/) on sys.path
        "-m",
        "brain.bridge.runner",
        "--persona-dir",
        str(persona_dir),
        "--client-origin",
        client_origin,
    ]
    if idle_shutdown_seconds is not None:
        cmd += ["--idle-shutdown-seconds", str(idle_shutdown_seconds)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            **popen_extra,
        )
        return proc.pid
    finally:
        log_fh.close()


@dataclass
class BridgeReadiness:
    """Verified-live snapshot of the freshly-spawned (or already-running) bridge.

    Returned via the `out` dict mutation pattern so cmd_start's int return
    code stays compatible with argparse handlers, while callers that need
    to immediately connect (the chat REPL) can grab the verified
    pid/port/auth_token directly without re-reading state_file. The
    re-read was the race in the 2026-05-05 audit-3 Bug B: state_file
    can be rewritten by the bridge supervisor between cmd_start's
    /health verification and the caller's read.
    """

    pid: int
    port: int
    auth_token: str | None


def cmd_start(args, *, out: dict | None = None) -> int:
    """Spawn the bridge daemon. Returns 0 on success, 2 on already-running, 1 on error.

    On success (or already-running), if `out` is provided, populates
    `out["readiness"]` with a BridgeReadiness carrying the verified
    pid/port/auth_token. Callers that need an immediate WS/HTTP connection
    should use that instead of re-reading state_file.
    """
    from brain.paths import get_log_dir, get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    if state_file.is_running(persona_dir):
        cur = state_file.read(persona_dir)
        print(f"bridge already running on port {cur.port} (pid {cur.pid})", file=sys.stderr)
        if out is not None and cur is not None and cur.pid is not None and cur.port is not None:
            out["readiness"] = BridgeReadiness(
                pid=cur.pid, port=cur.port, auth_token=cur.auth_token
            )
        return 2

    # Hold the lock through recovery (so two concurrent starters can't both
    # recover), then RELEASE it right before spawning. The bridge PROCESS — the
    # detached runner (see runner.main) — re-acquires it and holds it for its
    # lifetime, so pid-based stale recovery works. cmd_start must NOT hold the
    # lock across the spawn, or the child can't acquire it and dies (Bug 2a,
    # v0.0.36). The runner's lifetime lock is the real guard against two bridges.
    fd = acquire_lock(persona_dir)
    if fd is None:
        print("bridge already starting (lockfile held)", file=sys.stderr)
        return 2

    client_origin = getattr(args, "client_origin", "cli")
    if client_origin == "launchd":
        try:
            from brain.service.launchd import truncate_launchd_logs_if_large

            truncate_launchd_logs_if_large(args.persona)
        except Exception:
            logger.debug("launchd log truncation skipped", exc_info=True)

    drained = run_recovery_if_needed(persona_dir)
    if drained is not None:
        if drained > 0:
            print(f"recovered from interrupted shutdown - snapshotted {drained} active sessions")
        else:
            print("recovered from dirty shutdown (no orphan sessions to drain)")

    log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
    idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
    release_lock(persona_dir, fd)  # hand the lock to the child runner
    pid = spawn_detached(persona_dir, idle, client_origin, log_path)

    # Readiness window: Windows cold-boot (recovery + persona load + soul review)
    # routinely exceeds 5s; the old 5s deadline killed a healthy-but-slow bridge.
    # 50s stays inside the Rust caller's 60s outer timeout. (Bug 2b, v0.0.36.)
    deadline = time.time() + 50.0
    while time.time() < deadline:
        time.sleep(0.1)
        s = state_file.read(persona_dir)
        if s is not None and s.pid == pid and s.port:
            try:
                headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
                r = httpx.get(
                    f"http://127.0.0.1:{s.port}/health",
                    headers=headers,
                    timeout=1.0,
                )
                if r.status_code == 200:
                    print(f"bridge started on port {s.port} (pid {pid})")
                    if out is not None:
                        out["readiness"] = BridgeReadiness(
                            pid=pid, port=s.port, auth_token=s.auth_token
                        )
                    return 0
            except httpx.HTTPError:
                continue
    # Readiness failed — kill the orphan child and tell the user where to look.
    _kill_if_alive(pid, signal.SIGTERM)  # already-dead is fine either way
    print(
        f"bridge spawned (pid {pid}) but /health did not respond in 50s — "
        f"killed orphan child. Inspect log at {log_path}",
        file=sys.stderr,
    )
    return 1


def cmd_run(args) -> int:
    """Run the bridge in the foreground for OS service managers.

    Unlike ``cmd_start``, this does not fork or detach. The current process is
    the bridge process until uvicorn exits, which is exactly what launchd wants
    to supervise. A per-persona lock is held for the lifetime of the process so
    parallel starts fail cleanly and stale locks can be recovered by pid.
    """
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    if state_file.is_running(persona_dir):
        cur = state_file.read(persona_dir)
        print(f"bridge already running on port {cur.port} (pid {cur.pid})", file=sys.stderr)
        return 2

    fd = acquire_lock(persona_dir)
    if fd is None:
        print("bridge already starting (lockfile held)", file=sys.stderr)
        return 2

    try:
        client_origin = getattr(args, "client_origin", "cli")
        if client_origin == "launchd":
            try:
                from brain.service.launchd import truncate_launchd_logs_if_large

                truncate_launchd_logs_if_large(args.persona)
            except Exception:
                logger.debug("launchd log truncation skipped", exc_info=True)

        drained = run_recovery_if_needed(persona_dir)
        if drained is not None:
            if drained > 0:
                print(f"recovered from interrupted shutdown - snapshotted {drained} active sessions")
            else:
                print("recovered from dirty shutdown (no orphan sessions to drain)")

        from brain.bridge.runner import run_bridge_foreground

        idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
        return run_bridge_foreground(
            persona_dir,
            client_origin=client_origin,
            idle_shutdown_seconds=idle,
        )
    finally:
        release_lock(persona_dir, fd)


def _request_shutdown_via_http(s: state_file.BridgeState, *, timeout: float = 3.0) -> None:
    if s.port is None:
        raise RuntimeError("bridge state missing port")
    headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
    r = httpx.post(
        f"http://127.0.0.1:{s.port}/supervisor/shutdown",
        headers=headers,
        timeout=timeout,
    )
    r.raise_for_status()


def cmd_stop(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or s.pid is None or not state_file.pid_is_alive(s.pid):
        print("bridge not running")
        return 0
    try:
        _request_shutdown_via_http(s)
    except Exception as exc:
        if os.name == "nt":
            if getattr(args, "force", False):
                print(
                    "WARNING: forcing Windows termination; Python cleanup will NOT run. "
                    "Recovery will snapshot active sessions non-destructively on next start.",
                    file=sys.stderr,
                )
                if not _kill_if_alive(s.pid, signal.SIGTERM):  # TerminateProcess on Windows — explicit, logged, dirty by design
                    print("bridge not running")
                    return 0
            else:
                print(
                    f"bridge shutdown endpoint unreachable; refusing Windows hard kill because it would bypass cleanup: {exc}\n"
                    "If the bridge is wedged, re-run with --force to terminate it (recovery will snapshot sessions on next start).",
                    file=sys.stderr,
                )
                return 1
        else:
            logger.warning("shutdown endpoint failed; falling back to SIGTERM on POSIX", exc_info=True)
            if not _kill_if_alive(s.pid, signal.SIGTERM):
                print("bridge not running")
                return 0

    deadline = time.time() + args.timeout
    while time.time() < deadline:
        time.sleep(0.2)
        if not state_file.pid_is_alive(s.pid):
            print(f"bridge stopped (was pid {s.pid})")
            return 0
    print(f"bridge did not stop within {args.timeout}s", file=sys.stderr)
    return 1


def cmd_status(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None:
        print("bridge: not running (no state file)")
        return 0
    if state_file.is_running(persona_dir):
        try:
            headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
            r = httpx.get(
                f"http://127.0.0.1:{s.port}/health",
                headers=headers,
                timeout=1.0,
            )
            health = r.json()
            print(f"bridge: running pid={s.pid} port={s.port}")
            print(f"  uptime_s: {health['uptime_s']}")
            print(f"  sessions_active: {health['sessions_active']}")
            print(f"  supervisor: {health['supervisor_thread']}")
            print(f"  pending_alarms: {health['pending_alarms']}")
        except httpx.HTTPError as e:
            print(f"bridge: pid {s.pid} alive but /health unreachable: {e}", file=sys.stderr)
            return 1
    elif state_file.recovery_needed(persona_dir):
        print(f"bridge: previous process crashed dirty (pid {s.pid}) — next start will recover")
    else:
        print(f"bridge: stopped cleanly at {s.stopped_at}")
    return 0


def cmd_tail(args) -> int:
    """Subscribe to /events and print every event as a JSON line.

    Auth via Sec-WebSocket-Protocol: bearer, <token> — the only auth path
    the server accepts. Previously this used a ?token= query string, which
    (a) the server doesn't read, so tail was silently broken in any
    config with auth enabled, and (b) leaks the bearer token through
    process listings and proxy logs.
    """
    from websockets.sync.client import connect

    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or not state_file.is_running(persona_dir):
        print("bridge not running", file=sys.stderr)
        return 1
    url = f"ws://127.0.0.1:{s.port}/events"
    subprotocols = ["bearer", s.auth_token] if s.auth_token else None
    try:
        with connect(url, subprotocols=subprotocols) as ws:
            while True:
                msg = ws.recv()
                print(msg)
    except KeyboardInterrupt:
        return 0


def cmd_restart(args) -> int:
    """Stop the bridge, then start it again. Two-phase, gated on stop success.

    `cmd_stop` uses an endpoint-first stop (POST /supervisor/shutdown) with
    a POSIX SIGTERM fallback on non-Windows platforms. On Windows, if the
    shutdown endpoint is unreachable, `--force` is required as an explicit
    escape hatch (dirty termination via TerminateProcess; recovery will
    snapshot active sessions non-destructively on next start).

    `cmd_stop` collapses "no bridge was running" and "clean endpoint stop"
    into exit code 0; it returns 1 only when the bridge could not be stopped
    (endpoint unreachable on Windows without --force, or poll timeout). Restart
    proceeds to start ONLY when stop returned 0 — never over a wedged bridge.
    Restart's exit code is whatever `cmd_start` returned (0/1/2) on the
    success path, or stop's exit code on the bail path.
    """
    print("stopping bridge...")
    stop_rc = cmd_stop(args)
    if stop_rc != 0:
        print(f"restart aborted: stop failed (exit {stop_rc})", file=sys.stderr)
        return stop_rc
    print("starting bridge...")
    return cmd_start(args)


# Test seam: setting this Event from a test allows the follow-mode loop
# to exit cleanly without raising KeyboardInterrupt. Production exit
# from follow mode is handled by the KeyboardInterrupt catch in
# cmd_tail_log itself.
_follow_should_stop: threading.Event | None = None


def cmd_tail_log(args) -> int:
    """Print the last N lines of the bridge log; -f to follow.

    Cross-platform: pure Python loop, no shell `tail` (Windows CI lacks it).
    Follow mode polls every 200ms. KeyboardInterrupt is treated as a clean
    exit (returns 0). Tests can interrupt the loop by setting the module-level
    `_follow_should_stop` Event before the call.
    """
    from brain.paths import get_log_dir, get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    if not persona_dir.exists():
        print(f"persona directory not found: {persona_dir}", file=sys.stderr)
        return 1

    log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
    if not log_path.exists():
        print(
            f"bridge log not found at {log_path} — has the supervisor ever started?",
            file=sys.stderr,
        )
        return 1

    n = max(0, int(args.lines))
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            # deque(maxlen=n) keeps only the last n lines in memory as it
            # iterates the file — bounded regardless of log size, unlike
            # f.readlines() which materialised the whole file before slicing.
            if n > 0:
                for line in deque(f, maxlen=n):
                    print(line, end="")
            if not getattr(args, "follow", False):
                return 0
            # Follow mode: seek to end, poll for new content
            f.seek(0, 2)  # SEEK_END
            stop = _follow_should_stop or threading.Event()
            try:
                while not stop.is_set():
                    chunk = f.read()
                    if chunk:
                        print(chunk, end="")
                    else:
                        time.sleep(0.2)
            except KeyboardInterrupt:
                return 0
            return 0
    except OSError as e:
        print(f"error reading {log_path}: {e}", file=sys.stderr)
        return 1
