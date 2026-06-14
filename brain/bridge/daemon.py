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
from brain.bridge.provider import get_provider
from brain.ingest.pipeline import snapshot_stale_sessions
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

logger = logging.getLogger(__name__)

LOCKFILE = "bridge.json.lock"

_LOCK_STALE_SECONDS = 120.0
# Known UX edge (accepted): a crash followed by a relaunch within
# _LOCK_STALE_SECONDS of the lock's mtime blocks startup until the window
# passes. Single-user desktop app — acceptable; documented here.


def _lock_age_seconds(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return 0.0


def _archive_stale_lock(path: Path) -> None:
    """Rename a stale lock to a timestamped .stale-* sibling as evidence —
    never silently delete it."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = path.with_name(f"{path.name}.stale-{stamp}")
    try:
        path.replace(target)
    except FileNotFoundError:
        return


def _recorded_bridge_health(persona_dir: Path) -> bool:
    """True iff the recorded bridge port answers /health 200.

    Safe against PID/port reuse BECAUSE /health requires bearer auth
    (server.py — Depends(require_http_auth)): a stray process squatting the
    recorded port fails the token check → non-200 → lock recovers. If /health
    auth is ever relaxed, this recovery path silently breaks."""
    s = state_file.read(persona_dir)
    if s is None or s.port is None:
        return False
    headers = {"Authorization": f"Bearer {s.auth_token}"} if s.auth_token else {}
    try:
        r = httpx.get(f"http://127.0.0.1:{s.port}/health", headers=headers, timeout=0.5)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


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
    embeddings = EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(dim=256))
    config = PersonaConfig.load(persona_dir / "persona_config.json")
    provider = get_provider(config.provider, persona_dir=persona_dir)
    try:
        reports = snapshot_stale_sessions(
            persona_dir,
            silence_minutes=0,
            store=store,
            hebbian=hebbian,
            provider=provider,
            embeddings=embeddings,
        )
        return len(reports)
    finally:
        store.close()
        hebbian.close()
        embeddings.close()


def acquire_lock(persona_dir: Path) -> int | None:
    """Create the lockfile atomically. Returns fd on success, None on conflict."""
    path = persona_dir / LOCKFILE
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except FileExistsError:
        age = _lock_age_seconds(path)
        try:
            existing_text = path.read_text().strip()
            existing_pid = int(existing_text)
            if not state_file.pid_is_alive(existing_pid):
                # Double-read guard: best-effort, NOT atomic (TOCTOU race between
                # read and replace). Acceptable for a single-user desktop app; the
                # stale lock is archived as evidence, never silently deleted.
                if path.read_text().strip() != existing_text:
                    return None
                _archive_stale_lock(path)
                return acquire_lock(persona_dir)
            if age > _LOCK_STALE_SECONDS and not _recorded_bridge_health(persona_dir):
                logger.warning(
                    "recovering stale bridge lockfile with alive pid but dead health pid=%s age=%.1fs",
                    existing_pid,
                    age,
                )
                if path.read_text().strip() != existing_text:
                    return None
                _archive_stale_lock(path)
                return acquire_lock(persona_dir)
        except FileNotFoundError:
            return acquire_lock(persona_dir)
        except ValueError:
            if age > _LOCK_STALE_SECONDS:
                # Double-read guard (best-effort, not atomic — same TOCTOU caveat
                # as the dead/alive-pid branches): if a concurrent starter replaced
                # the corrupt lock with a valid one between our first read and now,
                # bail rather than archive their live lock.
                if path.read_text().strip() != existing_text:
                    return None
                logger.warning("recovering stale corrupt bridge lockfile age=%.1fs", age)
                _archive_stale_lock(path)
                return acquire_lock(persona_dir)
        except OSError:
            pass
        return None


def release_lock(persona_dir: Path, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        (persona_dir / LOCKFILE).unlink()
    except FileNotFoundError:
        pass


def spawn_detached(
    persona_dir: Path,
    idle_shutdown_seconds: float | None,
    client_origin: str,
    log_path: Path,
) -> int:
    """Spawn the bridge server in a detached process. Returns child pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "ab")  # noqa: SIM115
    cmd = [
        sys.executable,
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
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already dead
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
                try:
                    os.kill(s.pid, signal.SIGTERM)  # TerminateProcess on Windows — explicit, logged, dirty by design
                except ProcessLookupError:
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
            try:
                os.kill(s.pid, signal.SIGTERM)
            except ProcessLookupError:
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
