"""SP-7 bridge daemon orchestration — process spawn, stop, status, recovery.

Public surface used by CLI handlers in brain.cli:
  cmd_start(args)    -> int
  cmd_stop(args)     -> int
  cmd_restart(args)  -> int
  cmd_status(args)   -> int
  cmd_tail(args)     -> int
  cmd_tail_log(args) -> int

Internal:
  run_recovery_if_needed(persona_dir) — drain orphan buffers if previous bridge
    exited dirty.
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
from pathlib import Path

import httpx

from brain.bridge import state_file
from brain.bridge.provider import get_provider
from brain.ingest.pipeline import close_stale_sessions
from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig

logger = logging.getLogger(__name__)

LOCKFILE = "bridge.json.lock"


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
    provider = get_provider(config.provider)
    try:
        reports = close_stale_sessions(
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
        try:
            existing_pid = int(path.read_text().strip())
            if not state_file.pid_is_alive(existing_pid):
                path.unlink()
                return acquire_lock(persona_dir)
        except (ValueError, OSError):
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

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def cmd_start(args) -> int:
    from brain.paths import get_log_dir, get_persona_dir

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
        drained = run_recovery_if_needed(persona_dir)
        if drained is not None:
            if drained > 0:
                print(f"recovered from dirty shutdown — drained {drained} orphan sessions")
            else:
                print("recovered from dirty shutdown (no orphan sessions to drain)")

        log_path = get_log_dir() / f"bridge-{persona_dir.name}.log"
        idle = float(args.idle_shutdown) * 60 if args.idle_shutdown > 0 else None
        client_origin = getattr(args, "client_origin", "cli")
        pid = spawn_detached(persona_dir, idle, client_origin, log_path)

        deadline = time.time() + 5.0
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
                        return 0
                except httpx.HTTPError:
                    continue
        # H-D: readiness failed — kill the orphan child and tell the user where to look.
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already dead
        print(
            f"bridge spawned (pid {pid}) but /health did not respond in 5s — "
            f"killed orphan child. Inspect log at {log_path}",
            file=sys.stderr,
        )
        return 1
    finally:
        release_lock(persona_dir, fd)


def cmd_stop(args) -> int:
    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or s.pid is None or not state_file.pid_is_alive(s.pid):
        print("bridge not running")
        return 0
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
                f"http://127.0.0.1:{s.port}/health", headers=headers, timeout=1.0,
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
    """Subscribe to /events and print every event as a JSON line."""
    from websockets.sync.client import connect

    from brain.paths import get_persona_dir

    persona_dir = get_persona_dir(args.persona)
    s = state_file.read(persona_dir)
    if s is None or not state_file.is_running(persona_dir):
        print("bridge not running", file=sys.stderr)
        return 1
    token_qs = f"?token={s.auth_token}" if s.auth_token else ""
    url = f"ws://127.0.0.1:{s.port}/events{token_qs}"
    try:
        with connect(url) as ws:
            while True:
                msg = ws.recv()
                print(msg)
    except KeyboardInterrupt:
        return 0


def cmd_restart(args) -> int:
    """Stop the bridge, then start it again. Two-phase, gated on stop success.

    `cmd_stop` collapses "no bridge was running" and "clean SIGTERM stop"
    into exit code 0; it returns 1 only when the bridge ignored SIGTERM
    and timed out (wedged). Restart proceeds to start ONLY when stop
    returned 0 — never over a wedged bridge. Restart's exit code is
    whatever `cmd_start` returned (0/1/2) on the success path, or stop's
    exit code on the bail path.
    """
    print("stopping bridge...")
    stop_rc = cmd_stop(args)
    if stop_rc != 0:
        print(f"restart aborted: stop failed (exit {stop_rc})", file=sys.stderr)
        return stop_rc
    print("starting bridge...")
    return cmd_start(args)


# Hook used by tests + ctrl-c handler in follow mode. Tests can set this
# Event to exit follow mode cleanly without raising KeyboardInterrupt.
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
            lines = f.readlines()
            tail = lines[-n:] if n > 0 else []
            for line in tail:
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
