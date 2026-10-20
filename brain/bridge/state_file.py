"""Bridge daemon state file — bridge.json.

The file is persistent across restarts. `pid`/`port` get cleared on graceful
shutdown; `shutdown_clean` flips to True as the very last step. On the next
`bridge start`, if `shutdown_clean: false` AND the recorded pid is dead, the
previous bridge crashed — caller should run dirty-shutdown recovery.

Atomic writes go through brain.health.json_io.write_json so
partial writes don't corrupt state. Reads use read_json_result (HEAL) so corrupted
files restore from .bak1.

OG reference: NellBrain/nell_supervisor.py:53-101 (state file machinery,
shutdown_clean flag, pid helpers).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.health.adaptive import compute_treatment
from brain.health.json_io import Recovery, read_json_result, write_json
from brain.state_compat import from_known_fields

logger = logging.getLogger(__name__)

STATE_FILENAME = "bridge.json"


@dataclass
class BridgeState:
    """Full bridge.json schema."""

    persona: str
    pid: int | None
    port: int | None
    started_at: str
    stopped_at: str | None
    shutdown_clean: bool
    client_origin: str  # "cli" | "tauri" | "tests" | "launchd" | "systemd" | "task-scheduler"
    # H-C auth: ephemeral bearer token generated at start, persisted in
    # bridge.json so local clients (CLI / Tauri) can read it. Must NOT be
    # committed to git or logged. None for legacy state files (auth disabled
    # for backward compat — runner.py always generates one in real use).
    auth_token: str | None = None
    # Ingest errors of the last shutdown drain, written by versions before #456 (no drain any more, Q-B). Kept so an
    # older file of that shape still reads as "recovery needed" (#471); the next start's first idle pass saves the
    # sessions either way.
    drain_errors: int = 0
    # Process identity (#456): the token brain.process_identity computes for this bridge at start, so a reused pid
    # is never taken for the bridge. `identity` is "token" when pid_token is set, "unavailable" when this platform
    # could not read the bridge's own token (liveness then uses the command-line proof), and None in a file written
    # before this field existed (a pre-PR bridge, owner's decision D22).
    pid_token: str | None = None
    identity: str | None = None
    # Stop record (#456): "running" | "stopping" | "stopped"; why it stopped; when the stop was requested.
    phase: str | None = None
    stop_reason: str | None = None
    stop_requested_at: str | None = None
    # The supervisor's state (#456 Q6: SupervisorStatus.as_dict), so `status --json` reports it without HTTP.
    supervisor: dict | None = None


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / STATE_FILENAME


def _default_factory() -> dict:
    """Returned when bridge.json is missing AND all .bak files are corrupt.

    A "missing" bridge.json (never written) returns None from read() — we
    only fall through to default_factory in the heal when the file
    exists but is unrecoverable. In that case we return a stub that
    recovery_needed() sees as 'not running, not dirty'.
    """
    return {
        "persona": "",
        "pid": None,
        "port": None,
        "started_at": "",
        "stopped_at": None,
        "shutdown_clean": True,
        "client_origin": "cli",
        "auth_token": None,
        "drain_errors": 0,
    }


def write(persona_dir: Path, state: BridgeState) -> None:
    """Atomically write bridge.json with .bak rotation."""
    path = _state_path(persona_dir)
    try:
        treatment = compute_treatment(persona_dir, STATE_FILENAME)
        backup_count = treatment.backup_count
    except Exception:
        logger.warning("compute_treatment failed; defaulting to backup_count=3")
        backup_count = 3
    # mode=0o600 applies to the file and to every rotated backup (the token must never be wider).
    write_json(path, asdict(state), backups=backup_count, mode=0o600)
    _protect_state_dir(path)


# One read-modify-write of bridge.json at a time inside this process (C18 a). `_update_between` is a test hook called
# between the read and the write.
_update_lock = threading.Lock()
_update_between: Callable[[], None] | None = None


def update(persona_dir: Path, fn: Callable[[BridgeState], None]) -> bool:
    """Read-modify-write bridge.json for THIS bridge only (main's ruling: bridge.json is written only by its owner).

    Writes when the record's pid is this process and its token is this process's own token (both None on a platform
    where identity is unavailable); otherwise skips and returns False. Serialized under a module lock, so two updates
    in this process never lose each other.
    """
    from brain import process_identity

    with _update_lock:
        cur = read(persona_dir)
        if cur is None or cur.pid != os.getpid() or cur.pid_token != process_identity.own_token():
            return False
        if _update_between is not None:
            _update_between()
        fn(cur)
        write(persona_dir, cur)
        return True


def _protect_state_dir(path: Path) -> None:
    """Keep the bridge state directory owner-only (the files get 0o600 from write_json)."""
    try:
        path.parent.chmod(0o700)
    except OSError as exc:
        logger.warning("failed to chmod bridge state dir %s: %s", path.parent, exc)


def read(persona_dir: Path) -> BridgeState | None:
    """Read bridge.json. Returns None if file does not exist.

    If the file exists but is corrupt, the heal restores from .bak rotation
    and returns the recovered data. If all backups are corrupt, returns the
    default-factory output (treated as 'not running, clean shutdown').
    """
    path = _state_path(persona_dir)
    if not path.exists():
        return None
    res = read_json_result(path, default=_default_factory, recover=Recovery.HEAL)
    data, anomaly = res.data, res.anomaly
    if anomaly is not None:
        logger.warning(
            "bridge.json anomaly: %s (%s) — using recovered/default state",
            anomaly.kind,
            anomaly.action,
        )
    # Backward-compat: legacy files written before drain_errors existed.
    data.setdefault("drain_errors", 0)
    return from_known_fields(BridgeState, data)


def _windows_pid_is_alive(pid: int) -> bool:
    """Windows-safe PID liveness check that avoids os.kill(pid, 0)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    process_query_limited_information = 0x1000
    still_active = 259

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # ERROR_ACCESS_DENIED means the process exists but is not queryable by
        # this user/session. Treat it as alive, matching PermissionError on
        # POSIX. Other errors mean no live process was opened.
        return kernel32.GetLastError() == 5

    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def pid_is_alive(pid: int) -> bool:
    """Return True if the given pid is a live process owned by this user."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Pid exists but is owned by someone else — treat as alive.
        return True
    except OSError as exc:
        logger.debug("pid liveness check failed pid=%s err=%s", pid, exc)
        return False


def recovery_needed(persona_dir: Path) -> bool:
    """True iff the previous bridge process exited dirty and is gone.

    Dirty = ``shutdown_clean`` False, or (in a file written before #456) a shutdown drain that reported ingest
    errors. A dirty record whose pid was cleared (older versions cleared it on the way out, also after drain errors,
    #471) is gone by construction; one with a pid is gone when its identity says so (``bridge_liveness``).
    "Recovery" is the next start's first idle pass (the gated job ``session_snapshot_prune``), not a step at start."""
    state = read(persona_dir)
    if state is None:
        return False
    if state.shutdown_clean and state.drain_errors <= 0:
        return False
    if state.pid is None:
        return True
    return bridge_liveness(state, persona_dir) == GONE


def is_running(persona_dir: Path) -> bool:
    """True iff the recorded bridge is ALIVE (identity confirmed). UNKNOWN is not "running": start decisions then go
    to the OS lock, which a live bridge holds (#177 S2)."""
    return liveness(persona_dir) == ALIVE


# ---- process identity (#456, #177 S2, owner's decision D22) ----------------------------------------------------

ALIVE = "alive"
GONE = "gone"
UNKNOWN = "unknown"


def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.path.expanduser(str(path))))


def home_from_env(env: dict[str, str], platform: str | None = None) -> str | None:
    """The companion-emergence home a process with environment ``env`` uses (normalised): ``brain.paths.home_for_env``,
    the one rule ``get_home`` also uses (#456 review S15). None when it cannot be computed from that environment."""
    from brain import paths

    home = paths.home_for_env(env, platform)
    return _norm(home) if home is not None else None


def prove_bridge_cmdline(pid: int | None, persona_dir: Path, persona: str) -> bool:
    """True when ``pid`` runs a bridge for THIS persona, judged by its command line: ``brain.bridge.runner`` with
    ``--persona-dir`` equal to ``persona_dir``, or ``supervisor run --persona <persona>`` whose effective home (from
    its own environment) equals ours. Anything unreadable is "not proven" (never a name-only match)."""
    if pid is None or pid <= 0:
        return False
    try:
        import psutil

        proc = psutil.Process(pid)
        argv = [str(a) for a in proc.cmdline()]
    except Exception:  # noqa: BLE001 - NoSuchProcess / AccessDenied / ZombieProcess: unproven
        return False
    if "brain.bridge.runner" in argv and "--persona-dir" in argv:
        i = argv.index("--persona-dir")
        return i + 1 < len(argv) and _norm(argv[i + 1]) == _norm(persona_dir)
    for i in range(len(argv) - 3):
        if argv[i] == "supervisor" and argv[i + 1] == "run":
            rest = argv[i + 2 :]
            if (
                "--persona" in rest
                and rest.index("--persona") + 1 < len(rest)
                and rest[rest.index("--persona") + 1] == persona
            ):
                try:
                    env = proc.environ()
                except Exception:  # noqa: BLE001 - unreadable environment: unproven
                    return False
                from brain.paths import get_home

                theirs = home_from_env(env)
                return theirs is not None and theirs == _norm(get_home())
    return False


def bridge_liveness(state: BridgeState | None, persona_dir: Path) -> str:
    """ALIVE / GONE / UNKNOWN for the bridge recorded in ``state``.

    A record with a token: the live process must carry that token. A pre-PR record (no ``identity``) or a bridge whose
    identity was unavailable: the pid must be alive AND its command line must prove it is a bridge for this persona;
    an unproven live pid is GONE (stale) for start, stop and kill decisions."""
    from brain import process_identity as pi

    if state is None or state.pid is None:
        return GONE
    if state.pid_token:
        result = pi.matches(state.pid, state.pid_token)
        return ALIVE if result == "alive" else result
    if not pid_is_alive(state.pid):
        return GONE
    return (
        ALIVE
        if prove_bridge_cmdline(state.pid, persona_dir, state.persona or persona_dir.name)
        else GONE
    )


def liveness(persona_dir: Path) -> str:
    return bridge_liveness(read(persona_dir), persona_dir)


def is_legacy(state: BridgeState | None) -> bool:
    return state is not None and state.identity is None and not state.pid_token
