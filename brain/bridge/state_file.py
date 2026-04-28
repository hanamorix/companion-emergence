"""Bridge daemon state file — bridge.json.

The file is persistent across restarts. `pid`/`port` get cleared on graceful
shutdown; `shutdown_clean` flips to True as the very last step. On the next
`bridge start`, if `shutdown_clean: false` AND the recorded pid is dead, the
previous bridge crashed — caller should run dirty-shutdown recovery.

Atomic writes go through brain.health.attempt_heal.save_with_backup so
partial writes don't corrupt state. Reads use attempt_heal so corrupted
files restore from .bak1.

OG reference: NellBrain/nell_supervisor.py:53-101 (state file machinery,
shutdown_clean flag, pid helpers).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from brain.health.adaptive import compute_treatment
from brain.health.attempt_heal import attempt_heal, save_with_backup

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
    client_origin: str  # "cli" | "tauri" | "tests"
    # H-C auth: ephemeral bearer token generated at start, persisted in
    # bridge.json so local clients (CLI / Tauri) can read it. Must NOT be
    # committed to git or logged. None for legacy state files (auth disabled
    # for backward compat — runner.py always generates one in real use).
    auth_token: str | None = None


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / STATE_FILENAME


def _default_factory() -> dict:
    """Returned when bridge.json is missing AND all .bak files are corrupt.

    A "missing" bridge.json (never written) returns None from read() — we
    only fall through to default_factory in attempt_heal when the file
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
    save_with_backup(path, asdict(state), backup_count=backup_count)


def read(persona_dir: Path) -> BridgeState | None:
    """Read bridge.json. Returns None if file does not exist.

    If the file exists but is corrupt, attempt_heal restores from .bak rotation
    and returns the recovered data. If all backups are corrupt, returns the
    default-factory output (treated as 'not running, clean shutdown').
    """
    path = _state_path(persona_dir)
    if not path.exists():
        return None
    data, anomaly = attempt_heal(path, _default_factory)
    if anomaly is not None:
        logger.warning(
            "bridge.json anomaly: %s (%s) — using recovered/default state",
            anomaly.kind,
            anomaly.action,
        )
    return BridgeState(**data)


def pid_is_alive(pid: int) -> bool:
    """Return True if the given pid is a live process owned by this user."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Pid exists but is owned by someone else — treat as alive.
        return True


def recovery_needed(persona_dir: Path) -> bool:
    """True iff the previous bridge process exited dirty.

    Predicate: state file exists AND shutdown_clean is False AND
    the recorded pid is dead.
    """
    state = read(persona_dir)
    if state is None:
        return False
    if state.shutdown_clean:
        return False
    if state.pid is None:
        return False
    return not pid_is_alive(state.pid)


def is_running(persona_dir: Path) -> bool:
    """True iff a live bridge daemon is recorded for this persona."""
    state = read(persona_dir)
    if state is None or state.pid is None:
        return False
    return pid_is_alive(state.pid)
