"""The respawn guard (#456 D25): a service manager must not undo a user's kill.

After ``nell supervisor stop --kill`` (or an escalation) of a service-owned bridge, the manager may start the bridge again
on its own (systemd's restart, launchd's KeepAlive, a Task Scheduler retry). ``nell supervisor run`` with a service
origin therefore asks, before taking the bridge lock and again right after acquiring it (before the watcher files are
deleted), whether the kill should hold:

- the stop intent is ``committed`` with reason ``user_kill`` (a ``pending`` intent is a kill that never happened);
- it was written during THIS boot: Linux compares the recorded boot id with the current one (a null or unreadable boot id
  does not fire); macOS / Windows require the intent's ``at`` to be no earlier than ``psutil.boot_time()`` read here (a
  ``boot_time()`` that raises does not fire). No two boot-time readings are compared, so no tolerance is needed;
- the killed bridge (pid + token) is GONE.

When it fires, the run exits 0 without starting. It never deletes the intent: an explicit start does that.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SERVICE_ORIGINS = ("launchd", "systemd", "task-scheduler")


def _platform() -> str:
    return sys.platform


def _this_boot(intent: dict) -> bool:
    platform = _platform()
    if platform.startswith("linux"):
        from brain import process_identity

        current = process_identity.boot_id()
        return bool(current) and bool(intent.get("boot")) and intent.get("boot") == current
    try:
        import psutil

        boot_time = psutil.boot_time()
        at = datetime.fromisoformat(str(intent.get("at")))
    except Exception:  # noqa: BLE001 - an unknowable boot: never keep a bridge down on a guess
        return False
    return at.timestamp() >= boot_time


def _killed_bridge_gone(intent: dict, persona_dir: Path) -> bool:
    from brain import process_identity
    from brain.bridge import state_file

    pid, token = intent.get("pid"), intent.get("pid_token")
    if not isinstance(pid, int):
        return False
    if token:
        return process_identity.matches(pid, token) == process_identity.GONE
    # A pre-PR bridge (no token): gone when its pid is dead or no longer proves to be this persona's bridge.
    if not state_file.pid_is_alive(pid):
        return True
    return not state_file.prove_bridge_cmdline(pid, persona_dir, persona_dir.name)


def guard_fires(persona_dir: Path) -> bool:
    from brain.bridge import kill

    intent = kill.read_intent(Path(persona_dir))
    if not isinstance(intent, dict):
        return False
    if intent.get("state") != "committed" or intent.get("reason") != "user_kill":
        return False
    if not _this_boot(intent):
        return False
    return _killed_bridge_gone(intent, Path(persona_dir))
