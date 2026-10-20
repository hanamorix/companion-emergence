"""What every service backend does around a start or an install (#456: D20, D25, S-k, S5-i, must-fix 1 (c)).

- A service start (``start_service``) first deletes any stop intent: an explicit start means the user wants the bridge
  running, so a kill's "do not respawn" mark must not stop it (D25).
- An install never replaces a running bridge behind its back: with a terminal it first runs the normal stop and waits
  (the terminal watcher, plan I9); without one it refuses at once and changes nothing. On a stopped bridge it deletes
  any stop intent, then the backend writes and starts the service.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path


class BridgeRunningError(RuntimeError):
    """An install was asked for while the persona's bridge runs and no terminal can show the stop."""


def persona_dir_for(persona: str) -> Path:
    from brain.paths import get_persona_dir

    return Path(get_persona_dir(persona))


def clear_stop_intent(persona: str) -> None:
    from brain.bridge import shutdown_progress

    shutdown_progress.delete_record(persona_dir_for(persona), shutdown_progress.INTENT_FILENAME)


def bridge_running(persona: str) -> bool:
    from brain.bridge import state_file

    return state_file.liveness(persona_dir_for(persona)) == state_file.ALIVE


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


# The terminal watcher's stop (plan I9 sets it): stop the persona's bridge gracefully and wait, returning True once it
# has stopped. None until the CLI provides it.
interactive_stop: Callable[[str], bool] | None = None


def before_install(persona: str) -> None:
    """Run by every backend's ``install_service`` before it touches anything."""
    if bridge_running(persona):
        if _interactive() and interactive_stop is not None and interactive_stop(persona):
            pass
        else:
            raise BridgeRunningError("the bridge is running; stop it first")
    clear_stop_intent(persona)
