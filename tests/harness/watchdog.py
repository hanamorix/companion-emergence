"""The usage-reset watchdog — a Python state machine (ports ``usage_watchdog.sh``).

Detects the usage-maxed -> usage-reset (recovery) edge by pinging a cheap model. Cadence:

- ping every ``normal_s`` (baseline, 20 min);
- on a FAILED ping -> tighten to ``tight_s`` (5 min);
- after 4 CONSECUTIVE failures -> relax back to ``normal_s`` (it's down a while; stop hammering);
- when a ping FINALLY succeeds AFTER any failure -> RECOVERY: write a marker + return.

A success while NO failure has happened (pure baseline monitoring) does NOT return — the watchdog
only fires on the recovery edge, so it can be started while usage is fine and sit quietly.

The ping is INJECTED (``ping_fn() -> bool``) so unit tests drive the state machine with a fake ping
(no tokens); the real ping fn is ``claude -p --model {models.watchdog} "reply pong"``.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .config import DEFAULT_MODELS, DEFAULT_TIMEOUTS, ModelConfig, Timeouts


def watchdog_ping_argv(claude_bin: str, models: ModelConfig) -> list[str]:
    """The exact argv the real watchdog ping runs. Exposed so a unit test asserts the model
    threads through against the REAL argv the code builds (not a hand-rebuilt copy) — F1."""
    return [claude_bin, "-p", "--model", models.watchdog, "reply with exactly: pong"]


def real_ping_fn(claude_bin: str, models: ModelConfig, timeouts: Timeouts) -> Callable[[], bool]:
    """Build the real ping: ``claude -p --model {watchdog}`` returning True on a pong."""
    argv = watchdog_ping_argv(claude_bin, models)

    def _ping() -> bool:  # pragma: no cover - needs live CLI
        try:
            out = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeouts.watchdog_ping,
            )
        except Exception:
            return False
        return out.returncode == 0 and bool((out.stdout or "").strip())

    return _ping


class Watchdog:
    """Usage-reset detector. ``ping_fn`` is injected for testability."""

    def __init__(
        self,
        ping_fn: Callable[[], bool],
        marker_path: Path,
        *,
        models: ModelConfig = DEFAULT_MODELS,
        timeouts: Timeouts = DEFAULT_TIMEOUTS,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.ping_fn = ping_fn
        self.marker_path = Path(marker_path)
        self.models = models
        self.timeouts = timeouts
        self.sleep_fn = sleep_fn

    def step(self, state: dict) -> tuple[bool, dict]:
        """Advance the state machine one ping. Return (recovered, new_state).

        Pure (no sleep) so a unit test can step it deterministically. ``state`` keys:
        ``waiting`` (have we seen a failure yet?), ``cadence`` (seconds), ``tight_fails``.
        """
        waiting = state.get("waiting", False)
        tight_fails = state.get("tight_fails", 0)
        normal, tight = self.timeouts.watchdog_normal_s, self.timeouts.watchdog_tight_s

        if self.ping_fn():
            if waiting:
                self.marker_path.write_text("recovered\n")
                return True, {"waiting": False, "cadence": normal, "tight_fails": 0}
            return False, {"waiting": False, "cadence": normal, "tight_fails": 0}
        # failed ping
        if not waiting:
            return False, {"waiting": True, "cadence": tight, "tight_fails": 1}
        tight_fails += 1
        cadence = state.get("cadence", tight)
        if cadence == tight and tight_fails >= 4:
            cadence = normal
        return False, {"waiting": True, "cadence": cadence, "tight_fails": tight_fails}

    def run(self, max_iterations: int | None = None) -> bool:
        """Loop until recovery (or ``max_iterations``). Returns True on recovery. Uses ``sleep_fn``
        between pings (injectable so a test doesn't wait)."""
        state = {"waiting": False, "cadence": self.timeouts.watchdog_normal_s, "tight_fails": 0}
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            recovered, state = self.step(state)
            if recovered:
                return True
            self.sleep_fn(state["cadence"])
            iterations += 1
        return False
