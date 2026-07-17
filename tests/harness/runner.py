"""The re-entrant multi-arm runner — a Python state machine (ports ``run_t*_series.sh``).

Runs a list of arms (``setup -> drive -> archive`` per arm). Survives usage stalls: on a driver
exit-20 (usage limit) it checkpoints and pauses (resumable); on exit-10 (detector trip) it parks the
arm and continues; on exit-2 (invalid) it holds. Two modes off the same core:

- **assistant-in-the-loop**: ``run_once()`` processes arms until a stall/hold, then returns its
  disposition; the caller (a watchdog notifies them on usage reset) re-invokes ``run_once()``, which
  reads the state file and RESUMES the stalled arm.
- **headless autonomous**: ``run(wait_for_reset=fn)`` loops, calling ``wait_for_reset`` on a stall.

The state machine is driven by an injected ``drive_fn(arm, resume) -> exit_code`` so unit tests
exercise it with a FAKE engine — no server, no tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .bob import Bob
from .config import EXIT_DONE, EXIT_INVALID, EXIT_LIMIT, EXIT_REVIEW
from .detector import Detector
from .engine import atomic_write
from .fixture import PersonaSpec

# Runner dispositions (what run_once returns to its caller).
RUN_COMPLETE = "complete"      # all arms done
RUN_STALLED = "stalled"        # usage limit — re-invoke after reset
RUN_HELD = "held"              # invalid/unexpected — human hold


@dataclass
class ArmSpec:
    name: str
    persona: PersonaSpec
    bob: Bob
    detector: Detector
    max_turns: int = 30
    speech_mode: str = "clean"


@dataclass
class RunnerState:
    """Persisted across re-invocations. Exactly one writer (the runner process); serial re-entry."""

    idx: int = 0
    stage: str = "setup"          # "setup" | "resume"
    parked_trips: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)


class Runner:
    def __init__(
        self,
        arms: list[ArmSpec],
        state_path: Path,
        drive_fn: Callable[[ArmSpec, bool], int],
    ) -> None:
        self.arms = arms
        self.state_path = Path(state_path)
        self.drive_fn = drive_fn

    def _load_state(self) -> RunnerState | None:
        """Load persisted state. Missing file -> a fresh RunnerState (legitimate fresh start).
        A present-but-corrupt/unreadable file -> None (REFUSE: never silently restart, G4/L1).

        Tolerant of forward-compat drift (L2): unknown keys are dropped, missing keys default —
        a state written by a newer runner loads gracefully rather than splat-crashing.
        """
        if not self.state_path.exists():
            return RunnerState()
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        known = {"idx", "stage", "parked_trips", "completed"}
        filtered = {k: v for k, v in data.items() if k in known}
        try:
            st = RunnerState(**filtered)
        except TypeError:
            return None
        if not isinstance(st.idx, int) or st.idx < 0:
            return None
        return st

    def _save_state(self, st: RunnerState) -> None:
        atomic_write(self.state_path, json.dumps(st.__dict__, indent=2))

    def run_once(self) -> str:
        """Process arms until all done, a usage stall, or a hold. Return a ``RUN_*`` disposition.

        On a fresh call after a stall, the state file makes the runner RESUME the stalled arm
        (``stage == "resume"``) rather than restart it. A missing state file is a legitimate fresh
        start (idx 0); it is NOT a corrupt-resume (the driver, not the runner, refuses to resume
        without its own per-arm checkpoint).
        """
        st = self._load_state()
        if st is None:
            # A corrupt/unparseable checkpoint: refuse rather than lose constructed context.
            return RUN_HELD
        while st.idx < len(self.arms):
            arm = self.arms[st.idx]
            resume = st.stage == "resume"
            self._save_state(st)
            code = self.drive_fn(arm, resume)
            if code == EXIT_DONE:
                st.completed.append(arm.name)
                st.idx += 1
                st.stage = "setup"
                self._save_state(st)
            elif code == EXIT_REVIEW:
                # A real detector trip: park + continue (owner policy).
                st.parked_trips.append(arm.name)
                st.completed.append(arm.name)
                st.idx += 1
                st.stage = "setup"
                self._save_state(st)
            elif code == EXIT_LIMIT:
                st.stage = "resume"
                self._save_state(st)
                return RUN_STALLED
            elif code == EXIT_INVALID:
                self._save_state(st)
                return RUN_HELD
            else:
                self._save_state(st)
                return RUN_HELD
        # Complete: clear the state file so a re-invocation starts clean.
        if self.state_path.exists():
            self.state_path.unlink()
        return RUN_COMPLETE

    def run(self, wait_for_reset: Callable[[], None] | None = None) -> str:
        """Headless mode: loop, waiting on a usage reset between stalls. Returns the final
        disposition (``RUN_COMPLETE`` or ``RUN_HELD``)."""
        while True:
            disp = self.run_once()
            if disp == RUN_STALLED and wait_for_reset is not None:
                wait_for_reset()
                continue
            return disp
