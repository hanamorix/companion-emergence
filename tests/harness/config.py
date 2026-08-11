"""Framework configuration: the model toggle + standing defaults/timeouts.

`ModelConfig` is the single place the harness names which model each actor uses. It threads
through the persona fixture (Canary), DumbBob's ``claude -p --model`` argv, and the usage
watchdog ping — replacing the hardcoded ``sonnet``/``haiku`` scattered through the ad-hoc hunt
harness. Set once per run, overridable per-persona (``PersonaSpec.model``) or per-arm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Which model each actor uses. Overridable per-run / per-persona / per-arm.

    - ``canary`` — the persona under test (threads to ``write_persona_config(model=...)``).
    - ``bob`` — the LLM-simulated human (threads to DumbBob's ``claude -p --model``).
    - ``watchdog`` — the usage-reset ping model (cheap; threads to the watchdog).
    """

    canary: str = "sonnet"
    bob: str = "sonnet"
    watchdog: str = "haiku"


@dataclass(frozen=True)
class Timeouts:
    """Framework timeouts (seconds). Defaults mirror the hunt harness's proven values."""

    bob_call: float = 90.0
    ws_drive: float = 300.0
    ws_open: float = 30.0
    watchdog_ping: float = 45.0
    watchdog_normal_s: float = 1200.0  # 20 min baseline cadence
    watchdog_tight_s: float = 300.0    # 5 min after a failed ping


# The synthetic actors' fixed names. NEVER a real person's name in a fixture/harness.
SYNTHETIC_USER = "Bob"
PERSONA_NAME = "Canary"

# Synthetic identity used to de-identify the `claude` CLI's auto-injected context so it cannot leak the
# REAL account/project into the companion's STORED channels (memories.db, works/*.md, active-recall) at
# runtime. Single source of truth for the stored-channel de-id seal in ``sandbox.py`` (the next increment
# after the F1/F2 input-side $USER/$LOGNAME de-id). NEVER a real name/email/org/repo here.
SYNTHETIC_DISPLAY_NAME = SYNTHETIC_USER          # the CLI account display name (== "Bob")
SYNTHETIC_EMAIL = "bob@example.invalid"          # RFC-2606 reserved TLD → unmistakably synthetic
SYNTHETIC_ORG = "Bob's Test Org"
SYNTHETIC_PROJECT = "workspace"                  # neutral scratch-dir / project name (carrier A)
SYNTHETIC_ACCOUNT_UUID = "00000000-0000-4000-8000-000000000001"
SYNTHETIC_ORG_UUID = "00000000-0000-4000-8000-000000000002"

# Exit-code contract shared by the engine driver + runner (ported from the hunt drivers).
EXIT_DONE = 0
EXIT_REVIEW = 10   # a detector trip — pause for adjudication
EXIT_LIMIT = 20    # usage/session limit — checkpoint + resumable
EXIT_INVALID = 2   # the run is structurally broken (bridge errors, bad state)

DEFAULT_MODELS = ModelConfig()
DEFAULT_TIMEOUTS = Timeouts()
