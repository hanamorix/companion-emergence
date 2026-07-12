"""companion-emergence live-test harness.

A permanent, sandboxed framework for live tests that drive the REAL engine:
seed a throwaway persona ("Canary") -> stand up the real bridge -> drive it with an
LLM-simulated human ("Bob") -> detect symptoms in the replies -> orchestrate multi-arm
runs that survive usage-limit stalls. Generalizes an ad-hoc harness built during a prior
bug hunt. Can exercise ~anything except the GUI.

Design goal #1 is STRONG SANDBOX ISOLATION: the harness runs on developer laptops where a
real companion lives, so nothing it does may touch anything outside its temp sandbox. Every
run goes through `sandbox()` (see `sandbox.py`), which redirects KINDLED_HOME + CLAUDE_CONFIG_DIR,
disables the persona's external write/phone-home paths, and asserts post-run that no guarded
root outside the sandbox was mutated.

Public API is exported here as the modules land (see README.md for status + usage).
"""

from .bob import (
    AGENT_EFFORT,
    AgentBob,
    AgentSpawnSpec,
    Bob,
    BobContext,
    BobTurn,
    DumbBob,
    is_usage_limit,
)
from .config import (
    DEFAULT_MODELS,
    DEFAULT_TIMEOUTS,
    EXIT_DONE,
    EXIT_INVALID,
    EXIT_LIMIT,
    EXIT_REVIEW,
    PERSONA_NAME,
    SYNTHETIC_USER,
    ModelConfig,
    Timeouts,
)
from .detector import (
    DEFAULT_USER_NAME,
    CompositeDetector,
    Detector,
    DetectorGateError,
    Score,
    TurnContext,
    assert_detector_gate,
)
from .engine import BridgeServer, atomic_write, collect_reply, drive_ws, parse_ws_frame
from .fixture import LiveEnv, MemorySeed, PersonaSpec, build_persona
from .incident import IncidentResult, IncidentSpec, build_compacted_state
from .runner import ArmSpec, Runner, RunnerState
from .sandbox import (
    HARNESS_EDITABLE_SENTINEL,
    LIVE_CHECK_OFF,
    LIVE_CHECK_RAISE,
    LIVE_CHECK_WARN,
    EditablePathRefused,
    LiveServiceDetected,
    SandboxHandle,
    SandboxLeak,
    sandbox,
)
from .speech import CLEAN, REALISTIC, dyslexify
from .watchdog import Watchdog, real_ping_fn, watchdog_ping_argv

__all__: list[str] = [
    # sandbox (safety core)
    "sandbox", "SandboxHandle", "SandboxLeak", "LiveServiceDetected",
    "EditablePathRefused", "HARNESS_EDITABLE_SENTINEL",
    "LIVE_CHECK_RAISE", "LIVE_CHECK_WARN", "LIVE_CHECK_OFF",
    # config
    "ModelConfig", "Timeouts", "DEFAULT_MODELS", "DEFAULT_TIMEOUTS",
    "SYNTHETIC_USER", "PERSONA_NAME",
    "EXIT_DONE", "EXIT_REVIEW", "EXIT_LIMIT", "EXIT_INVALID",
    # fixture / incident
    "PersonaSpec", "MemorySeed", "LiveEnv", "build_persona",
    "IncidentSpec", "IncidentResult", "build_compacted_state",
    # bob
    "Bob", "BobTurn", "BobContext", "DumbBob", "AgentBob", "AgentSpawnSpec", "is_usage_limit",
    "AGENT_EFFORT",
    # speech
    "dyslexify", "CLEAN", "REALISTIC",
    # detector
    "Detector", "Score", "TurnContext", "assert_detector_gate", "DetectorGateError",
    "CompositeDetector", "DEFAULT_USER_NAME",
    # engine
    "BridgeServer", "atomic_write", "parse_ws_frame", "collect_reply", "drive_ws",
    # runner / watchdog
    "ArmSpec", "Runner", "RunnerState", "Watchdog", "real_ping_fn", "watchdog_ping_argv",
]
