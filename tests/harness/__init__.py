"""companion-emergence behavioral-test harness.

A permanent, sandboxed framework for behavioral tests that drive the REAL engine:
seed a throwaway persona ("Canary") -> stand up the real bridge -> drive it with an
LLM-simulated human ("Bob") -> detect symptoms in the replies -> orchestrate multi-arm
runs that survive usage-limit stalls. Generalizes the ad-hoc harness built during the
monologue-bleed hunt. Can exercise ~anything except the GUI.

Design goal #1 is STRONG SANDBOX ISOLATION: the harness runs on developer laptops where a
real companion lives, so nothing it does may touch anything outside its temp sandbox. Every
run goes through `sandbox()` (see `sandbox.py`), which redirects KINDLED_HOME + CLAUDE_CONFIG_DIR,
disables the persona's external write/phone-home paths, and asserts post-run that no guarded
root outside the sandbox was mutated.

Public API is exported here as the modules land (see README.md for status + usage).
"""

__all__: list[str] = []
