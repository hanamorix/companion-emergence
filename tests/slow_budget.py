"""The slow-test budget (#540, J12): in CI (the ubuntu job) a test's call phase may use at most 2 s of CPU unless it says why.

A slow test is not wrong, but an unmarked one is how a suite becomes slow without anyone choosing it: each new
two-second test is invisible in a diff and adds up. Where the budget applies, a test that PASSED but whose call phase used
more than the budget in CPU time is reported as FAILED with the time and the two ways out: make it faster, or mark it
`@pytest.mark.slow(reason="...")` with a reason a reviewer can weigh. The number of slow marks is pinned and can only go
down (`tests/unit/test_slow_test_budget.py`).

CPU time, not wall-clock time, on purpose. CI runs the suite in parallel on 3-4 core runners; there the wall-clock time of
a test is dominated by contention (the first CI run of this budget failed unmarked tests that take 10 to 300 ms alone,
at 2 to 6 s), so a wall-clock budget would fail innocent tests at random. The CPU time the process uses during the call
phase (all its threads, not its child processes) does not grow when the test waits for a core. What this does not catch is
a test that is slow because it WAITS (a fixed sleep, a subprocess it waits for): its wall time is shown in the weekly
timing report (`.github/workflows/timing.yml`), not gated here.

Where it applies: only when `KINDLED_SLOW_BUDGET` is set. `.github/workflows/test.yml` sets it for its python job on
UBUNTU only (the push and pull-request run and the manual run): CPU time is not contention-proof either, and on the
shared macOS and Windows runners a fixed line failed unmarked tests at random (the same code's median test used 2.1x
the CPU between two runs, the worst 14x). The CPU time of every test over 0.5 s is logged on all three systems
(`KINDLED_SLOW_BUDGET_LOG`, uploaded as `slow-budget-cpu-<os>`) whether or not the budget is enforced there. The other
workflows that run pytest (release gate, mutation testing, the weekly timing report, the 3.13 bridge job) do not enforce
it, and a developer sets the variable to try the budget locally.
`KINDLED_SLOW_BUDGET_S` overrides the 2.0 s (the planted tests use a small one). `KINDLED_SLOW_BUDGET_LOG=<file>` appends
one JSON line per test that used more than 0.5 s of CPU (node id, CPU seconds, wall seconds): the data the marks come from.

What is not budgeted: the setup and teardown phases (a fixture that does the heavy work, such as loading a model, is
not a slow test); a test that failed, skipped or xfailed (its own result stands); a test that carries a marker saying it
depends on something outside the process: `requires_network`, `requires_models`, `requires_claude_cli`, `live`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

BUDGET_S = 2.0
EXEMPT_MARKERS = ("requires_network", "requires_models", "requires_claude_cli", "live")


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return bool((os.environ if environ is None else environ).get("KINDLED_SLOW_BUDGET"))


def budget_s(environ: Mapping[str, str] | None = None) -> float:
    raw = (os.environ if environ is None else environ).get("KINDLED_SLOW_BUDGET_S")
    try:
        return float(raw) if raw else BUDGET_S
    except ValueError:
        return BUDGET_S


def over_budget(cpu_seconds: float, budget: float, *, has_slow_mark: bool, exempt: bool) -> str | None:
    """The failure message for a passing call phase that used `cpu_seconds` of CPU, or None when it is fine."""
    if has_slow_mark or exempt or cpu_seconds <= budget:
        return None
    return (
        f"this test used {cpu_seconds:.1f} s of CPU in its call phase; the budget is {budget:g} s. Make it faster (share "
        "the expensive setup in a fixture, shrink the data, patch the heavy call), or, if it is slow for a reason that stays, mark it "
        '@pytest.mark.slow(reason="why") on the test function. The number of slow marks is pinned and can only go down '
        "(tests/unit/test_slow_test_budget.py)."
    )
