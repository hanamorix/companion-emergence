"""The slow-test budget (#540, J12): in CI a test's call phase may take at most 2 s unless it says why it is slow.

A slow test is not wrong, but an unmarked one is how a suite becomes slow without anyone choosing it: each new
two-second test is invisible in a diff and adds up. Where the budget applies, a test that PASSED but whose call phase took
longer than the budget is reported as FAILED with the time and the two ways out: make it faster, or mark it
`@pytest.mark.slow(reason="...")` with a reason a reviewer can weigh. The number of slow marks is pinned and can only go
down (`tests/unit/test_slow_test_budget.py`).

Where it applies: only when `KINDLED_SLOW_BUDGET` is set. `.github/workflows/test.yml` sets it for its python job (the
push and pull-request run and the manual run); the other workflows that run pytest (release gate, mutation testing,
the weekly timing report, the 3.13 bridge job) deliberately do not, and a developer sets it to try the budget locally.
`KINDLED_SLOW_BUDGET_S` overrides the 2.0 s (the planted tests use a small one).

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


def over_budget(duration: float, budget: float, *, has_slow_mark: bool, exempt: bool) -> str | None:
    """The failure message for a passing call phase that took `duration` seconds, or None when it is fine."""
    if has_slow_mark or exempt or duration <= budget:
        return None
    return (
        f"this test took {duration:.1f} s in its call phase; the budget is {budget:g} s. Make it faster (patch the wait, "
        "share the expensive setup in a fixture, shrink the data), or, if it is slow for a reason that stays, mark it "
        '@pytest.mark.slow(reason="why") on the test function. The number of slow marks is pinned and can only go down '
        "(tests/unit/test_slow_test_budget.py)."
    )
