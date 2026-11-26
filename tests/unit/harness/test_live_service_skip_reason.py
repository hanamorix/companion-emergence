"""#540: the live-service skip carries its own explanation (no terminal-summary hook; it cannot work under xdist)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.harness.conftest import _skip_reason

_CONFTEST = Path(__file__).with_name("conftest.py")


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_skip_reason_names_each_live_bridge_and_says_what_to_do() -> None:
    reason = _skip_reason([(4242, "alpha"), (77, "beta")])
    assert "pid 4242 (persona alpha)" in reason and "pid 77 (persona beta)" in reason
    assert "Quit your companion bridge" in reason and "#264" in reason


@pytest.mark.no_base_failure(reason='test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code')
def test_the_guard_skips_with_that_reason_and_the_conftest_has_no_terminal_summary() -> None:
    tree = ast.parse(_CONFTEST.read_text(encoding="utf-8"))
    skips = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "pytest.skip"
    ]
    assert len(skips) == 1
    (arg,) = skips[0].args
    assert isinstance(arg, ast.Call) and ast.unparse(arg.func) == "_skip_reason"
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "pytest_terminal_summary" not in names
