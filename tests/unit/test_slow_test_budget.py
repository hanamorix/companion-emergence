"""The slow-test budget (#540, J12): the rule, the planted tests that show it bites, and the ratchet on the number of marks.

Planted test files run in an in-process inner pytest session with the budget switched on through the environment and
shrunk to 0.05 s of CPU so the plants stay quick (`KINDLED_SLOW_BUDGET_S`); they burn CPU in a loop (a sleep would not count).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import slow_budget
from tests.no_base_failure import marker_problems

REPO = Path(__file__).resolve().parents[2]
_REASON = "test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code"
_LOG_REASON = (
    "test-suite machinery (#790): the fix is in tests/conftest.py and tests/slow_budget.py, which the base check keeps at "
    "the PR head; before the fix these ended in an INTERNALERROR (FileNotFoundError from the patched open, IsADirectoryError) "
    "or an AttributeError (no log_min_s)"
)

# The number of `@pytest.mark.slow(...)` decorators under tests/. It can only go down: when a slow test is made fast and its
# mark removed, lower this number to the count the failure message shows. Raising it is a one-number edit that shows in the
# diff of the PR that adds a slow test, which is the point.
PINNED_SLOW_MARKS = 125


@pytest.mark.no_base_failure(reason=_REASON)
def test_the_budget_applies_only_to_an_unmarked_non_exempt_test_over_the_limit() -> None:
    assert slow_budget.over_budget(0.5, 2.0, has_slow_mark=False, exempt=False) is None
    assert slow_budget.over_budget(2.0, 2.0, has_slow_mark=False, exempt=False) is None
    message = slow_budget.over_budget(2.5, 2.0, has_slow_mark=False, exempt=False)
    assert message is not None and "2.5 s of CPU" in message and "@pytest.mark.slow" in message
    assert slow_budget.over_budget(9.0, 2.0, has_slow_mark=True, exempt=False) is None
    assert slow_budget.over_budget(9.0, 2.0, has_slow_mark=False, exempt=True) is None


@pytest.mark.no_base_failure(reason=_REASON)
def test_the_budget_is_on_only_when_asked_and_can_be_shrunk() -> None:
    assert not slow_budget.enabled({})
    assert not slow_budget.enabled({"GITHUB_ACTIONS": "true"})  # CI alone does not switch it on: test.yml sets it
    assert slow_budget.enabled({"KINDLED_SLOW_BUDGET": "1"})
    assert slow_budget.budget_s({}) == 2.0
    assert slow_budget.budget_s({"KINDLED_SLOW_BUDGET_S": "0.3"}) == 0.3
    assert slow_budget.budget_s({"KINDLED_SLOW_BUDGET_S": "nonsense"}) == 2.0


@pytest.mark.no_base_failure(reason=_LOG_REASON)
def test_the_cpu_log_threshold_is_half_a_second_and_can_be_lowered() -> None:
    """#790: the planted log tests lower it, so they need not burn half a second of CPU each."""
    assert slow_budget.log_min_s({}) == 0.5
    assert slow_budget.log_min_s({"KINDLED_SLOW_BUDGET_LOG_MIN_S": "0.05"}) == 0.05
    assert slow_budget.log_min_s({"KINDLED_SLOW_BUDGET_LOG_MIN_S": "nonsense"}) == 0.5


def _item(nodeid: str, *marks: dict, inherited: int = 0) -> SimpleNamespace:
    own = [SimpleNamespace(name="slow", kwargs=m) for m in marks]
    outer = [SimpleNamespace(name="slow", kwargs={"reason": "x"}) for _ in range(inherited)]
    return SimpleNamespace(nodeid=nodeid, own_markers=own, iter_markers=lambda name: [*own, *outer] if name == "slow" else [])


@pytest.mark.no_base_failure(reason=_REASON)
def test_a_slow_mark_needs_a_reason_and_must_sit_on_the_test_function() -> None:
    items = [_item("t::ok", {"reason": "3 s of real subprocess"}), _item("t::bare", {}), _item("t::module", inherited=1), _item("t::none")]
    problems = marker_problems(items, "slow")
    assert [p.split(": @")[0] for p in problems] == ["t::bare", "t::module"]
    assert all("pytest.mark.slow" in p for p in problems)


_INNER_CONFTEST = (
    "from tests.conftest import pytest_collection_modifyitems, pytest_runtest_call, pytest_runtest_makereport  # noqa: F401\n"
)
_INNER_INI = "[pytest]\nmarkers =\n    slow(reason): slow\n    requires_network: net\n"
_PLANTS = '''
import time

import pytest


def burn(seconds=0.08):
    start = time.process_time()
    while time.process_time() - start < seconds:
        pass


def test_fast():
    pass


def test_slow_unmarked():
    burn()


@pytest.mark.slow(reason="sleeps on purpose")
def test_slow_marked():
    burn()


@pytest.mark.requires_network
def test_slow_but_exempt():
    burn()


def test_slow_and_failing_keeps_its_own_failure():
    burn()
    assert False, "its own failure"


@pytest.mark.xfail(reason="x", strict=False)
def test_slow_xpass_is_left_alone():
    burn()


@pytest.fixture
def slow_setup():
    burn()
    yield
    burn()


def test_slow_setup_and_teardown_are_not_the_call_phase(slow_setup):
    pass


def test_a_long_wait_is_not_cpu_time():
    time.sleep(0.25)
'''


@pytest.mark.no_base_failure(reason=_REASON)
def test_with_the_budget_on_only_the_unmarked_slow_passing_test_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeini(_INNER_INI)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_PLANTS)
    # the environment is changed only around the inner session: this test's own call phase is judged by the outer budget
    with monkeypatch.context() as m:
        m.setenv("KINDLED_SLOW_BUDGET", "1")
        m.setenv("KINDLED_SLOW_BUDGET_S", "0.05")
        result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:tdd-guard", "-rf")
    result.assert_outcomes(passed=5, failed=2, xpassed=1)
    result.stdout.fnmatch_lines(
        [
            "*::test_slow_unmarked: this test used * s of CPU in its call phase; the budget is 0.05 s.*",
            "*AssertionError: its own failure*",
        ]
    )
    result.stdout.fnmatch_lines(["FAILED *::test_slow_unmarked*", "FAILED *::test_slow_and_failing_keeps_its_own_failure*"])
    assert "test_slow_marked" not in "\n".join(line for line in result.outlines if line.startswith("FAILED"))


@pytest.mark.no_base_failure(reason=_REASON)
def test_with_the_budget_off_everything_passes(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    pytester.makeini(_INNER_INI)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_PLANTS)
    with monkeypatch.context() as m:
        m.delenv("KINDLED_SLOW_BUDGET", raising=False)
        m.setenv("KINDLED_SLOW_BUDGET_S", "0.05")
        result = pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:tdd-guard")
    result.assert_outcomes(passed=6, failed=1, xpassed=1)  # only the test that fails by itself


_LOG_PLANTS = '''
import builtins
import time


def burn(seconds=0.08):
    start = time.process_time()
    while time.process_time() - start < seconds:
        pass


def test_patches_open_and_burns_cpu(monkeypatch):
    def unknown_path(*args, **kwargs):
        raise FileNotFoundError(args[0] if args else "?")

    monkeypatch.setattr(builtins, "open", unknown_path)  # like the cgroup fakes in test_reranker.py
    burn()
'''


def _run_logged(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, log: Path) -> pytest.RunResult:
    """The plant in an inner session, logging every call phase over 0.05 s of CPU to `log`, the budget itself off."""
    pytester.makeini(_INNER_INI)
    pytester.makeconftest(_INNER_CONFTEST)
    pytester.makepyfile(_LOG_PLANTS)
    with monkeypatch.context() as m:
        m.delenv("KINDLED_SLOW_BUDGET", raising=False)
        m.setenv("KINDLED_SLOW_BUDGET_LOG", str(log))
        m.setenv("KINDLED_SLOW_BUDGET_LOG_MIN_S", "0.05")
        return pytester.runpytest("-p", "no:cacheprovider", "-p", "no:randomly", "-p", "no:tdd-guard")


@pytest.mark.no_base_failure(reason=_LOG_REASON)
def test_a_test_that_patches_open_cannot_break_the_cpu_log(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """#790: the log is written while the test's monkeypatch is still on. Through a patched `builtins.open` the write
    raised, pytest reported an INTERNALERROR and the rest of the CI shard never ran."""
    log = pytester.path / "cpu.jsonl"
    result = _run_logged(pytester, monkeypatch, log)
    result.assert_outcomes(passed=1)
    assert result.ret == 0
    [line] = log.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["id"].endswith("::test_patches_open_and_burns_cpu")


@pytest.mark.no_base_failure(reason=_LOG_REASON)
def test_a_cpu_log_that_cannot_be_written_warns_and_the_run_goes_on(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """#790: a measurement log may not end the session. A path that cannot be opened (here a folder) is a warning."""
    log = pytester.mkdir("not-a-file")
    result = _run_logged(pytester, monkeypatch, log)
    result.assert_outcomes(passed=1, warnings=1)
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*the slow-test CPU log * could not be written*test_patches_open_and_burns_cpu*"])


def _slow_mark_scan() -> tuple[int, list[str]]:
    """(decorators of the form `@pytest.mark.slow(...)` directly on a function under tests/, other uses of `mark.slow`)."""
    direct = 0
    elsewhere: list[str] = []
    this_file = Path(__file__).resolve()
    for path in sorted((REPO / "tests").rglob("*.py")):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        if "mark.slow" not in text:
            continue
        tree = ast.parse(text)
        allowed: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    if _is_slow_call(dec):
                        allowed.add(id(dec.func))
                        direct += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "slow" and _is_mark(node.value) and id(node) not in allowed:
                elsewhere.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno}")
    return direct, elsewhere


def _is_mark(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "mark" and isinstance(node.value, ast.Name) and node.value.id == "pytest"


def _is_slow_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "slow" and _is_mark(node.func.value)


@pytest.fixture(scope="module")
def slow_mark_scan() -> tuple[int, list[str]]:
    """The scan of the real tests/ tree, run once in setup: reading every file and parsing those that mention `mark.slow`
    grows with the tree and once used 2.5 s of CPU on CI, which belongs to no one test (the call-phase budget, #787)."""
    return _slow_mark_scan()


@pytest.mark.no_base_failure(reason=_REASON)
def test_slow_marks_are_only_function_decorators_and_their_number_never_goes_up(slow_mark_scan: tuple[int, list[str]]) -> None:
    count, elsewhere = slow_mark_scan
    assert not elsewhere, f"mark a slow test with a decorator on the function itself, not like this: {elsewhere}"
    assert count <= PINNED_SLOW_MARKS, (
        f"{count} `@pytest.mark.slow` decorators, the pin is {PINNED_SLOW_MARKS}: a new slow test needs the pin raised in this file "
        "(a visible, reviewable edit); first try to make the test fast"
    )
    assert count == PINNED_SLOW_MARKS, f"{count} slow marks now: lower PINNED_SLOW_MARKS in {Path(__file__).name} to {count} (the pin only goes down)"


@pytest.mark.no_base_failure(
    reason="the change is in this test file, which the base check keeps at the PR head; against main's version of the "
    "file it fails (the test there takes no fixture and calls _slow_mark_scan() itself, #787)"
)
def test_the_slow_mark_scan_runs_in_setup_not_in_the_budgeted_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """#787: the scan reads every file under tests/ and parses those that mention `mark.slow`, which grows with the tree
    and used 2.5 s of the 2 s call-phase budget on a loaded runner. Setup is not budgeted, so the scan belongs in a module
    fixture, as the Windows-pitfalls lint's does (#665); the test itself only checks the result."""
    def scanned_in_the_call() -> tuple[int, list[str]]:
        raise AssertionError("the test called _slow_mark_scan() itself, in its budgeted call phase")

    monkeypatch.setattr(sys.modules[__name__], "_slow_mark_scan", scanned_in_the_call)
    test_slow_marks_are_only_function_decorators_and_their_number_never_goes_up(slow_mark_scan=(PINNED_SLOW_MARKS, []))
