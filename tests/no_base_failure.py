"""The ``no_base_failure`` marker's one rule: it must say why (#540).

``scripts/new_tests_fail_on_base.py`` runs the tests a pull request adds against the code the PR started from and
requires each to fail there. A test that passes on the old code by design (a refactor's guard, test infrastructure, a
snapshot of what already exists) carries ``@pytest.mark.no_base_failure(reason="...")`` and is exempt, counted and
listed in the check's output. A mark without a reason would be a silent opt-out, and a mark on a module or class would
exempt every test added there later, so the collection hook in ``tests/conftest.py`` refuses both.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MARKER = "no_base_failure"


def marker_problems(items: Iterable[Any], marker: str = MARKER) -> list[str]:
    """One line per item whose ``marker`` mark (default ``no_base_failure``) is not on the test itself, or has no non-empty ``reason=``.

    A mark on a module or class would exempt every test added to it later without anyone choosing that, so it is
    refused: mark the test function (or the ``pytest.param``). ``own_markers`` holds only the marks applied to the
    item's own function; ``iter_markers`` also walks the class and module.
    """
    problems = []
    for item in items:
        own = [m for m in getattr(item, "own_markers", []) if m.name == marker]
        if len(list(item.iter_markers(marker))) > len(own):
            problems.append(f"{item.nodeid}: @pytest.mark.{marker} is on a module or class; mark the test function itself")
            continue
        for mark in own:
            reason = mark.kwargs.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                problems.append(f"{item.nodeid}: @pytest.mark.{marker} needs reason=\"why\"")
                break
    return problems
