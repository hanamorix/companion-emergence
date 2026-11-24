"""pytest plugin for the test-growth guard (#746): record each collected test as a JSON row.

The guard loads this with ``-p guard_test_growth_plugin`` during ``pytest --collect-only`` and reads the rows from
the file named by ``TGG_ROWS_FILE``.
"""

from __future__ import annotations

import json
import os

MARKER = "large_grid"


def _function_marks(item):
    function = getattr(item, "function", None)
    return [m for m in getattr(function, "pytestmark", []) if m.name == MARKER]


def _row(item) -> dict:
    from tests.no_base_failure import marker_problems

    callspec = getattr(item, "callspec", None)
    params = {name: repr(value) for name, value in callspec.params.items()} if callspec else {}
    function_marks = _function_marks(item)
    problems = marker_problems([item], MARKER)
    own = [m for m in getattr(item, "own_markers", []) if m.name == MARKER]
    if any(m not in function_marks for m in own):
        problems.append(f"{item.nodeid}: @pytest.mark.{MARKER} is on a pytest.param; mark the test function itself")
    return {
        "nodeid": item.nodeid,
        "params": params,
        "large_grid": [m.kwargs["reason"] for m in function_marks if isinstance(m.kwargs.get("reason"), str)],
        "problems": problems,
    }


def pytest_collection_finish(session):
    path = os.environ.get("TGG_ROWS_FILE")
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([_row(item) for item in session.items], handle)
