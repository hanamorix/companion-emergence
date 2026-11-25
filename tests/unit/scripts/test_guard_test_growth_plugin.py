"""The test-growth guard's collection plugin: one JSON row per collected test (#746)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

SAMPLE = textwrap.dedent(
    """
    import pytest
    from pytest import mark

    parametrize = mark.parametrize


    def test_plain():
        pass


    @parametrize("a", [1, 2])
    @parametrize("b", [1, 2, 3])
    def test_cross(a, b):
        pass


    @pytest.mark.large_grid(reason="r")
    @parametrize("x", [1, 2])
    def test_justified(x):
        pass


    @pytest.mark.large_grid
    @parametrize("x", [1])
    def test_no_reason(x):
        pass


    @parametrize("x", [pytest.param(1, marks=pytest.mark.large_grid(reason="r"))])
    def test_param_mark(x):
        pass


    @parametrize("o", [object(), object()])
    def test_obj(o):
        pass


    @parametrize(
        "kind,path",
        [("a", "x"), ("a", "y"), ("b", "x"), ("b", "z"), ("c", "y"), ("c", "z"), ("a", "z")],
    )
    def test_table(kind, path):
        pass
    """
)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    root = tmp_path_factory.mktemp("tgg")
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "no_base_failure.py").write_text(
        (REPO / "tests" / "no_base_failure.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "tests" / "test_sample.py").write_text(SAMPLE, encoding="utf-8")
    out = root / "rows.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), str(REPO / "scripts")]),
        "TGG_ROWS_FILE": str(out),
    }
    subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "guard_test_growth_plugin", "tests/test_sample.py"],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(out.read_text(encoding="utf-8"))


def test_rows_record_nodeid_and_params(rows):
    by_id = {r["nodeid"]: r for r in rows}
    assert by_id["tests/test_sample.py::test_plain"]["params"] == {}
    # Stacked parametrize ids list the decorator nearest the function first, so a=1, b=3 is "[3-1]".
    assert by_id["tests/test_sample.py::test_cross[3-1]"]["params"] == {"a": "1", "b": "3"}
    assert len([r for r in rows if "test_cross[" in r["nodeid"]]) == 6


def test_large_grid_reason_on_function_is_recorded(rows):
    justified = [r for r in rows if "test_justified[" in r["nodeid"]]
    assert len(justified) == 2
    for row in justified:
        assert row["large_grid"] == ["r"]
        assert row["problems"] == []


def test_large_grid_without_reason_is_a_problem(rows):
    (row,) = [r for r in rows if "test_no_reason[" in r["nodeid"]]
    assert row["large_grid"] == []
    assert any("needs reason=" in p for p in row["problems"])


def test_large_grid_on_param_is_a_problem(rows):
    (row,) = [r for r in rows if "test_param_mark[" in r["nodeid"]]
    assert any("is on a pytest.param" in p for p in row["problems"])


def test_each_parametrize_decorator_is_recorded_as_one_group_of_names(rows):
    """One decorator over a written-out table is one axis, however many names it sets (ThinkerOfThoughts on #806)."""
    (table, *_) = [r for r in rows if "test_table[" in r["nodeid"]]
    assert table["groups"] == [["kind", "path"]]
    (cross, *_) = [r for r in rows if "test_cross[" in r["nodeid"]]
    assert sorted(cross["groups"]) == [["a"], ["b"]]


def test_object_params_stay_distinct(rows):
    first, second = [r for r in rows if "test_obj[" in r["nodeid"]]
    assert first["params"]["o"] != second["params"]["o"]
