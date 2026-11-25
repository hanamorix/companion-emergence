"""The counts and grid rules of the test-growth guard (#746): pure functions, tested directly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "guard_test_growth_rules", REPO / "scripts" / "guard_test_growth_rules.py"
)
rules = importlib.util.module_from_spec(_spec)
sys.modules["guard_test_growth_rules"] = rules
_spec.loader.exec_module(rules)


def test_count_files_counts_functions_ids_and_changes():
    base = {"tests/test_a.py::t[1]", "tests/test_a.py::old"}
    head = {"tests/test_a.py::t[1]", "tests/test_a.py::t[2]", "tests/test_a.py::u"}
    got = rules.count_files(["tests/test_a.py"], base, head)
    assert got == [rules.FileCount("tests/test_a.py", 2, 2, 2, 3, 2, 1)]


def test_count_files_new_file_has_zero_base_counts():
    got = rules.count_files(["tests/test_n.py"], set(), {"tests/test_n.py::t[1]", "tests/test_n.py::t[2]"})
    assert got == [rules.FileCount("tests/test_n.py", 0, 1, 0, 2, 2, 0)]


def test_count_files_deleted_file_has_zero_head_counts():
    got = rules.count_files(["tests/test_d.py"], {"tests/test_d.py::t"}, set())
    assert got == [rules.FileCount("tests/test_d.py", 1, 0, 1, 0, 0, 1)]


def test_is_cross_product_true_when_ids_exceed_sum_of_axes():
    assert rules.is_cross_product(6, {"a": 2, "b": 3}) is True


def test_is_cross_product_false_for_two_by_two_with_four_ids():
    assert rules.is_cross_product(4, {"a": 2, "b": 2}) is False


def test_is_cross_product_false_for_zipped_tuple_axes():
    assert rules.is_cross_product(3, {"a": 3, "b": 3}) is False


def test_single_axis_over_budget_when_new_ids_exceed_budget():
    assert rules.is_single_axis_over_budget(3, {"x": 3}, 2) is True


def test_single_axis_within_budget_is_not_flagged():
    assert rules.is_single_axis_over_budget(2, {"x": 2}, 2) is False


def test_single_id_is_not_a_grid():
    assert rules.is_single_axis_over_budget(1, {"x": 1}, 0) is False


def test_two_axes_are_not_a_single_axis_grid():
    assert rules.is_single_axis_over_budget(6, {"a": 2, "b": 3}, 0) is False


def test_axes_counts_distinct_reprs_per_param_name():
    rows = [
        {"nodeid": "t[1-x]", "params": {"a": "1", "b": "'x'"}},
        {"nodeid": "t[1-y]", "params": {"a": "1", "b": "'y'"}},
        {"nodeid": "t[2-x]", "params": {"a": "2", "b": "'x'"}},
    ]
    assert rules.axes(rows) == {"a": 2, "b": 2}


def test_is_product_path_true_for_brain_and_app_sources():
    assert rules.is_product_path("brain/x.py") is True
    assert rules.is_product_path("app/src/a.tsx") is True


def test_is_product_path_false_for_tests_docs_and_test_files():
    assert rules.is_product_path("tests/unit/x.py") is False
    assert rules.is_product_path("app/src/a.test.tsx") is False
    assert rules.is_product_path("app/src/__tests__/b.ts") is False
    assert rules.is_product_path("docs/x.md") is False


def test_product_lines_counts_non_blank_changes_in_product_files_only():
    diff = (
        "diff --git a/brain/x.py b/brain/x.py\n"
        "index 111..222 100644\n"
        "--- a/brain/x.py\n"
        "+++ b/brain/x.py\n"
        "@@ -3 +3,3 @@\n"
        "-old = 1\n"
        "+new = 1\n"
        "+\n"
        "+other = 2\n"
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
        "index 333..444 100644\n"
        "--- a/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+a\n+b\n+c\n+d\n+e\n"
    )
    assert rules.product_lines(diff) == 3


def test_product_lines_counts_removed_lines_of_a_deleted_file():
    diff = (
        "diff --git a/brain/gone.py b/brain/gone.py\n"
        "deleted file mode 100644\n"
        "index 555..000\n"
        "--- a/brain/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-one\n"
        "-two\n"
    )
    assert rules.product_lines(diff) == 2


def test_product_lines_counts_hunk_content_that_looks_like_a_header():
    diff = (
        "diff --git a/brain/x.py b/brain/x.py\n"
        "index 111..222 100644\n"
        "--- a/brain/x.py\n"
        "+++ b/brain/x.py\n"
        "@@ -1 +1 @@\n"
        "--- a/not/a/header.py\n"
        "+++ b/not/a/header.py\n"
    )
    assert rules.product_lines(diff) == 2


def test_product_lines_handles_a_c_quoted_non_ascii_path():
    diff = (
        'diff --git "a/\\303\\251.py" "b/\\303\\251.py"\n'
        "index 111..222 100644\n"
        '--- "a/\\303\\251.py"\n'
        '+++ "b/\\303\\251.py"\n'
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert rules.product_lines(diff) == 2


def test_product_lines_handles_a_path_with_a_space_and_trailing_tab():
    diff = (
        "diff --git a/x y.py b/x y.py\n"
        "index 111..222 100644\n"
        "--- a/x y.py\t\n"
        "+++ b/x y.py\t\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert rules.product_lines(diff) == 2


def test_scanner_tests_flags_a_glob_fed_parametrize():
    src = (
        "import pytest\n"
        "from pathlib import Path\n"
        "@pytest.mark.parametrize('p', sorted(Path('x').glob('*.py')))\n"
        "def test_a(p):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_flags_each_scanner_source():
    sources = [
        "os.walk('.')",
        "os.listdir('.')",
        "Path('.').iterdir()",
        "Path('.').rglob('*.py')",
        "subprocess.run(['git', 'ls-files'])",
    ]
    for values in sources:
        src = f"@pytest.mark.parametrize('p', {values})\ndef test_a(p):\n    pass\n"
        assert rules.scanner_tests(src) == {"test_a"}, values


def test_scanner_tests_ignores_a_literal_list():
    src = "@pytest.mark.parametrize('n', [1, 2, 3])\ndef test_a(n):\n    pass\n"
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_resolves_a_module_level_name():
    src = (
        "FILES = list(Path('.').rglob('*.json'))\n"
        "@pytest.mark.parametrize('f', FILES)\n"
        "def test_a(f):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_resolves_an_annotated_module_level_name():
    src = (
        "FILES: list[Path] = list(Path('.').glob('*'))\n"
        "@pytest.mark.parametrize('f', FILES)\n"
        "def test_a(f):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_recognises_a_bare_parametrize_name():
    src = (
        "from pytest.mark import parametrize\n"
        "@parametrize('p', os.listdir('.'))\n"
        "def test_a(p):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_resolves_a_class_body_constant():
    src = (
        "class TestT:\n"
        "    F = list(Path('.').glob('*'))\n"
        "    @pytest.mark.parametrize('p', F)\n"
        "    def test_m(self, p):\n"
        "        pass\n"
    )
    assert rules.scanner_tests(src) == {"TestT::test_m"}


def test_scanner_tests_resolves_a_same_file_function_call():
    src = (
        "def _discover():\n"
        "    return os.listdir('.')\n"
        "@pytest.mark.parametrize('f', _discover())\n"
        "def test_a(f):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_does_not_follow_an_imported_function():
    src = (
        "from helpers import discover\n"
        "@pytest.mark.parametrize('f', discover())\n"
        "def test_a(f):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_names_a_method_with_its_class():
    src = (
        "from pytest import mark\n"
        "class TestX:\n"
        "    @mark.parametrize('p', Path('.').glob('*'))\n"
        "    def test_m(self, p):\n"
        "        pass\n"
    )
    assert rules.scanner_tests(src) == {"TestX::test_m"}


def test_scanner_tests_reads_the_argvalues_keyword():
    src = "@pytest.mark.parametrize('p', argvalues=os.listdir('.'))\ndef test_a(p):\n    pass\n"
    assert rules.scanner_tests(src) == {"test_a"}


def test_scanner_tests_treats_a_literal_list_as_never_a_scan():
    src = "@pytest.mark.parametrize('p', ['ls-files', 'glob'])\ndef test_a(p):\n    pass\n"
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_treats_a_name_bound_to_a_literal_list_as_never_a_scan():
    src = (
        "N = ['git ls-files', 'git status']\n"
        "@pytest.mark.parametrize('x', N)\n"
        "def test_a(x):\n"
        "    pass\n"
        "@pytest.mark.parametrize('x', list(N))\n"
        "def test_b(x):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_ignores_an_ls_files_string_that_is_not_a_call_argument():
    src = (
        "CASES = {'git ls-files': 1, 'git status': 2}\n"
        "@pytest.mark.parametrize('x', sorted(CASES))\n"
        "def test_a(x):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_ignores_ast_walk_and_other_walk_methods():
    src = (
        "TREE = ast.parse('x = 1')\n"
        "@pytest.mark.parametrize('n', [n for n in ast.walk(TREE) if n])\n"
        "def test_a(n):\n"
        "    pass\n"
        "@pytest.mark.parametrize('n', list(graph.walk()))\n"
        "def test_b(n):\n"
        "    pass\n"
    )
    assert rules.scanner_tests(src) == set()


def test_scanner_tests_flags_a_bare_walk_and_an_ls_files_shell_string():
    for values in ("list(walk('.'))", "check_output('git ls-files tests', shell=True).split()"):
        src = f"@pytest.mark.parametrize('p', {values})\ndef test_a(p):\n    pass\n"
        assert rules.scanner_tests(src) == {"test_a"}, values


_DUP_A = (
    "def test_a():\n"
    '    """First."""\n'
    "    box = make(1, 2, name='x')\n"
    "    box.fill(3)\n"
    "    assert box.total == 6\n"
    "    assert box.name == 'x'\n"
)
_DUP_B = (
    "def test_b():\n"
    "    crate = make(7, 8, name='yy')\n"
    "    crate.fill(9)\n"
    "    assert crate.total == 24\n"
    "    assert crate.name == 'yy'\n"
)
_DUP_C = (
    "def test_c():\n"
    "    for item in load():\n"
    "        if item.ok:\n"
    "            print(item)\n"
    "        else:\n"
    "            raise ValueError(item.reason)\n"
)


def test_likely_duplicates_pairs_tests_that_differ_only_in_names_and_values():
    a = rules.test_bodies(_DUP_A)["test_a"]
    b = rules.test_bodies(_DUP_B)["test_b"]
    got = rules.likely_duplicates({"x.py::test_a": a}, {"y.py::test_b": b})
    assert len(got) == 1
    assert got[0][:2] == ("x.py::test_a", "y.py::test_b")
    assert got[0][2] >= 0.9


def test_likely_duplicates_ignores_tests_with_different_call_structure():
    a = rules.test_bodies(_DUP_A)["test_a"]
    c = rules.test_bodies(_DUP_C)["test_c"]
    assert rules.likely_duplicates({"x.py::test_a": a}, {"y.py::test_c": c}) == []


def test_likely_duplicates_skips_bodies_under_the_token_minimum_even_when_identical():
    short = rules.test_bodies("def test_s():\n    pass\n")["test_s"]
    assert len(short) < 20
    assert rules.likely_duplicates({"x.py::test_s": short}, {"y.py::test_s": short}) == []


def test_likely_duplicates_counts_source_tokens_so_a_two_line_body_is_skipped():
    src = "def test_p():\n    x = f(1)\n    assert x == 2\n"
    short = rules.test_bodies(src)["test_p"]
    assert len(short) < 20
    assert rules.likely_duplicates({"x.py::test_p": short}, {"y.py::test_p": short}) == []


def test_likely_duplicates_never_pairs_a_test_with_itself_or_repeats_a_pair():
    a = rules.test_bodies(_DUP_A)["test_a"]
    b = rules.test_bodies(_DUP_B)["test_b"]
    both = {"x.py::test_a": a, "y.py::test_b": b}
    got = rules.likely_duplicates(both, both)
    assert [t[:2] for t in got] == [("x.py::test_a", "y.py::test_b")]


def test_test_bodies_drops_the_docstring():
    plain = "def test_a():\n    box = make(1)\n    assert box.total == 1\n"
    documented = 'def test_a():\n    """Checks the total."""\n    box = make(1)\n    assert box.total == 1\n'
    assert rules.test_bodies(documented) == rules.test_bodies(plain)


def test_test_bodies_names_a_method_with_its_class_and_includes_async_tests():
    src = (
        "class TestX:\n"
        "    def test_m(self):\n"
        "        pass\n"
        "    def helper(self):\n"
        "        pass\n"
        "async def test_n():\n"
        "    pass\n"
        "def not_a_test():\n"
        "    pass\n"
    )
    assert set(rules.test_bodies(src)) == {"TestX::test_m", "test_n"}


_GRID_FN = "tests/unit/test_x.py::test_grid"


def _grid_rows(*, large_grid=None):
    rows = []
    for a in ("a1", "a2"):
        for b in ("b1", "b2", "b3"):
            rows.append(
                {
                    "nodeid": f"{_GRID_FN}[{a}-{b}]",
                    "params": {"a": a, "b": b},
                    "large_grid": list(large_grid or []),
                    "problems": [],
                }
            )
    return rows


def _report(rows, *, new_ids=None, scanners=None, duplicates=None, budget=0):
    ids = {r["nodeid"] for r in rows} if new_ids is None else new_ids
    return rules.build_report(
        counts=[],
        head_rows=rows,
        new_ids=ids,
        scanners=scanners or set(),
        budget=budget,
        duplicates=duplicates or [],
    )


def test_build_report_flags_a_new_two_by_three_function_as_cross_product():
    report = _report(_grid_rows())
    assert [(f.function, f.rule) for f in report.flags] == [(_GRID_FN, "cross-product")]
    assert "6 ids" in report.flags[0].detail
    assert "sum 5" in report.flags[0].detail
    assert report.exempt == []


def test_build_report_moves_a_large_grid_function_to_exempt_with_its_reason():
    report = _report(_grid_rows(large_grid=["full matrix is the point"]))
    assert report.flags == []
    assert [(f.function, f.rule, why) for f, why in report.exempt] == [
        (_GRID_FN, "cross-product", "full matrix is the point")
    ]


def test_build_report_never_flags_a_function_with_no_new_ids():
    report = _report(_grid_rows(), new_ids=set(), scanners={_GRID_FN})
    assert report.flags == []
    assert report.exempt == []


_SCAN_FN = "tests/unit/test_x.py::test_s"


def _scan_rows():
    return [{"nodeid": f"{_SCAN_FN}[a]", "params": {}, "large_grid": [], "problems": []}]


def test_build_report_flags_a_new_scanner_function():
    report = _report(_scan_rows(), scanners={_SCAN_FN})
    assert [(f.function, f.rule) for f in report.flags] == [(_SCAN_FN, "scanner")]
    assert report.flags[0].detail == "its parametrize values are built by scanning the repository (1 new id)"


def test_build_report_scanner_detail_counts_the_new_ids():
    rows = [{"nodeid": f"{_SCAN_FN}[{n}]", "params": {"p": n}, "large_grid": [], "problems": []} for n in "abc"]
    report = _report(rows, scanners={_SCAN_FN})
    assert [f.detail for f in report.flags if f.rule == "scanner"] == [
        "its parametrize values are built by scanning the repository (3 new ids)"
    ]


def test_exit_status_is_zero_without_enforce_even_with_a_scanner_flag():
    report = _report(_scan_rows(), scanners={_SCAN_FN})
    assert rules.exit_status(report, enforce=False) == 0


def test_exit_status_is_zero_with_enforce_when_the_only_scanner_flag_is_exempted():
    rows = [{"nodeid": f"{_SCAN_FN}[a]", "params": {}, "large_grid": ["one id per schema file is the point"], "problems": []}]
    report = _report(rows, scanners={_SCAN_FN})
    assert report.flags == [] and [f.rule for f, _ in report.exempt] == ["scanner"]
    assert rules.exit_status(report, enforce=True) == 0


def test_exit_status_is_one_with_enforce_and_an_unexempted_scanner_flag():
    report = _report(_scan_rows(), scanners={_SCAN_FN})
    assert rules.exit_status(report, enforce=True) == 1


def test_render_summary_has_totals_row_and_every_section_with_content():
    rows = _grid_rows() + _scan_rows() + _single_axis_rows()
    rows[0]["problems"] = [_no_reason(rows[0]["nodeid"])]
    exempt_rows = [
        {
            "nodeid": f"tests/unit/test_y.py::test_e[{a}-{b}]",
            "params": {"a": a, "b": b},
            "large_grid": ["whole matrix"],
            "problems": [],
        }
        for a in "xy"
        for b in "pqr"
    ]
    report = rules.build_report(
        counts=[
            rules.FileCount("tests/unit/test_x.py", 1, 3, 4, 14, 10, 0),
            rules.FileCount("tests/unit/test_y.py", 0, 1, 0, 6, 6, 0),
        ],
        head_rows=rows + exempt_rows,
        new_ids={r["nodeid"] for r in rows + exempt_rows},
        scanners={_SCAN_FN},
        budget=0,
        duplicates=[("a.py::t", "b.py::t", 0.953)],
    )
    text = rules.render_summary(report)
    assert "| tests/unit/test_x.py | 1 → 3 | 4 → 14 | +10 | −0 |" in text
    assert "| **Total** | 1 → 4 | 4 → 20 | +16 | −0 |" in text
    for heading in (
        "Flagged (blocks once enforced)",
        "Reported (never blocks)",
        "Justified with large_grid",
        "Marker problems",
        "Likely duplicates",
    ):
        assert heading in text
    assert "trim it, or add @pytest.mark.large_grid(reason=...)" in text
    assert "whole matrix" in text
    assert "a.py::t" in text and "0.95" in text


def test_render_summary_omits_empty_sections_and_says_nothing_was_flagged():
    text = rules.render_summary(_report(_scan_rows(), new_ids=set()))
    assert "Total" in text
    assert "Flagged" not in text and "Reported" not in text
    assert "Marker problems" not in text and "Likely duplicates" not in text
    assert "nothing was flagged" in text.lower()


_SINGLE_FN = "tests/unit/test_x.py::test_one_axis"


def _single_axis_rows():
    return [
        {"nodeid": f"{_SINGLE_FN}[{n}]", "params": {"n": n}, "large_grid": [], "problems": []}
        for n in ("1", "2", "3", "4")
    ]


def test_build_report_reports_single_axis_growth_over_the_budget():
    report = _report(_single_axis_rows(), budget=0)
    assert [(f.function, f.rule) for f in report.flags] == [(_SINGLE_FN, "single-axis")]
    assert "4 new ids against 0 changed product lines" in report.flags[0].detail


def test_exit_status_is_zero_with_enforce_for_single_axis_and_duplicates_only():
    report = _report(_single_axis_rows(), duplicates=[("a.py::t", "b.py::t", 0.95)])
    assert report.flags and report.duplicates
    assert rules.exit_status(report, enforce=True) == 0


def test_exit_status_is_one_with_enforce_and_a_marker_problem():
    rows = [{"nodeid": f"{_SCAN_FN}[a]", "params": {}, "large_grid": [], "problems": [_no_reason(f"{_SCAN_FN}[a]")]}]
    assert rules.exit_status(_report(rows), enforce=True) == 1


def _no_reason(nodeid):
    """The problem the plugin (``marker_problems``) records for a bare ``large_grid`` mark."""
    return f'{nodeid}: @pytest.mark.large_grid needs reason="why"'


def _on_param(nodeid):
    return f"{nodeid}: @pytest.mark.large_grid is on a pytest.param; mark the test function itself"


def test_build_report_lists_each_problem_once_per_function_and_only_for_new_functions():
    rows = [
        {"nodeid": f"{_GRID_FN}[{a}-{b}]", "params": {"a": a, "b": b}, "large_grid": [], "problems": [_no_reason(f"{_GRID_FN}[{a}-{b}]")]}
        for a in ("a1", "a2")
        for b in ("b1", "b2", "b3")
    ]
    rows[0]["problems"].append(_on_param(rows[0]["nodeid"]))
    old = "tests/unit/test_x.py::test_old[a]"
    rows.append({"nodeid": old, "params": {}, "large_grid": [], "problems": [_no_reason(old)]})
    report = _report(rows, new_ids={r["nodeid"] for r in rows if r["nodeid"] != old})
    assert report.problems == [_no_reason(_GRID_FN), _on_param(_GRID_FN)]
