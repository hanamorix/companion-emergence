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


def test_likely_duplicates_never_pairs_a_test_with_itself_or_repeats_a_pair():
    a = rules.test_bodies(_DUP_A)["test_a"]
    b = rules.test_bodies(_DUP_B)["test_b"]
    both = {"x.py::test_a": a, "y.py::test_b": b}
    got = rules.likely_duplicates(both, both)
    assert [t[:2] for t in got] == [("x.py::test_a", "y.py::test_b")]


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
