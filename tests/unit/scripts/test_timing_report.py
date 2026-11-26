"""The weekly timing report (#540, J12): a planted junit file in, the numbers and the slowest tests out."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("timing_report", REPO / "scripts" / "timing_report.py")
timing_report = importlib.util.module_from_spec(_spec)
sys.modules["timing_report"] = timing_report
_spec.loader.exec_module(timing_report)

_REASON = "test-suite machinery (#540): checks how the suite itself runs, which the product code does not affect, so it passes on the old code"
_JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="4">
<testcase classname="tests.unit.a" name="test_fast" time="0.010"/>
<testcase classname="tests.unit.a" name="test_one_and_a_half" time="1.500"/>
<testcase classname="tests.unit.b" name="test_three[x]" time="3.000"/>
<testcase classname="tests.unit.b" name="test_exactly_two" time="2.000"/>
</testsuite></testsuites>
"""


@pytest.mark.no_base_failure(reason=_REASON)
def test_the_report_counts_sums_and_ranks(tmp_path: Path) -> None:
    junit = tmp_path / "j.xml"
    junit.write_text(_JUNIT, encoding="utf-8")
    text = timing_report.report(timing_report.read_cases(junit), title="Linux", top=2)
    assert "### Linux" in text
    assert "- tests: 4" in text
    assert "- summed test time: 6.5 s" in text
    assert "- tests over 1 s: 3, over 2 s: 1" in text  # exactly 2.0 s is not over the budget
    ranked = [line for line in text.splitlines() if line.startswith("| ") and "`tests." in line]
    assert ranked[0].startswith("| 3.00 | `tests.unit.b::test_three[x]`")
    assert ranked[1].startswith("| 2.00 |")
    assert "tests.unit.a" not in "\n".join(ranked[:2])
    assert "| 5.0 | `tests.unit.b` |" in text


@pytest.mark.no_base_failure(reason=_REASON)
def test_the_command_line_prints_the_report_and_refuses_an_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junit = tmp_path / "j.xml"
    junit.write_text(_JUNIT, encoding="utf-8")
    assert timing_report.main([str(junit), "--title", "Windows", "--top", "1"]) == 0
    assert "### Windows" in capsys.readouterr().out
    empty = tmp_path / "e.xml"
    empty.write_text('<testsuites><testsuite name="pytest" tests="0"/></testsuites>', encoding="utf-8")
    assert timing_report.main([str(empty)]) == 1
