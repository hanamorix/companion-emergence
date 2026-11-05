#!/usr/bin/env python3
"""Turn a pytest junit XML file into a short Markdown timing report (#540).

    python scripts/timing_report.py pytest-junit.xml [--title TEXT] [--top N]

Prints: the test count and summed test time, how many tests took more than 1 s and more than 2 s, the slowest N tests with
their times, and the files with the largest summed time. A test's time in the junit file is setup + call + teardown.
Stdlib only; the weekly `timing.yml` workflow writes the output to the step summary and uploads it with the junit file.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path


def read_cases(path: Path) -> list[tuple[str, str, float]]:
    """``(classname, name, seconds)`` for every testcase in the junit file."""
    cases = []
    for case in ET.parse(path).getroot().iter("testcase"):
        cases.append((case.get("classname", ""), case.get("name", ""), float(case.get("time", "0") or 0)))
    return cases


def report(cases: Sequence[tuple[str, str, float]], *, title: str, top: int = 40) -> str:
    total = sum(t for _, _, t in cases)
    over1 = sum(1 for _, _, t in cases if t > 1.0)
    over2 = sum(1 for _, _, t in cases if t > 2.0)
    lines = [
        f"### {title}",
        "",
        f"- tests: {len(cases)}",
        f"- summed test time: {total:.1f} s ({total / 60:.1f} min)",
        f"- tests over 1 s: {over1}, over 2 s: {over2}",
        "",
        f"#### Slowest {min(top, len(cases))} tests",
        "",
        "| seconds | test |",
        "|---:|---|",
    ]
    for classname, name, seconds in sorted(cases, key=lambda c: -c[2])[:top]:
        lines.append(f"| {seconds:.2f} | `{classname}::{name}` |")
    by_file: dict[str, float] = defaultdict(float)
    for classname, _, seconds in cases:
        by_file[classname] += seconds
    lines += ["", f"#### Largest {min(15, len(by_file))} test modules by summed time", "", "| seconds | module (and class) |", "|---:|---|"]
    for classname, seconds in sorted(by_file.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"| {seconds:.1f} | `{classname}` |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("junit", type=Path)
    ap.add_argument("--title", default="Test timing")
    ap.add_argument("--top", type=int, default=40)
    ns = ap.parse_args(argv)
    cases = read_cases(ns.junit)
    if not cases:
        print(f"no testcase in {ns.junit}", file=sys.stderr)
        return 1
    sys.stdout.write(report(cases, title=ns.title, top=ns.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
