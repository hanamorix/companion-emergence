"""Rebuild tests/shard_durations.json from Windows junit reports (#746).

    uv run python scripts/refresh_shard_durations.py pytest-junit-*.xml [--out tests/shard_durations.json]

The shard plugin (tests/shard.py) balances the Windows CI shards with these timings. Download the junit artifacts of
every Windows shard of one green run on main (`gh run download <run> -p 'pytest-junit-windows-latest-*'`) and pass them
all. Only tests that took at least --threshold seconds get an entry; every other test weighs `default_seconds`, the mean
of those, so the file stays small. Stale timings only unbalance the shards; they never drop or repeat a test.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def build(junit_paths: list[Path], root: Path, *, threshold: float) -> dict:
    """The shard_durations.json content for these junit reports: tests of at least ``threshold`` seconds by node id,
    and ``default_seconds``, the mean of the others (both to the millisecond)."""
    seconds: dict[str, float] = {}
    for path in junit_paths:
        for case in ET.parse(path).getroot().iter("testcase"):  # our own CI's reports, not untrusted input
            nodeid = nodeid_of(case.get("classname", ""), case.get("name", ""), root)
            if nodeid is not None:
                seconds[nodeid] = float(case.get("time") or 0.0)
    rest = [s for s in seconds.values() if s < threshold]
    return {
        "default_seconds": round(sum(rest) / len(rest), 3) if rest else 0.0,
        "durations": {k: round(v, 3) for k, v in sorted(seconds.items()) if v >= threshold},
    }


def nodeid_of(classname: str, name: str, root: Path) -> str | None:
    """The pytest node id of a junit testcase, without xdist's ``@<group>`` suffix, or None when it names no test
    (a module-level entry with no classname, or a module that no longer exists under ``root``)."""
    if not classname:
        return None
    if name.rfind("@") > name.rfind("]"):  # xdist's loadgroup suffix (the same rule as tests/shard.py base_nodeid)
        name = name.rsplit("@", 1)[0]
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):  # the longest dotted prefix that is a module file; the rest are classes
        module = Path(*parts[:cut]).with_suffix(".py")
        if (root / module).is_file():
            return "::".join([module.as_posix(), *parts[cut:], name])
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path, help="junit XML reports of every Windows shard of one run")
    parser.add_argument("--out", type=Path, default=repo / "tests" / "shard_durations.json")
    parser.add_argument("--root", type=Path, default=repo, help="the checkout the node ids are resolved against")
    parser.add_argument("--threshold", type=float, default=0.5, help="seconds a test needs to get its own entry")
    args = parser.parse_args(argv)
    built = build(args.reports, args.root, threshold=args.threshold)
    args.out.write_text(json.dumps(built, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(built['durations'])} tests of at least {args.threshold} s; the rest weigh {built['default_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
