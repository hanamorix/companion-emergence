"""Split the suite across CI runners: ``--shard i/N`` (#746).

Every test lands in exactly one shard, and the tests of one ``xdist_group`` stay together. The split depends only on
the collected set and the recorded timings (``tests/shard_durations.json``), never on the order, so the random order
(pytest-randomly) can differ on every shard and every xdist worker still computes the same split.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

_SHARD = re.compile(r"([0-9]+)/([0-9]+)")
# Recorded Windows timings (scripts/refresh_shard_durations.py): stale ones only unbalance the shards, never drop a test.
DURATIONS_FILE = Path(__file__).with_name("shard_durations.json")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        type=_shard_arg,
        default=None,
        metavar="i/N",
        help="Run only shard i of N (#746): every test is in exactly one shard and an xdist_group stays whole, "
        "whatever the test order.",
    )


def _shard_arg(value: str) -> tuple[int, int]:
    try:
        return parse_shard(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep this shard's tests, in their current order; the rest are reported deselected. Last, so after marker
    deselection and the random shuffle (the split does not depend on either's order)."""
    shard = config.getoption("--shard")
    if shard is None:
        return
    index, count = shard
    durations, default = load_durations(DURATIONS_FILE)
    keep = set(select([(item.nodeid, _group(item)) for item in items], durations, default, index, count))
    dropped = [item for i, item in enumerate(items) if i not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = [item for i, item in enumerate(items) if i in keep]


def load_durations(path: Path) -> tuple[dict[str, float], float]:
    """(recorded seconds by node id, default seconds for a test with no entry) from ``path``; with no file every test
    weighs the same."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, 1.0
    return {str(k): float(v) for k, v in data["durations"].items()}, float(data["default_seconds"])


def _group(item: pytest.Item) -> str | None:
    """The item's xdist group name as xdist's loadgroup names it (several marks joined, sorted), or None."""
    names = {
        str(mark.args[0] if mark.args else mark.kwargs.get("name", "default")) for mark in item.iter_markers("xdist_group")
    }
    return "_".join(sorted(names)) if names else None


def parse_shard(value: str) -> tuple[int, int]:
    """``"i/N"`` as ``(i, N)``, 1 <= i <= N; anything else is a ValueError."""
    m = _SHARD.fullmatch(value)
    if m is None:
        raise ValueError(f"--shard takes i/N (for example 2/4), not {value!r}")
    index, count = int(m.group(1)), int(m.group(2))
    if not 1 <= index <= count:
        raise ValueError(f"--shard {value}: i must be between 1 and N")
    return index, count


def base_nodeid(nodeid: str) -> str:
    """``nodeid`` without the ``@<group>`` suffix xdist's loadgroup adds (xdist's own rule: an ``@`` after the last ``]``,
    so a parameter that contains ``@`` is kept)."""
    return nodeid.rsplit("@", 1)[0] if nodeid.rfind("@") > nodeid.rfind("]") else nodeid


def select(
    tests: list[tuple[str, str | None]], durations: dict[str, float], default: float, index: int, count: int
) -> list[int]:
    """The positions in ``tests`` ((node id, xdist group or None) pairs) that belong to shard ``index`` of ``count``, in
    their given order. A test weighs its recorded seconds, or ``default``; a group is one unit of its tests' sum."""
    unit_of: list[str] = []
    # Whole milliseconds: an integer sum is the same in any order, a float sum is not (0.1 + 0.2 + 0.3 != 0.3 + 0.2 +
    # 0.1), and every shard sees the tests in its own random order, so a float tie could split them differently.
    weights: dict[str, int] = {}
    for nodeid, group in tests:
        base = base_nodeid(nodeid)
        unit = f"group:{group}" if group else f"test:{base}"
        unit_of.append(unit)
        weights[unit] = weights.get(unit, 0) + round(1000 * durations.get(base, default))
    shards = assign(weights, count)
    return [i for i, unit in enumerate(unit_of) if shards[unit] == index]


def assign(weights: dict[str, int] | dict[str, float], count: int) -> dict[str, int]:
    """Map each unit to a shard 1..``count``: heaviest first (ties by key), each to the least-loaded shard (ties to the
    lowest index)."""
    loads = [0.0] * count
    shards: dict[str, int] = {}
    for key in sorted(weights, key=lambda k: (-weights[k], k)):
        index = min(range(count), key=lambda i: (loads[i], i))
        loads[index] += weights[key]
        shards[key] = index + 1
    return shards
