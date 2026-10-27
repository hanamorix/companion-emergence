"""A loud warning when a dev constant changes (#540, J3).

``brain/dev_constants.py`` holds the build-time judgment calls (timeouts, budgets, file names) that are not user
tunables. This test compares every constant in it with the approved values in ``tests/snapshots/dev_constants.json``
and fails, naming each constant that changed, was added or was removed. A deliberate change is approved with one command
(it rewrites the snapshot, so the change is part of the commit and its diff is reviewed there):

    uv run python tests/unit/test_dev_constants_snapshot.py --approve

Constants are compared as canonical text (``repr`` of numbers and strings, items of sets sorted), so ``5`` and ``5.0``
are different values. The values come from a private copy of the file loaded from its path, because the suite patches
some of them in ``tests/conftest.py`` (the RAM fit safety factor is 1.0 inside a test). The names come from the module's
top-level assignments, so an import in the file is never mistaken for a constant.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "brain" / "dev_constants.py"
SNAPSHOT = REPO / "tests" / "snapshots" / "dev_constants.json"
APPROVE_COMMAND = "uv run python tests/unit/test_dev_constants_snapshot.py --approve"


def canonical(value: Any) -> str:
    """Text that changes whenever the value or its type changes, and not with hash order."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(canonical(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    if isinstance(value, list):
        return "[" + ", ".join(canonical(v) for v in value) + "]"
    if isinstance(value, (set, frozenset)):
        return f"{type(value).__name__}({{" + ", ".join(sorted(canonical(v) for v in value)) + "})"
    if isinstance(value, dict):
        return "{" + ", ".join(sorted(f"{canonical(k)}: {canonical(v)}" for k, v in value.items())) + "}"
    raise TypeError(f"a {type(value).__name__} cannot be snapshotted: teach canonical() about it on purpose")


def constant_names(source: str) -> list[str]:
    """The public names assigned at the top level of the module (imports and underscore names are not constants)."""
    names: list[str] = []
    for node in ast.parse(source).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        names += [t.id for t in targets if isinstance(t, ast.Name) and not t.id.startswith("_")]
    return sorted(set(names))


def current_constants(path: Path = SOURCE) -> dict[str, str]:
    """``{name: canonical text}`` of the file's constants, read from a private copy of the module."""
    spec = importlib.util.spec_from_file_location("_dev_constants_pristine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out: dict[str, str] = {}
    for name in constant_names(path.read_text(encoding="utf-8")):
        try:
            out[name] = canonical(getattr(module, name))
        except TypeError as exc:
            raise TypeError(f"dev constant {name}: {exc}") from exc
    return out


def differences(approved: dict[str, str], current: dict[str, str]) -> list[str]:
    """One line per constant that changed, is new or was removed."""
    lines = [
        f"dev constant {name} changed: {approved[name]} -> {current[name]}"
        for name in sorted(approved.keys() & current.keys())
        if approved[name] != current[name]
    ]
    lines += [f"dev constant {name} is new: {current[name]}" for name in sorted(current.keys() - approved.keys())]
    lines += [f"dev constant {name} was removed (it was {approved[name]})" for name in sorted(approved.keys() - current.keys())]
    return lines


def test_dev_constants_match_the_approved_snapshot() -> None:
    from tests.snapshot_support import read_snapshot

    try:
        current = current_constants()
    except TypeError as exc:
        pytest.fail(str(exc), pytrace=False)
    try:
        approved = read_snapshot(SNAPSHOT)
    except ValueError as exc:
        pytest.fail(f"{exc}. Rewrite it with: {APPROVE_COMMAND}", pytrace=False)
    if approved is None:
        pytest.fail(f"there is no approved snapshot at {SNAPSHOT}. Create it with: {APPROVE_COMMAND}", pytrace=False)
    lines = differences(approved, current)
    if lines:
        pytest.fail(
            "brain/dev_constants.py differs from tests/snapshots/dev_constants.json:\n  "
            + "\n  ".join(lines)
            + f"\nIf this is intended, approve it and commit the changed snapshot: {APPROVE_COMMAND}",
            pytrace=False,
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--approve"]:
        sys.path.insert(0, str(REPO))
        from tests.snapshot_support import write_snapshot

        write_snapshot(SNAPSHOT, current_constants())
        print(f"rewrote {SNAPSHOT}: review it with git diff, then commit it")
    else:
        print(f"usage: {APPROVE_COMMAND}")
        raise SystemExit(2)
