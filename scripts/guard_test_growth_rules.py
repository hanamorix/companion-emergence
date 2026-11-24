"""Counts and grid rules of the test-growth guard (#746): pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass


def function_of(nodeid: str) -> str:
    """The node id without a trailing ``[...]`` parameter suffix."""
    if nodeid.endswith("]") and "[" in nodeid:
        return nodeid[: nodeid.index("[")]
    return nodeid


@dataclass(frozen=True)
class FileCount:
    path: str
    base_functions: int
    head_functions: int
    base_ids: int
    head_ids: int
    added_ids: int
    removed_ids: int


def count_files(files: list[str], base_ids: set[str], head_ids: set[str]) -> list[FileCount]:
    """Per-file function and id counts at base and head; ids are already on head paths."""
    out = []
    for path in files:
        prefix = path + "::"
        base = {i for i in base_ids if i.startswith(prefix)}
        head = {i for i in head_ids if i.startswith(prefix)}
        out.append(
            FileCount(
                path,
                len({function_of(i) for i in base}),
                len({function_of(i) for i in head}),
                len(base),
                len(head),
                len(head - base),
                len(base - head),
            )
        )
    return out


def axes(rows: list[dict]) -> dict[str, int]:
    """Per param name, the number of distinct reprs across the rows (read from ``params``, never the id text)."""
    seen: dict[str, set[str]] = {}
    for row in rows:
        for name, value in row.get("params", {}).items():
            seen.setdefault(name, set()).add(value)
    return {name: len(values) for name, values in seen.items()}


def is_cross_product(ids: int, axis_sizes: dict[str, int]) -> bool:
    """At least two axes and more ids than the sum of the axis sizes."""
    return len(axis_sizes) >= 2 and ids > sum(axis_sizes.values())


def is_single_axis_over_budget(new_ids: int, axis_sizes: dict[str, int], budget: int) -> bool:
    """Exactly one axis, at least two new ids, and more new ids than the budget (changed product lines)."""
    return len(axis_sizes) == 1 and new_ids >= 2 and new_ids > budget
