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


PRODUCT_SUFFIXES = (".py", ".ts", ".tsx", ".rs")


def is_product_path(path: str) -> bool:
    """A product source file: .py/.ts/.tsx/.rs, outside tests/, *.test.* files and __tests__/ dirs."""
    if not path.endswith(PRODUCT_SUFFIXES):
        return False
    parts = path.split("/")
    if parts[0] == "tests" or "__tests__" in parts[:-1]:
        return False
    return ".test." not in parts[-1]


_C_ESCAPES = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13, "t": 9, "v": 11, "\\": 92, '"': 34}


def _unquote_c(text: str) -> str:
    """Decode a git C-quoted path body (octal bytes and simple escapes) to text."""
    out = bytearray()
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "\\" or i + 1 >= len(text):
            out += ch.encode("utf-8")
            i += 1
        elif text[i + 1] in _C_ESCAPES:
            out.append(_C_ESCAPES[text[i + 1]])
            i += 2
        elif text[i + 1 : i + 4].isdigit():
            out.append(int(text[i + 1 : i + 4], 8) & 0xFF)
            i += 4
        else:
            out += text[i : i + 2].encode("utf-8")
            i += 2
    return out.decode("utf-8", errors="replace")


def _header_path(line: str, marker: str, prefix: str) -> str | None:
    """The path of a ``--- ``/``+++ `` header line (plain or C-quoted), without its a/ or b/ prefix."""
    rest = line[len(marker) :]
    if rest.startswith('"'):
        end = rest.rfind('"')
        path = _unquote_c(rest[1:end] if end > 0 else rest[1:])
    else:
        path = rest.rstrip("\t")
    return path[len(prefix) :] if path.startswith(prefix) else None


def product_lines(diff_u0: str) -> int:
    """Non-blank added/removed lines in product files of a ``git diff -U0`` text."""
    total = 0
    path = ""
    in_hunk = False
    for line in diff_u0.splitlines():
        if line.startswith("diff --git "):
            path, in_hunk = "", False
        elif line.startswith("@@"):
            in_hunk = True
        elif not in_hunk:
            for marker, prefix in (("--- ", "a/"), ("+++ ", "b/")):
                if line.startswith(marker):
                    found = _header_path(line, marker, prefix)
                    if found is not None:
                        path = found
        elif line[:1] in "+-" and line[1:].strip() and is_product_path(path):
            total += 1
    return total


def is_single_axis_over_budget(new_ids: int, axis_sizes: dict[str, int], budget: int) -> bool:
    """Exactly one axis, at least two new ids, and more new ids than the budget (changed product lines)."""
    return len(axis_sizes) == 1 and new_ids >= 2 and new_ids > budget
