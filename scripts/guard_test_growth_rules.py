"""Counts and grid rules of the test-growth guard (#746): pure functions, no I/O."""

from __future__ import annotations

import ast
import copy
import difflib
import io
import tokenize
from dataclasses import dataclass

_SKIP_TOKENS = {
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
    tokenize.ENDMARKER,
}


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


SCANNER_CALLS = frozenset({"glob", "rglob", "iterdir", "listdir", "scandir", "walk"})


def _names_ls_files(arg: ast.expr) -> bool:
    """A string constant containing ``ls-files``, alone or as an element of a list or tuple literal."""
    elements = arg.elts if isinstance(arg, (ast.List, ast.Tuple)) else [arg]
    return any(isinstance(e, ast.Constant) and isinstance(e.value, str) and "ls-files" in e.value for e in elements)


def _is_scan(node: ast.AST) -> bool:
    """True when the subtree calls a filesystem scanner, or passes ``ls-files`` to a call (a git subprocess)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "walk":
                # Only os.walk or a bare walk (from os import walk): ast.walk and other .walk methods don't scan.
                on_os = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "os"
                if on_os or isinstance(func, ast.Name):
                    return True
            elif name in SCANNER_CALLS:
                return True
            if any(_names_ls_files(arg) for arg in [*sub.args, *(kw.value for kw in sub.keywords)]):
                return True
    return False


def _parametrize_values(dec: ast.expr) -> ast.expr | None:
    """The values argument of a ``*.parametrize(...)`` decorator, else None."""
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not (
        getattr(func, "attr", None) == "parametrize"
        or (isinstance(func, ast.Name) and func.id == "parametrize")
    ):
        return None
    if len(dec.args) >= 2:
        return dec.args[1]
    for kw in dec.keywords:
        if kw.arg == "argvalues":
            return kw.value
    return None


_LITERALS = (ast.List, ast.Tuple, ast.Set)


def _resolved(
    values: ast.expr, assigns: dict[str, ast.expr], funcs: dict[str, ast.AST]
) -> list[ast.AST]:
    """The values expression plus, one level down, each module-level name's value and same-file callee's body.

    A literal list, tuple or set is never a scan, whether it is written inline or bound to a name.
    """
    if isinstance(values, _LITERALS):
        return []
    nodes: list[ast.AST] = [values]
    for sub in ast.walk(values):
        if isinstance(sub, ast.Name) and sub.id in assigns and not isinstance(assigns[sub.id], _LITERALS):
            nodes.append(assigns[sub.id])
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in funcs:
            nodes.append(funcs[sub.func.id])
    return nodes


def _scope(body: list[ast.stmt]) -> tuple[dict[str, ast.expr], dict[str, ast.AST]]:
    """The assignments (plain and annotated) and function defs directly in a body."""
    assigns: dict[str, ast.expr] = {}
    funcs: dict[str, ast.AST] = {}
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[stmt.name] = stmt
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assigns[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name):
                assigns[stmt.target.id] = stmt.value
    return assigns, funcs


def scanner_tests(source: str) -> set[str]:
    """Qualnames (``func`` or ``Class::func``) of tests whose parametrize values scan the filesystem."""
    tree = ast.parse(source)
    module_assigns, module_funcs = _scope(tree.body)
    found: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str, assigns: dict, funcs: dict) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                class_assigns, class_funcs = _scope(node.body)
                # The enclosing class body is looked up first, then the module.
                visit(
                    node.body,
                    f"{prefix}{node.name}::",
                    {**module_assigns, **class_assigns},
                    {**module_funcs, **class_funcs},
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    values = _parametrize_values(dec)
                    if values is not None and any(_is_scan(n) for n in _resolved(values, assigns, funcs)):
                        found.add(f"{prefix}{node.name}")

    visit(tree.body, "", module_assigns, module_funcs)
    return found


def _normalised_tokens(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    body = copy.deepcopy(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    names: dict[str, str] = {}

    def rename(ident: str) -> str:
        return names.setdefault(ident, f"v{len(names)}")

    tokens: list[str] = []
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                node.id = rename(node.id)
            elif isinstance(node, ast.arg):
                node.arg = rename(node.arg)
            elif isinstance(node, ast.Attribute):
                node.attr = rename(node.attr)
            elif isinstance(node, ast.keyword) and node.arg is not None:
                node.arg = rename(node.arg)
            elif isinstance(node, ast.Constant):
                node.value = f"c:{type(node.value).__name__}"
        readline = io.StringIO(ast.unparse(stmt) + "\n").readline
        tokens += [t.string for t in tokenize.generate_tokens(readline) if t.type not in _SKIP_TOKENS]
    return tokens


def test_bodies(source: str) -> dict[str, list[str]]:
    """Qualname to normalised body tokens for every ``test*`` function or method."""
    tree = ast.parse(source)
    out: dict[str, list[str]] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}::")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                out[f"{prefix}{node.name}"] = _normalised_tokens(node)

    visit(tree.body, "")
    return out


def likely_duplicates(
    new: dict[str, list[str]],
    pool: dict[str, list[str]],
    *,
    threshold: float = 0.9,
    min_tokens: int = 20,
) -> list[tuple[str, str, float]]:
    """Pairs of tests (keys ``path::qualname``) whose normalised bodies match at least ``threshold``."""
    found = []
    seen: set[frozenset[str]] = set()
    for a, a_tokens in new.items():
        if len(a_tokens) < min_tokens:
            continue
        for b, b_tokens in pool.items():
            pair = frozenset((a, b))
            if a == b or pair in seen or len(b_tokens) < min_tokens:
                continue
            seen.add(pair)
            # a is the new body, b the pool body: ratio() is not symmetric, so the order is fixed. The two quick
            # ratios are upper bounds of ratio(), so a pair under either can't reach the threshold.
            sm = difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False)
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            ratio = sm.ratio()
            if ratio >= threshold:
                found.append((a, b, ratio))
    return sorted(found, key=lambda t: -t[2])


def is_single_axis_over_budget(new_ids: int, axis_sizes: dict[str, int], budget: int) -> bool:
    """Exactly one axis, at least two new ids, and more new ids than the budget (changed product lines)."""
    return len(axis_sizes) == 1 and new_ids >= 2 and new_ids > budget


@dataclass
class Flag:
    function: str
    rule: str  # "scanner", "cross-product" or "single-axis"
    detail: str


@dataclass
class Report:
    counts: list[FileCount]
    flags: list[Flag]
    exempt: list[tuple[Flag, str]]
    problems: list[str]
    duplicates: list[tuple[str, str, float]]
    budget: int


def build_report(
    *,
    counts: list[FileCount],
    head_rows: list[dict],
    new_ids: set[str],
    scanners: set[str],
    budget: int,
    duplicates: list[tuple[str, str, float]],
) -> Report:
    """Flag the functions that gained ids in ``new_ids``; large_grid-marked ones move to ``exempt``."""
    by_function: dict[str, list[dict]] = {}
    for row in head_rows:
        by_function.setdefault(function_of(row["nodeid"]), []).append(row)

    flags: list[Flag] = []
    exempt: list[tuple[Flag, str]] = []
    problems: list[str] = []
    for function, rows in by_function.items():
        new_rows = [r for r in rows if r["nodeid"] in new_ids]
        if not new_rows:
            continue
        for row in rows:
            for problem in row.get("problems", []):
                text = f"{row['nodeid']}: {problem}"
                if text not in problems:
                    problems.append(text)
        axis_sizes = axes(new_rows)
        found: list[Flag] = []
        if function in scanners:
            found.append(
                Flag(function, "scanner", "its parametrize values are built by scanning the repository")
            )
        if is_cross_product(len(new_rows), axis_sizes):
            sizes = ", ".join(f"{k}={v}" for k, v in sorted(axis_sizes.items()))
            detail = f"{len(new_rows)} ids from axes {sizes} (sum {sum(axis_sizes.values())})"
            found.append(Flag(function, "cross-product", detail))
        if not found and is_single_axis_over_budget(len(new_rows), axis_sizes, budget):
            detail = f"{len(new_rows)} new ids against {budget} changed product lines"
            found.append(Flag(function, "single-axis", detail))
        reason = next(
            (r["large_grid"][0] for r in sorted(rows, key=lambda r: r["nodeid"]) if r.get("large_grid")),
            None,
        )
        for flag in found:
            if reason is None:
                flags.append(flag)
            else:
                exempt.append((flag, reason))
    flags.sort(key=lambda f: (f.function, f.rule))
    exempt.sort(key=lambda e: (e[0].function, e[0].rule))
    return Report(counts, flags, exempt, problems, duplicates, budget)


_TRIM = "trim it, or add @pytest.mark.large_grid(reason=...)"


def render_summary(report: Report) -> str:
    """Markdown for the PR step summary: counts table, then a section for each non-empty list."""
    lines = [
        "| File | Functions base → head | Ids base → head | +added | −removed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in report.counts:
        lines.append(
            f"| {c.path} | {c.base_functions} → {c.head_functions} | {c.base_ids} → {c.head_ids}"
            f" | +{c.added_ids} | −{c.removed_ids} |"
        )
    totals = report.counts
    lines.append(
        f"| **Total** | {sum(c.base_functions for c in totals)} → {sum(c.head_functions for c in totals)}"
        f" | {sum(c.base_ids for c in totals)} → {sum(c.head_ids for c in totals)}"
        f" | +{sum(c.added_ids for c in totals)} | −{sum(c.removed_ids for c in totals)} |"
    )

    def flag_line(flag: Flag) -> str:
        return f"- `{flag.function}` [{flag.rule}]: {flag.detail}; {_TRIM}"

    blocking = [f for f in report.flags if f.rule in ("scanner", "cross-product")]
    reported = [f for f in report.flags if f.rule == "single-axis"]
    if blocking:
        lines += ["", "### Flagged (blocks once enforced)", ""] + [flag_line(f) for f in blocking]
    if reported:
        lines += ["", "### Reported (never blocks)", ""] + [flag_line(f) for f in reported]
    if report.exempt:
        lines += ["", "### Justified with large_grid", ""]
        lines += [f"- `{f.function}` [{f.rule}]: {f.detail}; reason: {why}" for f, why in report.exempt]
    if report.problems:
        lines += ["", "### Marker problems", ""] + [f"- {p}" for p in report.problems]
    if report.duplicates:
        lines += ["", "### Likely duplicates", ""]
        lines += [f"- `{a}` and `{b}` (ratio {ratio:.2f})" for a, b, ratio in report.duplicates]
    if not (report.flags or report.exempt or report.problems or report.duplicates):
        lines += ["", "Nothing was flagged."]
    return "\n".join(lines) + "\n"


def exit_status(report: Report, *, enforce: bool) -> int:
    """0 unless enforcing; enforcing, 1 on an unexempted scanner or cross-product flag or any problem."""
    if not enforce:
        return 0
    if report.problems or any(f.rule in ("scanner", "cross-product") for f in report.flags):
        return 1
    return 0
