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


def _is_scan(node: ast.AST) -> bool:
    """True when the subtree calls a filesystem scanner or names ``ls-files``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in SCANNER_CALLS:
                return True
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and "ls-files" in sub.value:
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


def _resolved(
    values: ast.expr, assigns: dict[str, ast.expr], funcs: dict[str, ast.AST]
) -> list[ast.AST]:
    """The values expression plus, one level down, each module-level name's value and same-file callee's body."""
    if isinstance(values, (ast.List, ast.Tuple, ast.Set)):
        return []
    nodes: list[ast.AST] = [values]
    for sub in ast.walk(values):
        if isinstance(sub, ast.Name) and sub.id in assigns:
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
            ratio = difflib.SequenceMatcher(None, a_tokens, b_tokens, autojunk=False).ratio()
            if ratio >= threshold:
                found.append((a, b, ratio))
    return sorted(found, key=lambda t: -t[2])


def is_single_axis_over_budget(new_ids: int, axis_sizes: dict[str, int], budget: int) -> bool:
    """Exactly one axis, at least two new ids, and more new ids than the budget (changed product lines)."""
    return len(axis_sizes) == 1 and new_ids >= 2 and new_ids > budget
