"""Guard (#781): no file in ``brain/`` opens a file in append mode outside the shared durable appenders.

``brain.health.json_io.append_jsonl`` and ``append_text`` lock the file, start a new line after a torn tail and
fsync every append, so an unclean stop cannot leave a just-written record as NUL bytes (#618), and a write cannot
land while the start-up repair or a trim rewrites the file. A writer that opens the file in append mode itself
gets none of that. This test fails on any append-mode open in ``brain/`` that does not carry an in-line marker
``# append: ok -- <reason>`` on one of the call's lines.

Forms seen: ``open(p, "a…")``, ``io.open``/``gzip.open``/``codecs.open`` with an append mode, ``p.open("a…")`` on a
path, ``mode="a…"`` as a keyword, ``os.fdopen(fd, "a…")``, and ``os.open(…, … O_APPEND …)``. A mode that is an
expression counts when any string constant in it is an append mode (``"a" if op == "append" else "w"``).

Known limits: a mode held in a variable (``m = "a"; open(p, m)``), ``getattr(builtins, "open")`` and
``pathlib.Path.open(p, "a")`` written as an unbound call are not seen.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MARKER = re.compile(r"#\s*append:\s*ok\s*--\s*\S")
_MODE_CHARS = set("rwxabtU+")
_MODULE_OPENS = {"io", "gzip", "bz2", "lzma", "codecs", "builtins"}  # mod.open(path, mode): the mode is the 2nd arg


def _is_append_mode(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for c in ast.walk(node):
        if isinstance(c, ast.Constant) and isinstance(c.value, str):
            v = c.value
            if v and set(v) <= _MODE_CHARS and "a" in v:
                return True
    return False


def _mode_arg(call: ast.Call, position: int) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[position] if len(call.args) > position else None


def _mentions_o_append(call: ast.Call) -> bool:
    nodes = [*call.args, *(kw.value for kw in call.keywords)]
    return any(
        (isinstance(n, ast.Attribute) and n.attr == "O_APPEND") or (isinstance(n, ast.Name) and n.id == "O_APPEND")
        for arg in nodes
        for n in ast.walk(arg)
    )


def append_opens(source: str) -> list[int]:
    """Line numbers of the append-mode opens in ``source`` that carry no ``# append: ok --`` marker."""
    lines = source.splitlines()
    found = []
    for call in ast.walk(ast.parse(source)):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        hit = False
        if isinstance(func, ast.Name) and func.id == "open":
            hit = _is_append_mode(_mode_arg(call, 1))
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            owner = func.value.id if isinstance(func.value, ast.Name) else None
            if owner == "os":
                hit = _mentions_o_append(call)
            elif owner in _MODULE_OPENS:
                hit = _is_append_mode(_mode_arg(call, 1))
            else:  # a path's own .open(mode)
                hit = _is_append_mode(_mode_arg(call, 0))
        elif isinstance(func, ast.Attribute) and func.attr == "fdopen":
            hit = _is_append_mode(_mode_arg(call, 1))
        if not hit:
            continue
        span = lines[call.lineno - 1 : (call.end_lineno or call.lineno)]
        if not any(MARKER.search(line) for line in span):
            found.append(call.lineno)
    return sorted(found)


def test_brain_has_no_unmarked_append_mode_open() -> None:
    offenders = []
    for path in sorted((REPO / "brain").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        for line in append_opens(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{line}")
    assert not offenders, (
        "append-mode open outside the shared appenders; use brain.health.json_io.append_jsonl or append_text, "
        "or mark a deliberate one with `# append: ok -- <reason>`: " + ", ".join(offenders)
    )



@pytest.mark.no_base_failure(reason="#781 test infrastructure: it checks this file's own scanner, not brain code")
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('open(p, "a")', [1]),
        ('open(p, "ab+", encoding="utf-8")', [1]),
        ('open(p, mode="at")', [1]),
        ('p.open("a", encoding="utf-8")', [1]),
        ('(d / "x.log").open(mode="ab")', [1]),
        ('io.open(p, "a")', [1]),
        ('gzip.open(p, "ab")', [1]),
        ('os.fdopen(fd, "a" if op == "append" else "w")', [1]),
        ("os.open(p, os.O_WRONLY | os.O_APPEND)", [1]),
        ("os.open(p, flags | O_APPEND)", [1]),
        ('with p.open(\n    "a",\n    encoding="utf-8",\n) as f:\n    pass', [1]),
        ('open(p, "a")  # append: ok -- a reason', []),
        ('with p.open(\n    "a",  # append: ok -- a reason\n) as f:\n    pass', []),
        ('open(p, "a")  # append: ok --', [1]),  # a marker needs a reason
        ('open(p, "r")', []),
        ('open(p, "wb")', []),
        ('p.open("r", encoding="a")', []),  # "a" is an encoding here, not the mode
        ('p.open(encoding="utf-8")', []),
        ("os.open(p, os.O_RDONLY)", []),
        ('d.get("a")', []),
        ('open(p, "abc-not-a-mode")', []),
    ],
)
def test_the_scanner_sees_each_append_form(source: str, expected: list[int]) -> None:
    assert append_opens(source) == expected
