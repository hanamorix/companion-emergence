"""#416 canary: every tool name in prompt-facing prose is backticked.

On the MCP path the provider rewrites `name` to `mcp__brain-tools__name` (the only name the
CLI can call), but only backticked names: prose and history are not touched. A bare name in
prose survives the rewrite, and the model calls it and fails ("No such tool available": 32 bare
read_full_memory calls in the real login's October CLI logs, from epistemic_instruction).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from brain.tools import NELL_TOOL_NAMES

BRAIN = Path(__file__).resolve().parents[4] / "brain"
# "boot" is also an English word ("Full session boot"); its own inventory entry is backticked.
_NAMES = "|".join(n for n in NELL_TOOL_NAMES if n != "boot")
_BARE = re.compile(rf"(?<![`\w])({_NAMES})(?![\w`])")


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _prose_literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
        and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
    }
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings and " " in n.value:
            yield n.lineno, n.value


def test_chat_prompt_strings_backtick_tool_names() -> None:
    chat = tomllib.loads((BRAIN / "prompt_strings.toml").read_text(encoding="utf-8"))["chat"]
    bad = [m.group(0) + " in: " + s[:80] for s in _strings(chat) for m in _BARE.finditer(s)]
    assert not bad, bad


def test_chat_path_literals_backtick_tool_names() -> None:
    files = sorted((BRAIN / "chat").glob("*.py")) + [BRAIN / "self_model" / "ambient.py"]
    bad = [
        f"{f.relative_to(BRAIN)}:{line}: {m.group(0)}"
        for f in files
        for line, s in _prose_literals(f)
        if not s.lstrip().startswith(("%s", "{"))  # log/format templates are not prompt text
        for m in _BARE.finditer(s)
    ]
    assert not bad, bad
