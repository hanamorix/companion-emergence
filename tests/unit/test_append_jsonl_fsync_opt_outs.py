"""#618: every ``append_jsonl(..., fsync=<not True>)`` in brain/ must say why with a ``# fsync off:`` comment."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARKER = "# fsync off:"


def _is_append_jsonl(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Name) and f.id == "append_jsonl") or (
        isinstance(f, ast.Attribute) and f.attr == "append_jsonl"
    )


def test_every_fsync_opt_out_states_its_reason() -> None:
    missing: list[str] = []
    for path in sorted((REPO / "brain").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and _is_append_jsonl(node)):
                continue
            for kw in node.keywords:
                if kw.arg == "fsync" and not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
                    first = node.lineno
                    window = lines[max(first - 4, 0) : first]  # the call's first line and the 3 above it
                    if not any(MARKER in line for line in window):
                        missing.append(f"{path.relative_to(REPO).as_posix()}:{first}")
    assert not missing, f"append_jsonl fsync opt-outs without a '{MARKER} <reason>' comment: {missing}"
