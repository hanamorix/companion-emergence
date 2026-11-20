"""#618: every ``append_jsonl(..., fsync=<not True>)`` in brain/ must say why with a ``# fsync off:`` comment."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARKER = "# fsync off:"
REASONED = re.compile(r"# fsync off:\s*\S")


def _is_append_jsonl(call: ast.Call) -> bool:
    f = call.func
    return (isinstance(f, ast.Name) and f.id == "append_jsonl") or (
        isinstance(f, ast.Attribute) and f.attr == "append_jsonl"
    )


def find_unreasoned_opt_outs(source: str) -> list[int]:
    """Line numbers of ``append_jsonl`` opt-outs whose nearest comment (call line or the 3 above) has no reason."""
    lines = source.splitlines()
    offending: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and _is_append_jsonl(node)):
            continue
        for kw in node.keywords:
            if kw.arg == "fsync" and not (isinstance(kw.value, ast.Constant) and kw.value.value is True):
                first = node.lineno
                window = lines[max(first - 4, 0) : first]
                nearest = next((line for line in reversed(window) if "#" in line), "")
                if not REASONED.search(nearest):
                    offending.append(first)
    return sorted(offending)


def test_the_checker_wants_a_reason_on_the_nearest_comment() -> None:
    reasoned = "# fsync off: a rebuildable cache\nappend_jsonl(p, r, fsync=False)\n"
    bare = "# fsync off:\nappend_jsonl(p, r, fsync=False)\n"
    belongs_to_another_call = (
        "# fsync off: for the call above\n"
        "append_jsonl(p, r, fsync=False)\n"
        "x = 1\n"
        "# an ordinary comment\n"
        "append_jsonl(p, r, fsync=False)\n"
    )
    assert find_unreasoned_opt_outs(reasoned) == []
    assert find_unreasoned_opt_outs(bare) == [2]
    assert find_unreasoned_opt_outs(belongs_to_another_call) == [5]


def test_every_fsync_opt_out_states_its_reason() -> None:
    missing: list[str] = []
    for path in sorted((REPO / "brain").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        missing += [f"{path.relative_to(REPO).as_posix()}:{n}" for n in find_unreasoned_opt_outs(source)]
    assert not missing, f"append_jsonl fsync opt-outs without a '{MARKER} <reason>' comment: {missing}"
