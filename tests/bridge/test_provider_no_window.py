"""Windows console-flash guard.

Since v0.0.33 the bridge runs windowless (pythonw / Task Scheduler, no
console). On Windows a console child launched from a windowless parent gets a
*fresh visible console* unless told otherwise, so every ``claude`` CLI call
(soul review, dreams, research, monologue, chat, file writes) flashed a small
"claude" window that could steal keyboard focus.

Fix: pass ``creationflags=CREATE_NO_WINDOW`` on each ``claude`` spawn. The flag
only exists on Windows, so it is read as ``getattr(subprocess, "CREATE_NO_WINDOW", 0)``
— a no-op (0) on macOS/Linux. stdout/stderr capture is unaffected.

This is an AST guard over ``provider.py`` so it covers all current launches AND
catches a future ``claude`` call site that forgets the flag (mirrors the
grep-style regression tests already in the suite). The ``subprocess.run(cmd, ...)``
inside ``_system_prompt_tempfile``'s docstring is a string literal, not a Call
node, so it is correctly ignored.
"""

import ast
from pathlib import Path

_PROVIDER = Path(__file__).resolve().parents[2] / "brain" / "bridge" / "provider.py"


def _spawn_calls(tree: ast.AST) -> list[ast.Call]:
    """Every real subprocess.run / subprocess.Popen call node."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in ("run", "Popen")
        ):
            out.append(node)
    return out


def _passes_no_window(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "creationflags":
            return isinstance(kw.value, ast.Name) and kw.value.id == "_NO_WINDOW"
    return False


def test_no_window_constant_defined() -> None:
    src = _PROVIDER.read_text(encoding="utf-8")
    assert '_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)' in src


def test_every_claude_spawn_passes_creationflags() -> None:
    tree = ast.parse(_PROVIDER.read_text(encoding="utf-8"))
    calls = _spawn_calls(tree)
    assert calls, "expected subprocess.run/Popen launches in provider.py"
    bad = [c.lineno for c in calls if not _passes_no_window(c)]
    assert not bad, f"claude spawn(s) missing creationflags=_NO_WINDOW at lines {bad}"
