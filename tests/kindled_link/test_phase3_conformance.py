"""Guarded-change Phase 3 conformance oracle (spec §7.1 / §7.2). Static checks
that no test can drift past: peer-session modules IMPORT/REFERENCE no tool path
(AST-based, so prose mentions are allowed), and the chat path imports no
kindled_link."""
import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PEER_MODULES = [
    _ROOT / "brain" / "kindled_link" / "peer_prompt.py",
    _ROOT / "brain" / "kindled_link" / "session_engine.py",
]
# the genuine injection-hole names (verified in the red-team against the real code)
_FORBIDDEN = {"dispatch", "tool_recruit", "reach_for_capability",
              "read_file", "NELL_TOOL_NAMES"}


def _referenced_names(tree: ast.AST) -> set[str]:
    """Collect names actually USED as code: imports, attribute accesses, and
    bare name references. Strings, comments, and docstrings are excluded by
    construction (ast never yields them as Name/Attribute/import nodes)."""
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                used.add(alias.name.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                used.add(node.module.split(".")[-1])
            for alias in node.names:
                used.add(alias.name)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    return used


def test_peer_modules_reference_no_tool_path():
    # CRITERION 1 (static, AST): the deepest injection holes are not importable
    # or referenced as code. A docstring naming them is allowed.
    for mod in _PEER_MODULES:
        tree = ast.parse(mod.read_text(encoding="utf-8"), filename=str(mod))
        hits = _FORBIDDEN & _referenced_names(tree)
        assert not hits, f"{mod.name} references forbidden tool path(s) {hits!r}"


def test_chat_path_does_not_import_kindled_link():
    # §7.2: Phase 3 cannot regress the chat token path — it isn't reachable from it.
    targets = list((_ROOT / "brain" / "chat").rglob("*.py"))
    targets.append(_ROOT / "brain" / "bridge" / "provider.py")
    for py in targets:
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "kindled_link" not in node.module, f"{py} imports kindled_link"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "kindled_link" not in alias.name, f"{py} imports kindled_link"
