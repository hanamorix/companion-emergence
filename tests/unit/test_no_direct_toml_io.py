"""Ratchet: every TOML read or write in the product goes through ``brain/utils/toml_io.py`` (#529).

The rule (owner, 2026-10-21): the dedicated reader/writer lives in ``brain/utils/`` and a test enforces its use. This test
parses ``brain/**/*.py`` and ``scripts/**/*.py`` and fails on:

* any import of a TOML library (``tomllib``, ``tomli``, ``tomli_w``, ``tomlkit``, ``toml``), at module or function
  level, aliased or not, and ``__import__("tomllib")`` / ``importlib.import_module("tomllib")``;
* any file operation (``open``, ``Path.open``, ``read_text``, ``read_bytes``, ``write_text``, ``write_bytes``,
  ``os.replace``, ``os.rename``, ``Path.replace``, ``Path.rename``, ``Path.touch``, ``shutil.copy``, ``copy2``,
  ``copyfile``, ``move``) whose path argument is, or derives from, a string literal ending in ``.toml``. "Derives from"
  follows module constants (one level), local variables assigned in the same function, ``/`` and ``+`` joins,
  f-strings, ``os.path.join``, ``Path(...)``, ``str(...)``, ``joinpath``, ``with_name`` and ``with_suffix``.

The only exemption is ``brain/utils/toml_io.py`` and the allowlist is EMPTY: an entry needs a reason line, and the last
test below fails if one is ever added without it.

Scope: ``tests/``, ``wizard-validation/`` and ``examples/`` are out on purpose (they legitimately read ``pyproject.toml``
or fixtures).

Stated blind spots (as the JSON ratchet states its own): names are followed within one function and one level of module
constants only; a path built by another function and returned is not followed; a path read from a config value or an
environment variable is not seen; ``getattr(os, "replace")`` and ``from tomllib import *`` through an exec string are
not seen. A ``.toml`` literal that is only text (a docstring, a comment, a log message, a prompt) is not a path argument
and never flags.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("brain", "scripts")
EXEMPT = {"brain/utils/toml_io.py"}
# path -> reason. Must stay empty: the helper replaces the readers, nothing is grandfathered.
ALLOWLIST: dict[str, str] = {}

TOML_LIBS = {"tomllib", "tomli", "tomli_w", "tomlkit", "toml"}
# Method names on a path-like receiver (the receiver is the path), and name/attribute functions (the arguments are).
METHOD_OPS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "replace", "rename", "touch"}
FUNC_OPS = {"open", "replace", "rename", "copy", "copy2", "copyfile", "move"}
PASS_THROUGH_CALLS = {"Path", "str", "PurePath", "PosixPath", "WindowsPath", "fspath", "join", "joinpath"}
PATH_METHODS = {"with_name", "with_suffix", "joinpath", "resolve", "expanduser", "absolute", "relative_to"}


def _module_constants(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out[node.target.id] = node.value
    return out


def _local_assignments(fn: ast.AST | None) -> dict[str, list[ast.AST]]:
    out: dict[str, list[ast.AST]] = {}
    if fn is None:
        return out
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            out.setdefault(node.target.id, []).append(node.value)
    return out


def _derives_toml(
    expr: ast.AST,
    local: dict[str, list[ast.AST]],
    module: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Does this expression, used as a path, derive from a string literal ending in ``.toml``?"""
    if isinstance(expr, ast.Constant):
        return isinstance(expr.value, str) and expr.value.endswith(".toml")
    if isinstance(expr, ast.Name):
        if expr.id in seen:
            return False
        seen = seen | {expr.id}
        if expr.id in local:
            return any(_derives_toml(v, local, module, seen) for v in local[expr.id])
        if expr.id in module:
            return _derives_toml(module[expr.id], {}, module, seen)
        return False
    if isinstance(expr, ast.BinOp):
        return _derives_toml(expr.left, local, module, seen) or _derives_toml(expr.right, local, module, seen)
    if isinstance(expr, ast.JoinedStr):
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str) and part.value.endswith(".toml"):
                return True
            if isinstance(part, ast.FormattedValue) and _derives_toml(part.value, local, module, seen):
                return True
        return False
    if isinstance(expr, ast.Call):
        func = expr.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in PASS_THROUGH_CALLS or name in PATH_METHODS:
            if isinstance(func, ast.Attribute) and _derives_toml(func.value, local, module, seen):
                return True
            return any(_derives_toml(a, local, module, seen) for a in expr.args)
        return False
    if isinstance(expr, (ast.Attribute, ast.Subscript)):
        return _derives_toml(expr.value, local, module, seen)
    return False


def _imports_toml_library(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in TOML_LIBS:
                return alias.name
    if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in TOML_LIBS:
        return node.module
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if (
            name in ("__import__", "import_module")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.split(".")[0] in TOML_LIBS
        ):
            return str(node.args[0].value)
    return None


def _enclosing_functions(tree: ast.Module) -> dict[int, ast.AST | None]:
    """Map id(node) -> the innermost function that contains it (None at module level)."""
    owner: dict[int, ast.AST | None] = {}

    def visit(node: ast.AST, fn: ast.AST | None) -> None:
        for child in ast.iter_child_nodes(node):
            inner = child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            owner[id(child)] = inner
            visit(child, inner)

    visit(tree, None)
    return owner


def scan_source(source: str, tree: ast.Module | None = None) -> list[str]:
    """The violations (one short text each) found in one module's source (``tree``: its parse, to skip parsing again)."""
    tree = ast.parse(source) if tree is None else tree
    module = _module_constants(tree)
    owner = _enclosing_functions(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        lib = _imports_toml_library(node)
        if lib:
            found.append(f"line {node.lineno}: imports {lib}")
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        local = _local_assignments(owner.get(id(node)))
        path_args: list[ast.AST] = []
        op: str | None = None
        if isinstance(func, ast.Attribute) and func.attr in METHOD_OPS:
            # `x.open()` / `x.replace(dst)`: the receiver is the path; `os.replace(src, dst)`, `shutil.copy(...)`: the arguments are.
            if isinstance(func.value, ast.Name) and func.value.id in {"os", "shutil", "io"}:
                op, path_args = func.attr, list(node.args[:2])
            else:
                op, path_args = func.attr, [func.value, *node.args[:1]] if func.attr in ("replace", "rename") else [func.value]
        elif isinstance(func, ast.Attribute) and func.attr in FUNC_OPS and isinstance(func.value, ast.Name) and func.value.id in {"os", "shutil", "io"}:
            op, path_args = func.attr, list(node.args[:2])
        elif isinstance(func, ast.Name) and func.id in FUNC_OPS:
            op, path_args = func.id, list(node.args[:2])
        if op is None:
            continue
        if any(_derives_toml(a, local, module) for a in path_args):
            found.append(f"line {node.lineno}: {op}() on a .toml path")
    return found


def scan_tree(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sub in SCAN_DIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if rel in EXEMPT or rel in ALLOWLIST or "__pycache__" in path.parts:
                continue
            hits = scan_source(path.read_text(encoding="utf-8"))
            if hits:
                out[rel] = hits
    return out


def scan_repo() -> dict[str, list[str]]:
    """``scan_tree`` of this repository, from the process-wide parsed copy of its source (tests/source_tree.py)."""
    from tests.source_tree import tree_asts, tree_sources

    out: dict[str, list[str]] = {}
    for sub in SCAN_DIRS:
        trees = tree_asts(sub)
        for rel, source in tree_sources(sub).items():
            if rel in EXEMPT or rel in ALLOWLIST:
                continue
            hits = scan_source(source, trees[rel])
            if hits:
                out[rel] = hits
    return out


# ---------------------------------------------------------------------------------------------------------------
# The product tree
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.slow(reason='5.3 s of CPU in its call phase on a 4-core dev VM, 5.3 s wall (serial run for #540, J12)')
def test_no_toml_io_outside_the_helper() -> None:
    offenders = scan_repo()
    assert not offenders, (
        "TOML is read and written only through brain/utils/toml_io.py; offenders:\n"
        + "\n".join(f"  {rel}: {', '.join(hits)}" for rel, hits in offenders.items())
    )


def test_the_allowlist_is_empty_and_any_entry_has_a_reason() -> None:
    assert all(isinstance(reason, str) and reason.strip() for reason in ALLOWLIST.values()), (
        "an allowlist entry needs a reason line"
    )
    assert ALLOWLIST == {}, "the helper replaces the readers; nothing is grandfathered"
    assert EXEMPT == {"brain/utils/toml_io.py"}, "the helper is the only exemption"


# ---------------------------------------------------------------------------------------------------------------
# Self-test: every pattern has a violating fixture that is flagged and a conforming one that passes
# ---------------------------------------------------------------------------------------------------------------

VIOLATING = {
    "import tomllib": "import tomllib\n",
    "from tomllib import load": "from tomllib import load\n",
    "aliased import": "import tomllib as t\n",
    "tomli": "import tomli\n",
    "tomli_w": "import tomli_w\n",
    "tomlkit": "import tomlkit\n",
    "toml": "import toml\n",
    "function-local import": "def f():\n    import tomllib\n    return tomllib\n",
    "__import__": "def f():\n    return __import__('tomllib')\n",
    "importlib.import_module": "import importlib\n\ndef f():\n    return importlib.import_module('tomllib')\n",
    "open of a literal": "def f():\n    return open('a.toml', 'rb')\n",
    "Path.open": "from pathlib import Path\n\ndef f():\n    return Path('a.toml').open('rb')\n",
    "read_text": "from pathlib import Path\n\ndef f():\n    return Path('a.toml').read_text()\n",
    "read_bytes": "from pathlib import Path\n\ndef f():\n    return Path('a.toml').read_bytes()\n",
    "write_text": "from pathlib import Path\n\ndef f():\n    Path('a.toml').write_text('x')\n",
    "write_bytes": "from pathlib import Path\n\ndef f():\n    Path('a.toml').write_bytes(b'x')\n",
    "os.replace": "import os\n\ndef f():\n    os.replace('a.tmp', 'a.toml')\n",
    "os.rename": "import os\n\ndef f():\n    os.rename('a.toml', 'b.old')\n",
    "Path.replace": "from pathlib import Path\n\ndef f():\n    Path('a.tmp').replace(Path('a.toml'))\n",
    "Path.rename": "from pathlib import Path\n\ndef f():\n    Path('a.toml').rename('b')\n",
    "Path.touch": "from pathlib import Path\n\ndef f():\n    Path('a.toml').touch()\n",
    "shutil.copy": "import shutil\n\ndef f():\n    shutil.copy('a.toml', 'b')\n",
    "shutil.copy2": "import shutil\n\ndef f():\n    shutil.copy2('b', 'a.toml')\n",
    "shutil.copyfile": "import shutil\n\ndef f():\n    shutil.copyfile('a.toml', 'b')\n",
    "shutil.move": "import shutil\n\ndef f():\n    shutil.move('a.toml', 'b')\n",
    "module constant": "from pathlib import Path\nNAME = 'x.toml'\n\ndef f(d):\n    return (d / NAME).read_text()\n",
    "local variable (config.py shape)": (
        "def f(persona_dir):\n"
        "    p = persona_dir / 'persona.toml'\n"
        "    if not p.exists():\n"
        "        return {}\n"
        "    with p.open('rb') as fh:\n"
        "        return fh.read()\n"
    ),
    "f-string path": "def f(name):\n    return open(f'{name}.toml')\n",
    "with_suffix": "def f(p):\n    return p.with_suffix('.toml').read_text()\n",
    "with_name": "def f(p):\n    return p.with_name('x.toml').read_bytes()\n",
    "os.path.join": "import os\n\ndef f(d):\n    return open(os.path.join(d, 'x.toml'))\n",
    "str() wrapped": "def f(d):\n    return open(str(d / 'x.toml'))\n",
}

CONFORMING = {
    "a toml_io call": "from brain.utils import toml_io\n\ndef f(p):\n    return toml_io.read_toml(p)\n",
    "a .toml word in a docstring": "def f():\n    '''Reads prompt_strings.toml.'''\n    return 1\n",
    "a .toml word in a comment": "# reads x.toml\nx = 1\n",
    "a prompt text containing .toml": "PROMPT = 'edit your settings.toml'\n\ndef f():\n    return PROMPT\n",
    "a .toml literal in a log message": (
        "import logging\nlogger = logging.getLogger(__name__)\n\ndef f():\n    logger.warning('could not read a.toml')\n"
    ),
    "a path that is not TOML": "from pathlib import Path\n\ndef f(d):\n    return (d / 'x.json').read_text()\n",
    "open of a non-toml variable": "def f(p):\n    with open(p) as fh:\n        return fh.read()\n",
    "a JSON replace": "import os\n\ndef f():\n    os.replace('a.tmp', 'a.json')\n",
}


@pytest.mark.parametrize("name", sorted(VIOLATING))
def test_each_pattern_has_a_violating_fixture_that_is_flagged(name: str) -> None:
    assert scan_source(VIOLATING[name]), f"{name}: the checker did not flag its violating fixture"


@pytest.mark.parametrize("name", sorted(CONFORMING))
def test_each_pattern_has_a_conforming_fixture_that_passes(name: str) -> None:
    assert scan_source(CONFORMING[name]) == [], f"{name}: the checker flagged its conforming fixture"


def test_the_config_py_shape_is_flagged_by_the_operation_rule_alone() -> None:
    """The persona.toml reader opens the file through a local variable; with no import in sight, the operation rule must still flag it."""
    source = VIOLATING["local variable (config.py shape)"]
    assert "import" not in source
    hits = scan_source(source)
    assert hits and all("import" not in h for h in hits)


def test_a_planted_tomllib_import_in_a_brain_file_is_flagged(tmp_path: Path) -> None:
    (tmp_path / "brain").mkdir()
    (tmp_path / "brain" / "throwaway.py").write_text("import tomllib\n", encoding="utf-8")
    assert "brain/throwaway.py" in scan_tree(tmp_path)


def test_the_exempt_file_is_not_scanned(tmp_path: Path) -> None:
    (tmp_path / "brain" / "utils").mkdir(parents=True)
    (tmp_path / "brain" / "utils" / "toml_io.py").write_text("import tomllib\n", encoding="utf-8")
    assert scan_tree(tmp_path) == {}
