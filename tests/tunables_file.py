"""Helpers for tests that need a ``tunables.toml`` in a sandbox data home (#529).

``write_values`` lays down the file the way a person would type it (one quoted entry per line, no recorded defaults),
which is what most tests want: a few values the reader must honour. ``write_text`` writes exact text for tests that care
about layout. Tests may use ``toml_io`` (they are outside the enforcement scan) but must not import ``tomllib``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from brain.utils import toml_io

FILE_NAME = "tunables.toml"


def path_in(home: Path) -> Path:
    return home / FILE_NAME


def write_values(home: Path, values: dict[str, Any]) -> Path:
    """Write ``values`` as flat quoted entries (replacing any existing file) and return the file's path."""
    text = "".join(f"{toml_io.dumps_value(k)} = {toml_io.dumps_value(v)}\n" for k, v in values.items())
    return write_text(home, text)


def write_text(home: Path, text: str | bytes) -> Path:
    path = path_in(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return path


def read_values(home: Path) -> dict[str, Any]:
    """The file's values as dotted keys (what the reader sees)."""
    return toml_io.flatten_toml(toml_io.read_toml(path_in(home)))


_REGISTER = "brain.tunables.register"


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a chain of names and attributes, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def register_calls_in_source(source: str, rel: str) -> list[ast.Call]:
    """Every call of ``brain.tunables.register`` in ``source`` (a file at ``rel`` inside the repo), resolved by AST.

    Imported names are followed, so these all count: ``tunables.register(...)`` after ``from brain import tunables``, an
    alias (``from brain import tunables as t``, ``import brain.tunables as bt``), the full dotted call after
    ``import brain.tunables``, ``from brain.tunables import register`` (also ``as``), a relative import, a star import,
    and any of them inside a function. The bare ``register(...)`` inside ``brain/tunables.py`` itself counts. Not followed:
    a name passed around as a value, or ``importlib`` at run time (the descriptions test and the boot test would not see
    those; no brain module does either).
    """
    tree = ast.parse(source)
    package = rel.split("/")[:-1]
    aliases: dict[str, str] = {}
    star = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)] if node.level - 1 <= len(package) else []
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                module = node.module or ""
            for a in node.names:
                if a.name == "*":
                    star = star or module == "brain.tunables"
                else:
                    aliases[a.asname or a.name] = f"{module}.{a.name}" if module else a.name
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        if name is None:
            continue
        first, _dot, rest = name.partition(".")
        resolved = f"{aliases[first]}.{rest}" if first in aliases and rest else aliases.get(first, name)
        if first not in aliases and name == "register" and (star or rel == "brain/tunables.py"):
            resolved = _REGISTER
        if resolved == _REGISTER:
            found.append(node)
    return found


def register_calls(repo_root: Path) -> list[tuple[str, int, ast.Call]]:
    """``(path relative to the repo, line, call)`` for every ``brain.tunables.register`` call under ``repo_root/brain``."""
    out: list[tuple[str, int, ast.Call]] = []
    for path in sorted((repo_root / "brain").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        out.extend((rel, call.lineno, call) for call in register_calls_in_source(path.read_text(encoding="utf-8"), rel))
    return out


def registered_literal_keys(repo_root: Path) -> set[str]:
    """The literal first arguments of those calls (a registration with a non-literal key is not in the set)."""
    return {
        call.args[0].value
        for _rel, _line, call in register_calls(repo_root)
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
    }


def differing_from_defaults() -> dict[str, Any]:
    """The model settings in the sandbox home's file that are not their shipped default (the file lists every setting now, so
    "which settings has the user changed" is a comparison). Defaults come from the job table, not the registry."""
    from brain import tunables
    from brain.bridge.model_tier import _CLAUDE_JOBS

    path = tunables._file_path()
    if not path.exists():
        return {}
    defaults = dict(_CLAUDE_JOBS.values())
    values = toml_io.flatten_toml(toml_io.read_toml(path))
    return {k: v for k, v in values.items() if k in defaults and v != defaults[k]}
