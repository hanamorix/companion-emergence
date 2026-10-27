"""The product source tree, read and parsed once per process for the tests that scan it (#540).

About thirty tests walk ``brain/`` (and a few ``scripts/`` or ``app/src``) to prove a rule holds everywhere. Each used to
read and ``ast.parse`` all of it on its own: ``ast.parse`` of the 369 files of ``brain/`` takes about 0.5 s, so a serial
run paid it thirty times and every ``pytest-xdist`` worker that drew one of those tests paid it again. The functions here
cache the result for the life of the process, so each worker pays it once.

Everything returned is shared between tests: read it, never change it. The mappings are read-only; an ``ast`` tree is
not, so a scanner that needs to rewrite nodes must parse its own copy. The parsed trees of ``brain/`` hold about 110 MB,
which is why trees are kept only for the roots a scanner asks for.

Keys are paths relative to the repository root in posix form, ``brain/x/y.py``. A test that pins names relative to
``brain/`` (``bridge/server.py``) strips the prefix itself. ``exclude`` entries are substrings matched against that
relative path, so ``"node_modules"`` skips any path containing it.

Scanners that scan a synthetic tree under ``tmp_path`` keep their own path-taking functions; only the calls that scan the
real repository come here.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

REPO = Path(__file__).resolve().parents[1]


def tree_sources(
    root: str, suffixes: tuple[str, ...] = (".py",), exclude: tuple[str, ...] = ()
) -> Mapping[str, str]:
    """``{path relative to the repository, posix style: text}`` of every file under ``root`` with one of ``suffixes``.

    Sorted by path. ``__pycache__`` is skipped, and so is any file whose relative path contains one of the ``exclude``
    substrings (they are tested before the file is read). ``root`` is relative to the repository (``"brain"``,
    ``"scripts"``, ``"app/src"``).
    """
    return _read(root, tuple(suffixes), tuple(exclude))


@functools.cache
def _read(root: str, suffixes: tuple[str, ...], exclude: tuple[str, ...]) -> Mapping[str, str]:
    base = REPO / root
    out: dict[str, str] = {}
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if path.suffix not in suffixes or "__pycache__" in path.parts or not path.is_file():
                continue
            rel = path.relative_to(REPO).as_posix()
            if any(part in rel for part in exclude):
                continue
            out[rel] = path.read_text(encoding="utf-8")
    return MappingProxyType(out)


def brain_sources() -> Mapping[str, str]:
    """``{"brain/x/y.py": text}`` for every Python file of the product package."""
    return tree_sources("brain", (".py",))


@functools.cache
def tree_asts(root: str) -> Mapping[str, ast.Module]:
    """``{path: parsed module}`` for the Python files under ``root``, parsed from the text of :func:`tree_sources`."""
    return MappingProxyType({rel: ast.parse(text) for rel, text in tree_sources(root, (".py",)).items()})


def brain_trees() -> Mapping[str, ast.Module]:
    """``{"brain/x/y.py": parsed module}``. Do not modify the trees."""
    return tree_asts("brain")
