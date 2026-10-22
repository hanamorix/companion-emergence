"""Helpers for tests that need a ``tunables.toml`` in a sandbox data home (#529).

``write_values`` lays down the file the way a person would type it (one quoted entry per line, no recorded defaults),
which is what most tests want: a few values the reader must honour. ``write_text`` writes exact text for tests that care
about layout. Tests may use ``toml_io`` (they are outside the enforcement scan) but must not import ``tomllib``.
"""

from __future__ import annotations

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
