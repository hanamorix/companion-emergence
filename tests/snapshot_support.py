"""Read and write a committed snapshot file for the tests that warn loudly when something changes (#540).

A snapshot is a JSON object of strings, kept in ``tests/snapshots/``. The tests that use it compare the current text with
the approved text and, on a difference, fail with a message that says how to approve the new text: a one-shot script
mode (``uv run python <test file> --approve``) that rewrites the file, so the change shows up in the diff of the commit
and is reviewed there. There is no environment variable to leave switched on.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_snapshot(path: Path) -> dict[str, str] | None:
    """The approved snapshot, or ``None`` when there is no file yet. A file that is not a JSON object of strings raises."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f"{path} is not a JSON object of strings")
    return data


def write_snapshot(path: Path, data: dict[str, str]) -> None:
    """Write ``data`` as sorted, indented utf-8 JSON with ``\\n`` line ends on every platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
