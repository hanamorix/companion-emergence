"""list_directory tool — read-only, audited. Non-recursive; entry-capped."""
from __future__ import annotations

import os
from pathlib import Path

from brain.tools.impls.read_file import _audit

_LIST_ENTRY_CAP = 200


def list_directory(path: str, *, persona_dir: Path, **_) -> dict:
    raw = path
    try:
        p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    except Exception as exc:  # noqa: BLE001
        _audit(persona_dir, tool="list_directory", path=raw, resolved="", bytes_=0, ok=False, error=str(exc))
        return {"error": f"bad path: {exc}"}

    if not p.exists():
        _audit(persona_dir, tool="list_directory", path=raw, resolved=str(p), bytes_=0, ok=False, error="not found")
        return {"error": f"path does not exist: {p}"}

    if not p.is_dir():
        _audit(
            persona_dir,
            tool="list_directory",
            path=raw,
            resolved=str(p),
            bytes_=0,
            ok=False,
            error="not a directory",
        )
        return {"error": f"not a directory: {p}"}

    entries = []
    try:
        for child in sorted(p.iterdir(), key=lambda c: c.name)[:_LIST_ENTRY_CAP]:
            try:
                entries.append(
                    {
                        "name": child.name,
                        "type": "dir" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else None,
                    }
                )
            except OSError:
                continue
    except OSError as exc:
        _audit(
            persona_dir,
            tool="list_directory",
            path=raw,
            resolved=str(p),
            bytes_=0,
            ok=False,
            error=str(exc),
        )
        return {"error": f"list failed: {exc}"}

    _audit(persona_dir, tool="list_directory", path=raw, resolved=str(p), bytes_=0, ok=True)
    return {"path": str(p), "entries": entries}
