"""read_file tool — read-only, guarded, audited. Used only when the user asks."""
from __future__ import annotations

import difflib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

_FILE_READ_MAX_BYTES = 256 * 1024
_SUGGEST_MAX = 10


def _suggest(target: Path) -> list[str]:
    """Case-insensitive / fuzzy filename suggestions from the target's parent dir.
    Keeps the model from crawling parent dirs when it guesses a wrong name."""
    parent = target.parent
    try:
        if not parent.is_dir():
            return []
        names = [c.name for c in parent.iterdir() if c.is_file()]
    except OSError:
        return []
    stem = target.name.casefold()
    # substring matches first, then close fuzzy matches, deduped, capped.
    subs = [n for n in names if stem in n.casefold() or n.casefold() in stem]
    fuzzy = difflib.get_close_matches(target.name, names, n=_SUGGEST_MAX, cutoff=0.6)
    out: list[str] = []
    for n in [*subs, *fuzzy]:
        if n not in out:
            out.append(n)
    return out[:_SUGGEST_MAX]


def _audit(
    persona_dir: Path,
    *,
    tool: str,
    path: str,
    resolved: str,
    bytes_: int,
    ok: bool,
    error: str | None = None,
) -> None:
    try:
        persona_dir.mkdir(parents=True, exist_ok=True)
        with (persona_dir / "file_access.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "tool": tool,
                        "path": path,
                        "resolved_path": resolved,
                        "bytes": bytes_,
                        "ok": ok,
                        "error": error,
                    }
                )
                + "\n"
            )
    except Exception:  # noqa: BLE001
        pass


def read_file(path: str, *, persona_dir: Path, **_) -> dict:
    """Read a text file's contents (read-only). Refuses files over the size cap."""
    raw = path
    try:
        p = Path(os.path.expandvars(os.path.expanduser(path))).resolve()
    except Exception as exc:  # noqa: BLE001
        _audit(persona_dir, tool="read_file", path=raw, resolved="", bytes_=0, ok=False, error=str(exc))
        return {"error": f"bad path: {exc}"}

    if not p.exists() or not p.is_file():
        suggestions = _suggest(p)
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=0,
            ok=False,
            error="not a readable file",
        )
        return {"error": f"not a readable file: {p}", "did_you_mean": suggestions}

    size = p.stat().st_size
    if size > _FILE_READ_MAX_BYTES:
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=size,
            ok=False,
            error="too large",
        )
        return {"error": f"file too large ({size} bytes > {_FILE_READ_MAX_BYTES} cap) — not shown"}

    try:
        data = p.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            _audit(
                persona_dir,
                tool="read_file",
                path=raw,
                resolved=str(p),
                bytes_=size,
                ok=True,
                error="binary",
            )
            return {"path": str(p), "note": f"(binary file, {size} bytes — not shown)"}

        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
        return {"path": str(p), "content": content}

    except OSError as exc:
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=size,
            ok=False,
            error=str(exc),
        )
        return {"error": f"read failed: {exc}"}
