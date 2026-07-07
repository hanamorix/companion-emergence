"""read_file tool — read-only, guarded, audited. Used only when the user asks."""
from __future__ import annotations

import difflib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from brain import tunables
from brain.tools.impls import _read_cache

_FILE_READ_MAX_BYTES = tunables.register("files.read_max_bytes", 256 * 1024)
_SUGGEST_MAX = 10
_DEFAULT_HEAD_LINES = 400


def _file_read_max_bytes() -> int:
    return tunables.get_tunable("files.read_max_bytes", _FILE_READ_MAX_BYTES)


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


def read_file(path: str, *, persona_dir: Path, max_lines: int | None = None,
              offset: int = 0, **_) -> dict:
    """Read a text file's contents (read-only). Refuses files over the size cap.

    max_lines: optional — return at most this many lines (ranged read).
    offset: 0-based line to start reading from (used with max_lines).
    Large files without max_lines are head-capped at _DEFAULT_HEAD_LINES.
    """
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

    cap = _file_read_max_bytes()
    size = p.stat().st_size
    if size > cap:
        _audit(
            persona_dir,
            tool="read_file",
            path=raw,
            resolved=str(p),
            bytes_=size,
            ok=False,
            error="too large",
        )
        return {"error": f"file too large ({size} bytes > {cap} cap) — not shown"}

    # Platform-correct case handling: normcase lowercases on Windows (case-
    # insensitive FS) and is a no-op on macOS/Linux, where realpath already
    # canonicalises case on the case-insensitive macOS FS and keeps genuinely
    # distinct files distinct on case-sensitive Linux. Do NOT casefold here — on
    # Linux that would collide two different files (Notes.md vs notes.md).
    _dedup_key = os.path.normcase(os.path.realpath(str(p)))
    if _read_cache.seen_recently(_dedup_key):
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=0, ok=True, error="deduped")
        return {
            "path": str(p),
            "deduped": True,
            "note": "you already read this file moments ago this turn — its content is above.",
        }
    _read_cache.mark(_dedup_key)

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

        lines = content.splitlines(keepends=True)
        total = len(lines)
        start = max(0, int(offset or 0))
        if max_lines is not None:
            window = lines[start:start + max(0, int(max_lines))]
            truncated = (start + len(window)) < total or start > 0
        elif total > _DEFAULT_HEAD_LINES:
            window = lines[:_DEFAULT_HEAD_LINES]
            truncated = True
        else:
            window = lines[start:] if start else lines
            truncated = start > 0
        sliced = "".join(window)
        _audit(persona_dir, tool="read_file", path=raw, resolved=str(p), bytes_=size, ok=True)
        out: dict = {"path": str(p), "content": sliced, "total_lines": total}
        if truncated:
            out["truncated"] = True
            out["note"] = (
                f"showing lines {start}-{start + len(window)} of {total}; "
                "pass offset/max_lines to read more"
            )
        return out

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
