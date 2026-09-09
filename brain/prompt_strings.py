"""Dev-facing prompt-strings registry — centralized access to prompt/template
text externalized out of code and into ``prompt_strings.toml``.

Spec: issue #129 stage 1 (relocation only; see docs / hunts/129-externalize).

Call sites register their key at import time and use the returned value
directly — there is no separate "read at call time" step, because (unlike
``tunables.py``) there is nothing to override at runtime:

    _MY_PROMPT: str = prompt_strings.register("area.my_prompt")
    ...
    text = _MY_PROMPT.format(...)

This is the DEV-facing sibling of ``brain/tunables.py`` (the USER-facing
ops-tunables mechanism, backed by ``$KINDLED_HOME/tunables.json`` with an
in-code default as a delete-guard fallback). The two are deliberately
separate, parallel mechanisms:

  * ``tunables.py``      — user-editable, has an in-code fallback default,
                            fails open (a missing/corrupt file silently
                            falls back to the code default).
  * ``prompt_strings.py`` (this module) — developer-editable, defaults live
                            ONLY in ``prompt_strings.toml`` (no in-code
                            fallback), fails CLOSED: a missing key or
                            missing/malformed file is a hard error. These
                            are prompt strings shipped with the code; a
                            break here should be immediately obvious, not
                            silently patched over.

The "setter" is editing ``prompt_strings.toml`` by hand — code never writes
this file (same as ``tunables.py``'s "defaults" section is documentation,
not this file's whole purpose).
"""

from __future__ import annotations

import threading
import tomllib
from pathlib import Path
from typing import Any

_FILE_NAME = "prompt_strings.toml"

_lock = threading.Lock()
_registry: set[str] = set()
_cache: dict[str, Any] | None = None
_cache_mtime: float | None = None


def _file_path() -> Path:
    return Path(__file__).resolve().parent / _FILE_NAME


class PromptStringError(RuntimeError):
    """Raised when a registered/requested prompt-string key cannot be
    resolved — a missing key, a missing file, or a malformed TOML file.
    Deliberately NOT swallowed: this file ships with the code, so a broken
    reference here is a bug that should surface immediately."""


def _load_locked() -> dict[str, Any]:
    """Return the parsed TOML document, re-parsing only when the file's
    mtime changes. Any read/parse failure raises PromptStringError — no
    fail-open behavior (see module docstring)."""
    global _cache, _cache_mtime
    path = _file_path()
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise PromptStringError(f"prompt_strings: cannot stat {path}: {exc}") from exc
    if _cache is not None and mtime == _cache_mtime:
        return _cache
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:  # noqa: BLE001 — re-raised as our own type below
        raise PromptStringError(f"prompt_strings: cannot parse {path}: {exc}") from exc
    _cache, _cache_mtime = data, mtime
    return _cache


def get(key: str) -> str:
    """Return the string value at dotted *key* (e.g. ``"chat.compaction.summary_prompt"``).

    Missing key, wrong type, or unreadable/malformed file all raise
    PromptStringError — a hard error, by design (no code fallback).
    """
    with _lock:
        data = _load_locked()
    node: Any = data
    consumed: list[str] = []
    for part in key.split("."):
        consumed.append(part)
        if not isinstance(node, dict) or part not in node:
            raise PromptStringError(
                f"prompt_strings: missing key {'.'.join(consumed)!r} "
                f"(looking up {key!r} in {_file_path()})"
            )
        node = node[part]
    if not isinstance(node, str):
        raise PromptStringError(
            f"prompt_strings: key {key!r} is not a string (got {type(node).__name__})"
        )
    return node


def register(key: str) -> str:
    """Record that *key* is used by a call site, and return its resolved
    value immediately — so a missing/broken key fails at import time,
    at the call site that declares it, rather than deep in a request."""
    with _lock:
        _registry.add(key)
    return get(key)


def _reset_for_tests() -> None:
    global _cache, _cache_mtime
    with _lock:
        _registry.clear()
        _cache, _cache_mtime = None, None
