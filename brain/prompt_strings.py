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
ops-tunables mechanism, backed by ``$KINDLED_HOME/tunables.toml`` with an
in-code default as a delete-guard fallback). The two are deliberately
separate, parallel mechanisms (both read TOML only through
``brain.utils.toml_io``):

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

The "setter" is editing ``prompt_strings.toml`` by hand; code never writes
this file.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from brain.utils import toml_io

_FILE_NAME = "prompt_strings.toml"

_lock = threading.Lock()
_registry: set[str] = set()


def _file_path() -> Path:
    return Path(__file__).resolve().parent / _FILE_NAME


class PromptStringError(RuntimeError):
    """Raised when a registered/requested prompt-string key cannot be
    resolved — a missing key, a missing file, or a malformed TOML file.
    Deliberately NOT swallowed: this file ships with the code, so a broken
    reference here is a bug that should surface immediately."""


def _load_locked() -> dict[str, Any]:
    """Return the parsed TOML document (``toml_io`` re-parses only when the file changes). Any read or parse failure raises
    PromptStringError, naming the file and the failing line: no fail-open behavior (see module docstring)."""
    try:
        return toml_io.read_toml_cached(_file_path())
    except toml_io.TomlFileError as exc:
        raise PromptStringError(f"prompt_strings: cannot read {exc}") from exc


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


def get_segments(key: str) -> list[str]:
    """Return the list-of-strings value at dotted *key*.

    For a straight-line interpolation (``"literal" + var + "literal"`` —
    f-string / concat / ``.format`` with no branching or looping that
    changes which literal text appears), the fixed literal SEGMENTS are
    stored here as an array; the call site reassembles
    ``seg[0] + var_1 + seg[1] + var_2 + ... + seg[n]`` with the same
    variables, in the same order, the original expression interpolated.

    Missing key, wrong type (not a list, or containing a non-string
    element), or unreadable/malformed file all raise PromptStringError —
    a hard error, by design (no code fallback; see module docstring).
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
    if not isinstance(node, list) or not all(isinstance(x, str) for x in node):
        raise PromptStringError(
            f"prompt_strings: key {key!r} is not a list of strings (got {type(node).__name__})"
        )
    return node


def register_segments(key: str) -> list[str]:
    """Record that *key* is used by a call site, and return its resolved
    segment list immediately — the ``get_segments`` counterpart to
    ``register``, same fail-at-import-time rationale."""
    with _lock:
        _registry.add(key)
    return get_segments(key)


def _reset_for_tests() -> None:
    with _lock:
        _registry.clear()
