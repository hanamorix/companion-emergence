"""Platform-aware path resolution for companion-emergence.

All user-facing paths route through this module so we never hard-code
OS-specific locations. Uses platformdirs for the OS-appropriate default
with NELLBRAIN_HOME env var for full override.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from platformdirs import PlatformDirs

_APP_NAME = "companion-emergence"
_APP_AUTHOR = "hanamorix"

_PERSONA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def validate_persona_name(name: str) -> None:
    """Raise ValueError if `name` would land outside <home>/personas/.

    Persona names become directory names. Reject anything that could
    escape the personas/ root (slashes, dotdot, empty, oversize) or
    break str.format_map prompt rendering (literal '{' / '}'). The
    grammar is ``[A-Za-z0-9_-]{1,40}`` — no slashes, dots, spaces, or
    special characters. This is the single shared validator used by
    both :func:`get_persona_dir` and ``brain.setup.validate_persona_name``
    (which re-exports it).
    """
    if not isinstance(name, str) or not _PERSONA_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid persona name {name!r} — must match "
            f"[A-Za-z0-9_-]{{1,40}} (no slashes, dots, or spaces)"
        )

# PlatformDirs properties are evaluated lazily (env vars read at
# property-access time, not at construction), so module-level
# instantiation is safe under monkeypatching in tests.
_dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR)


def get_home() -> Path:
    """Root directory for all companion-emergence state.

    Resolution order:
    1. NELLBRAIN_HOME env var if set (supports ~ expansion)
    2. platformdirs user_data_path for the current OS

    Both branches return a fully resolved canonical path so symlinks
    collapse consistently (matters on Linux CI runners with symlinked
    home dirs).
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return _dirs.user_data_path.resolve()


def get_persona_dir(name: str) -> Path:
    """Return the directory for a specific persona's private data.

    Delegates name validation to :func:`validate_persona_name` — the
    single source of truth for the persona-name grammar, shared with
    ``brain.setup``. Raises ValueError on any input that doesn't match
    ``[A-Za-z0-9_-]{1,40}``.
    """
    validate_persona_name(name)
    return get_home() / "personas" / name


def get_cache_dir() -> Path:
    """Return the cache directory (embeddings, computed matrices, etc).

    Resolution order matches get_home():
    1. NELLBRAIN_HOME / "cache" if NELLBRAIN_HOME is set (supports ~ expansion)
    2. platformdirs user_cache_path for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return (Path(override).expanduser() / "cache").resolve()
    return _dirs.user_cache_path.resolve()


def get_log_dir() -> Path:
    """Return the log file directory (per-persona bridge logs etc).

    Resolution order matches get_home():
    1. NELLBRAIN_HOME / "logs" if NELLBRAIN_HOME is set (supports ~ expansion)
    2. platformdirs user_log_path for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return (Path(override).expanduser() / "logs").resolve()
    return _dirs.user_log_path.resolve()
