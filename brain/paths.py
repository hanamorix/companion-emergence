"""Platform-aware path resolution for companion-emergence.

All user-facing paths route through this module so we never hard-code
OS-specific locations. Uses platformdirs for the OS-appropriate default
with NELLBRAIN_HOME env var for full override.
"""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import PlatformDirs

_APP_NAME = "companion-emergence"
_APP_AUTHOR = "hanamorix"

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

    Raises ValueError if `name` could escape the personas/ root via path
    traversal ('/' or '\\') or break str.format_map prompt rendering
    (literal '{' or '}'), or equals '.' / '..'.
    """
    if not name:
        raise ValueError("Persona name cannot be empty.")
    if "/" in name or "\\" in name or "{" in name or "}" in name or name in (".", ".."):
        raise ValueError(
            f"Invalid persona name: {name!r} "
            "(must not contain '/', '\\\\', '{', '}', or be '.' / '..')."
        )
    return get_home() / "personas" / name


def get_cache_dir() -> Path:
    """Return the cache directory (embeddings, computed matrices, etc)."""
    return _dirs.user_cache_path.resolve()


def get_log_dir() -> Path:
    """Return the log file directory."""
    return _dirs.user_log_path.resolve()
