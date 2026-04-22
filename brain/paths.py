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

_dirs = PlatformDirs(appname=_APP_NAME, appauthor=_APP_AUTHOR)


def get_home() -> Path:
    """Root directory for all companion-emergence state.

    Resolution order:
    1. NELLBRAIN_HOME env var if set (supports ~ expansion)
    2. platformdirs user_data_dir for the current OS
    """
    override = os.environ.get("NELLBRAIN_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return Path(_dirs.user_data_dir)


def get_persona_dir(name: str) -> Path:
    """Return the directory for a specific persona's private data."""
    return get_home() / "personas" / name


def get_cache_dir() -> Path:
    """Return the cache directory (embeddings, computed matrices, etc)."""
    return Path(_dirs.user_cache_dir)


def get_log_dir() -> Path:
    """Return the log file directory."""
    return Path(_dirs.user_log_dir)
