"""Platform-aware path resolution for companion-emergence.

All user-facing paths route through this module so we never hard-code
OS-specific locations. Uses platformdirs for the OS-appropriate default
with KINDLED_HOME env var for full override. NELLBRAIN_HOME is
honored as a backwards-compat fallback through the v0.0.13 series
(removed in v0.0.14).
"""

from __future__ import annotations

import os
import re
import warnings
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


def _resolve_home_override() -> str | None:
    """Return the env-var override for the home directory, or None.

    Priority (v0.0.13 transition):
      1. KINDLED_HOME — the canonical var as of v0.0.13.
      2. NELLBRAIN_HOME — backwards-compat fallback, emits
         DeprecationWarning. Removed in v0.0.14.
      3. None — caller uses platformdirs default.
    """
    if v := os.environ.get("KINDLED_HOME"):
        return v
    if v := os.environ.get("NELLBRAIN_HOME"):
        warnings.warn(
            "NELLBRAIN_HOME is deprecated; use KINDLED_HOME. "
            "Backwards-compat fallback will be removed in v0.0.14.",
            DeprecationWarning,
            stacklevel=3,
        )
        return v
    return None


def get_home() -> Path:
    """Root directory for all companion-emergence state.

    Resolution order:
    1. KINDLED_HOME env var if set (supports ~ expansion)
    2. NELLBRAIN_HOME env var (deprecated, warns; removed in v0.0.14)
    3. platformdirs user_data_path for the current OS
    """
    override = _resolve_home_override()
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

    Resolution order matches get_home(): KINDLED_HOME first, then
    NELLBRAIN_HOME (deprecated), then platformdirs.
    """
    override = _resolve_home_override()
    if override:
        return (Path(override).expanduser() / "cache").resolve()
    return _dirs.user_cache_path.resolve()


def get_log_dir() -> Path:
    """Return the log file directory (per-persona bridge logs etc).

    Resolution order matches get_home(): KINDLED_HOME first, then
    NELLBRAIN_HOME (deprecated), then platformdirs.
    """
    override = _resolve_home_override()
    if override:
        return (Path(override).expanduser() / "logs").resolve()
    return _dirs.user_log_path.resolve()
