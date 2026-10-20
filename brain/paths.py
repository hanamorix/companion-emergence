"""Platform-aware path resolution for companion-emergence.

All user-facing paths route through this module so we never hard-code
OS-specific locations. Uses platformdirs for the OS-appropriate default
with KINDLED_HOME env var for full override. NELLBRAIN_HOME is
honored as a backwards-compat fallback (deprecated; back-compat
fallback retained).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Mapping
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


def _override_from(env: Mapping[str, str], *, warn: bool) -> str | None:
    """The env-var override for the home directory in ``env``, or None.

    Priority (v0.0.13 transition):
      1. KINDLED_HOME — the canonical var as of v0.0.13.
      2. NELLBRAIN_HOME — backwards-compat fallback, emits DeprecationWarning (``warn``). Deprecated; back-compat
         fallback retained.
      3. None — caller uses the platform default.
    """
    if v := env.get("KINDLED_HOME"):
        return v
    if v := env.get("NELLBRAIN_HOME"):
        if warn:
            warnings.warn(
                "NELLBRAIN_HOME is deprecated; use KINDLED_HOME. "
                "Backwards-compat fallback is retained but may be removed in a future release.",
                DeprecationWarning,
                stacklevel=4,
            )
        return v
    return None


def _resolve_home_override() -> str | None:
    """This process's home override (see :func:`_override_from`)."""
    return _override_from(os.environ, warn=True)


def home_for_env(
    env: Mapping[str, str], platform: str | None = None, *, current: bool = False
) -> Path | None:
    """THE home rule, for this process (``current=True``) or for another process whose environment is ``env``.

    Resolution order:
    1. KINDLED_HOME if set (supports ~ expansion)
    2. NELLBRAIN_HOME (deprecated; warns for this process; back-compat fallback retained)
    3. the platform default: platformdirs' user_data_path for this process; for another environment the same
       default computed from that environment (Linux XDG_DATA_HOME or ~/.local/share, macOS ~/Library/Application
       Support, Windows LOCALAPPDATA\\<author>), None when ``env`` lacks what it needs. A test checks the computed
       default against platformdirs on every platform (#456 review S15: one rule, no drift).
    """
    override = _override_from(env, warn=current)
    if override:
        return Path(override).expanduser()
    if current:
        return _dirs.user_data_path
    platform = platform or sys.platform
    if platform.startswith("linux"):
        base = env.get("XDG_DATA_HOME") or (os.path.join(env["HOME"], ".local", "share") if env.get("HOME") else None)
        return Path(base, _APP_NAME) if base else None
    if platform == "darwin":
        home = env.get("HOME")
        return Path(home, "Library", "Application Support", _APP_NAME) if home else None
    if platform == "win32":
        base = env.get("LOCALAPPDATA")
        return Path(base, _APP_AUTHOR, _APP_NAME) if base else None
    return None


def get_home() -> Path:
    """Root directory for all companion-emergence state: :func:`home_for_env` for this process."""
    home = home_for_env(os.environ, current=True)
    assert home is not None  # current=True always yields a path
    return home.resolve()


def get_persona_dir(name: str) -> Path:
    """Return the directory for a specific persona's private data.

    Delegates name validation to :func:`validate_persona_name` — the
    single source of truth for the persona-name grammar, shared with
    ``brain.setup``. Raises ValueError on any input that doesn't match
    ``[A-Za-z0-9_-]{1,40}``.
    """
    validate_persona_name(name)
    return get_home() / "personas" / name


def list_persona_names() -> list[str]:
    """Return the sorted list of valid persona names installed under
    <home>/personas/.

    A name is "valid" iff:
      - the entry is a directory
      - the entry's basename matches the persona-name grammar
        ([A-Za-z0-9_-]{1,40})

    Missing personas/ root returns [] rather than raising.
    """
    root = get_home() / "personas"
    if not root.is_dir():
        return []
    names: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not _PERSONA_NAME_RE.fullmatch(entry.name):
            continue
        names.append(entry.name)
    names.sort()
    return names


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


_CADENCE_SUBDIR = "cadence"


def cadence_state_path(persona_dir: Path, filename: str) -> Path:
    """Resolve a persona-scoped cadence state file under ``<persona>/cadence/``.

    One resolver for every persisted cadence (the generic
    ``brain/bridge/persisted_cadence.py`` files plus the soul and self-model
    bespoke ones) so the layout lives in one place (#178). A legacy root-level
    ``<persona>/<filename>`` from before the subdirectory existed is moved into
    place the first time it is resolved, unless a file already exists at the
    new path (then the legacy copy is left alone rather than silently lost).
    Migration is best-effort: on any OSError the new path is still returned
    and the caller's missing-file handling (due-now) takes over.
    """
    new = persona_dir / _CADENCE_SUBDIR / filename
    legacy = persona_dir / filename
    if not new.exists() and legacy.is_file():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            legacy.replace(new)
        except OSError:
            warnings.warn(
                f"cadence_state_path: could not migrate {legacy} → {new}; "
                "cadence will start due-now",
                RuntimeWarning,
                stacklevel=2,
            )
    return new


def presence_state_path(persona_dir: Path) -> Path:
    """Resolve the persona-scoped ``presence_state.json`` sidecar path (#225).

    One resolver for this file, mirroring ``cadence_state_path``'s "one
    resolver, one place" convention. Unlike the cadence files, this sidecar
    lives directly under ``<persona_dir>/`` rather than the ``cadence/``
    subdirectory: it holds live signal state (last-seen timestamp, the
    active-hour histogram, the reply-lag running mean), not a next-due
    cadence pointer.
    """
    return persona_dir / "presence_state.json"
