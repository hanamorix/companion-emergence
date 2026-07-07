"""Ops-tier tunables — centralized overrides for operational constants.

Spec: docs/superpowers/specs/2026-07-04-ops-tunables-design.md

Call sites register their key at import time and read at call time:

    _MY_TIMEOUT: float = tunables.register("area.my_timeout_seconds", 60.0)
    ...
    timeout = tunables.get_tunable("area.my_timeout_seconds", _MY_TIMEOUT)

Overrides live in $KINDLED_HOME/tunables.json under "overrides".
The "defaults" section is documentation, rewritten at bridge boot
(write_defaults_section) and never read back — this avoids the
frozen-defaults trap where a seeded value shadows a later code change.

Ops tier ONLY. Physiology (forgetting, emotion, salience, cadences) is
fenced by the user-surface principle and must not gain keys here.
Fail-open throughout: this module never raises into a turn.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from brain import paths

logger = logging.getLogger(__name__)

_FILE_NAME = "tunables.json"

_lock = threading.Lock()
_registry: dict[str, Any] = {}
_overrides_cache: dict[str, Any] = {}
_cache_mtime: float | None = None
_warned_file = False
_warned_keys: set[str] = set()


def _file_path() -> Path:
    return paths.get_home() / _FILE_NAME


def register(key: str, default: Any) -> Any:
    """Record (key, default) in the registry; return default unchanged."""
    with _lock:
        _registry[key] = default
    return default


def get_tunable(key: str, default: Any) -> Any:
    """Return the override for *key* if present and type-compatible, else *default*."""
    with _lock:
        _registry.setdefault(key, default)
        overrides = _load_overrides_locked()
    if key not in overrides:
        return default
    value = overrides[key]
    if not _type_ok(default, value):
        _warn_key(key, default, value)
        return default
    if isinstance(default, float) and isinstance(value, int):
        return float(value)
    return value


def _type_ok(default: Any, value: Any) -> bool:
    # bool is an int subclass — check it first and strictly, both directions.
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if isinstance(default, float):
        return isinstance(value, (int, float))
    if isinstance(default, int):
        return isinstance(value, int)
    if isinstance(default, str):
        return isinstance(value, str)
    return True


def _warn_key(key: str, default: Any, value: Any) -> None:
    with _lock:
        if key in _warned_keys:
            return
        _warned_keys.add(key)
    logger.warning(
        "tunables: override %r=%r has wrong type (expected %s) — ignored",
        key, value, type(default).__name__,
    )


def _load_overrides_locked() -> dict[str, Any]:
    """Return the overrides dict, re-parsing only when the file mtime changes."""
    global _overrides_cache, _cache_mtime, _warned_file
    path = _file_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _overrides_cache, _cache_mtime = {}, None
        return _overrides_cache
    if mtime == _cache_mtime:
        return _overrides_cache
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        overrides = data.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("'overrides' is not an object")
        _overrides_cache, _cache_mtime = overrides, mtime
        _warned_file = False
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        if not _warned_file:
            logger.warning("tunables: cannot read %s (%s) — using code defaults", path, exc)
            _warned_file = True
        _overrides_cache, _cache_mtime = {}, mtime  # cache the failure until the file changes
    return _overrides_cache


def _reset_for_tests() -> None:
    global _overrides_cache, _cache_mtime, _warned_file
    with _lock:
        _registry.clear()
        _warned_keys.clear()
        _overrides_cache, _cache_mtime, _warned_file = {}, None, False


_README = (
    "Edit 'overrides' only. 'defaults' is rewritten by the brain at boot — "
    "it documents the current code defaults and is never read back."
)


def write_defaults_section() -> None:
    """Rewrite the defaults section from the registry; preserve overrides.

    Called once at bridge boot. Creates the file if missing. Atomic write.
    Fail-open: any error is logged and swallowed — boot must not break.
    """
    path = _file_path()
    try:
        overrides: dict[str, Any] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("overrides"), dict):
                overrides = data["overrides"]
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 — corrupt file: rebuild, drop nothing readable
            logger.warning("tunables: %s unreadable at boot (%s) — rebuilding", path, exc)
        with _lock:
            defaults = dict(_registry)
        payload = {"_readme": _README, "defaults": defaults, "overrides": overrides}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 — never break bridge startup
        logger.exception("tunables: write_defaults_section failed (ignored)")
