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
fenced by the user-surface principle and must not gain keys here — with
ONE named exception: ``chat.idle_lull_seconds`` (below, ``CHAT_IDLE_LULL_SECONDS``),
the single ops-timing key that gates every background/cadence caller via
``is_chat_idle`` (brain/bridge/cli_throttle.py). No other ops-timing key
may be registered here (S51/I7/C27).
Fail-open throughout: this module never raises into a turn.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import paths
from brain.health.json_io import JsonFileError, Recovery, read_json, read_json_result, write_json
from brain.utils.file_lock import file_lock
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

_FILE_NAME = "tunables.json"
_NOTICE_FILE_NAME = "tunables.notice.json"

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


# The ONE ops-timing key this module's docstring fence allows (S47/S51/S68/C27):
# the chat-idle lull duration. Read in exactly one place —
# brain/bridge/cli_throttle.py's private `_lull_seconds()`, called only from
# `is_chat_idle` (C4(b)) — never re-exported or read elsewhere.
CHAT_IDLE_LULL_SECONDS: float = register("chat.idle_lull_seconds", 600.0)


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
    # An unreadable or malformed file is read_json's default ({}): the module logs it, fail-open by design. It is also
    # persisted as the "unreadable" notice (once per file change), so a hand edit that breaks the file while the bridge runs
    # is not silently ignored until the next restart (#462). The runtime never heals or rewrites the file.
    res = read_json_result(path, default=dict, expect=dict)
    overrides = res.data.get("overrides", {})
    broken = res.outcome == "defaulted" and res.kind != "empty"  # a zero-byte file carries no overrides
    if isinstance(overrides, dict):
        _warned_file = False
    else:
        if not _warned_file:
            logger.warning("tunables: %s: 'overrides' is not an object — using code defaults", path)
            _warned_file = True
        overrides = {}
        broken = True
    if broken:
        _note_unreadable_at_runtime(path)
    _overrides_cache, _cache_mtime = overrides, mtime  # cache the result (failures too) until the file changes
    return _overrides_cache


def _reset_overrides_cache() -> None:
    """Forget the cached overrides (and the warn-once flag) but keep the registry."""
    global _overrides_cache, _cache_mtime, _warned_file
    with _lock:
        _overrides_cache, _cache_mtime, _warned_file = {}, None, False


def _reset_for_tests() -> None:
    with _lock:
        _registry.clear()
        _warned_keys.clear()
    _reset_overrides_cache()


_README = (
    "Edit 'overrides' only. 'defaults' is rewritten by the brain at boot — "
    "it documents the current code defaults and is never read back."
)


def _overrides_is_object(data: dict) -> None:
    """Reject a file whose `overrides` is present but not an object (absent or null means "no overrides")."""
    overrides = data.get("overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("'overrides' is not an object")


def _notice_path() -> Path:
    return _file_path().with_name(_NOTICE_FILE_NAME)


def _write_marker(record: dict[str, Any]) -> None:
    """Persist the notice beside tunables.json, best effort: a failure is logged and NEVER stops the defaults rewrite."""
    path = _notice_path()
    try:
        with file_lock(path):
            write_json(path, record, indent=2, fsync=False, nonfinite="null")
    except OSError:  # the filesystem (a lock or a write): telling the user must never break bridge startup
        logger.warning(
            "tunables: could not write the notice file %s; the app will not be told about the unreadable tunables.json",
            path, exc_info=True,
        )


def _record_heal(res: Any, overrides: dict[str, Any]) -> None:
    """A heal replaced or restored a damaged tunables.json: remember it for every bridge, and say so in the log once."""
    action = str(res.anomaly.action)
    kind = "restored" if action.startswith("restored") else "reset"
    kept_as = res.anomaly.quarantine_path
    _write_marker({
        "kind": kind,
        "kept_as": kept_as,
        "at": datetime.now(UTC).isoformat(),
        "overrides_at_heal": overrides,
    })
    if kind == "reset":
        logger.warning(
            "tunables: tunables.json could not be read, so your overrides are not in effect; the damaged file was kept as %s",
            kept_as,
        )
    else:
        logger.warning(
            "tunables: tunables.json could not be read, so an older saved copy was put in its place; the damaged file was kept as %s",
            kept_as,
        )


def _record_unreadable(path: Path) -> None:
    """The file could not be read at all (an I/O fault) and was left as it is: remember it for every bridge."""
    try:
        mtime_ns: int | None = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    _write_marker({
        "kind": "unreadable",
        "kept_as": None,
        "at": datetime.now(UTC).isoformat(),
        "mtime_ns": mtime_ns,
    })


def _note_unreadable_at_runtime(path: Path) -> None:
    """The runtime reload found tunables.json unusable: persist the "unreadable" notice once per file change (best effort)."""
    try:
        mtime_ns = path.stat().st_mtime_ns
        marker = _read_marker()
        if marker is not None and marker["kind"] == "unreadable" and marker.get("mtime_ns") == mtime_ns:
            return  # already recorded for this very file state
        _record_unreadable(path)
    except OSError:
        logger.warning("tunables: could not record that %s is unreadable", path, exc_info=True)


def _read_marker() -> dict[str, Any] | None:
    marker = read_json(_notice_path(), default=lambda: None, expect=dict)
    if marker is None or marker.get("kind") not in ("reset", "restored", "unreadable"):
        return None
    return marker


def _marker_is_active(marker: dict[str, Any]) -> bool:
    """Is the user still unaware of / not done with what the marker says?

    reset / restored: the kept-aside file still exists AND the live file's overrides are still the ones written at the heal
    (the user has not edited them). unreadable: the runtime reloads overrides only when the file's mtime changes (it caches a
    failed read until then), so the notice ends only when the mtime changed AND the file now reads cleanly; a permissions fix
    alone does not reload, and a restart's boot drops the marker.
    """
    home = _file_path().parent
    path = _file_path()
    if marker["kind"] in ("reset", "restored"):
        kept = marker.get("kept_as")
        if not isinstance(kept, str) or Path(kept).name != kept or not (home / kept).exists():
            return False
        current = read_json(path, default=lambda: None, expect=dict)
        if current is None:
            return True  # cannot tell: keep telling
        overrides = current.get("overrides")
        return (overrides if isinstance(overrides, dict) else {}) == marker.get("overrides_at_heal", {})
    try:
        now = path.stat().st_mtime_ns
    except OSError:
        return True
    if marker.get("mtime_ns") == now:
        return True
    try:
        read_json(path, recover=Recovery.RAISE, expect=dict)
    except JsonFileError:
        return True
    return False


def current_notice() -> dict[str, Any] | None:
    """The persisted notice about tunables.json, or None. A PURE READ: never writes, never raises.

    Called on every `/persona/state` poll by every bridge, so a fact recorded by whichever bridge healed the file reaches the
    user whichever persona they are looking at.
    """
    try:
        marker = _read_marker()
        if marker is None or not _marker_is_active(marker):
            return None
        return {"kind": marker["kind"], "kept_as": marker.get("kept_as")}
    except Exception:  # noqa: BLE001 — the state endpoint must never fail on this
        warn_throttled(logger, "tunables:notice", "tunables: could not read the notice file; reporting none", exc_info=True)
        return None


def _drop_inactive_marker() -> None:
    """Delete the marker once it is inactive. Re-evaluated under the marker lock so a newer marker is never deleted."""
    try:
        marker = _read_marker()
        if marker is None or _marker_is_active(marker):
            return
        with file_lock(_notice_path()):
            marker = _read_marker()
            if marker is not None and not _marker_is_active(marker):
                _notice_path().unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 — housekeeping must never break bridge startup
        logger.warning("tunables: could not clear the notice file", exc_info=True)


def write_defaults_section() -> None:
    """Rewrite the defaults section from the registry; preserve overrides.

    Called once at bridge boot. Creates the file if missing. Atomic write.
    Fail-open: any error is logged and swallowed — boot must not break.

    A damaged tunables.json is healed, not silently rebuilt (#462): HEAL keeps the damaged bytes as
    ``tunables.json.corrupt-<ts>`` (or restores a valid backup), and the fact is persisted in ``tunables.notice.json``
    for every bridge to report. An I/O fault leaves the file as it is and is persisted as an ``unreadable`` notice.
    """
    path = _file_path()
    try:
        try:
            res = read_json_result(
                path, default=dict, recover=Recovery.HEAL, expect=dict, validate=_overrides_is_object
            )
        except JsonFileError:
            _record_unreadable(path)  # the file is left as it is; the raise goes to the handler below
            raise
        data = res.data
        overrides: dict[str, Any] = data["overrides"] if isinstance(data.get("overrides"), dict) else {}
        if res.anomaly is not None and res.kind != "empty":  # a zero-byte file carried no overrides: nothing to tell
            _record_heal(res, overrides)
        with _lock:
            defaults = dict(_registry)
        payload = {"_readme": _README, "defaults": defaults, "overrides": overrides}
        write_json(path, payload, indent=2, sort_keys=True, fsync=False, nonfinite="null")
        _drop_inactive_marker()
    except Exception:  # noqa: BLE001 — never break bridge startup
        logger.exception("tunables: write_defaults_section failed (ignored)")
