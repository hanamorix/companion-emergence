"""Backward compatibility for the old JSON settings file (#529). Permanent: there is no way to know which version an install
is upgrading from, now or later.

Before ``tunables.toml`` the install's settings lived in ``$KINDLED_HOME/tunables.json`` (an ``overrides`` object next to a
``defaults`` section that was only documentation). ``migrate_legacy_json(home, complete_registry=...)`` brings such a file
forward the first time it is seen, and a JSON that turns up later (a restored backup, a copied install, imported data) is
folded the same way:

1. No JSON file: nothing happens (no lock, no write).
2. ``tunables.sync_file()`` runs first, so the TOML exists with every default and its recorded default.
3. The JSON is read with ``Recovery.RAISE``. One that cannot be read (syntax error, not UTF-8, a persisting I/O fault) is
   LEFT ALONE exactly like an unreadable TOML (owner ruling E9): not repaired, not renamed, one throttled WARNING per start
   naming the line that failed; the next start tries again. A zero-byte file or one with nothing in ``overrides`` carries
   nothing and is renamed without a warning.
4. The older retirement of the pre-lull idle keys is applied (below), then every value the TOML cannot hold (null, NaN or
   Infinity, a string with a lone surrogate, a nested object) is skipped by key with a log, so one poison value never blocks
   the others or the rename.
5. The bridge (``complete_registry=True``: it imports every module) applies each registered key with
   ``tunables.set_values(..., only_if_default=True)``: a key whose TOML value still equals its default takes the old value, a
   key the user already changed keeps its value and the old one is logged and discarded. A key this process does not register
   is retired code: logged and DISCARDED, never inserted. A CLI process (``complete_registry=False``) registers only the keys
   it imported, so it folds ALL-OR-NOTHING: if every key is registered it folds exactly as the bridge does, otherwise it
   applies nothing, leaves the JSON and logs once at INFO that the bridge will fold it. A CLI that had applied the keys it
   could would later re-apply them over a change the user made in between.
6. The JSON is renamed ``tunables.json.migrated-<UTC stamp>`` (the cheap backup, never deleted by the app) after step 5
   completed. A crash between 5 and 6 makes the next run re-apply with ``only_if_default`` (values already equal: a no-op) and
   rename. If step 5 raises (lock timeout, unreadable TOML) the JSON is not renamed and the next start retries.

Retired keys, kept from the earlier retirement (ram-spike-fix INC-6): ``throttle.background_min_idle_seconds`` (old default
300.0) carries to ``chat.idle_lull_seconds`` iff its value differs from 300.0; ``chat.pass2_min_idle_seconds``,
``self_model.articulate_min_idle_seconds`` and ``judge_selftune.gate_handful_decisions`` are discarded.

Concurrency: the JSON is read before any lock and never modified, only renamed; every write goes through
``tunables.set_values`` (the tunables lock, bounded wait), so a concurrent ``set_values`` lands before or after the fold's
write and is never lost, and the fold never rewrites the whole file. Nothing here raises into startup.
"""
from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from brain import tunables
from brain.health.json_io import JsonFileError, Recovery, read_json_result
from brain.utils import toml_io
from brain.utils.log_throttle import warn_throttled

logger = logging.getLogger(__name__)

LEGACY_FILE_NAME = "tunables.json"  # the old settings file this module folds; the one place the name is spelled

_OLD_THROTTLE_KEY = "throttle.background_min_idle_seconds"
_OLD_THROTTLE_DEFAULT = 300.0
_NEW_LULL_KEY = "chat.idle_lull_seconds"

# Keys whose VALUE is never carried forward, only ever dropped.
_DROP_ONLY_KEYS = (
    "chat.pass2_min_idle_seconds",
    "self_model.articulate_min_idle_seconds",
    "judge_selftune.gate_handful_decisions",
)
_RETIRED_KEYS = [_OLD_THROTTLE_KEY, *_DROP_ONLY_KEYS]


def _after_retirements(overrides: dict[str, Any]) -> dict[str, Any]:
    """The overrides without the retired idle keys; the old throttle value carries to the lull key when it was customized."""
    carried = {k: v for k, v in overrides.items() if k not in _RETIRED_KEYS}
    if _NEW_LULL_KEY not in overrides and _OLD_THROTTLE_KEY in overrides:
        old = overrides[_OLD_THROTTLE_KEY]
        if isinstance(old, (int, float)) and not isinstance(old, bool) and old != _OLD_THROTTLE_DEFAULT:
            carried[_NEW_LULL_KEY] = float(old)
        # else: the old default (or malformed) means "never customized": the new code default applies.
    return carried


def _nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return any(_nonfinite(v) for v in value)
    return False


def _cannot_hold(value: Any) -> str | None:
    """Why the TOML cannot hold this value, or None when it can."""
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "a nested object"
    if _nonfinite(value):
        return "not a finite number"
    try:
        toml_io.dumps_value(value)
    except toml_io.TomlFileError as exc:
        return exc.reason or "not a value the file can hold"
    return None


def _archive(legacy: Path) -> None:
    """Rename the folded JSON to ``tunables.json.migrated-<stamp>``; never overwrites an earlier copy."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = legacy.with_name(f"{legacy.name}.migrated-{stamp}")
    n = 1
    while target.exists():
        target = legacy.with_name(f"{legacy.name}.migrated-{stamp}-{n}")
        n += 1
    try:
        legacy.rename(target)
    except FileNotFoundError:
        return  # another bridge folded it first
    except OSError as exc:
        logger.warning("tunables_migration: could not rename %s (%s); its values are already in tunables.toml", legacy, exc)
        return
    logger.info("tunables_migration: folded %s into tunables.toml; the old file was kept as %s", legacy.name, target.name)


def _fold(legacy: Path, complete_registry: bool) -> None:
    tunables.sync_file()  # the TOML exists with every default and its recorded default before anything is applied
    try:
        res = read_json_result(legacy, default=dict, recover=Recovery.RAISE, expect=dict)
    except JsonFileError as exc:
        if exc.kind == "missing":
            return  # folded by another process meanwhile
        if exc.kind == "empty":
            _archive(legacy)  # a zero-byte file carries nothing
            return
        where = f" at line {exc.line}" if exc.line is not None else ""
        warn_throttled(
            logger,
            f"tunables_migration:unreadable:{legacy}",
            "tunables_migration: %s could not be read%s (%s); it was left as it is and is tried again at the next start",
            legacy, where, exc.detail or exc.kind,
            exc_info=False,
        )
        return

    raw = res.data.get("overrides")
    overrides = raw if isinstance(raw, dict) else {}
    carried: dict[str, Any] = {}
    for key, value in _after_retirements(overrides).items():
        reason = _cannot_hold(value)
        if reason is not None:
            logger.info("tunables_migration: %s in %s skipped (%s); the other settings are still carried", key, legacy.name, reason)
            continue
        carried[key] = value

    registered = tunables.registered_keys()
    known = {k: v for k, v in carried.items() if k in registered}
    unknown = sorted(set(carried) - set(known))
    if unknown and not complete_registry:
        logger.info(
            "tunables_migration: %s holds settings this command cannot place (%s); nothing was applied and the bridge folds it "
            "at its next start",
            legacy.name, ", ".join(unknown),
        )
        return
    for key in unknown:
        logger.info("tunables_migration: %s in %s is no longer a setting and is discarded", key, legacy.name)
    if known:
        try:
            written = tunables.set_values(known, only_if_default=True)
        except tunables.TunablesWriteError as exc:
            logger.warning("tunables_migration: could not fold %s (%s); tried again at the next start", legacy.name, exc)
            return
        for key, value in known.items():
            if key not in written and not toml_io.same_value(tunables.get_raw(key), value):
                logger.info(
                    "tunables_migration: %s was changed in tunables.toml already, so the old value %r in %s is discarded",
                    key, value, legacy.name,
                )
    _archive(legacy)


def migrate_legacy_json(home: Path, *, complete_registry: bool) -> None:
    """Fold ``home/tunables.json`` into ``tunables.toml`` (see the module docstring). Never raises.

    ``complete_registry`` is True where every module that registers a setting has been imported (the bridge) and False for
    a CLI process, which folds all-or-nothing.
    """
    legacy = home / LEGACY_FILE_NAME
    if not legacy.exists():
        return  # one exists() call per start; no lock, no write
    try:
        _fold(legacy, complete_registry)
    except Exception:  # noqa: BLE001 - startup and the CLI must never fail on this
        logger.warning("tunables_migration: folding %s failed (ignored; tried again at the next start)", legacy, exc_info=True)
