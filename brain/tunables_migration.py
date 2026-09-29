"""One-time, idempotent migration of the pre-lull idle-tuning tunable keys
into the single ``chat.idle_lull_seconds`` key (ram-spike-fix INC-6, spec §7,
ledger S25/S30/S37/S52/S71).

Retired keys (ram-spike-fix INC-6, C4):
  - ``throttle.background_min_idle_seconds`` (old default 300.0) — carried
    forward as ``chat.idle_lull_seconds``'s override IFF its value differs
    from the old default (a genuine user customization); a plain-default
    override (300.0, indistinguishable from "never customized") is dropped,
    landing on the new code default (600.0) instead of carrying a stale
    number forward as if it were still meaningful.
  - ``chat.pass2_min_idle_seconds`` — removed, value not carried (pass 2 now
    shares the one lull like every other caller).
  - ``self_model.articulate_min_idle_seconds`` — removed, value not carried
    (same reason).
  - ``judge_selftune.gate_handful_decisions`` (S71) — removed, value not
    carried (this key moved to ``brain/dev_constants.py`` in INC-1 and must
    not be a registered tunable at all, C13/C26).

Called synchronously in the bridge lifespan BEFORE the supervisor thread
starts (server.py, S37) — the migration must complete before any code can
read the (possibly still-old) file.

Fail-safe: any error during the rewrite restores the original file from the
backup taken just before writing (if the on-disk file was touched at all)
and logs a WARNING — this module must never raise into bridge startup.
Idempotent: if none of the retired keys are present, this is a no-op (no
write at all, not even a no-op rewrite) — running it twice produces
byte-identical files.

Known residual risk (round-2 code red-team MINOR, accepted, not fixed):
this migration runs against the shared, cross-persona KINDLED_HOME/
tunables.json (S74 only scoped the FTS health check to bridge-only, not
this migration). Two bridge processes for DIFFERENT personas starting at
the same instant could both read the same original bytes before either
writes, racing on the shared `.bak-migrate` sidecar. The write itself stays
safe (atomic temp+os.replace, never a torn file) and the migration is
idempotent (a lost/overwritten write just re-runs identically at the next
bridge start), so the blast radius is self-healing, not data-corrupting —
accepted as a low-probability, bounded residual rather than adding a new
cross-process lock for a one-time startup migration.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FILE_NAME = "tunables.json"
_BACKUP_SUFFIX = ".bak-migrate"

_OLD_THROTTLE_KEY = "throttle.background_min_idle_seconds"
_OLD_THROTTLE_DEFAULT = 300.0
_NEW_LULL_KEY = "chat.idle_lull_seconds"

# Keys whose VALUE is never carried forward — only ever removed.
_DROP_ONLY_KEYS = (
    "chat.pass2_min_idle_seconds",
    "self_model.articulate_min_idle_seconds",
    "judge_selftune.gate_handful_decisions",
)


def migrate_idle_keys(home: Path) -> None:
    """Migrate ``home/tunables.json``'s retired idle-tuning keys in place.

    No-op (no read past the initial stat/parse, no write) when the file is
    absent or none of the retired keys are present in its ``overrides``.
    """
    path = home / _FILE_NAME
    try:
        original_bytes = path.read_bytes()
    except FileNotFoundError:
        return  # nothing to migrate — a fresh persona has no old keys
    except OSError as exc:
        logger.warning("tunables_migration: cannot read %s (%s) — skipping", path, exc)
        return

    try:
        data = json.loads(original_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — corrupt file: leave it for tunables.py's own repair
        logger.warning(
            "tunables_migration: %s unreadable (%s) — skipping (not this module's job to repair)",
            path, exc,
        )
        return

    if not isinstance(data, dict):
        logger.warning("tunables_migration: %s is not a JSON object — skipping", path)
        return

    overrides = data.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}

    retired_keys_present = [_OLD_THROTTLE_KEY, *_DROP_ONLY_KEYS]
    if not any(k in overrides for k in retired_keys_present):
        return  # idempotent no-op — nothing to migrate, no write at all

    new_overrides: dict[str, Any] = {
        k: v for k, v in overrides.items() if k not in retired_keys_present
    }

    if _NEW_LULL_KEY not in overrides and _OLD_THROTTLE_KEY in overrides:
        old_val = overrides[_OLD_THROTTLE_KEY]
        if isinstance(old_val, (int, float)) and not isinstance(old_val, bool) and old_val != _OLD_THROTTLE_DEFAULT:
            new_overrides[_NEW_LULL_KEY] = float(old_val)
        # else: old value equals the old default (or is malformed) — treated
        # as "never customized"; the new code default (600.0) applies, no
        # override key written.

    new_data = dict(data)
    new_data["overrides"] = new_overrides

    backup_path = path.with_name(path.name + _BACKUP_SUFFIX)
    try:
        backup_path.write_bytes(original_bytes)
    except OSError as exc:
        logger.warning(
            "tunables_migration: could not write backup %s (%s) — skipping migration "
            "(original left untouched)",
            backup_path, exc,
        )
        return

    try:
        tmp = path.with_suffix(path.suffix + ".tmp-migrate")
        tmp.write_text(json.dumps(new_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

        # Re-load via the SAME parser tunables.py uses and verify.
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        reloaded_overrides = reloaded.get("overrides", {})
        for k in retired_keys_present:
            if k in reloaded_overrides:
                raise ValueError(f"retired key {k!r} still present after migration")
        if _NEW_LULL_KEY in new_overrides:
            if reloaded_overrides.get(_NEW_LULL_KEY) != new_overrides[_NEW_LULL_KEY]:
                raise ValueError("migrated chat.idle_lull_seconds value did not round-trip")
    except Exception as exc:  # noqa: BLE001 — fail-safe: restore original, never raise
        logger.warning(
            "tunables_migration: migration of %s failed (%s) — restoring original", path, exc,
        )
        try:
            if path.read_bytes() != original_bytes:
                path.write_bytes(original_bytes)
        except OSError as restore_exc:  # noqa: BLE001 — last-resort log only
            logger.error(
                "tunables_migration: could not restore original %s after a failed "
                "migration (%s) — the file may be in a partially-migrated state",
                path, restore_exc,
            )
        return
    finally:
        try:
            backup_path.unlink(missing_ok=True)
        except OSError:
            pass

    # mtime changed under tunables.py's own cache — the next get_tunable()
    # call re-reads (tunables.py:_load_overrides_locked), no extra reset needed.
    logger.info(
        "tunables_migration: migrated %s (%d retired key(s) removed%s)",
        path,
        len(retired_keys_present),
        f", chat.idle_lull_seconds={new_overrides[_NEW_LULL_KEY]!r} carried forward"
        if _NEW_LULL_KEY in new_overrides
        else "",
    )
